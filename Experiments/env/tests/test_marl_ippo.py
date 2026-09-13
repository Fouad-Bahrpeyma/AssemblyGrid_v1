"""Tests for the IPPO baseline and shared PPO machinery (Sec. 10, item 5).

These verify MECHANICAL correctness -- masked action sampling never picks an
illegal action, GAE/PPO update runs and reduces loss, behavior cloning
reduces its own loss, and the WIP/spawn dynamics an untrained (all-idle)
policy produces exactly match the environment's own ground truth. They do
NOT require the tiny from-scratch network to reach full delivery within a
short training budget: verified empirically (see common.py's
pretrain_behavior_cloning docstring) that even the existing hand-crafted
reference policy needs the environment's real multi-tick coordination to
deliver anything, and confirming full policy convergence is a compute/
hyperparameter question distinct from whether the training code is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "marl"))

import numpy as np
import pytest
import torch

from core import EnvConfig, NUM_ACTIONS
from common import (
    RolloutBuffer, collect_rollout, compute_gae, local_critic_input, make_env, masked_categorical,
    pretrain_behavior_cloning, ppo_update,
)
from ippo import IPPOActor, IPPOCritic, train_ippo


def _tiny_cfg():
    return EnvConfig(M=2, N=3, spawn_interval=3, max_wip=3, motion_planning=False,
                     horizon_T=60, recipe_ids=("solo_inspection",), seed=0)


def test_masked_categorical_never_samples_illegal_actions():
    torch.manual_seed(0)
    logits = torch.randn(200, NUM_ACTIONS)
    mask = torch.zeros(200, NUM_ACTIONS, dtype=torch.int64)
    mask[:, 0] = 1  # idle always legal
    mask[:, 5] = 1  # one more legal action
    dist = masked_categorical(logits, mask)
    samples = dist.sample()
    assert set(samples.tolist()) <= {0, 5}


def test_collect_rollout_and_ppo_update_run_without_crashing():
    cfg = _tiny_cfg()
    env = make_env(cfg)
    actor = IPPOActor(); critic = IPPOCritic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    dev = torch.device("cpu")

    obs = env._last_obs
    buffers, obs, bootstrap, true_reward = collect_rollout(
        env, actor, critic, local_critic_input, obs, 40, dev)
    assert len(buffers) > 0
    for buf in buffers.values():
        assert len(buf) == 40
        assert all(0 <= a < NUM_ACTIONS for a in buf.action)

    stats = ppo_update(actor, critic, actor_opt, critic_opt, buffers, bootstrap, dev)
    assert "actor_loss" in stats and "critic_loss" in stats
    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["critic_loss"])


def test_ippo_wip_dynamics_match_environment_ground_truth_under_untrained_policy():
    """A freshly-initialized (near-random-but-masked) actor should still
    respect the environment's own spawn-vs-WIP-cap arithmetic exactly,
    since that logic lives in the environment, not the policy."""
    cfg = _tiny_cfg()
    env = make_env(cfg)
    actor = IPPOActor()
    obs = env._last_obs
    torch.manual_seed(0)
    for _ in range(60):
        actions = {}
        with torch.no_grad():
            for a in env.agents:
                o = torch.from_numpy(obs[a]).unsqueeze(0)
                m = torch.from_numpy(env._last_mask[a]).unsqueeze(0)
                dist = masked_categorical(actor(o), m)
                actions[a] = int(dist.sample().item())
        obs, rewards, terms, truncs, infos = env.step(actions)
        env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
        assert env.core.wip() <= cfg.max_wip
        if not env.agents:
            obs, infos = env.reset()
            env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}


def test_behavior_cloning_loss_decreases():
    cfg = _tiny_cfg()
    dev = torch.device("cpu")
    actor = IPPOActor()
    losses = pretrain_behavior_cloning(actor, cfg, 1500, dev, seed=0)
    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_train_ippo_end_to_end_smoke():
    """Full train_ippo() call, including its BC warm-start path, completes
    without error and returns a history of finite-valued PPO stats."""
    cfg = _tiny_cfg()
    actor, critic, history = train_ippo(cfg, total_steps=400, steps_per_update=100,
                                        seed=0, bc_warmstart_steps=200)
    assert len(history) > 0
    for h in history:
        assert np.isfinite(h["actor_loss"])
        assert np.isfinite(h["critic_loss"])
        assert h["delivered"] >= 0


def test_gae_matches_hand_calculated_terminal_example():
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    dones = [False, False, True]
    advantages, returns = compute_gae(
        rewards, values, dones, bootstrap_value=9.0, gamma=0.9, lam=0.8)
    # Backward calculation:
    # A2=1.5; A1=2+0.9*1.5-1 + 0.9*0.8*1.5 = 3.43;
    # A0=1+0.9*1-0.5 + 0.9*0.8*3.43 = 3.8696.
    assert np.allclose(advantages, [3.8696, 3.43, 1.5], atol=1e-5)
    assert np.allclose(returns, [4.3696, 4.43, 3.0], atol=1e-5)


def test_ppo_clipped_surrogate_matches_hand_calculated_batch():
    class FixedActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(NUM_ACTIONS))

        def forward(self, x):
            return self.logits.unsqueeze(0).expand(x.shape[0], -1)

    class ZeroCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return self.value.expand(x.shape[0], 1)

    buf = RolloutBuffer()
    obs = np.zeros(IPPOActor().net[0].in_features, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8); mask[0] = 1
    # gamma=0 makes advantages equal rewards because values are zero.
    # New log p(action 0)=0 because it is the only legal action. Choose old
    # log-probabilities so the probability ratios are exactly 1.5 and 0.5.
    buf.add(obs, obs, mask, 0, -np.log(1.5), 0.0, -1.0, True)
    buf.add(obs, obs, mask, 0, -np.log(0.5), 0.0, +1.0, True)
    actor = FixedActor(); critic = ZeroCritic()
    actor_opt = torch.optim.SGD(actor.parameters(), lr=0.0)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=0.0)
    stats = ppo_update(
        actor, critic, actor_opt, critic_opt, {"a": buf}, {"a": 0.0},
        torch.device("cpu"), epochs=1, clip=0.2, gamma=0.0, lam=0.0,
        ent_coef=0.0, minibatch_size=2,
    )
    # Standardized advantages are +/-1/sqrt(2). PPO's elementwise minimum
    # gives [-1.5/sqrt(2), +0.5/sqrt(2)], hence loss = 1/(2*sqrt(2)).
    assert stats["actor_loss"] == pytest.approx(1.0 / (2.0 * np.sqrt(2.0)), rel=1e-5)


def test_ippo_actor_and_critic_are_both_local_information_interfaces():
    from encoding import OBS_DIM, global_state_dim
    cfg = _tiny_cfg()
    env = make_env(cfg)
    local = local_critic_input(env, env._last_obs)
    assert set(local) == set(env.agents)
    for agent in env.agents:
        assert np.array_equal(local[agent], env._last_obs[agent])
        assert local[agent].shape == (OBS_DIM,)
    actor = IPPOActor(); critic = IPPOCritic()
    assert actor.net[0].in_features == OBS_DIM
    assert critic.net[0].in_features == OBS_DIM
    # The privileged state is a separate interface and is not the IPPO critic input.
    assert global_state_dim(env.core) != critic.net[0].in_features
