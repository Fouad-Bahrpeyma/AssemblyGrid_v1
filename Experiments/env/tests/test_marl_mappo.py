"""Tests for the MAPPO baseline (Sec. 10, item 6)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "marl"))

import numpy as np
import torch

from core import EnvConfig
from common import collect_rollout, global_critic_input, make_env, ppo_update
from encoding import global_state_dim
from ippo import IPPOActor
from mappo import MAPPOCritic, train_mappo


def _tiny_cfg():
    return EnvConfig(M=2, N=3, spawn_interval=3, max_wip=3, motion_planning=False,
                     horizon_T=60, recipe_ids=("solo_inspection",), seed=0)


def test_mappo_critic_is_shared_across_agents_at_a_given_timestep():
    """The whole point of a centralized critic: every agent should see the
    identical global-state input at a given timestep, unlike IPPO where each
    agent's local observation differs."""
    cfg = _tiny_cfg()
    env = make_env(cfg)
    critic_inputs = global_critic_input(env, env._last_obs)
    values = list(critic_inputs.values())
    for v in values[1:]:
        assert np.array_equal(v, values[0])
    assert values[0].shape == (global_state_dim(env.core),)


def test_collect_rollout_and_ppo_update_work_with_centralized_critic():
    cfg = _tiny_cfg()
    env = make_env(cfg)
    actor = IPPOActor()
    critic = MAPPOCritic(global_state_dim(env.core))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    dev = torch.device("cpu")

    buffers, obs, bootstrap, true_reward = collect_rollout(
        env, actor, critic, global_critic_input, env._last_obs, 40, dev)
    stats = ppo_update(actor, critic, actor_opt, critic_opt, buffers, bootstrap, dev)
    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["critic_loss"])


def test_train_mappo_end_to_end_smoke():
    cfg = _tiny_cfg()
    actor, critic, history = train_mappo(cfg, total_steps=400, steps_per_update=100, seed=0)
    assert len(history) > 0
    for h in history:
        assert np.isfinite(h["actor_loss"])
        assert np.isfinite(h["critic_loss"])


def test_mappo_actor_input_is_local_even_when_critic_is_global():
    """Runtime spy: collect_rollout must feed the actor the canonical local
    observation batch, never the privileged global-state critic batch."""
    from encoding import OBS_DIM

    class SpyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from core import NUM_ACTIONS
            self.bias = torch.nn.Parameter(torch.zeros(NUM_ACTIONS))
            self.seen = []

        def forward(self, x):
            self.seen.append(x.detach().cpu().numpy().copy())
            return self.bias.unsqueeze(0).expand(x.shape[0], -1)

    cfg = _tiny_cfg()
    env = make_env(cfg)
    initial_agents = list(env.agents)
    initial_local = np.stack([env._last_obs[a] for a in initial_agents])
    actor = SpyActor()
    critic = MAPPOCritic(global_state_dim(env.core))
    collect_rollout(env, actor, critic, global_critic_input, env._last_obs, 1, torch.device("cpu"))
    assert actor.seen
    assert actor.seen[0].shape[1] == OBS_DIM
    assert np.array_equal(actor.seen[0], initial_local)
    assert critic.net[0].in_features == global_state_dim(env.core)
    assert actor.seen[0].shape[1] != critic.net[0].in_features
