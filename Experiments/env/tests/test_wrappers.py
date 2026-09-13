"""Wrapper smoke tests.  They skip cleanly when optional API deps are absent."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")

from core import EnvConfig, NUM_ACTIONS
from encoding import OBS_DIM, global_state_dim
from gym_env import AssemblyGridGymEnv
from pettingzoo_env import AssemblyGridParallelEnv
from policies import Tier1Policy


def test_gym_env_exposes_true_global_state_and_joint_masks():
    cfg = EnvConfig(M=4, N=6, seed=1, episode_mode="horizon", horizon_T=20,
                    motion_planning=False, trajectory_conflicts=False)
    env = AssemblyGridGymEnv(cfg)
    obs, info = env.reset(seed=1)
    assert obs.shape == (global_state_dim(env.core),)
    assert obs.shape != (cfg.M * cfg.N * OBS_DIM,)
    assert info["action_masks"].shape == (cfg.M * cfg.N, NUM_ACTIONS)
    for _ in range(20):
        masks = info["action_masks"]
        action = np.asarray([np.flatnonzero(m)[0] for m in masks], dtype=int)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert env.core.t == cfg.horizon_T


def test_pettingzoo_env_shapes_and_horizon_truncation():
    cfg = EnvConfig(M=4, N=6, seed=1, episode_mode="horizon", horizon_T=30,
                    motion_planning=False, trajectory_conflicts=False)
    env = AssemblyGridParallelEnv(cfg)
    obs, infos = env.reset(seed=1)
    assert len(env.agents) == cfg.M * cfg.N
    for a in env.agents:
        assert obs[a].shape == (OBS_DIM,)
        assert infos[a]["action_mask"].shape == (NUM_ACTIONS,)
    while env.agents:
        actions = {a: int(np.flatnonzero(infos[a]["action_mask"])[0]) for a in env.agents}
        obs, _, _, _, infos = env.step(actions)
    assert env.core.t == cfg.horizon_T


@pytest.mark.parametrize("baseline", ["greedy", "parallel_aware", "edd", "queue_balancing"])
def test_dispatching_rule_survives_pettingzoo_roundtrip(baseline):
    cfg = EnvConfig(M=5, N=8, seed=7, episode_mode="horizon", horizon_T=120,
                    motion_planning=False, trajectory_conflicts=False)
    policy = Tier1Policy(baseline, seed=7)
    env = AssemblyGridParallelEnv(cfg, priority_fn=policy.priority_fn)
    assert env.core.priority_fn is policy.priority_fn
    _, infos = env.reset(seed=7)
    while env.agents:
        core_obs = env.core._all_observations()
        joint = policy.act(env.core, core_obs)
        actions = {a: joint[env._agent_to_coord[a]] for a in env.agents}
        _, _, _, _, infos = env.step(actions)
    assert env.core.unmasked_invalid_attempts == 0
