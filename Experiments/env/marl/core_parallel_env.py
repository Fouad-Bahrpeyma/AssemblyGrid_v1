"""Dependency-free parallel adapter for AssemblyGrid MARL training.

This adapter mirrors the small runtime surface used by the reference MARL
implementations without importing Gymnasium or PettingZoo.  It is an
algorithm-support interface only: it delegates all state transitions,
feasibility, observations, rewards and KPIs to :class:`AssemblyGridCore`.
When PettingZoo is installed, the standard ``AssemblyGridParallelEnv`` remains
available and should be used for external API-compliance tests.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core import AssemblyGridCore, EnvConfig
from encoding import action_mask_to_array, global_state_to_array, observation_to_array
from training_reward import TrainingRewardConfig


def _agent_id(i: int, j: int) -> str:
    return f"r{i}_c{j}"


class CoreParallelEnv:
    """Read/write adapter whose *only* dynamics are those of AssemblyGridCore."""

    def __init__(self, cfg: Optional[EnvConfig] = None, priority_fn=None,
                 training_reward: Optional[TrainingRewardConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.core = AssemblyGridCore(self.cfg, priority_fn=priority_fn,
                                     training_reward=training_reward)
        self.possible_agents = [
            _agent_id(i, j) for i in range(self.cfg.M) for j in range(self.cfg.N)
        ]
        self.agents = list(self.possible_agents)
        self._agent_to_coord = {
            _agent_id(i, j): (i, j)
            for i in range(self.cfg.M) for j in range(self.cfg.N)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        core_obs = self.core.reset(seed=seed)
        self.agents = list(self.possible_agents)
        obs = {
            a: observation_to_array(core_obs[c], self.cfg.M, self.cfg.N)
            for a, c in self._agent_to_coord.items()
        }
        infos = {
            a: {"action_mask": action_mask_to_array(core_obs[c].action_mask)}
            for a, c in self._agent_to_coord.items()
        }
        return obs, infos

    def step(self, actions: Dict[str, int]):
        joint = {
            self._agent_to_coord[a]: int(act)
            for a, act in actions.items()
            if a in self._agent_to_coord
        }
        core_obs, reward, terminated, truncated, env_info = self.core.step(joint)
        current_agents = list(self.agents)
        obs = {
            a: observation_to_array(core_obs[self._agent_to_coord[a]], self.cfg.M, self.cfg.N)
            for a in current_agents
        }
        infos = {
            a: {
                "action_mask": action_mask_to_array(
                    core_obs[self._agent_to_coord[a]].action_mask
                ),
                "env_info": env_info,
            }
            for a in current_agents
        }
        rewards = {a: float(reward) for a in current_agents}
        terminations = {a: bool(terminated) for a in current_agents}
        truncations = {a: bool(truncated) for a in current_agents}
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return global_state_to_array(self.core)

    def algorithm_support_snapshot(self):
        return self.core.algorithm_support_snapshot()

    def close(self):
        pass
