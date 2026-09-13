"""Decentralized PettingZoo ParallelEnv wrapper for AssemblyGrid."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from core import AssemblyGridCore, EnvConfig, NUM_ACTIONS
from encoding import (OBS_DIM, action_mask_to_array, global_state_dim,
                      global_state_to_array, observation_to_array)
from training_reward import TrainingRewardConfig


def _agent_id(i: int, j: int) -> str:
    return f"r{i}_c{j}"


class AssemblyGridParallelEnv(ParallelEnv):
    metadata = {"name": "assembly_grid_v1", "render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None, priority_fn=None,
                 training_reward: Optional[TrainingRewardConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.core = AssemblyGridCore(self.cfg, priority_fn=priority_fn, training_reward=training_reward)
        self.possible_agents = [_agent_id(i, j) for i in range(self.cfg.M) for j in range(self.cfg.N)]
        self.agents = list(self.possible_agents)
        self._agent_to_coord = {_agent_id(i, j): (i, j) for i in range(self.cfg.M) for j in range(self.cfg.N)}
        # Explicit privileged interface for CTDE/centralized training. It is
        # deliberately separate from every agent's canonical local observation.
        self.state_space = spaces.Box(
            low=0.0, high=1.0, shape=(global_state_dim(self.core),), dtype=np.float32
        )

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        core_obs = self.core.reset(seed=seed)
        self.agents = list(self.possible_agents)
        obs = {a: observation_to_array(core_obs[c], self.cfg.M, self.cfg.N) for a, c in self._agent_to_coord.items()}
        infos = {a: {"action_mask": action_mask_to_array(core_obs[c].action_mask)} for a, c in self._agent_to_coord.items()}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        # PettingZoo permits callers to omit actions for already absent agents;
        # the core fills omitted active robots with idle.
        joint = {self._agent_to_coord[a]: int(act) for a, act in actions.items() if a in self._agent_to_coord}
        core_obs, reward, terminated, truncated, env_info = self.core.step(joint)
        current_agents = list(self.agents)
        obs = {a: observation_to_array(core_obs[self._agent_to_coord[a]], self.cfg.M, self.cfg.N) for a in current_agents}
        infos = {a: {
            "action_mask": action_mask_to_array(core_obs[self._agent_to_coord[a]].action_mask),
            "env_info": env_info,
        } for a in current_agents}
        rewards = {a: float(reward) for a in current_agents}
        terminations = {a: bool(terminated) for a in current_agents}
        truncations = {a: bool(truncated) for a in current_agents}
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos


    def state(self) -> np.ndarray:
        """Privileged global state for CTDE/centralized methods (non-canonical)."""
        return global_state_to_array(self.core)

    def algorithm_support_snapshot(self):
        """Non-canonical read-only structured state for algorithm adapters."""
        return self.core.algorithm_support_snapshot()

    def render(self):
        raise NotImplementedError("Use the canonical GUI adapter or an external renderer.")

    def close(self):
        pass
