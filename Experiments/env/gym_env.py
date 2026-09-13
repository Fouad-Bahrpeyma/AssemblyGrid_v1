"""Privileged centralized Gymnasium wrapper for the AssemblyGrid oracle.

Unlike the previous implementation, this wrapper exposes an explicit global
state (robots, q/model state, active product recipe progress, token locations,
load and production counters) rather than concatenating local observations.
The joint action still uses the same parameterized local action slots as the
decentralized environment so the decentralization gap is not confounded by a
different low-level action vocabulary.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core import AssemblyGridCore, EnvConfig, NUM_ACTIONS
from encoding import global_state_dim, global_state_to_array
from training_reward import TrainingRewardConfig


class AssemblyGridGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None, priority_fn=None,
                 training_reward: Optional[TrainingRewardConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.core = AssemblyGridCore(self.cfg, priority_fn=priority_fn, training_reward=training_reward)
        self._coords = [(i, j) for i in range(self.cfg.M) for j in range(self.cfg.N)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(global_state_dim(self.core),), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS] * len(self._coords))

    def action_masks(self) -> np.ndarray:
        cands = self.core.candidate_operations()
        return np.stack([np.asarray(self.core.action_mask(*c, cands), dtype=np.int8) for c in self._coords])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return global_state_to_array(self.core), {"action_masks": self.action_masks()}

    def step(self, action):
        joint = {c: int(a) for c, a in zip(self._coords, action)}
        _, reward, terminated, truncated, info = self.core.step(joint)
        info = dict(info); info["action_masks"] = self.action_masks()
        return global_state_to_array(self.core), float(reward), bool(terminated), bool(truncated), info

    def algorithm_support_snapshot(self):
        """Non-canonical read-only structured state for algorithm adapters."""
        return self.core.algorithm_support_snapshot()

    def render(self):
        raise NotImplementedError("Use the canonical GUI adapter or an external renderer.")
