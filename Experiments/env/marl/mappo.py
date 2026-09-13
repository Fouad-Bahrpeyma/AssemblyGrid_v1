"""MAPPO (Sec. 10, item 6): centralized critic with decentralized shared actors.

Identical actor to IPPO (a shared-parameter policy conditioned only on each
robot's own local observation, so execution is still fully decentralized),
but the critic is conditioned on the privileged global state
(``encoding.global_state_to_array``) instead of the local observation --
the standard CTDE (centralized training, decentralized execution) recipe,
and the paper's own stated definition of this baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .common import (
        build_mlp, collect_rollout, global_critic_input, make_env, ppo_update,
        pretrain_behavior_cloning,
    )
    from .ippo import IPPOActor  # actor architecture is identical to IPPO's
except ImportError:  # standalone execution
    from common import (
        build_mlp, collect_rollout, global_critic_input, make_env, ppo_update,
        pretrain_behavior_cloning,
    )
    from ippo import IPPOActor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import EnvConfig
from training_reward import TrainingRewardConfig
from encoding import global_state_dim


class MAPPOCritic(nn.Module):
    def __init__(self, global_dim: int, hidden: int = 128):
        super().__init__()
        self.net = build_mlp(global_dim, 1, hidden)

    def forward(self, global_state):
        return self.net(global_state)


def train_mappo(cfg: Optional[EnvConfig] = None, total_steps: int = 4000,
                steps_per_update: int = 256, lr: float = 3e-4, seed: int = 0,
                device: str = "cpu", bc_warmstart_steps: int = 0,
                training_reward: Optional[TrainingRewardConfig] = None,
                progress_shaping_coef: float = 0.05, ent_coef: float = 0.01):
    torch.manual_seed(seed)
    dev = torch.device(device)
    actor = IPPOActor().to(dev)
    if bc_warmstart_steps > 0:
        pretrain_behavior_cloning(actor, cfg, bc_warmstart_steps, dev, seed=seed)
    env = make_env(cfg, training_reward=training_reward)
    critic = MAPPOCritic(global_state_dim(env.core)).to(dev)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    obs = env._last_obs
    history = []
    steps_done = 0
    while steps_done < total_steps:
        n = min(steps_per_update, total_steps - steps_done)
        buffers, obs, bootstrap, environment_training_return = collect_rollout(
            env, actor, critic, global_critic_input, obs, n, dev, shaping_coef=progress_shaping_coef)
        stats = ppo_update(actor, critic, actor_opt, critic_opt, buffers, bootstrap,
                           dev, ent_coef=ent_coef)
        steps_done += n
        stats["steps"] = steps_done
        stats["delivered"] = env.core.delivered_count
        stats["training_env_return"] = environment_training_return
        history.append(stats)
        env._last_obs = obs
    return actor, critic, history


if __name__ == "__main__":
    _, _, history = train_mappo(total_steps=4000)
    for h in history:
        print(h)
