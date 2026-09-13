"""IPPO (Sec. 10, item 5): shared-parameter independent policy baseline.

One actor and one critic, shared across every robot (parameter sharing is
what makes this tractable on a grid with tens of agents), each critic
prediction conditioned only on that robot's own local observation -- the
"independent" in IPPO. Trained with the environment's real dynamics via
``pettingzoo_env.AssemblyGridParallelEnv``.
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
        NUM_ACTIONS, OBS_DIM, build_mlp, collect_rollout, local_critic_input,
        make_env, masked_categorical, ppo_update, pretrain_behavior_cloning,
    )
except ImportError:  # standalone execution
    from common import (
        NUM_ACTIONS, OBS_DIM, build_mlp, collect_rollout, local_critic_input,
        make_env, masked_categorical, ppo_update, pretrain_behavior_cloning,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import EnvConfig
from training_reward import TrainingRewardConfig


class IPPOActor(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = build_mlp(OBS_DIM, NUM_ACTIONS, hidden)

    def forward(self, obs):
        return self.net(obs)


class IPPOCritic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = build_mlp(OBS_DIM, 1, hidden)

    def forward(self, obs):
        return self.net(obs)


def train_ippo(cfg: Optional[EnvConfig] = None, total_steps: int = 4000,
               steps_per_update: int = 256, lr: float = 3e-4, seed: int = 0,
               device: str = "cpu", bc_warmstart_steps: int = 0,
               training_reward: Optional[TrainingRewardConfig] = None,
               progress_shaping_coef: float = 0.05, ent_coef: float = 0.01):
    torch.manual_seed(seed)
    dev = torch.device(device)
    actor = IPPOActor().to(dev)
    critic = IPPOCritic().to(dev)
    if bc_warmstart_steps > 0:
        pretrain_behavior_cloning(actor, cfg, bc_warmstart_steps, dev, seed=seed)
    env = make_env(cfg, training_reward=training_reward)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    obs = env._last_obs
    history = []
    steps_done = 0
    while steps_done < total_steps:
        n = min(steps_per_update, total_steps - steps_done)
        buffers, obs, bootstrap, environment_training_return = collect_rollout(env, actor, critic, local_critic_input, obs, n, dev, shaping_coef=progress_shaping_coef)
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
    _, _, history = train_ippo(total_steps=4000)
    for h in history:
        print(h)
