"""AAG038 stable IPPO and MAPPO methods.

This module changes only algorithm internals. The AssemblyGrid observation,
flat action, mask, transition, reward configuration, and evaluation contracts
remain untouched.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENV_MARL = Path(__file__).resolve().parent.parent / "env" / "marl"
if ENV_MARL.is_dir():
    sys.path.insert(0, str(ENV_MARL))

from core import (
    ACTION_DELIVER,
    ACTION_HANDOFF_BASE,
    ACTION_IDLE,
    ACTION_OPERATION_BASE,
    ACTION_PICK_BASE,
    ACTION_RECEIVE_BASE,
    EnvConfig,
    NUM_ACTIONS,
)
from encoding import OBS_DIM, global_state_dim, global_state_to_array
from marl.common import RolloutBuffer, compute_gae, make_env, masked_categorical, pretrain_behavior_cloning
from training_reward import TrainingRewardConfig


def orthogonal_init(module: nn.Module, gain: float = np.sqrt(2.0)) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)
    return module


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = orthogonal_init(nn.Linear(width, width))
        self.fc2 = orthogonal_init(nn.Linear(width, width), gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.tanh(self.fc1(self.norm(x)))
        return residual + torch.tanh(self.fc2(x))


class StructuredActor(nn.Module):
    """Masked flat policy with an internal action family decomposition."""

    def __init__(self, width: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(OBS_DIM)
        self.input = orthogonal_init(nn.Linear(OBS_DIM, width))
        self.blocks = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.family_head = orthogonal_init(nn.Linear(width, 6), gain=0.01)
        self.slot_head = orthogonal_init(nn.Linear(width, NUM_ACTIONS), gain=0.01)
        action_family = torch.empty(NUM_ACTIONS, dtype=torch.long)
        action_family[ACTION_IDLE] = 0
        action_family[ACTION_DELIVER] = 1
        action_family[ACTION_PICK_BASE:ACTION_HANDOFF_BASE] = 2
        action_family[ACTION_HANDOFF_BASE:ACTION_RECEIVE_BASE] = 3
        action_family[ACTION_RECEIVE_BASE:ACTION_OPERATION_BASE] = 4
        action_family[ACTION_OPERATION_BASE:] = 5
        self.register_buffer("action_family", action_family)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input(self.input_norm(obs)))
        hidden = self.blocks(hidden)
        family = self.family_head(hidden)
        return self.slot_head(hidden) + family.index_select(1, self.action_family)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, width: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.input = orthogonal_init(nn.Linear(input_dim, width))
        self.blocks = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.output = orthogonal_init(nn.Linear(width, 1), gain=1.0)

    def forward(self, value_input: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input(self.norm(value_input)))
        return self.output(self.blocks(hidden))


def local_value_inputs(env, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return dict(obs)


def agent_conditioned_global_inputs(env, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    state = global_state_to_array(env.core)
    return {agent: np.concatenate((state, obs[agent])).astype(np.float32, copy=False) for agent in env.agents}


@dataclass
class StablePPOConfig:
    steps_per_update: int = 1600
    epochs: int = 4
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.10
    value_clip: float = 0.20
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    entropy_start: float = 0.001
    entropy_end: float = 0.0
    clone_kl_start: float = 0.25
    clone_kl_end: float = 0.01
    critic_warmup_steps: int = 4800


def collect_rollout(env, actor, critic, value_input_fn, obs, n_steps, device, shaping_coef):
    agents = list(env.agents)
    buffers = {agent: RolloutBuffer() for agent in agents}
    if not hasattr(env, "_aag038_episode"):
        env._aag038_episode = 0
        env._aag038_total_steps = 0
        env._aag038_components = {"delivery": 0.0, "progress": 0.0, "steps": 0}
        env._aag038_completed = []
    for _ in range(n_steps):
        value_inputs = value_input_fn(env, obs)
        active = list(env.agents)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.stack([obs[a] for a in active])).to(device)
            mask_tensor = torch.from_numpy(np.stack([env._last_mask[a] for a in active])).to(device)
            distribution = masked_categorical(actor(obs_tensor), mask_tensor)
            actions_tensor = distribution.sample()
            log_prob = distribution.log_prob(actions_tensor)
            value_tensor = critic(torch.from_numpy(np.stack([value_inputs[a] for a in active])).to(device)).squeeze(-1)
        actions = {agent: int(actions_tensor[i].item()) for i, agent in enumerate(active)}
        progress_before = env.core.progress_events
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        progress_reward = shaping_coef * (env.core.progress_events - progress_before)
        delivery_reward = float(next(iter(rewards.values()), 0.0)) if rewards else 0.0
        shaped_rewards = {agent: float(rewards[agent]) + progress_reward for agent in active}
        for i, agent in enumerate(active):
            buffers[agent].add(
                obs[agent], value_inputs[agent], env._last_mask[agent], actions[agent],
                float(log_prob[i].item()), float(value_tensor[i].item()), shaped_rewards[agent],
                bool(terms[agent] or truncs[agent]),
            )
        components = env._aag038_components
        components["delivery"] += delivery_reward
        components["progress"] += progress_reward
        components["steps"] += 1
        env._aag038_total_steps += 1
        env._last_mask = {agent: infos[agent]["action_mask"] for agent in env.agents}
        obs = next_obs
        if not env.agents:
            env._aag038_completed.append({
                "episode": int(env._aag038_episode),
                "end_step": int(env._aag038_total_steps),
                "episode_length": int(components["steps"]),
                "delivery_return": float(components["delivery"]),
                "progress_reward": float(components["progress"]),
                "shaped_return": float(components["delivery"] + components["progress"]),
                "delivered": int(env.core.delivered_count),
            })
            env._aag038_episode += 1
            env._aag038_components = {"delivery": 0.0, "progress": 0.0, "steps": 0}
            obs, infos = env.reset()
            env._last_mask = {agent: infos[agent]["action_mask"] for agent in env.agents}
            for agent in env.agents:
                buffers.setdefault(agent, RolloutBuffer())
    bootstrap = {}
    if env.agents:
        value_inputs = value_input_fn(env, obs)
        with torch.no_grad():
            values = critic(torch.from_numpy(np.stack([value_inputs[a] for a in env.agents])).to(device)).squeeze(-1)
        bootstrap = {agent: float(values[i].item()) for i, agent in enumerate(env.agents)}
    return buffers, obs, bootstrap


def stable_update(actor, critic, teacher, actor_optimizer, critic_optimizer, buffers, bootstrap,
                  device, config: StablePPOConfig, entropy_coef: float, clone_kl_coef: float,
                  update_actor: bool) -> dict:
    obs, value_inputs, masks, actions, old_log_probs, old_values, advantages, returns = [], [], [], [], [], [], [], []
    for agent, buffer in buffers.items():
        if not buffer:
            continue
        advantage, target_return = compute_gae(buffer.reward, buffer.value, buffer.done, bootstrap.get(agent, 0.0),
                                               config.gamma, config.gae_lambda)
        obs.extend(buffer.obs); value_inputs.extend(buffer.critic_in); masks.extend(buffer.mask)
        actions.extend(buffer.action); old_log_probs.extend(buffer.logp); old_values.extend(buffer.value)
        advantages.append(advantage); returns.append(target_return)
    obs_t = torch.from_numpy(np.stack(obs)).to(device)
    value_input_t = torch.from_numpy(np.stack(value_inputs)).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    action_t = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_prob_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    old_value_t = torch.tensor(old_values, dtype=torch.float32, device=device)
    advantage_t = torch.from_numpy(np.concatenate(advantages)).to(device)
    return_t = torch.from_numpy(np.concatenate(returns)).to(device)
    advantage_mean, advantage_std = advantage_t.mean(), advantage_t.std()
    if float(advantage_std.item()) > 1e-4:
        advantage_t = (advantage_t - advantage_mean) / advantage_std
    else:
        advantage_t = advantage_t - advantage_mean
    with torch.no_grad():
        teacher_logits = teacher(obs_t) if teacher is not None else None
    indices = torch.arange(obs_t.shape[0], device=device)
    totals = {"actor_loss": [], "critic_loss": [], "entropy": [], "approx_kl": [], "clone_kl": [], "clip_fraction": [], "actor_grad_norm": [], "critic_grad_norm": []}
    early_stop = False
    for _ in range(config.epochs):
        for batch in indices[torch.randperm(len(indices), device=device)].split(config.minibatch_size):
            distribution = masked_categorical(actor(obs_t[batch]), mask_t[batch])
            new_log_prob = distribution.log_prob(action_t[batch])
            log_ratio = new_log_prob - old_log_prob_t[batch]
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio * advantage_t[batch],
                                      ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio) * advantage_t[batch])
            entropy = distribution.entropy().mean()
            teacher_distribution = (masked_categorical(teacher_logits[batch], mask_t[batch])
                                    if teacher_logits is not None else None)
            clone_kl = (torch.distributions.kl_divergence(teacher_distribution, distribution).mean()
                        if teacher_distribution is not None else torch.zeros((), device=device))
            actor_loss = -surrogate.mean() - entropy_coef * entropy + clone_kl_coef * clone_kl
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            if update_actor:
                actor_optimizer.zero_grad(); actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                actor_optimizer.step()
            else:
                actor_grad = torch.zeros((), device=device)
            value = critic(value_input_t[batch]).squeeze(-1)
            value_clipped = old_value_t[batch] + (value-old_value_t[batch]).clamp(-config.value_clip, config.value_clip)
            raw_loss = F.smooth_l1_loss(value, return_t[batch], reduction="none")
            clipped_loss = F.smooth_l1_loss(value_clipped, return_t[batch], reduction="none")
            critic_loss = torch.maximum(raw_loss, clipped_loss).mean()
            critic_optimizer.zero_grad(); critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()
            for key, value_stat in (("actor_loss", actor_loss), ("critic_loss", critic_loss), ("entropy", entropy),
                                    ("approx_kl", approx_kl), ("clone_kl", clone_kl), ("clip_fraction", clip_fraction),
                                    ("actor_grad_norm", actor_grad), ("critic_grad_norm", critic_grad)):
                totals[key].append(float(value_stat.detach().item()))
            if update_actor and float(approx_kl.item()) > 1.5 * config.target_kl:
                early_stop = True
                break
        if early_stop:
            break
    predicted = critic(value_input_t).squeeze(-1).detach()
    return_variance = torch.var(return_t)
    explained_variance = 1.0 - torch.var(return_t-predicted) / return_variance if return_variance > 1e-8 else torch.zeros((), device=device)
    return {key: float(np.mean(values)) for key, values in totals.items()} | {
        "advantage_mean": float(advantage_mean.item()), "advantage_std": float(advantage_std.item()),
        "explained_variance": float(explained_variance.item()), "kl_early_stop": bool(early_stop),
        "actor_updated": bool(update_actor),
    }


def train_stable_ppo(method: str, cfg: Optional[EnvConfig], total_steps: int, seed: int, device: str,
                     training_reward: Optional[TrainingRewardConfig], progress_shaping_coef: float = 0.05,
                     bc_warmstart_steps: int = 10000, width: int = 256,
                     ppo_config: Optional[StablePPOConfig] = None,
                     phase_evaluator: Optional[Callable[[nn.Module, str], dict]] = None):
    if method not in {"ippo", "mappo"}:
        raise ValueError(method)
    config = ppo_config or StablePPOConfig()
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    actor = StructuredActor(width).to(dev)
    phase_history: List[dict] = [{"phase": "random_initialization", "steps": 0,
                                 "evaluation": phase_evaluator(actor, "random_initialization") if phase_evaluator else None}]
    bc_losses = pretrain_behavior_cloning(actor, cfg, bc_warmstart_steps, dev, seed=seed) if bc_warmstart_steps else []
    phase_history.append({"phase": "behavior_cloning_complete", "steps": 0,
                          "environment_steps": int(bc_warmstart_steps),
                          "updates": len(bc_losses), "final_loss": bc_losses[-1] if bc_losses else None,
                          "evaluation": phase_evaluator(actor, "behavior_cloning_complete") if phase_evaluator else None})
    teacher = copy.deepcopy(actor).eval() if bc_warmstart_steps else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    env = make_env(cfg, training_reward=training_reward)
    if method == "ippo":
        value_input_fn: Callable = local_value_inputs
        critic_dim = OBS_DIM
    else:
        value_input_fn = agent_conditioned_global_inputs
        critic_dim = global_state_dim(env.core) + OBS_DIM
    critic = ValueNetwork(critic_dim, width).to(dev)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr, eps=1e-5)
    obs = env._last_obs
    history = []
    steps_done = 0
    while steps_done < total_steps:
        rollout_steps = min(config.steps_per_update, total_steps-steps_done)
        buffers, obs, bootstrap = collect_rollout(env, actor, critic, value_input_fn, obs, rollout_steps, dev,
                                                  progress_shaping_coef)
        next_steps = steps_done + rollout_steps
        fraction = next_steps / max(1, total_steps)
        entropy_coef = config.entropy_start + fraction * (config.entropy_end-config.entropy_start)
        clone_coef = ((config.clone_kl_start + fraction*(config.clone_kl_end-config.clone_kl_start))
                      if teacher is not None else 0.0)
        stats = stable_update(actor, critic, teacher, actor_optimizer, critic_optimizer, buffers, bootstrap, dev,
                              config, entropy_coef, clone_coef, update_actor=next_steps > config.critic_warmup_steps)
        steps_done = next_steps
        stats.update({"phase": "reinforcement_learning", "steps": steps_done,
                      "entropy_coefficient": entropy_coef, "clone_kl_coefficient": clone_coef,
                      "completed_episodes": list(env._aag038_completed)})
        env._aag038_completed.clear(); history.append(stats); env._last_obs = obs
    phase_history.append({"phase": "reinforcement_learning_complete", "steps": int(total_steps),
                          "evaluation": phase_evaluator(actor, "reinforcement_learning_complete") if phase_evaluator else None})
    return actor, critic, {"phase_history": phase_history, "training_history": history,
                           "method": method, "architecture": "structured-residual-256-v1",
                           "config": config.__dict__}
