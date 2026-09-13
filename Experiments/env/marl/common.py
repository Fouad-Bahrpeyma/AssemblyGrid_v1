"""Shared PPO machinery for IPPO and MAPPO: masked action sampling, rollout
collection against AssemblyGridParallelEnv, and generalized advantage
estimation (GAE). IPPO and MAPPO differ only in what the critic conditions
on (local observation vs. privileged global state), so both build on the
same ``train_ppo`` loop with a different ``critic_input`` function.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import EnvConfig, NUM_ACTIONS
from encoding import OBS_DIM, global_state_dim
try:
    from pettingzoo_env import AssemblyGridParallelEnv as _ParallelEnv
    PARALLEL_ENV_BACKEND = "pettingzoo"
except ModuleNotFoundError as exc:
    if exc.name not in {"gymnasium", "pettingzoo", "pettingzoo.utils.env"}:
        raise
    from core_parallel_env import CoreParallelEnv as _ParallelEnv
    PARALLEL_ENV_BACKEND = "core"

AssemblyGridParallelEnv = _ParallelEnv
from training_reward import TrainingRewardConfig

NEG_INF = -1e9


def build_mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """A categorical distribution over only the actions this action_mask
    allows. Every AssemblyGrid observation guarantees ACTION_IDLE is legal,
    so the mask is never all-False and this never produces NaNs."""
    masked_logits = logits.masked_fill(mask == 0, NEG_INF)
    return torch.distributions.Categorical(logits=masked_logits)


@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    critic_in: List[np.ndarray] = field(default_factory=list)
    mask: List[np.ndarray] = field(default_factory=list)
    action: List[int] = field(default_factory=list)
    logp: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    done: List[bool] = field(default_factory=list)

    def add(self, obs, critic_in, mask, action, logp, value, reward, done):
        self.obs.append(obs); self.critic_in.append(critic_in); self.mask.append(mask)
        self.action.append(action); self.logp.append(logp); self.value.append(value)
        self.reward.append(reward); self.done.append(done)

    def __len__(self):
        return len(self.obs)


def compute_gae(rewards: List[float], values: List[float], dones: List[bool],
                bootstrap_value: float, gamma: float = 0.99, lam: float = 0.95
                ) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(T)):
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def collect_rollout(
    env: AssemblyGridParallelEnv, actor: nn.Module, critic: nn.Module,
    critic_input_fn: Callable[[AssemblyGridParallelEnv, Dict[str, np.ndarray]], Dict[str, np.ndarray]],
    obs: Dict[str, np.ndarray], n_steps: int, device: torch.device,
    shaping_coef: float = 0.05,
) -> Tuple[Dict[str, RolloutBuffer], Dict[str, np.ndarray], Dict[str, float], float]:
    """Collect rollout experience using batched per-agent network inference.

    Batching changes only execution efficiency: each agent still receives its
    own canonical observation/mask, samples from its own masked categorical,
    and stores an independent rollout item. The environment sees the same
    joint action dictionary as the previous per-agent loop.
    """
    agents = list(env.agents)
    buffers = {a: RolloutBuffer() for a in agents}
    environment_training_return = 0.0
    for _ in range(n_steps):
        critic_inputs = critic_input_fn(env, obs)
        active = list(agents)
        with torch.no_grad():
            obs_batch = torch.from_numpy(np.stack([obs[a] for a in active])).to(device)
            mask_batch = torch.from_numpy(np.stack([env._last_mask[a] for a in active])).to(device)
            logits = actor(obs_batch)
            dist = masked_categorical(logits, mask_batch)
            acts = dist.sample()
            logp_batch = dist.log_prob(acts)
            critic_batch = torch.from_numpy(np.stack([critic_inputs[a] for a in active])).to(device)
            value_batch = critic(critic_batch).squeeze(-1)
        actions = {a: int(acts[i].item()) for i, a in enumerate(active)}
        logps = {a: float(logp_batch[i].item()) for i, a in enumerate(active)}
        values = {a: float(value_batch[i].item()) for i, a in enumerate(active)}

        progress_before = env.core.progress_events
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        shaping = shaping_coef * (env.core.progress_events - progress_before)
        environment_training_return += next(iter(rewards.values()), 0.0) if rewards else 0.0
        shaped_rewards = {a: r + shaping for a, r in rewards.items()}
        for a in active:
            buffers[a].add(obs[a], critic_inputs[a], env._last_mask[a], actions[a], logps[a],
                           values[a], shaped_rewards[a], terms[a] or truncs[a])
        env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
        obs = next_obs
        if not env.agents:
            obs, infos = env.reset()
            env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
            agents = list(env.agents)
            for a in agents:
                buffers.setdefault(a, RolloutBuffer())

    bootstrap = {}
    if agents:
        with torch.no_grad():
            critic_inputs = critic_input_fn(env, obs)
            ci_batch = torch.from_numpy(np.stack([critic_inputs[a] for a in agents])).to(device)
            vals = critic(ci_batch).squeeze(-1)
            bootstrap = {a: float(vals[i].item()) for i, a in enumerate(agents)}
    return buffers, obs, bootstrap, environment_training_return

def local_critic_input(env: AssemblyGridParallelEnv, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """IPPO: each agent's critic sees only that agent's own local observation."""
    return dict(obs)


def global_critic_input(env: AssemblyGridParallelEnv, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """MAPPO: every agent's critic sees the same privileged global state."""
    from encoding import global_state_to_array
    g = global_state_to_array(env.core)
    return {a: g for a in env.agents}


def ppo_update(actor: nn.Module, critic: nn.Module, actor_opt, critic_opt,
               buffers: Dict[str, RolloutBuffer], bootstrap: Dict[str, float],
               device: torch.device, epochs: int = 4, clip: float = 0.2,
               gamma: float = 0.99, lam: float = 0.95, ent_coef: float = 0.01,
               minibatch_size: int = 2048) -> dict:
    all_obs, all_critic_in, all_mask, all_act, all_logp, all_adv, all_ret = [], [], [], [], [], [], []
    for a, buf in buffers.items():
        if len(buf) == 0:
            continue
        adv, ret = compute_gae(buf.reward, buf.value, buf.done, bootstrap.get(a, 0.0), gamma, lam)
        all_obs += buf.obs; all_critic_in += buf.critic_in; all_mask += buf.mask
        all_act += buf.action; all_logp += buf.logp
        all_adv.append(adv); all_ret.append(ret)

    obs_t = torch.from_numpy(np.stack(all_obs)).to(device)
    critic_in_t = torch.from_numpy(np.stack(all_critic_in)).to(device)
    mask_t = torch.from_numpy(np.stack(all_mask)).to(device)
    act_t = torch.tensor(all_act, dtype=torch.long, device=device)
    old_logp_t = torch.tensor(all_logp, dtype=torch.float32, device=device)
    adv_t = torch.from_numpy(np.concatenate(all_adv)).to(device)
    ret_t = torch.from_numpy(np.concatenate(all_ret)).to(device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs_t.shape[0]
    last_stats = {}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            logits = actor(obs_t[idx])
            dist = masked_categorical(logits, mask_t[idx])
            new_logp = dist.log_prob(act_t[idx])
            ratio = torch.exp(new_logp - old_logp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx]
            actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * dist.entropy().mean()
            value = critic(critic_in_t[idx]).squeeze(-1)
            critic_loss = F.mse_loss(value, ret_t[idx])

            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            last_stats = {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item()),
                         "entropy": float(dist.entropy().mean().item())}
    return last_stats


def pretrain_behavior_cloning(actor: nn.Module, cfg: Optional[EnvConfig], n_steps: int,
                              device: torch.device, lr: float = 1e-3, seed: int = 0) -> List[float]:
    """Warm-start the actor by imitating the Tier-1 parallel_aware reference
    policy for n_steps env steps before PPO fine-tuning.

    Behaviour cloning is an optional algorithm-side warm start for difficult
    exploration regimes. It does not change the AssemblyGrid benchmark
    definition. PPO fine-tuning subsequently uses the experiment-selected
    TrainingRewardConfig (plus any explicitly reported training-only shaping).
    """
    from policies import Tier1Policy
    env = make_env(cfg)
    expert = Tier1Policy("parallel_aware", seed=seed)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    obs = env._last_obs
    losses = []
    batch_o, batch_m, batch_a = [], [], []
    for _ in range(n_steps):
        core_obs = env.core._all_observations()
        expert_actions = expert.act(env.core, core_obs)
        for a in env.agents:
            rc = env._agent_to_coord[a]
            batch_o.append(obs[a]); batch_m.append(env._last_mask[a]); batch_a.append(expert_actions[rc])
        next_obs, _rewards, terms, truncs, infos = env.step({a: expert_actions[env._agent_to_coord[a]] for a in env.agents})
        env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
        obs = next_obs
        if not env.agents:
            obs, infos = env.reset()
            env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
        if len(batch_o) >= 512:
            o_t = torch.from_numpy(np.stack(batch_o)).to(device)
            m_t = torch.from_numpy(np.stack(batch_m)).to(device)
            a_t = torch.tensor(batch_a, dtype=torch.long, device=device)
            logits = actor(o_t)
            dist = masked_categorical(logits, m_t)
            # Most robots are idle most of the time in the reference
            # trajectory, so plain average cross-entropy is dominated by the
            # idle class and barely penalizes missing the rare-but-essential
            # pick/handoff/operation/deliver actions. Upweight every non-idle
            # expert action so imitation actually captures them.
            from core import ACTION_IDLE
            weights = torch.where(a_t == ACTION_IDLE, 1.0, 8.0)
            per_sample_loss = -dist.log_prob(a_t)
            loss = (per_sample_loss * weights).sum() / weights.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
            batch_o, batch_m, batch_a = [], [], []
    return losses


def make_env(cfg: Optional[EnvConfig] = None,
             training_reward: Optional[TrainingRewardConfig] = None) -> AssemblyGridParallelEnv:
    env = AssemblyGridParallelEnv(cfg, training_reward=training_reward)
    obs, infos = env.reset()
    env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
    env._last_obs = obs
    return env
