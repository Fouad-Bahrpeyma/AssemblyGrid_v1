"""QMIX controller with shared per-agent Q networks and a monotonic team mixer."""
from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .common import NEG_INF, build_mlp, make_env
except ImportError:  # standalone execution
    from common import NEG_INF, build_mlp, make_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import EnvConfig, NUM_ACTIONS
from encoding import OBS_DIM, global_state_dim, global_state_to_array
from training_reward import TrainingRewardConfig


class QNetwork(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = build_mlp(OBS_DIM, NUM_ACTIONS, hidden)

    def forward(self, obs):
        return self.net(obs)




class QMixMixer(nn.Module):
    """Monotonic mixing network conditioned on the privileged global state,
    via a hypernetwork producing non-negative first-layer weights (softplus)
    so the mix is monotonic in every agent's Q-value."""

    def __init__(self, n_agents: int, global_dim: int, mix_hidden: int = 32,
                 hyper_hidden: int = 64, normalize_weights: bool = False):
        super().__init__()
        self.n_agents = n_agents
        self.mix_hidden = mix_hidden
        self.normalize_weights = normalize_weights
        self.hyper_w1 = build_mlp(global_dim, n_agents * mix_hidden, hyper_hidden)
        self.hyper_b1 = nn.Linear(global_dim, mix_hidden)
        self.hyper_w2 = build_mlp(global_dim, mix_hidden, hyper_hidden)
        self.hyper_b2 = build_mlp(global_dim, 1, hyper_hidden)

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        bsz = agent_qs.shape[0]
        w1 = F.softplus(self.hyper_w1(global_state)).view(bsz, self.n_agents, self.mix_hidden)
        if self.normalize_weights:
            w1 = w1 / w1.sum(dim=1, keepdim=True).clamp_min(1e-6)
        b1 = self.hyper_b1(global_state).view(bsz, 1, self.mix_hidden)
        # This is the same batched vector-matrix product as bmm, expressed as
        # multiply-and-reduce.  Recent CUDA/PyTorch builds may route the tiny
        # bmm shape through a just-in-time Triton kernel and unexpectedly
        # require a host C compiler at runtime.  The explicit reduction is
        # portable, differentiable, and preserves QMIX exactly.
        hidden = F.elu((agent_qs.unsqueeze(-1) * w1).sum(dim=1, keepdim=True) + b1)
        w2 = F.softplus(self.hyper_w2(global_state)).view(bsz, self.mix_hidden, 1)
        if self.normalize_weights:
            w2 = w2 / w2.sum(dim=1, keepdim=True).clamp_min(1e-6)
        b2 = self.hyper_b2(global_state).view(bsz, 1, 1)
        q_tot = (hidden.squeeze(1) * w2.squeeze(-1)).sum(dim=1) + b2.view(bsz)
        return q_tot


class JointReplayBuffer:
    def __init__(self, capacity: int, n_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buf = deque(maxlen=capacity)

    def add(self, obs, mask, actions, reward, next_obs, next_mask, done, gstate, next_gstate):
        self.buf.append((obs, mask, actions, reward, next_obs, next_mask, done, gstate, next_gstate))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs = np.stack([b[0] for b in batch])            # (B, n_agents, OBS_DIM)
        mask = np.stack([b[1] for b in batch])            # (B, n_agents, NUM_ACTIONS)
        actions = np.stack([b[2] for b in batch])         # (B, n_agents)
        reward = np.asarray([b[3] for b in batch], dtype=np.float32)
        next_obs = np.stack([b[4] for b in batch])
        next_mask = np.stack([b[5] for b in batch])
        done = np.asarray([b[6] for b in batch], dtype=np.float32)
        gstate = np.stack([b[7] for b in batch])
        next_gstate = np.stack([b[8] for b in batch])
        return obs, mask, actions, reward, next_obs, next_mask, done, gstate, next_gstate


def epsilon_greedy_actions(qnet: nn.Module, obs: Dict[str, np.ndarray], masks: Dict[str, np.ndarray],
                           epsilon: float, device: torch.device, agents: List[str]) -> Dict[str, int]:
    """Epsilon-greedy selection with one batched Q-network forward pass."""
    actions = {}
    with torch.no_grad():
        obs_batch = torch.from_numpy(np.stack([obs[a] for a in agents])).to(device)
        q_batch = qnet(obs_batch).cpu().numpy()
    for i, a in enumerate(agents):
        m = masks[a]
        if random.random() < epsilon:
            legal = np.flatnonzero(m)
            actions[a] = int(np.random.choice(legal))
        else:
            q = np.where(m == 0, NEG_INF, q_batch[i])
            actions[a] = int(np.argmax(q))
    return actions


def pretrain_q_behavior_cloning(qnet: nn.Module, cfg: Optional[EnvConfig], n_steps: int,
                                device: torch.device, lr: float = 1e-3,
                                seed: int = 0, return_anchor: bool = False,
                                anchor_capacity: int = 8192,
                                demo_seed_count: int = 1,
                                non_idle_weight: float = 8.0,
                                operation_weight: float = 8.0):
    """Optionally warm-start local Q logits from the declared Tier-1 expert."""
    from policies import Tier1Policy
    from core import ACTION_IDLE, ACTION_OPERATION_BASE
    demo_seed_count = max(1, int(demo_seed_count))
    segment_steps = max(1, n_steps // demo_seed_count)
    env = expert = obs = None
    active_demo_seed = -1
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    batch_o, batch_m, batch_a, losses = [], [], [], []
    anchor_o, anchor_m, anchor_a = (deque(maxlen=anchor_capacity),
                                    deque(maxlen=anchor_capacity),
                                    deque(maxlen=anchor_capacity))
    for step in range(n_steps):
        demo_seed_index = min(demo_seed_count - 1, step // segment_steps)
        if demo_seed_index != active_demo_seed:
            active_demo_seed = demo_seed_index
            variant = cfg
            if cfg is not None and demo_seed_count > 1:
                generation_base = cfg.generation_seed if cfg.generation_seed is not None else (cfg.seed or seed)
                execution_base = cfg.execution_seed if cfg.execution_seed is not None else (cfg.seed or seed)
                variant = replace(cfg, generation_seed=generation_base + 1009 * demo_seed_index,
                                  execution_seed=execution_base + 1013 * demo_seed_index,
                                  seed=None)
            env = make_env(variant)
            expert = Tier1Policy("parallel_aware", seed=seed + demo_seed_index)
            obs = env._last_obs
        core_obs = env.core._all_observations()
        expert_actions = expert.act(env.core, core_obs)
        for agent in env.agents:
            rc = env._agent_to_coord[agent]
            batch_o.append(obs[agent]); batch_m.append(env._last_mask[agent])
            batch_a.append(expert_actions[rc])
            anchor_o.append(obs[agent]); anchor_m.append(env._last_mask[agent])
            anchor_a.append(expert_actions[rc])
        next_obs, _rewards, _terms, _truncs, infos = env.step(
            {agent: expert_actions[env._agent_to_coord[agent]] for agent in env.agents})
        env._last_mask = {agent: infos[agent]["action_mask"] for agent in env.agents}
        obs = next_obs
        if not env.agents:
            obs, infos = env.reset()
            env._last_mask = {agent: infos[agent]["action_mask"] for agent in env.agents}
        if len(batch_o) >= 512:
            o_t = torch.from_numpy(np.stack(batch_o)).to(device)
            m_t = torch.from_numpy(np.stack(batch_m)).to(device)
            a_t = torch.tensor(batch_a, dtype=torch.long, device=device)
            logits = qnet(o_t).masked_fill(m_t == 0, NEG_INF)
            weights = torch.where(a_t == ACTION_IDLE, 1.0, non_idle_weight)
            weights = torch.where(a_t >= ACTION_OPERATION_BASE, operation_weight, weights)
            per_sample = F.cross_entropy(logits, a_t, reduction="none")
            loss = (per_sample * weights).sum() / weights.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
            opt.step(); losses.append(float(loss.item()))
            batch_o, batch_m, batch_a = [], [], []
    if not return_anchor:
        return losses
    anchor = (np.stack(anchor_o), np.stack(anchor_m), np.asarray(anchor_a, dtype=np.int64))
    return losses, anchor

def train_value_decomposition(
    algo: str, cfg: Optional[EnvConfig] = None, total_steps: int = 4000,
    batch_size: int = 32, buffer_capacity: int = 5000, warmup_steps: int = 200,
    train_every: int = 4, target_sync_every: int = 200, lr: float = 5e-4,
    gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 3000, seed: int = 0, device: str = "cpu",
    training_reward: Optional[TrainingRewardConfig] = None,
    progress_shaping_coef: float = 0.05,
    normalize_mixer_weights: bool = False, double_q: bool = False,
    huber_loss: bool = False, bc_warmstart_steps: int = 0,
    bc_anchor_coef: float = 0.0, bc_anchor_batch_size: int = 128,
    bc_parameter_anchor_coef: float = 0.0,
    bc_demo_seed_count: int = 1,
    bc_non_idle_weight: float = 8.0, bc_operation_weight: float = 8.0,
    q_hidden: int = 64,
):
    """Train the QMIX controller."""
    assert algo == "qmix"
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    env = make_env(cfg, training_reward=training_reward)
    agents = list(env.agents)
    n_agents = len(agents)

    qnet = QNetwork(hidden=q_hidden).to(dev)
    target_qnet = QNetwork(hidden=q_hidden).to(dev)
    bc_losses = []
    bc_anchor = None
    if bc_warmstart_steps > 0:
        if bc_anchor_coef > 0:
            bc_losses, bc_anchor = pretrain_q_behavior_cloning(
                qnet, cfg, bc_warmstart_steps, dev, seed=seed, return_anchor=True,
                demo_seed_count=bc_demo_seed_count,
                non_idle_weight=bc_non_idle_weight,
                operation_weight=bc_operation_weight)
        else:
            bc_losses = pretrain_q_behavior_cloning(
                qnet, cfg, bc_warmstart_steps, dev, seed=seed,
                demo_seed_count=bc_demo_seed_count,
                non_idle_weight=bc_non_idle_weight,
                operation_weight=bc_operation_weight)
    bc_parameter_anchor = ({name: parameter.detach().clone()
                            for name, parameter in qnet.named_parameters()}
                           if bc_parameter_anchor_coef > 0 else None)
    target_qnet.load_state_dict(qnet.state_dict())
    gdim = global_state_dim(env.core)
    mixer = QMixMixer(n_agents, gdim, normalize_weights=normalize_mixer_weights).to(dev)
    target_mixer = QMixMixer(n_agents, gdim, normalize_weights=normalize_mixer_weights).to(dev)
    target_mixer.load_state_dict(mixer.state_dict())

    params = list(qnet.parameters()) + list(mixer.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    buf = JointReplayBuffer(buffer_capacity, n_agents)

    obs = env._last_obs
    masks = dict(env._last_mask)
    history = []
    for step in range(total_steps):
        eps = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * step / max(1, epsilon_decay_steps))
        actions = epsilon_greedy_actions(qnet, obs, masks, eps, dev, agents)
        gstate = global_state_to_array(env.core)
        progress_before = env.core.progress_events
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        next_masks = {a: infos[a]["action_mask"] for a in env.agents}
        done = bool(terms.get(agents[0], False) or truncs.get(agents[0], False)) if agents else False
        next_gstate = global_state_to_array(env.core) if not done else gstate
        reward = next(iter(rewards.values()), 0.0) if rewards else 0.0
        reward += progress_shaping_coef * (env.core.progress_events - progress_before)

        obs_arr = np.stack([obs[a] for a in agents])
        mask_arr = np.stack([masks[a] for a in agents])
        act_arr = np.array([actions[a] for a in agents])
        if done or not env.agents:
            next_obs_full, infos2 = env.reset()
            nxt_masks_full = {a: infos2[a]["action_mask"] for a in env.agents}
            next_obs_arr = np.stack([next_obs_full[a] for a in agents]) if all(a in next_obs_full for a in agents) else obs_arr
            next_mask_arr = np.stack([nxt_masks_full[a] for a in agents]) if all(a in nxt_masks_full for a in agents) else mask_arr
        else:
            next_obs_arr = np.stack([next_obs[a] for a in agents])
            next_mask_arr = np.stack([next_masks[a] for a in agents])

        buf.add(obs_arr, mask_arr, act_arr, reward, next_obs_arr, next_mask_arr, float(done), gstate, next_gstate)

        if done or not env.agents:
            obs, infos = env.reset()
            masks = {a: infos[a]["action_mask"] for a in env.agents}
        else:
            obs, masks = next_obs, next_masks

        if len(buf) >= warmup_steps and step % train_every == 0:
            stats = _train_step(qnet, target_qnet, mixer, target_mixer, opt, buf,
                                batch_size, gamma, dev, double_q=double_q,
                                huber_loss=huber_loss, bc_anchor=bc_anchor,
                                bc_anchor_coef=bc_anchor_coef,
                                bc_anchor_batch_size=bc_anchor_batch_size,
                                bc_parameter_anchor=bc_parameter_anchor,
                                bc_parameter_anchor_coef=bc_parameter_anchor_coef,
                                bc_non_idle_weight=bc_non_idle_weight,
                                bc_operation_weight=bc_operation_weight)
            stats["step"] = step
            stats["epsilon"] = eps
            stats["delivered"] = env.core.delivered_count
            history.append(stats)

        if step % target_sync_every == 0:
            target_qnet.load_state_dict(qnet.state_dict())
            target_mixer.load_state_dict(mixer.state_dict())

    if history and bc_losses:
        history[0]["bc_updates"] = len(bc_losses)
        history[0]["bc_final_loss"] = bc_losses[-1]
    return qnet, mixer, history


def _train_step(qnet, target_qnet, mixer, target_mixer, opt, buf: JointReplayBuffer,
                batch_size: int, gamma: float, dev: torch.device,
                double_q: bool = False, huber_loss: bool = False,
                bc_anchor=None, bc_anchor_coef: float = 0.0,
                bc_anchor_batch_size: int = 128, bc_parameter_anchor=None,
                bc_parameter_anchor_coef: float = 0.0,
                bc_non_idle_weight: float = 8.0,
                bc_operation_weight: float = 8.0) -> dict:
    obs, mask, actions, reward, next_obs, next_mask, done, gstate, next_gstate = buf.sample(batch_size)
    B, n_agents = actions.shape

    obs_t = torch.from_numpy(obs).to(dev)
    mask_t = torch.from_numpy(mask).to(dev)
    act_t = torch.from_numpy(actions).long().to(dev)
    reward_t = torch.from_numpy(reward).to(dev)
    next_obs_t = torch.from_numpy(next_obs).to(dev)
    next_mask_t = torch.from_numpy(next_mask).to(dev)
    done_t = torch.from_numpy(done).to(dev)
    gstate_t = torch.from_numpy(gstate).to(dev)
    next_gstate_t = torch.from_numpy(next_gstate).to(dev)

    q_all = qnet(obs_t.view(B * n_agents, -1)).view(B, n_agents, NUM_ACTIONS)
    chosen_q = q_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, n_agents)
    q_tot = mixer(chosen_q, gstate_t)

    with torch.no_grad():
        next_q_all = target_qnet(next_obs_t.view(B * n_agents, -1)).view(B, n_agents, NUM_ACTIONS)
        next_q_all = next_q_all.masked_fill(next_mask_t == 0, NEG_INF)
        if double_q:
            online_next = qnet(next_obs_t.view(B * n_agents, -1)).view(B, n_agents, NUM_ACTIONS)
            online_next = online_next.masked_fill(next_mask_t == 0, NEG_INF)
            next_actions = online_next.argmax(dim=-1, keepdim=True)
            next_max_q = next_q_all.gather(-1, next_actions).squeeze(-1)
        else:
            next_max_q = next_q_all.max(dim=-1).values  # (B, n_agents)
        next_q_tot = target_mixer(next_max_q, next_gstate_t)
        target = reward_t + gamma * (1.0 - done_t) * next_q_tot

    td_loss = F.smooth_l1_loss(q_tot, target) if huber_loss else F.mse_loss(q_tot, target)
    bc_loss = None
    loss = td_loss
    if bc_anchor is not None and bc_anchor_coef > 0:
        from core import ACTION_IDLE, ACTION_OPERATION_BASE
        anchor_obs, anchor_mask, anchor_actions = bc_anchor
        count = min(bc_anchor_batch_size, len(anchor_actions))
        indices = np.random.randint(0, len(anchor_actions), size=count)
        anchor_obs_t = torch.from_numpy(anchor_obs[indices]).to(dev)
        anchor_mask_t = torch.from_numpy(anchor_mask[indices]).to(dev)
        anchor_actions_t = torch.from_numpy(anchor_actions[indices]).long().to(dev)
        anchor_logits = qnet(anchor_obs_t).masked_fill(anchor_mask_t == 0, NEG_INF)
        anchor_weights = torch.where(anchor_actions_t == ACTION_IDLE, 1.0, bc_non_idle_weight)
        anchor_weights = torch.where(anchor_actions_t >= ACTION_OPERATION_BASE,
                                     bc_operation_weight, anchor_weights)
        per_sample = F.cross_entropy(anchor_logits, anchor_actions_t, reduction="none")
        bc_loss = (per_sample * anchor_weights).sum() / anchor_weights.sum()
        loss = td_loss + bc_anchor_coef * bc_loss
    parameter_anchor_loss = None
    if bc_parameter_anchor is not None and bc_parameter_anchor_coef > 0:
        parameter_anchor_loss = sum(
            (parameter - bc_parameter_anchor[name]).square().mean()
            for name, parameter in qnet.named_parameters())
        loss = loss + bc_parameter_anchor_coef * parameter_anchor_loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(qnet.parameters()) + list(mixer.parameters()), 10.0)
    opt.step()
    result = {"td_loss": float(td_loss.item()), "mean_q_tot": float(q_tot.mean().item())}
    if bc_loss is not None:
        result["bc_anchor_loss"] = float(bc_loss.item())
    if parameter_anchor_loss is not None:
        result["bc_parameter_anchor_loss"] = float(parameter_anchor_loss.item())
    return result


if __name__ == "__main__":
    qnet, mixer, history = train_value_decomposition("qmix", total_steps=2000)
    for h in history[::5]:
        print(h)
