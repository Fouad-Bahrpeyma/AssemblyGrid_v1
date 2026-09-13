"""Protocol-matched from-scratch IPPO/MAPPO coalition runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--env-root", type=Path, required=True)
parser.add_argument("--stable-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--method", choices=("ippo", "mappo"), required=True)
parser.add_argument("--total-steps", type=int, default=1536000)
parser.add_argument("--critic-warmup-steps", type=int, default=3200)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--direct-easy", action="store_true")
parser.add_argument("--direct-medium", action="store_true",
                    help="train the entire budget directly on coalition/medium")
parser.add_argument("--direct-hard", action="store_true",
                    help="train the entire budget directly on coalition/hard")
args = parser.parse_args()
for path in (args.env_root / "env", args.env_root / "assgrdims3a",
             args.stable_root, HERE):
    sys.path.insert(0, str(path))

from encoding import OBS_DIM, global_state_dim
from marl.common import RolloutBuffer, make_env, masked_categorical
from marl_reference import _eval_local_model
from stable_ppo import (StablePPOConfig, StructuredActor, ValueNetwork,
                        agent_conditioned_global_inputs, local_value_inputs, stable_update)
from standards import official_instance_suite
from training_reward import sparse_delivery_reward

PROTOCOL_ID = "assemblygrid-coalition-old-ppo-new-reward-scratch-v2"


def cell(index, difficulty):
    return next(row for row in official_instance_suite(index)
                if row["family"] == "coalition" and row["difficulty"] == difficulty)


def metric_means(rows):
    keys = ("delivered", "completion_rate", "throughput", "coalition_success_rate",
            "team_formation_failures", "team_proposal_churn", "deadlock_ticks",
            "robot_occupancy")
    return {key: float(np.mean([float(row.get(key, 0.0) or 0.0) for row in rows]))
            for key in keys}


def evaluate(actor, difficulty, phase, indices):
    rows = [_eval_local_model(actor, cell(i, difficulty)["config"], args.method,
                              max_steps=cell(i, difficulty)["config"].horizon_T,
                              device=args.device) for i in indices]
    return {"phase": phase, "family": "coalition", "difficulty": difficulty,
            "evaluation_role": "final_test" if phase == "final_test" else "validation",
            "instance_indices": list(indices),
            "mean": metric_means(rows), "episodes": rows}


def completed_operations(core):
    return sum(len(product.completed_ops) for product in core.products.values())


def collect(env, actor, critic, value_input_fn, obs, n_steps, device, gamma):
    agents = list(env.agents)
    buffers = {agent: RolloutBuffer() for agent in agents}
    if not hasattr(env, "_aag042_episode"):
        env._aag042_episode = 0; env._aag042_steps = 0; env._aag042_completed = []
        env._aag042_sum = {"canonical": 0., "milestone": 0., "steps": 0,
                          "teams": 0, "operations": 0, "deliveries": 0}
    for _ in range(n_steps):
        active = list(env.agents)
        value_inputs = value_input_fn(env, obs)
        with torch.no_grad():
            obs_t = torch.from_numpy(np.stack([obs[a] for a in active])).to(device)
            mask_t = torch.from_numpy(np.stack([env._last_mask[a] for a in active])).to(device)
            distribution = masked_categorical(actor(obs_t), mask_t)
            action_t = distribution.sample(); logp_t = distribution.log_prob(action_t)
            value_t = critic(torch.from_numpy(np.stack([value_inputs[a] for a in active])).to(device)).squeeze(-1)
        actions = {agent: int(action_t[i]) for i, agent in enumerate(active)}
        before_proposal = {a: env.core._robot(env._agent_to_coord[a]).proposal_id for a in active}
        before_progress = env.core.progress_events
        before_teams = env.core.team_formation_count
        before_ops = completed_operations(env.core)
        before_deliveries = env.core.delivered_count
        before_failures = env.core.team_formation_failures
        before_churn = env.core.team_proposal_churn
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        team_delta = env.core.team_formation_count - before_teams
        op_delta = completed_operations(env.core) - before_ops
        delivery_delta = env.core.delivered_count - before_deliveries
        progress_delta = env.core.progress_events - before_progress
        failure_delta = env.core.team_formation_failures - before_failures
        churn_delta = env.core.team_proposal_churn - before_churn
        # Global verified milestones solve the temporal-credit problem. Delivery
        # remains the dominant objective; handshake rewards are deliberately
        # smaller and cannot be farmed once a team commits.
        team_milestone = (0.10 * progress_delta + 2.0 * team_delta +
                          3.0 * op_delta + 10.0 * delivery_delta -
                          0.25 * failure_delta - 0.05 * churn_delta - 0.001)
        local = {}
        for agent in active:
            rc = env._agent_to_coord[agent]
            after = env.core._robot(rc).proposal_id
            local[agent] = 0.05 if before_proposal[agent] is None and after is not None else 0.0
        for i, agent in enumerate(active):
            shaped = float(rewards[agent]) + team_milestone + local[agent]
            buffers[agent].add(obs[agent], value_inputs[agent], env._last_mask[agent],
                               actions[agent], float(logp_t[i]), float(value_t[i]),
                               shaped, bool(terms[agent] or truncs[agent]))
        summary = env._aag042_sum
        summary["canonical"] += float(next(iter(rewards.values()), 0.0))
        summary["milestone"] += team_milestone + float(np.mean(list(local.values())))
        summary["steps"] += 1; summary["teams"] += team_delta
        summary["operations"] += op_delta; summary["deliveries"] += delivery_delta
        env._aag042_steps += 1
        env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}; obs = next_obs
        if not env.agents:
            env._aag042_completed.append({"episode": env._aag042_episode,
                "end_step": env._aag042_steps, "episode_length": summary["steps"],
                "canonical_return": summary["canonical"],
                "milestone_return": summary["milestone"],
                "shaped_return": summary["canonical"] + summary["milestone"],
                "teams_formed": summary["teams"], "operations_completed": summary["operations"],
                "delivered": summary["deliveries"]})
            env._aag042_episode += 1
            env._aag042_sum = {"canonical": 0., "milestone": 0., "steps": 0,
                               "teams": 0, "operations": 0, "deliveries": 0}
            obs, infos = env.reset(); env._last_mask = {a: infos[a]["action_mask"] for a in env.agents}
    bootstrap = {}
    if env.agents:
        values_in = value_input_fn(env, obs)
        with torch.no_grad():
            values = critic(torch.from_numpy(np.stack([values_in[a] for a in env.agents])).to(device)).squeeze(-1)
        bootstrap = {a: float(values[i]) for i, a in enumerate(env.agents)}
    return buffers, obs, bootstrap


def save_checkpoint(path, actor, critic, actor_opt, critic_opt, steps, stage):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".pt.tmp")
    torch.save({"protocol_id": PROTOCOL_ID, "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(), "actor_optimizer": actor_opt.state_dict(),
                "critic_optimizer": critic_opt.state_dict(), "steps": steps, "stage": stage,
                "behavior_cloning_steps": 0, "expert_trajectories": 0}, temporary)
    temporary.replace(path)


torch.manual_seed(args.seed); np.random.seed(args.seed); device = torch.device(args.device)
actor = StructuredActor(256).to(device)
ppo = StablePPOConfig(steps_per_update=1600, epochs=4, minibatch_size=4096,
                      gamma=.99, gae_lambda=.95, clip_ratio=.10, value_clip=.20,
                      actor_lr=1e-4, critic_lr=3e-4, max_grad_norm=.5,
                      target_kl=.02, entropy_start=.015, entropy_end=.001,
                      clone_kl_start=0., clone_kl_end=0., critic_warmup_steps=args.critic_warmup_steps)
actor_opt = torch.optim.AdamW(actor.parameters(), lr=ppo.actor_lr, eps=1e-5)
if sum((args.direct_easy,args.direct_medium,args.direct_hard)) != 1:
    raise ValueError("choose exactly one direct-training target")
if args.total_steps <= 0 or args.total_steps % (5 * ppo.steps_per_update):
    raise ValueError("total steps must be positive and divisible by 8,000 for five matched realizations")
if args.direct_medium and args.direct_hard:
    raise ValueError("choose only one direct-training target")
target_difficulty = "hard" if args.direct_hard else ("medium" if args.direct_medium else "easy")
if args.smoke:
    stages = [(target_difficulty, 1600, 1)]
else:
    stages = [(target_difficulty, args.total_steps // 5, 5)]
total_steps = sum(per_instance * count for _, per_instance, count in stages)
validation_indices = range(200,205); final_test_indices = range(1000,1050)
history = []; phases = [evaluate(actor, target_difficulty, "random_initialization", validation_indices)]
steps_done = 0; critic = critic_opt = None; started = time.perf_counter()
for stage_index, (difficulty, per_instance, count) in enumerate(stages):
    for realization in range(count):
        cfg = replace(cell(realization, difficulty)["config"],
                      generation_seed=80_000 + args.seed * 100 + stage_index * 10 + realization,
                      execution_seed=90_000 + args.seed * 100 + stage_index * 10 + realization,
                      official_result=False)
        env = make_env(cfg, training_reward=sparse_delivery_reward())
        value_input_fn = local_value_inputs if args.method == "ippo" else agent_conditioned_global_inputs
        critic_dim = OBS_DIM if args.method == "ippo" else global_state_dim(env.core) + OBS_DIM
        if critic is None or critic.norm.normalized_shape[0] != critic_dim:
            critic = ValueNetwork(critic_dim, 256).to(device)
            critic_opt = torch.optim.AdamW(critic.parameters(), lr=ppo.critic_lr, eps=1e-5)
        obs = env._last_obs; block_done = 0
        while block_done < per_instance:
            n = min(ppo.steps_per_update, per_instance - block_done)
            buffers, obs, bootstrap = collect(env, actor, critic, value_input_fn, obs, n, device, ppo.gamma)
            steps_done += n; block_done += n; fraction = steps_done / total_steps
            entropy = ppo.entropy_start + fraction * (ppo.entropy_end - ppo.entropy_start)
            stats = stable_update(actor, critic, None, actor_opt, critic_opt, buffers,
                                  bootstrap, device, ppo, entropy, 0.,
                                  update_actor=steps_done > ppo.critic_warmup_steps)
            episodes = list(env._aag042_completed); env._aag042_completed.clear(); env._last_obs = obs
            stats.update({"steps": steps_done, "stage": difficulty,
                          "realization": realization, "completed_episodes": episodes})
            history.append(stats)
            if len(history) % 10 == 0:
                recent = [e for h in history[-10:] for e in h["completed_episodes"]]
                print(json.dumps({"steps": steps_done, "total_steps": total_steps,
                    "stage": difficulty, "realization": realization,
                    "mean_teams": float(np.mean([e["teams_formed"] for e in recent])) if recent else 0.,
                    "mean_operations": float(np.mean([e["operations_completed"] for e in recent])) if recent else 0.,
                    "mean_deliveries": float(np.mean([e["delivered"] for e in recent])) if recent else 0.,
                    "actor_grad_norm": stats["actor_grad_norm"]}), flush=True)
    stage_eval = evaluate(actor, difficulty, f"{difficulty}_complete", validation_indices)
    phases.append(stage_eval)
    save_checkpoint(args.output_dir / f"checkpoint_after_{difficulty}.pt", actor, critic,
                    actor_opt, critic_opt, steps_done, difficulty)

final = evaluate(actor, target_difficulty, "final_test", final_test_indices); phases.append(final)
checkpoint_path = args.output_dir / "aag042_final.pt"
save_checkpoint(checkpoint_path, actor, critic, actor_opt, critic_opt, steps_done, "complete")
payload = {"protocol_id": PROTOCOL_ID, "terminal_status": "complete",
           "method": args.method, "family": "coalition",
           "target_difficulty": target_difficulty, "trained_from_scratch": True,
           "behavior_cloning_steps": 0, "expert_trajectories": 0,
           "training_steps": steps_done, "expected_training_episodes": steps_done // 400,
           "direct_easy": bool(args.direct_easy), "direct_medium": bool(args.direct_medium),
           "direct_hard": bool(args.direct_hard),
           "stages": [{"difficulty": d, "steps_per_realization": s, "realizations": c}
                      for d, s, c in stages], "ppo_config": asdict(ppo),
           "phase_history": phases, "training_history": history,
           "training_seconds": time.perf_counter() - started,
           "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size,
              "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}}
args.output_dir.mkdir(parents=True, exist_ok=True)
output = args.output_dir / "aag042_result.json"; temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"); temporary.replace(output)
print(json.dumps({"result": str(output), f"{target_difficulty}_held": final["mean"]}), flush=True)
