from __future__ import annotations
import numpy as np
from dataclasses import replace
from ag3_common import held_evaluation_config, scenario_config, record, write_json

REFERENCE_PROGRESS_SHAPING = 0.05
REFERENCE_REWARD_PROFILE = "sparse-delivery-v1"

def _deps():
    try:
        __import__("torch")
    except Exception as exc:
        raise RuntimeError("AssGrDims#3.A.3 requires PyTorch") from exc

def _eval_local_model(model, cfg, kind, max_steps=5000, device="cpu"):
    import torch
    from training_reward import sparse_delivery_reward
    try:
        from pettingzoo_env import AssemblyGridParallelEnv
        backend = "pettingzoo"
    except ModuleNotFoundError as exc:
        if exc.name not in {"gymnasium", "pettingzoo", "pettingzoo.utils.env"}:
            raise
        from marl.core_parallel_env import CoreParallelEnv as AssemblyGridParallelEnv
        backend = "core"
    env=AssemblyGridParallelEnv(cfg,training_reward=sparse_delivery_reward())
    obs,infos=env.reset(); masks={a:infos[a]["action_mask"] for a in env.agents}
    steps=0
    with torch.no_grad():
        while env.agents and steps<max_steps:
            agents=list(env.agents)
            if kind in ("ippo","mappo","qmix"):
                o=torch.from_numpy(np.stack([obs[a] for a in agents])).float().to(device)
                logits=model(o).cpu().numpy()
                actions={}
                for i,a in enumerate(agents):
                    q=np.where(masks[a]==0,-1e30,logits[i])
                    actions[a]=int(np.argmax(q))
            elif kind=="graph":
                from marl.graph_marl import build_adjacency_matrix
                order=agents
                A=build_adjacency_matrix(env.core,order,env._agent_to_coord).to(device)
                o=torch.from_numpy(np.stack([obs[a] for a in order])).float().to(device)
                logits=model(o,A).cpu().numpy(); actions={}
                for i,a in enumerate(order):
                    q=np.where(masks[a]==0,-1e30,logits[i]); actions[a]=int(np.argmax(q))
            else: raise ValueError(kind)
            obs,rewards,terms,truncs,infos=env.step(actions)
            if env.agents: masks={a:infos[a]["action_mask"] for a in env.agents}
            steps+=1
    metrics={"t":env.core.t,"delivered":env.core.delivered_count,"spawned":env.core.spawned_count}
    metrics.update(env.core.metrics()); return metrics

def run(algorithms, seeds, total_steps=4000, scenario="continuing_standard", max_eval_steps=5000, device="cpu"):
    _deps()
    from training_reward import sparse_delivery_reward
    rows=[]
    reward_cfg=sparse_delivery_reward()
    for seed in seeds:
        cfg=scenario_config(scenario,seed)
        for algo in algorithms:
            kwargs=dict(cfg=cfg,total_steps=total_steps,seed=seed,device=device,
                        training_reward=reward_cfg,progress_shaping_coef=REFERENCE_PROGRESS_SHAPING)
            if algo=="ippo":
                from marl.ippo import train_ippo
                model,_,history=train_ippo(**kwargs)
            elif algo=="mappo":
                from marl.mappo import train_mappo
                model,_,history=train_mappo(**kwargs)
            elif algo=="qmix":
                from marl.vdn_qmix import train_value_decomposition
                model,_,history=train_value_decomposition("qmix",train_every=8,epsilon_decay_steps=15000,**kwargs)
            elif algo=="graph":
                from marl.graph_marl import train_graph_marl
                model,_,history=train_graph_marl(**kwargs)
            else: raise ValueError(algo)
            # Same paired evaluation seed across methods, distinct from training/generation seed.
            eval_cfg=held_evaluation_config(scenario, seed)
            metrics=_eval_local_model(model,eval_cfg,algo,max_steps=max_eval_steps,device=device)
            try:
                from marl.common import PARALLEL_ENV_BACKEND
            except Exception:
                PARALLEL_ENV_BACKEND = "unknown"
            rows.append(record(scenario,seed,algo,eval_cfg,metrics,
                               training={"steps":total_steps,"reward_profile":REFERENCE_REWARD_PROFILE,
                                         "progress_shaping_coef":REFERENCE_PROGRESS_SHAPING,
                                         "parallel_env_backend":PARALLEL_ENV_BACKEND,
                                         "ppo_minibatch_size":2048 if algo in ("ippo","mappo") else None,
                                         "ppo_epochs":4 if algo in ("ippo","mappo") else None,
                                         "qmix_train_every":8 if algo=="qmix" else None,
                                         "qmix_epsilon_decay_steps":15000 if algo=="qmix" else None,
                                         "final_stats":history[-1] if history else {}}))
            print(scenario,seed,algo,"delivered",metrics.get("delivered"),"throughput",metrics.get("throughput"))
    p=write_json("3a_marl_eval.json",rows); print("Wrote",p); return rows
