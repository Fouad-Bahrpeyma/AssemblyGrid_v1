from __future__ import annotations
import json, math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env"
RESULTS = Path(__file__).resolve().parent / "results"

import sys
if str(ENV) not in sys.path:
    sys.path.insert(0, str(ENV))

from benchmark import load_config

SCENARIOS = {
    "finite_standard": ROOT / "configs" / "ag_core.json",
    "continuing_standard": ROOT / "configs" / "ag_core.json",
    "parallel_branch": ROOT / "configs" / "ag_parallel.json",
    "xor_batch": ROOT / "configs" / "ag_core.json",
    "marl_pair_compact": ROOT / "configs" / "ag_core.json",
}


def scenario_config(name: str, seed: int):
    cfg = load_config(SCENARIOS[name])
    if name == "finite_standard":
        cfg = replace(cfg, episode_mode="batch", batch_Z=4, horizon_T=600,
                      recipe_ids=("standard_ab",), max_wip=4, spawn_interval=3)
    elif name == "continuing_standard":
        cfg = replace(cfg, episode_mode="horizon", horizon_T=250,
                      recipe_ids=("standard_ab",), max_wip=6, spawn_interval=4)
    elif name == "parallel_branch":
        cfg = replace(cfg, episode_mode="horizon", horizon_T=300,
                      recipe_ids=("parallel_branch",), max_wip=8, spawn_interval=3)
    elif name == "xor_batch":
        cfg = replace(cfg, episode_mode="batch", batch_Z=4, horizon_T=600,
                      recipe_ids=("alternative_route",), max_wip=4, spawn_interval=3)
    elif name == "marl_pair_compact":
        # Minimal cooperative learnability case: four agents in one Moore clique
        # must form k=2 teams to assemble and deliver pair_assembly products.
        # This is an experimental reference scenario, not a new benchmark rule.
        cfg = replace(cfg, M=2, N=2, episode_mode="horizon", horizon_T=200,
                      recipe_ids=("pair_assembly",), max_wip=4, spawn_interval=4)
    return replace(cfg, generation_seed=seed, execution_seed=seed, seed=seed,
                   official_result=False)



def held_evaluation_seeds(seed: int) -> tuple[int, int]:
    """Frozen first-paper held realization/trajectory pair for compact evaluation."""
    return 10_000 + int(seed), 20_000 + int(seed)


def held_evaluation_config(name: str, seed: int):
    """Return a scenario config on the frozen held evaluation realization.

    The training/method seed remains ``seed`` for bookkeeping; generation and
    execution streams are the first-paper held pair shared by every compared
    method.
    """
    cfg = scenario_config(name, seed)
    g, e = held_evaluation_seeds(seed)
    return replace(cfg, generation_seed=g, execution_seed=e, seed=seed)

def record(scenario: str, seed: int, method: str, cfg, metrics: dict, **extra):
    return {"assgrdims":"3.A", "scenario":scenario, "seed":seed, "method":method,
            "config":asdict(cfg), "metrics":metrics, **extra}


def write_json(name: str, rows: list) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return p

CORE_KPIS = ["completion_rate", "throughput", "makespan", "mean_flow_time", "wip_mean",
             "productive_parallelism_utilization"]
