"""AssGrDims#2 conformance audits for AssemblyGrid v1.

These are mechanism checks, not publication-performance experiments.
Run from ``Experiments/env`` with::

    python audits.py --all --output assgrdims2_audit_results.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable

from benchmark import track_config
from official import load_catalogue, load_official_config
from core import (
    ACTION_IDLE,
    AssemblyGridCore,
    EnvConfig,
    MAX_OPERATION_OPTIONS,
    MAX_PICK_OPTIONS,
    Product,
)
from policies import Tier1Policy
from recipe import OperationSpec, RecipeLibrary, RecipeSpec


DEFAULT_TRACKS = (
    "AG-Core", "AG-Team", "AG-Parallel", "AG-Recipes",
    "AG-Architecture", "AG-Motion", "AG-Scale",
)
DEFAULT_SEEDS = (0, 1, 7)


def addressability_audit(
    tracks: Iterable[str] = DEFAULT_TRACKS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    ticks: int = 180,
) -> dict:
    """Measure pre-encoding local option counts over representative v1 runs.

    ``core._pick_options_for`` and ``core._operation_options_for`` raise on
    overflow, so an audit can never silently pass a truncated state.
    """
    rows = []
    global_pick = global_op = 0
    overflow = []
    for track in tracks:
        for seed in seeds:
            cfg = track_config(track, seed=seed, generation_seed=seed, execution_seed=seed)
            if cfg.episode_mode == "horizon":
                cfg = EnvConfig(**{**asdict(cfg), "horizon_T": max(ticks, cfg.horizon_T),
                                   "robot_models": cfg.robot_models})
            policy = Tier1Policy("parallel_aware", seed=seed)
            core = AssemblyGridCore(cfg, priority_fn=policy.priority_fn)
            obs = core.reset(seed=seed)
            max_pick = max_op = 0
            ran = 0
            try:
                for _ in range(ticks):
                    cands = core.candidate_operations()
                    for rc in obs:
                        max_pick = max(max_pick, len(core._pick_options_for(rc)))
                        max_op = max(max_op, sum(rc in c.coalition for c in cands))
                    obs, _reward, term, trunc, _info = core.step(policy.act(core, obs))
                    ran += 1
                    if term or trunc:
                        break
            except RuntimeError as exc:
                overflow.append({"track": track, "seed": seed, "error": str(exc)})
            global_pick = max(global_pick, max_pick)
            global_op = max(global_op, max_op)
            rows.append({"track": track, "seed": seed, "ticks": ran,
                         "max_pick_options": max_pick, "max_operation_options": max_op})
    return {
        "status": "PASS" if not overflow else "FAIL",
        "capacity": {"pick": MAX_PICK_OPTIONS, "operation": MAX_OPERATION_OPTIONS},
        "observed_max": {"pick": global_pick, "operation": global_op},
        "overflow": overflow,
        "runs": rows,
        "interpretation": (
            "Representative pre-freeze v1 track audit. Runtime overflow is an explicit error, "
            "never silent truncation; official release realizations must be re-audited when frozen."
        ),
    }



def official_addressability_audit(catalogue_path, ticks: int = 180) -> dict:
    """Audit addressability on every frozen official realization.

    This is the post-freeze counterpart to :func:`addressability_audit`.  It
    verifies the exact published instance catalogue rather than proxy track
    configurations.  The same runtime overflow guards make silent truncation
    impossible.
    """
    catalogue_path = Path(catalogue_path)
    catalogue = load_catalogue(catalogue_path)
    rows = []
    overflow = []
    global_pick = global_op = 0
    for record in catalogue.get("realizations", []):
        instance_id = record["instance_id"]
        cfg = load_official_config(catalogue_path, instance_id)
        policy = Tier1Policy("parallel_aware", seed=int(record.get("seed_index", 0)))
        core = AssemblyGridCore(cfg, priority_fn=policy.priority_fn)
        obs = core.reset(seed=cfg.execution_seed)
        max_pick = max_op = 0
        ran = 0
        try:
            for _ in range(ticks):
                cands = core.candidate_operations()
                for rc in obs:
                    max_pick = max(max_pick, len(core._pick_options_for(rc)))
                    max_op = max(max_op, sum(rc in c.coalition for c in cands))
                obs, _reward, term, trunc, _info = core.step(policy.act(core, obs))
                ran += 1
                if term or trunc:
                    break
        except RuntimeError as exc:
            overflow.append({"instance_id": instance_id, "error": str(exc)})
        global_pick = max(global_pick, max_pick)
        global_op = max(global_op, max_op)
        rows.append({
            "instance_id": instance_id,
            "family": record.get("family"),
            "seed_index": record.get("seed_index"),
            "ticks": ran,
            "max_pick_options": max_pick,
            "max_operation_options": max_op,
        })
    return {
        "status": "PASS" if rows and not overflow else "FAIL",
        "catalogue": str(catalogue_path),
        "realizations": len(rows),
        "capacity": {"pick": MAX_PICK_OPTIONS, "operation": MAX_OPERATION_OPTIONS},
        "observed_max": {"pick": global_pick, "operation": global_op},
        "overflow": overflow,
        "runs": rows,
        "interpretation": (
            "Post-freeze audit of every official v1 realization. Runtime overflow is an "
            "explicit error; PASS therefore verifies that no official realization silently "
            "drops eligible pick or operation choices during the audited trajectory."
        ),
    }

def _exact_arity_library(k: int) -> RecipeLibrary:
    roles = tuple(["holder"] + [f"support_{i}" for i in range(1, k)])
    op = OperationSpec(
        id=f"joint_{k}", kind="assemble", inputs=("A",), output="FINAL",
        kappa=k, kappa_max=k, roles=roles, role_inputs={"holder": "A"},
        duration=2,
    )
    return RecipeLibrary((RecipeSpec(id=f"arity_{k}", raw_tokens=("A",),
                                     operations=(op,), final_token="FINAL"),))


def _inject_ready_product(core: AssemblyGridCore, recipe_id: str) -> None:
    core.products.clear()
    core.spawned_count = 1
    core.delivered_count = 0
    p = Product(0, recipe_id, core.t, 999, source_rows={"A": 0})
    p.raw_picked.add("A")
    p.token_positions["A"] = (0, 0)
    core.products[0] = p
    core._robot((0, 0)).holding = (0, "A")
    core.next_product_id = 1
    core._invalidate()


def coalition_formation_sanity(seeds: Iterable[int] = range(20), max_ticks: int = 12) -> dict:
    """Check exact-arity k=2,3,4 formation with and without support visibility.

    This deliberately checks mechanism viability rather than algorithm quality.
    Equal outcomes are possible because deterministic candidate identity/order
    remains a focal mechanism in the no-support condition.
    """
    conditions: Dict[str, dict] = {}
    for k in (2, 3, 4):
        for support_visible in (True, False):
            key = f"k{k}-{'support' if support_visible else 'no-support'}"
            success = 0
            latencies = []
            failures = []
            churn = []
            for seed in seeds:
                lib = _exact_arity_library(k)
                cfg = EnvConfig(
                    M=2, N=2, recipe_ids=(f"arity_{k}",), spawn_interval=999,
                    horizon_T=max_ticks + 2, proposal_support_visible=support_visible,
                    generation_seed=seed, execution_seed=seed,
                    motion_planning=True, trajectory_conflicts=True,
                    geometry_profile="abstract-v1", profile_enforced=True,
                )
                policy = Tier1Policy("parallel_aware", seed=seed)
                core = AssemblyGridCore(cfg, priority_fn=policy.priority_fn, recipe_library=lib)
                _inject_ready_product(core, f"arity_{k}")
                obs = core._all_observations()
                formed_at = None
                for tick in range(max_ticks):
                    obs, _r, _term, _trunc, _info = core.step(policy.act(core, obs))
                    if core.team_formation_count > 0:
                        formed_at = tick + 1
                        break
                if formed_at is not None:
                    success += 1
                    latencies.append(formed_at)
                failures.append(core.team_formation_failures)
                churn.append(core.team_proposal_churn)
            n = len(tuple(seeds)) if not isinstance(seeds, range) else len(seeds)
            conditions[key] = {
                "runs": n,
                "successes": success,
                "success_rate": success / max(1, n),
                "mean_ticks_to_form": mean(latencies) if latencies else None,
                "mean_timeouts_or_failures": mean(failures) if failures else 0.0,
                "mean_proposal_churn": mean(churn) if churn else 0.0,
            }
    return {
        "status": "PASS" if all(v["successes"] > 0 for v in conditions.values()) else "FAIL",
        "conditions": conditions,
        "interpretation": (
            "Sanity check only. The no-support condition removes proposal support from both the "
            "local view and support-first ordering, but retains deterministic candidate ordering; "
            "therefore it is not a test of 'no coordination'."
        ),
    }


def run_all(official_catalogue=None) -> dict:
    result = {
        "addressability": addressability_audit(),
        "coalition_formation": coalition_formation_sanity(),
    }
    if official_catalogue:
        result["official_addressability"] = official_addressability_audit(official_catalogue)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run all AssGrDims#2 audits")
    ap.add_argument("--addressability", action="store_true")
    ap.add_argument("--coalitions", action="store_true")
    ap.add_argument("--official-catalogue", type=Path,
                    help="also audit every frozen official realization in this catalogue")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not any((args.all, args.addressability, args.coalitions)):
        args.all = True
    result = {}
    if args.all or args.addressability:
        result["addressability"] = addressability_audit()
    if args.all or args.coalitions:
        result["coalition_formation"] = coalition_formation_sanity()
    if args.official_catalogue:
        result["official_addressability"] = official_addressability_audit(args.official_catalogue)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
