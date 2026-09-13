"""Normative instance classification and result-record helpers."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

from benchmark import instance_definition
from core import EnvConfig
from profiles import apply_geometry_profile
from recipe import RecipeLibrary, default_recipe_library


OFFICIAL_FAMILIES = ("flow", "coalition", "concurrency")
# Geometry is a historical workload family kept outside the official suite.  It
# varies release/WIP pressure together with conflict radius, so it never isolated
# geometry as a single factor, and coalition/medium and geometry/medium are the
# same problem under two labels.  In AssemblyGrid v1 it is a declared historical
# extension (historical_realizations.json), excluded from all official aggregates.
# The conflict-radius-only causal check lives in matched_geometry_control_suite().
HISTORICAL_FAMILIES = ("geometry",)
DIFFICULTY_LEVELS = ("easy", "medium", "hard")
FIDELITY_LABELS = (
    "abstract-v1",
    "geometry-off-control",
    "ur10-case-v1",
    "ur10-kinematic-validation-v1",
)

MANDATORY_METRIC_GROUPS = {
    "completion": ("delivered", "completion_rate", "unfinished_count"),
    "throughput": ("throughput",),
    "time": ("makespan", "restricted_mean_flow_time"),
    "coalition": ("coalition_success_rate", "coalition_formation_latency"),
    "geometry": ("geometry_rejection_count", "geometry_conflict_count", "geometry_delay_ticks"),
}

OPTIONAL_METRIC_GROUPS = {
    "productive_concurrency": ("productive_concurrency_utilization", "parallelism_exact_opportunity_samples"),
    "backlog_age": ("unfinished_mean_age", "unfinished_max_age"),
    "routing": ("handoffs_per_delivered_product", "motion_path_length"),
    "workload": ("robot_occupancy", "team_load_cv"),
    "compute": ("environment_steps_per_second", "policy_inference_seconds", "planner_seconds"),
}


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def instance_sha256(cfg) -> str:
    return hashlib.sha256(canonical_json_bytes(instance_definition(cfg))).hexdigest()


def structural_difficulty(cfg, recipes: RecipeLibrary,
                          declared_level: Optional[str] = None,
                          family: Optional[str] = None) -> dict:
    """Return algorithm-independent descriptors for a predeclared level.

    Difficulty is ordinal *within a mechanism family*, not a global scalar:
    an easy geometry control and an easy flow control need not have comparable
    operation counts.  The suite freezes the label before rollout and records
    the descriptors that justify that ordering; no algorithm outcome enters.
    """
    selected = [recipes.get(rid) for rid in cfg.recipe_ids]
    operations = [op for recipe in selected for op in recipe.operations]
    op_count = len(operations)
    depth = max((recipe.nominal_critical_path for recipe in selected), default=0)
    max_arity = max((op.kappa for op in operations), default=1)
    coalition_fraction = sum(op.kappa > 1 for op in operations) / max(1, op_count)
    alternative_groups = sum(len(op.all_alternative_groups) for op in operations)
    parallel_width_proxy = max(1, op_count - depth // max(1, min(op.duration for op in operations)))
    pressure = cfg.max_wip / max(1, cfg.M * cfg.N)
    spacing_to_reach = cfg.cell_spacing / max(1e-9, cfg.robot_models[0].reach_radius)
    geometric_conflict_ratio = cfg.workspace_conflict_radius / max(1e-9, cfg.cell_spacing)
    if declared_level is not None and declared_level not in DIFFICULTY_LEVELS:
        raise ValueError(f"unknown declared difficulty {declared_level!r}")
    bases = {
        "flow": "recipe depth, grid size, WIP and release pressure",
        "coalition": "required arity and optional support-role complexity",
        "concurrency": "parallel branch opportunity, WIP and release pressure",
        "geometry": (
            "predeclared geometry-stress structure (conflict radius plus the published "
            "release/WIP settings); use matched_geometry_control_suite() for a causal "
            "conflict-radius-only control"
        ),
    }
    return {
        "level": declared_level or "unclassified",
        "comparison_scope": "within_family",
        "basis": bases.get(family, "published structural descriptors"),
        "recipe_operation_count": op_count,
        "nominal_critical_path": depth,
        "maximum_required_arity": max_arity,
        "coalition_operation_fraction": coalition_fraction,
        "alternative_group_count": alternative_groups,
        "parallel_width_proxy": parallel_width_proxy,
        "wip_per_robot": pressure,
        "spacing_to_reach": spacing_to_reach,
        "geometric_conflict_ratio": geometric_conflict_ratio,
    }


def _suite_base(seed: int) -> EnvConfig:
    return EnvConfig(
        M=5, N=6, topology="moore", geometry_profile="abstract-v1",
        profile_enforced=True, official_result=True,
        motion_planning=True, trajectory_conflicts=True,
        generation_seed=10_000 + seed, execution_seed=20_000 + seed,
        seed=None, horizon_T=400, spawn_interval=5, max_wip=5,
    )


def _cells_to_records(cells, seed: int) -> list[dict]:
    return [
        {
            "instance_id": f"ag32-{family}-{difficulty}-s{seed:02d}",
            "family": family,
            "difficulty": difficulty,
            "config": cfg,
        }
        for family, difficulty, cfg in cells
    ]


def official_instance_suite(seed: int = 0) -> list[dict]:
    """Return the frozen AssemblyGrid v1 official suite: 3 families x 3 levels.

    Families: flow, coalition, concurrency.  Geometry is a historical extension,
    see ``historical_instance_suite``.
    """
    base = _suite_base(seed)
    cells = [
        ("flow", "easy", replace(base, M=4, N=5, recipe_ids=("solo_inspection",), max_wip=3, spawn_interval=8)),
        ("flow", "medium", replace(base, recipe_ids=("flow_serial_3",), max_wip=5, spawn_interval=5)),
        ("flow", "hard", replace(base, M=6, N=8, recipe_ids=("flow_serial_5",), max_wip=10, spawn_interval=2)),
        ("coalition", "easy", replace(base, recipe_ids=("pair_assembly",), max_wip=4, spawn_interval=7)),
        ("coalition", "medium", replace(base, recipe_ids=("supported_insert",), max_wip=4, spawn_interval=5)),
        ("coalition", "hard", replace(base, M=6, N=8, recipe_ids=("standard_abc",), max_wip=1, spawn_interval=8)),
        ("concurrency", "easy", replace(base, recipe_ids=("parallel_branch_light",), max_wip=2, spawn_interval=10)),
        ("concurrency", "medium", replace(base, recipe_ids=("parallel_branch_light",), max_wip=5, spawn_interval=5)),
        ("concurrency", "hard", replace(base, M=6, N=8, recipe_ids=("parallel_branch_light",), max_wip=10, spawn_interval=2)),
    ]
    return _cells_to_records(cells, seed)


def historical_instance_suite(seed: int = 0) -> list[dict]:
    """Return the historical geometry family (3 levels).

    Retained for traceability only.  Not part of the AssemblyGrid v1 official suite
    and excluded from every official aggregate.  Instance definitions are frozen, so
    the published digests are stable.
    """
    base = _suite_base(seed)
    cells = [
        ("geometry", "easy", apply_geometry_profile(replace(base, recipe_ids=("supported_insert",), max_wip=3, spawn_interval=7), "abstract-v1", "conflict-low")),
        ("geometry", "medium", apply_geometry_profile(replace(base, recipe_ids=("supported_insert",), max_wip=4, spawn_interval=5), "abstract-v1", "conflict-medium")),
        ("geometry", "hard", apply_geometry_profile(replace(base, recipe_ids=("supported_insert",), max_wip=3, spawn_interval=5), "abstract-v1", "conflict-high")),
    ]
    return _cells_to_records(cells, seed)



def matched_geometry_control_suite(seed: int = 0) -> list[dict]:
    """Return a non-headline, conflict-radius-only geometry mechanism control.

    The historical geometry family varies release/WIP pressure across its three
    structural levels, so it cannot isolate geometry as a single factor.  Here every
    problem-defining field is held fixed except the declared geometry variant (and
    therefore ``workspace_conflict_radius``), so this suite may be used for a clean
    causal mechanism check.

    The suite is intentionally marked ``official_result=False`` and uses distinct
    IDs.  It is supplementary validation evidence, not a replacement for or silent
    mutation of the frozen historical result matrix.
    """
    base = EnvConfig(
        M=5, N=6, topology="moore", geometry_profile="abstract-v1",
        profile_enforced=True, official_result=False,
        motion_planning=True, trajectory_conflicts=True,
        generation_seed=10_000 + seed, execution_seed=20_000 + seed,
        seed=None, horizon_T=400, spawn_interval=5, max_wip=4,
        recipe_ids=("supported_insert",),
    )
    variants = (
        ("easy", "conflict-low"),
        ("medium", "conflict-medium"),
        ("hard", "conflict-high"),
    )
    return [
        {
            "instance_id": f"ag32-geometry-matched-{difficulty}-s{seed:02d}",
            "family": "geometry",
            "difficulty": difficulty,
            "config": apply_geometry_profile(base, "abstract-v1", variant),
            "control_kind": "matched_conflict_radius_only",
        }
        for difficulty, variant in variants
    ]


def _catalogue_records(suite_fn, seed_indices) -> list[dict]:
    library = default_recipe_library()
    records = []
    for seed in seed_indices:
        for cell in suite_fn(seed):
            cfg = cell["config"]
            records.append({
                "instance_id": cell["instance_id"],
                "family": cell["family"],
                "difficulty": cell["difficulty"],
                "instance_definition": instance_definition(cfg),
                "instance_sha256": instance_sha256(cfg),
                "structural_difficulty": structural_difficulty(
                    cfg, library, cell["difficulty"], cell["family"]
                ),
                "reference_execution_seed": cfg.execution_seed,
                "reference_run_configuration": asdict(cfg),
            })
    return records


def official_suite_catalogue(seed_indices=range(5)) -> dict:
    return {
        "schema": "assemblygrid-official-realizations-v4",
        "suite": "assemblygrid-aag032-rc1",
        "official_families": list(OFFICIAL_FAMILIES),
        "note": "v4 relocates the geometry family to historical_realizations.json; official aggregates use these records only.",
        "realizations": _catalogue_records(official_instance_suite, seed_indices),
    }


def historical_suite_catalogue(seed_indices=range(5)) -> dict:
    return {
        "schema": "assemblygrid-official-realizations-v4",
        "suite": "assemblygrid-aag032-rc1",
        "historical_families": list(HISTORICAL_FAMILIES),
        "note": "Pre-v1 AAG032 geometry family. Traceability only; excluded from every AssemblyGrid v1 official aggregate.",
        "realizations": _catalogue_records(historical_instance_suite, seed_indices),
    }


def build_result_record(*, benchmark_version: str, cfg, instance_id: str,
                        family: str, difficulty: str, method_name: str,
                        execution_information: str, training_information: str,
                        algorithm_seed: Optional[int], environment_steps: int,
                        evaluation_episodes: int, fidelity: str, status: str,
                        metrics: Mapping, reward: Optional[Mapping] = None) -> dict:
    if family not in OFFICIAL_FAMILIES + HISTORICAL_FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    if fidelity not in FIDELITY_LABELS:
        raise ValueError(f"unknown fidelity {fidelity!r}")
    missing = sorted({x for group in MANDATORY_METRIC_GROUPS.values() for x in group} - set(metrics))
    if missing:
        raise ValueError(f"result is missing mandatory metrics: {missing}")
    return {
        "schema": "assemblygrid-result-v1",
        "benchmark": "AssemblyGrid",
        "benchmark_version": benchmark_version,
        "instance": {
            "id": instance_id,
            "sha256": instance_sha256(cfg),
            "family": family,
            "difficulty": difficulty,
            "episode_mode": cfg.episode_mode,
        },
        "seeds": {
            "generation": cfg.generation_seed if cfg.generation_seed is not None else cfg.seed,
            "execution": cfg.execution_seed if cfg.execution_seed is not None else cfg.seed,
            "algorithm": algorithm_seed,
        },
        "method": {
            "name": method_name,
            "execution_information": execution_information,
            "training_information": training_information,
        },
        "budget": {"environment_steps": environment_steps, "evaluation_episodes": evaluation_episodes},
        "fidelity": fidelity,
        "status": status,
        "reward": dict(reward) if reward is not None else None,
        "metrics": dict(metrics),
    }


def save_result_record(record: Mapping, path) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
