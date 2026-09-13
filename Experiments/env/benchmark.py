"""Benchmark tracks, configuration I/O, matched suites and ID/OOD helpers."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Iterable, Sequence

from core import EnvConfig, TOPOLOGY_MAX_ARITY, URModel
from profiles import apply_geometry_profile


LEGACY_REWARD_FIELDS = {
    "reward_alpha", "reward_beta", "reward_eta", "reward_zeta", "product_value",
}


TRACKS = {
    "AG-Core": {
        "recipe_ids": ("standard_ab",), "topology": "moore",
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
        "robot_failure_prob": 0.0, "operation_failure_prob": 0.0,
    },
    "AG-Team": {
        "recipe_ids": ("standard_ab", "standard_abc"), "formation_timeout": 4,
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
    },
    "AG-Parallel": {
        "recipe_ids": ("standard_ab", "standard_abc", "parallel_branch"),
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
        "spawn_interval": 2, "max_wip": 8,
    },
    "AG-Motion": {
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
    },
    "AG-Recipes": {
        "recipe_ids": ("standard_ab", "standard_abc", "parallel_branch", "solo_inspection", "alternative_route"),
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
    },
    "AG-Skills": {
        "recipe_ids": ("standard_ab", "standard_abc", "parallel_branch"),
        "motion_planning": False, "trajectory_conflicts": False,
        "skill_mode": "learning", "skill_initial_level": 0.65,
    },
    "AG-Scale": {
        "recipe_ids": ("standard_ab", "standard_abc"), "max_wip": 10,
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
    },
    "AG-Generalize": {
        "recipe_ids": ("standard_ab", "standard_abc", "parallel_branch"),
        "motion_planning": False, "trajectory_conflicts": False,
        "held_out_axes": ("geometry", "recipe", "scale"),
    },
    "AG-Robust": {
        "motion_planning": False, "trajectory_conflicts": False,
        "robot_failure_prob": 0.002, "operation_failure_prob": 0.03,
        "handoff_failure_prob": 0.02,
        "duration_noise": 0.10, "stockout_prob": 0.01,
    },
    "AG-Architecture": {
        "recipe_ids": ("pair_assembly",), "topology": "moore",
        "geometry_profile": "abstract-v1", "profile_enforced": True,
        "motion_planning": True, "trajectory_conflicts": True,
    },
}


def track_config(name: str, **overrides) -> EnvConfig:
    if name not in TRACKS:
        raise ValueError(f"unknown track {name!r}")
    data = dict(TRACKS[name]); data.update(overrides)
    return EnvConfig(benchmark_track=name, **data)


def track_suite(name: str, seed: int = 0) -> list[EnvConfig]:
    """Small canonical factor suite for the named track.

    These are manifests, not a claim that every paper must run every cell.
    They make the previously label-only tracks executable and reproducible.
    """
    base = track_config(name, seed=seed)
    if name == "AG-Architecture":
        # A κ=2-only recipe avoids making lower-connectivity architectures
        # trivially infeasible solely because a four-clique is impossible.
        return [replace(base, topology=t) for t in ("line", "von_neumann", "moore")]
    if name == "AG-Scale":
        return [replace(base, M=m, N=n, max_wip=w) for m, n, w in ((3, 5, 4), (5, 8, 8), (8, 12, 14))]
    if name == "AG-Motion":
        # Geometry sweeps use declared, versioned variants rather than free
        # mutation of profile-owned fields.
        return [
            apply_geometry_profile(base, "abstract-v1", variant, profile_enforced=True)
            for variant in ("spacing-0.75", "spacing-1.00", "spacing-1.25")
        ]
    if name == "AG-Parallel":
        return [replace(base, spawn_interval=s) for s in (6, 4, 2, 1)]
    if name == "AG-Recipes":
        return [replace(base, recipe_ids=(rid,)) for rid in base.recipe_ids]
    if name == "AG-Robust":
        return [
            replace(base, robot_failure_prob=0, operation_failure_prob=0, handoff_failure_prob=0, stockout_prob=0),
            base,
            replace(base, robot_failure_prob=0.01, operation_failure_prob=0.08, handoff_failure_prob=0.05,
                    stockout_prob=0.03),
        ]
    return [base]


def save_config(cfg: EnvConfig, path) -> None:
    data = asdict(cfg)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def env_config_from_dict(data) -> EnvConfig:
    """Construct :class:`EnvConfig` from a serialized runtime configuration.

    ``max_coalition_size`` is accepted only as a legacy redundant ceiling and
    is discarded.  The live value is derived from the manipulation topology.
    """
    data = dict(data)
    legacy_reward = sorted(LEGACY_REWARD_FIELDS.intersection(data))
    if legacy_reward:
        raise ValueError(
            "training reward fields are no longer part of EnvConfig: %s. "
            "Move them to an algorithm-layer TrainingRewardConfig." % ", ".join(legacy_reward)
        )
    legacy_cap = data.pop("max_coalition_size", None)
    if legacy_cap is not None and int(legacy_cap) != 4:
        raise ValueError(
            "legacy max_coalition_size was an implementation ceiling fixed at 4; "
            "canonical coalition capacity is now derived from topology"
        )
    data["robot_models"] = tuple(URModel(**{**x, "tools": tuple(x.get("tools", ("gripper",)))}) for x in data.get("robot_models", [{}]))
    for key in ("held_out_axes", "recipe_ids"):
        data[key] = tuple(data.get(key, ()))
    return EnvConfig(**data)


def load_config(path) -> EnvConfig:
    return env_config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def instance_definition(cfg: EnvConfig) -> dict:
    """Return the normative problem/realization definition for hashing.

    Diagnostic limits, labels, validation switches, execution/algorithm seeds,
    and disabled extension parameters are deliberately excluded.  Geometry
    owned by a named profile is represented by profile identity rather than a
    second copy of its numeric fields.
    """
    cfg.validate()
    definition = {
        "schema": "assemblygrid-instance-v1",
        "spatial": {
            "M": cfg.M, "N": cfg.N, "topology": cfg.topology,
            "cooperation_radius": cfg.cooperation_radius,
            "geometry_profile": cfg.geometry_profile,
            "geometry_variant": cfg.geometry_variant,
            "reference_reach": cfg.reference_reach,
            "reference_time": cfg.reference_time,
            "role_target_offset": cfg.role_target_offset,
        },
        "robot_population": ({
            "count": cfg.M * cfg.N,
            "assignment": cfg.robot_model_assignment,
            "model_source": "geometry_profile",
        } if (cfg.profile_enforced or cfg.official_result) else {
            "count": cfg.M * cfg.N,
            "assignment": cfg.robot_model_assignment,
            "models": [asdict(m) for m in cfg.robot_models],
        }),
        "process": {"recipe_ids": list(cfg.recipe_ids)},
        "material_resources": {
            "shelf_cap": cfg.shelf_cap,
            "shelf_replenish_interval": cfg.shelf_replenish_interval,
            "reach_extra_prob": cfg.reach_extra_prob,
            "product_mass_min": cfg.product_mass_min,
            "product_mass_max": cfg.product_mass_max,
            "resource_capacities": dict(sorted(cfg.resource_capacities.items())),
        },
        "demand": {"spawn_interval": cfg.spawn_interval, "max_wip": cfg.max_wip},
        "timing": {
            "pick_duration": cfg.pick_duration, "deliver_duration": cfg.deliver_duration,
            "handoff_duration": cfg.handoff_duration,
        },
        "cooperation": {
            "formation_timeout": cfg.formation_timeout,
            "commitment_horizon": cfg.commitment_horizon,
            "require_bilateral_handoff": cfg.require_bilateral_handoff,
        },
        "information": {"proposal_support_visible": cfg.proposal_support_visible},
        "evaluation_mode": {
            "episode_mode": cfg.episode_mode, "horizon_T": cfg.horizon_T, "batch_Z": cfg.batch_Z,
        },
        "generation_seed": cfg.generation_seed if cfg.generation_seed is not None else cfg.seed,
    }
    if not (cfg.profile_enforced or cfg.official_result):
        definition["geometry_overrides"] = {
            "cell_spacing": cfg.cell_spacing,
            "workspace_conflict_radius": cfg.workspace_conflict_radius,
            "reference_reach": cfg.reference_reach,
            "reference_time": cfg.reference_time,
            "physical_reference_reach_m": cfg.physical_reference_reach_m,
            "role_target_offset": cfg.role_target_offset,
            "retreat_fraction": cfg.retreat_fraction,
            "motion_planning": cfg.motion_planning,
            "trajectory_conflicts": cfg.trajectory_conflicts,
        }
    # Only active extensions belong to the problem definition.
    if cfg.skill_mode != "disabled":
        definition["skills_extension"] = {
            "mode": cfg.skill_mode, "threshold": cfg.skill_threshold,
            "initial_level": cfg.skill_initial_level, "learning_rate": cfg.skill_learning_rate,
        }
    disturbance = {
        "robot_failure_prob": cfg.robot_failure_prob, "repair_duration": cfg.repair_duration,
        "operation_failure_prob": cfg.operation_failure_prob,
        "handoff_failure_prob": cfg.handoff_failure_prob, "duration_noise": cfg.duration_noise,
        "stockout_prob": cfg.stockout_prob,
    }
    if any(v != 0 for k, v in disturbance.items() if k != "repair_duration"):
        definition["disturbance_extension"] = disturbance
    if cfg.allow_inclusive_or_extension:
        definition["inclusive_or_extension"] = True
    return definition


def _is_ood(cfg: EnvConfig, axes: set[str]) -> bool:
    conditions = []
    if "scale" in axes:
        conditions.append((cfg.M, cfg.N) not in {(5, 8), (4, 6)})
    if "robot_model" in axes or "identity" in axes:
        conditions.append(cfg.robot_model_assignment != "homogeneous" or len(cfg.robot_models) > 1)
    if "geometry" in axes:
        conditions.append(abs(cfg.cell_spacing - 1.0) > 1e-12)
    if "recipe" in axes:
        conditions.append(any(r not in {"standard_ab", "standard_abc"} for r in cfg.recipe_ids))
    if "demand" in axes:
        conditions.append(cfg.spawn_interval not in {4, 6})
    if "faults" in axes:
        conditions.append(any(x > 0 for x in (cfg.robot_failure_prob, cfg.operation_failure_prob,
                                               cfg.handoff_failure_prob, cfg.stockout_prob)))
    if "architecture" in axes or "topology" in axes:
        conditions.append(cfg.topology != "moore")
    if "skills" in axes:
        conditions.append(cfg.skill_mode != "disabled")
    return any(conditions)


def generalization_split(configs: Sequence[EnvConfig], held_out_axes: Iterable[str]):
    """Leakage-free config-only split; rollout outcomes are never inspected."""
    axes = set(held_out_axes); train, test = [], []
    for cfg in configs:
        (test if _is_ood(cfg, axes) else train).append(cfg)
    return train, test


METRIC_DICTIONARY = {
    "throughput": "completed products / elapsed simulation time",
    "value_throughput": "delivered product value / elapsed simulation time",
    "completion_rate": "delivered / spawned products",
    "mean_flow_time": "mean completion minus spawn time",
    "restricted_mean_flow_time": "mean observed time in system with unfinished products censored at evaluation horizon",
    "mean_tardiness": "mean positive lateness",
    "makespan": "completion time of the last delivered batch product relative to first release",
    "mean_wip": "time-average work in process",
    "robot_occupancy": "time-average fraction of robots busy in persistent physical/logistics activity",
    "cooperative_concurrency_mean": "mean concurrently executing recipe operations with coalition size >=2",
    "blocked_ready_task_rate": "fraction of logically ready operation samples with no currently feasible team realization",
    "ready_task_feasible_team_fraction": "fraction of logically ready operation samples with at least one feasible team realization",
    "K_feasible_star_mean": "mean state-conditioned maximum operation concurrency (exact when flagged)",
    "K_realized_mean": "mean concurrently executing recipe operations",
    "productive_concurrency_capacity_mean": "canonical unweighted feasible productive-operation concurrency",
    "productive_concurrency_realized_mean": "canonical unweighted realized productive-operation concurrency",
    "P_productive_star_mean": "optional weighted productive-concurrency diagnostic",
    "P_realized_mean": "optional weighted productive value actually executing",
    "parallelism_utilization": "mean K_realized/K_feasible* over exact states with operation opportunity",
    "productive_concurrency_utilization": "canonical v1 unweighted productive-operation utilization; same K ratio",
    "weighted_productive_parallelism_utilization": "optional weighted P_realized/P* diagnostic",
    "productive_parallelism_utilization": "legacy alias for the optional weighted P_realized/P* diagnostic",
    "capacity_exact_fraction": "fraction of capacity snapshots solved exactly rather than approximated",
    "team_formation_latency": "mean ticks from first proposal to committed team",
    "coalition_success_rate": "committed coalitions / (committed + expired nonempty proposals)",
    "coalition_formation_latency": "mean ticks from first proposal to commitment; null when no coalition commits",
    "team_formation_failures": "expired/invalid nonempty team proposals",
    "team_abort_rate": "failed committed recipe teams / committed teams",
    "team_proposal_churn": "robot switches/withdrawals from pending proposals",
    "team_load_cv": "coefficient of variation of team participations across robots",
    "trajectory_conflicts": "legacy aggregate: activities blocked by a committed activity, any cause",
    "geometry_rejection_count": "agreed candidates rejected by normalized geometric reach/feasibility",
    "geometry_conflict_count": "legacy aggregate alias of trajectory_conflicts, kept for archived records",
    "blocked_activity_count": "activities blocked by a committed activity, any cause",
    "geometry_block_count": "blocked activities whose only cause is abstract workspace overlap",
    "resource_block_count": "blocked activities caused by an exclusive or exhausted finite-capacity resource",
    "robot_block_count": "blocked activities caused by a robot committed to another activity",
    "geometry_delay_ticks": "aggregate normalized approach plus retreat delay, excluding declared process duration",
    "motion_path_length": "aggregate end-effector proxy path length",
    "motion_duration_total": "aggregate planned persistent motion/process duration",
    "minimum_reach_margin": "minimum reach-radius minus target distance encountered",
    "operation_failures": "failed pick/transformation/delivery events",
    "handoff_failures": "failed bilateral transfer events",
    "robot_failures": "injected robot downtime events",
    "deadlock_ticks": "ticks with WIP but no progress and no active recipe team",
}
