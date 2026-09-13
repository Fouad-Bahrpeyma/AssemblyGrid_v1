"""Read-only algorithm-support views for AssemblyGrid.

This module exposes a structured superset of environment information for
algorithm wrappers without changing canonical v1 observations, actions,
transitions, feasibility, or metrics.  Snapshots contain copied primitive
values/tuples only; callers never receive mutable references to core state.

The benchmark/algorithm boundary is:

    core truth -> canonical interface + read-only support views -> wrappers

Learning state (network memory, replay buffers, advantages, gradients, etc.)
remains the responsibility of the algorithm and is intentionally absent here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Tuple

Coord = Tuple[int, int]


def _pairs(mapping) -> tuple:
    return tuple(sorted((str(k), _freeze(v)) for k, v in mapping.items()))


def _freeze(value):
    """Recursively copy common values into immutable, deterministic forms."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(v) for v in value))
    return value


@dataclass(frozen=True)
class AlgorithmSupportSnapshot:
    """Immutable structured snapshot intended for algorithm-side adapters."""
    metadata: Any
    canonical: Any
    privileged: Any
    entities: Any
    topology: Any
    recipes: Any
    candidates: Any
    motion: Any
    resources: Any
    diagnostics: Any
    extensions: Any

    @property
    def available_groups(self) -> tuple[str, ...]:
        return (
            "metadata", "canonical", "privileged", "entities", "topology",
            "recipes", "candidates", "motion", "resources", "diagnostics",
            "extensions",
        )

    def to_dict(self, groups: Optional[Iterable[str]] = None) -> dict:
        """Return detached plain data for wrappers/serialization.

        ``groups`` lets an adapter select only the information family it needs.
        The returned dictionary is a copy; mutating it cannot alter the core.
        """
        selected = set(self.available_groups if groups is None else groups)
        unknown = selected.difference(self.available_groups)
        if unknown:
            raise KeyError(f"unknown algorithm-support groups: {sorted(unknown)}")
        raw = asdict(self)
        return {name: raw[name] for name in self.available_groups if name in selected}


def _motion_plan_view(plan) -> tuple:
    return (
        ("target", tuple(plan.target)),
        ("approach_duration", int(plan.approach_duration)),
        ("process_duration", int(plan.process_duration)),
        ("retreat_duration", int(plan.retreat_duration)),
        ("total_duration", int(plan.total_duration)),
        ("min_reach_margin", float(plan.min_reach_margin)),
        ("path_length", float(plan.path_length)),
        ("robot_motions", tuple(
            (
                ("robot", tuple(m.robot)), ("start", tuple(m.start)),
                ("target", tuple(m.target)), ("q_target", tuple(m.q_target)),
                ("travel_distance", float(m.travel_distance)),
                ("travel_duration", int(m.travel_duration)),
                ("reach_margin", float(m.reach_margin)),
                ("segment", tuple(tuple(x) for x in m.segment)),
            ) for m in plan.robot_motions
        )),
    )


def build_algorithm_support_snapshot(core) -> AlgorithmSupportSnapshot:
    """Build a detached, read-only superset view of ``core``.

    This function performs observations only.  It does not draw from the
    environment RNGs, apply actions, reserve resources, or alter transition
    state. Candidate caching may be populated exactly as by a normal
    observation call, but has no semantic effect on the trajectory.
    """
    # Imported lazily to avoid making the core depend on NumPy at import time.
    from encoding import global_state_to_array, observation_to_array

    cands = tuple(core.candidate_operations())
    observations = core._all_observations()

    from core import (
        ACTION_DELIVER, ACTION_HANDOFF_BASE, ACTION_IDLE, ACTION_OPERATION_BASE,
        ACTION_PICK_BASE, ACTION_RECEIVE_BASE, DIRECTIONS, MAX_OPERATION_OPTIONS,
        MAX_PICK_OPTIONS, NUM_ACTIONS,
    )
    action_layout = (
        ("idle", ACTION_IDLE), ("deliver", ACTION_DELIVER),
        ("pick_base", ACTION_PICK_BASE), ("pick_capacity", MAX_PICK_OPTIONS),
        ("handoff_base", ACTION_HANDOFF_BASE), ("handoff_directions", tuple(DIRECTIONS)),
        ("receive_base", ACTION_RECEIVE_BASE), ("receive_directions", tuple(DIRECTIONS)),
        ("operation_base", ACTION_OPERATION_BASE), ("operation_capacity", MAX_OPERATION_OPTIONS),
        ("num_actions", NUM_ACTIONS),
    )
    terminated = bool(
        core.cfg.episode_mode == "batch"
        and core.spawned_count >= core.cfg.batch_Z
        and core.delivered_count >= core.cfg.batch_Z
    )
    truncated = bool(core.cfg.episode_mode == "horizon" and core.t >= core.cfg.horizon_T)
    metadata = (
        ("time", int(core.t)), ("shape", (int(core.cfg.M), int(core.cfg.N))),
        ("topology", core.cfg.topology),
        ("geometry_profile", core.cfg.geometry_profile),
        ("geometry_variant", core.cfg.geometry_variant),
        ("episode_mode", core.cfg.episode_mode),
        ("horizon_T", int(core.cfg.horizon_T)), ("batch_Z", int(core.cfg.batch_Z)),
        ("generation_seed", core.cfg.generation_seed),
        ("execution_seed", core.cfg.execution_seed), ("fallback_seed", core.cfg.seed),
        ("terminated", terminated), ("truncated", truncated),
        ("action_layout", action_layout),
        ("config", _freeze(asdict(core.cfg))),
    )

    canonical_agents = []
    for rc, obs in sorted(observations.items()):
        canonical_agents.append((
            ("robot", tuple(rc)), ("time", int(obs.t)),
            ("holding_kind", obs.holding_kind), ("holding_pid", obs.holding_pid),
            ("busy", bool(obs.busy)), ("failed", bool(obs.failed)),
            ("proposal_id", obs.proposal_id),
            ("neighbors", tuple((tuple(nrc), _freeze(info)) for nrc, info in sorted(obs.neighbors.items()))),
            ("pick_options", tuple(_freeze(asdict(x)) for x in obs.pick_options)),
            ("operation_options", tuple(_freeze(asdict(x)) for x in obs.operation_options)),
            ("action_mask", tuple(bool(x) for x in obs.action_mask)),
            ("observation_vector", tuple(float(x) for x in observation_to_array(obs, core.cfg.M, core.cfg.N))),
        ))
    canonical = (("agents", tuple(canonical_agents)),)

    privileged = (
        ("global_state_vector", tuple(float(x) for x in global_state_to_array(core))),
        ("global_state_semantics", "non-canonical privileged/CTDE state"),
    )

    robot_entities = []
    for row in core.robots:
        for r in row:
            robot_entities.append((
                ("coord", tuple(r.coord)), ("holding", _freeze(r.holding)),
                ("busy_until", int(r.busy_until)), ("failed_until", int(r.failed_until)),
                ("role", r.role), ("operation_kind", r.op),
                ("q", tuple(float(x) for x in r.q)),
                ("ee_pos", None if r.ee_pos is None else tuple(float(x) for x in r.ee_pos)),
                ("model", _freeze(asdict(r.model))),
                ("skills", _freeze(r.skill_levels)), ("skill_uses", _freeze(r.skill_uses)),
                ("proposal_id", r.proposal_id), ("proposal_until", int(r.proposal_until)),
                ("proposal_lock_until", int(r.proposal_lock_until)),
                ("incoming_transfer", _freeze(r.incoming_transfer)),
                ("role_switches", int(r.role_switches)), ("team_uses", int(r.team_uses)),
            ))

    product_entities = []
    for p in sorted(core.products.values(), key=lambda x: x.id):
        ach = core.achievement_state(p.id)
        product_entities.append((
            ("id", int(p.id)), ("recipe_id", p.recipe_id),
            ("spawn_time", int(p.spawn_time)), ("due_time", int(p.due_time)),
            ("mass", float(p.mass)), ("value", float(p.value)),
            ("source_rows", _freeze(p.source_rows)),
            ("raw_picked", _freeze(p.raw_picked)), ("reserved_raw", _freeze(p.reserved_raw)),
            ("token_positions", _freeze(p.token_positions)),
            ("completed_ops", _freeze(p.completed_ops)), ("in_progress_ops", _freeze(p.in_progress_ops)),
            ("route_locked_ops", _freeze(core._route_locked.get(p.id, set()))),
            ("delivered", bool(p.delivered)), ("complete_time", p.complete_time),
            ("achievement", _freeze(asdict(ach))),
        ))
    entities = (("robots", tuple(robot_entities)), ("products", tuple(product_entities)))

    def edges_for(fn):
        edges = set()
        for i in range(core.cfg.M):
            for j in range(core.cfg.N):
                a = (i, j)
                for b in fn(a):
                    edges.add(tuple(sorted((a, b))))
        return tuple(sorted(edges))

    candidate_membership = tuple(
        sorted((cand.id, tuple(rc), cand.role_map.get(rc, "")) for cand in cands for rc in cand.coalition)
    )
    topology = (
        ("observation_edges", edges_for(core.obs_neighbors)),
        ("handoff_edges", edges_for(core.handoff_neighbors)),
        ("manipulation_edges", edges_for(core.manip_neighbors)),
        ("candidate_membership_edges", candidate_membership),
    )

    recipe_views = []
    precedence_edges = []
    for rid in core.cfg.recipe_ids:
        rec = core.recipe_library.get(rid)
        ops = []
        for op in rec.operations:
            ops.append(_freeze(asdict(op)))
            precedence_edges += [(rid, pred, op.id, "and") for pred in op.predecessors]
            precedence_edges += [(rid, pred, op.id, "xor") for group in op.predecessor_any for pred in group]
            precedence_edges += [(rid, pred, op.id, "inclusive_or_extension") for group in op.predecessor_any_inclusive for pred in group]
        recipe_views.append((
            ("id", rec.id), ("raw_tokens", tuple(rec.raw_tokens)),
            ("final_token", rec.final_token), ("product_value_extension", float(rec.product_value)),
            ("nominal_critical_path", int(rec.nominal_critical_path)),
            ("operations", tuple(ops)),
        ))
    recipes = (("recipes", tuple(recipe_views)), ("precedence_edges", tuple(sorted(precedence_edges))))

    candidate_views = []
    for cand in cands:
        prop = core.proposals.get(cand.id)
        candidate_views.append((
            ("id", cand.id), ("product_id", cand.product_id),
            ("recipe_id", core.products[cand.product_id].recipe_id),
            ("operation_id", cand.operation_id), ("kind", cand.kind),
            ("coalition", tuple(cand.coalition)), ("roles", tuple(cand.roles)),
            ("workspace", _freeze(cand.workspace)), ("resources", _freeze(cand.resources)),
            ("kappa_min", int(cand.kappa_min)), ("size", int(cand.size)),
            ("proposal_support", 0.0 if prop is None else len(prop.votes) / max(1, cand.size)),
            ("supporter_ids_privileged", tuple(sorted(prop.votes)) if prop is not None else tuple()),
            ("weighted_productive_value_extension", float(cand.productive_value)),
            ("motion", _motion_plan_view(cand.motion_plan)),
        ))
    proposals = tuple((
        ("candidate_id", p.candidate_id), ("product_id", p.product_id), ("operation_id", p.operation_id),
        ("members", tuple(p.members)), ("created_at", p.created_at), ("expires_at", p.expires_at),
        ("votes_privileged", tuple(sorted(p.votes))),
    ) for p in sorted(core.proposals.values(), key=lambda x: x.candidate_id))
    teams = tuple((
        ("id", t.id), ("candidate_id", t.candidate_id), ("product_id", t.product_id),
        ("operation_id", t.operation_id), ("kind", t.operation), ("members", tuple(t.members)),
        ("roles", _freeze(t.roles)), ("kappa_min", t.kappa_min),
        ("formed_at", t.formed_at), ("first_proposed_at", t.first_proposed_at),
        ("committed_until", t.committed_until), ("status", t.status),
        ("resources", _freeze(t.resources)), ("motion", _motion_plan_view(t.motion_plan)),
    ) for t in sorted(core.teams.values(), key=lambda x: x.id))
    candidates = (("operation_candidates", tuple(candidate_views)), ("proposals", proposals), ("teams", teams))

    reservations = tuple((
        ("id", r.id), ("kind", r.kind), ("members", tuple(r.members)),
        ("resources", _freeze(r.resources)), ("until", r.until), ("motion", _motion_plan_view(r.motion_plan)),
    ) for r in sorted(core.active_reservations.values(), key=lambda x: x.id))
    motion = (
        ("motion_planning_enabled", bool(core.cfg.motion_planning)),
        ("trajectory_conflicts_enabled", bool(core.cfg.trajectory_conflicts)),
        ("cell_spacing", float(core.cfg.cell_spacing)),
        ("workspace_conflict_radius", float(core.cfg.workspace_conflict_radius)),
        ("active_reservations", reservations),
    )

    resource_names = set(core.cfg.resource_capacities)
    for cand in cands:
        resource_names.update(cand.resources)
    for r in core.active_reservations.values():
        resource_names.update(r.resources)
    resources = (
        ("capacities", tuple((name, core._resource_capacity(name)) for name in sorted(resource_names))),
        ("active_counts", tuple((name, core._active_resource_count(name)) for name in sorted(resource_names))),
        ("shelves", _freeze(core.shelves)),
    )

    diagnostics = (
        ("metrics", _freeze(core.metrics())),
        ("progress_events", int(core.progress_events)),
        ("scheduled_events", _freeze(core.scheduled)),
        ("last_pulses", _freeze(core.last_pulses)),
    )

    extension_flags = (
        ("skills_enabled", core.cfg.skill_mode != "disabled"),
        ("inclusive_or_enabled", bool(core.cfg.allow_inclusive_or_extension)),
        ("disturbances_enabled", any(float(x) > 0 for x in (
            core.cfg.robot_failure_prob, core.cfg.operation_failure_prob,
            core.cfg.handoff_failure_prob, core.cfg.duration_noise, core.cfg.stockout_prob,
        ))),
        ("heterogeneous_robots", core.cfg.robot_model_assignment != "homogeneous" or len(core.cfg.robot_models) > 1),
        ("due_dates_available", True), ("product_values_available", True),
    )
    extensions = (("flags", extension_flags),)

    return AlgorithmSupportSnapshot(
        metadata=metadata, canonical=canonical, privileged=privileged,
        entities=entities, topology=topology, recipes=recipes,
        candidates=candidates, motion=motion, resources=resources,
        diagnostics=diagnostics, extensions=extensions,
    )
