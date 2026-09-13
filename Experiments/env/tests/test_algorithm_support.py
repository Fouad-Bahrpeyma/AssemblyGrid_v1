from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core import ACTION_IDLE, AssemblyGridCore, EnvConfig
from encoding import observation_to_array


def _semantics(core):
    robots = tuple(
        (
            r.coord, r.holding, r.busy_until, r.failed_until, r.role, r.op,
            tuple(round(x, 12) for x in r.q),
            None if r.ee_pos is None else tuple(round(x, 12) for x in r.ee_pos),
            r.proposal_id, r.proposal_until, r.proposal_lock_until,
            r.incoming_transfer,
        )
        for row in core.robots for r in row
    )
    products = tuple(
        (
            p.id, p.recipe_id, p.spawn_time, p.due_time, tuple(sorted(p.source_rows.items())),
            round(p.mass, 12), round(p.value, 12), tuple(sorted(p.raw_picked)),
            tuple(sorted(p.reserved_raw.items())), tuple(sorted(p.token_positions.items())),
            tuple(sorted(p.completed_ops)), tuple(sorted(p.in_progress_ops)),
            p.delivered, p.complete_time,
        )
        for p in sorted(core.products.values(), key=lambda x: x.id)
    )
    proposals = tuple(
        (p.candidate_id, p.product_id, p.operation_id, p.members, p.created_at,
         p.expires_at, tuple(sorted(p.votes)))
        for p in sorted(core.proposals.values(), key=lambda x: x.candidate_id)
    )
    teams = tuple(
        (t.id, t.candidate_id, t.product_id, t.operation_id, t.members,
         tuple(sorted(t.roles.items())), t.kappa_min, t.formed_at,
         t.first_proposed_at, t.committed_until, t.status)
        for t in sorted(core.teams.values(), key=lambda x: x.id)
    )
    route_locked = tuple((pid, tuple(sorted(xs))) for pid, xs in sorted(core._route_locked.items()))
    # Scheduled plans contain wall-clock planning_seconds, which is instrumentation
    # rather than semantic state, so compare only event semantics.
    scheduled = tuple(sorted(
        (e.get("at"), e.get("kind"), e.get("pid"), e.get("op_id"), e.get("token"),
         e.get("team"), e.get("sender"), e.get("receiver"), e.get("success"),
         e.get("candidate_id"))
        for e in core.scheduled
    ))
    return core.t, robots, products, proposals, teams, route_locked, scheduled


def test_support_snapshot_has_named_algorithm_groups_and_rich_state():
    core = AssemblyGridCore(EnvConfig(M=3, N=4, max_wip=3, horizon_T=30, seed=7))
    snap = core.algorithm_support_snapshot()

    assert snap.available_groups == (
        "metadata", "canonical", "privileged", "entities", "topology",
        "recipes", "candidates", "motion", "resources", "diagnostics", "extensions",
    )
    selected = snap.to_dict(["entities", "topology", "privileged"])
    assert set(selected) == {"entities", "topology", "privileged"}

    entities = dict(snap.entities)
    robots = entities["robots"]
    products = entities["products"]
    assert robots and products
    robot0 = dict(robots[0])
    product0 = dict(products[0])
    assert "q" in robot0 and "model" in robot0 and "skills" in robot0
    assert "achievement" in product0 and "due_time" in product0 and "value" in product0

    topo = dict(snap.topology)
    assert "observation_edges" in topo
    assert "handoff_edges" in topo
    assert "manipulation_edges" in topo
    assert "candidate_membership_edges" in topo

    privileged = dict(snap.privileged)
    assert len(privileged["global_state_vector"]) > 0


def test_support_snapshot_is_detached_and_group_selection_is_safe():
    core = AssemblyGridCore(EnvConfig(M=3, N=4, max_wip=3, horizon_T=30, seed=8))
    snap = core.algorithm_support_snapshot()
    before = _semantics(core)

    detached = snap.to_dict(["metadata", "entities"])
    detached["metadata"] = ("modified",)
    detached["entities"] = ("modified",)
    assert _semantics(core) == before

    with pytest.raises(KeyError):
        snap.to_dict(["not-a-group"])
    with pytest.raises(FrozenInstanceError):
        snap.metadata = ()


def test_querying_rich_support_does_not_change_environment_trajectory():
    cfg = EnvConfig(M=3, N=4, max_wip=3, horizon_T=40, seed=11)
    queried = AssemblyGridCore(cfg)
    plain = AssemblyGridCore(cfg)

    for _ in range(12):
        # Query every rich group on only one copy before choosing/executing the
        # exact same joint action on both copies.
        queried.algorithm_support_snapshot()
        obs = plain._all_observations()
        joint = {}
        for rc, ob in sorted(obs.items()):
            legal = [a for a, ok in enumerate(ob.action_mask) if ok]
            # Deterministic non-learning controller: prefer first non-idle legal
            # action when one exists, otherwise idle.
            joint[rc] = next((a for a in legal if a != ACTION_IDLE), ACTION_IDLE)

        queried.step(joint)
        plain.step(joint)
        assert _semantics(queried) == _semantics(plain)


def test_richer_internal_joint_state_does_not_expand_canonical_observation():
    core = AssemblyGridCore(EnvConfig(M=3, N=4, max_wip=3, horizon_T=30, seed=13))
    rc = (1, 1)
    before = observation_to_array(core.observation(*rc), core.cfg.M, core.cfg.N).copy()
    core._robot(rc).q = (1.0, -0.5, 0.25, 0.7, -1.2, 0.4)
    after = observation_to_array(core.observation(*rc), core.cfg.M, core.cfg.N)
    assert (before == after).all()

    # The rich entity view still exposes the changed joint state to an explicit
    # motion/privileged wrapper.
    snap = core.algorithm_support_snapshot()
    robots = dict(snap.entities)["robots"]
    own = next(dict(x) for x in robots if dict(x)["coord"] == rc)
    assert own["q"] == core._robot(rc).q


def test_support_metadata_exposes_stable_action_layout_and_episode_status():
    from core import NUM_ACTIONS, MAX_PICK_OPTIONS, MAX_OPERATION_OPTIONS
    core = AssemblyGridCore(EnvConfig(M=3, N=4, max_wip=3, horizon_T=30, seed=17))
    meta = dict(core.algorithm_support_snapshot().metadata)
    layout = dict(meta["action_layout"])
    assert layout["num_actions"] == NUM_ACTIONS
    assert layout["pick_capacity"] == MAX_PICK_OPTIONS
    assert layout["operation_capacity"] == MAX_OPERATION_OPTIONS
    assert meta["terminated"] is False
    assert meta["truncated"] is False


def test_thin_algorithm_selectors_use_only_declared_support_groups():
    import sys
    from pathlib import Path
    marl_dir = str(Path(__file__).resolve().parents[1] / "marl")
    if marl_dir not in sys.path:
        sys.path.insert(0, marl_dir)
    from support_adapters import canonical_view, ctde_view, graph_view, motion_view, full_research_view

    core = AssemblyGridCore(EnvConfig(M=3, N=4, max_wip=3, horizon_T=30, seed=19))
    assert set(canonical_view(core)) == {"metadata", "canonical"}
    assert set(ctde_view(core)) == {"metadata", "canonical", "privileged"}
    assert set(graph_view(core)) == {"metadata", "entities", "topology", "recipes", "candidates", "resources"}
    assert set(motion_view(core)) == {"metadata", "canonical", "entities", "motion", "resources"}
    assert set(full_research_view(core)) == set(core.algorithm_support_snapshot().available_groups)
