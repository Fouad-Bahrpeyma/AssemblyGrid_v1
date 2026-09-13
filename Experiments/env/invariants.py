"""Executable semantic invariants for the revised AssemblyGrid benchmark."""
from __future__ import annotations

from itertools import combinations


def check_coalitions(core):
    for team in core.teams.values():
        assert 1 <= len(team.members) <= core.cfg.max_coalition_size
        assert len(set(team.members)) == len(team.members)
        # Canonical semantics: every pair in one direct-manipulation coalition
        # must be directly connected in the declared local topology.
        assert all(core._local(a, b) for a, b in combinations(team.members, 2))
        assert set(team.roles) == set(team.members)
        p = core.products[team.product_id]
        op = core.operation_spec(p, team.operation_id)
        # op.kappa is the MINIMUM team size the operation needs, not a fixed
        # headcount: a larger team (up to max_coalition_size) is allowed and
        # finishes the operation faster.
        assert len(team.members) >= op.kappa
        assert team.operation_id in p.in_progress_ops


def check_coalition_lifetime(core, _seen={}):
    """Point 1: a committed coalition's membership must not change while the
    operation runs. Call every tick on the same core; it remembers each team
    id's membership and roles at first sight and asserts they never change
    afterwards. Membership may only disappear (terminal event), never mutate.
    Pass a fresh dict as _seen (or use a new core) to reset between episodes.
    """
    key = id(core)
    seen = _seen.setdefault(key, {})
    for tid, team in core.teams.items():
        if team.status != "committed":
            continue
        snapshot = (tuple(team.members), tuple(sorted(team.roles.items())), team.operation_id)
        if tid in seen:
            assert seen[tid] == snapshot, (
                f"coalition {tid} changed mid-operation: {seen[tid]} -> {snapshot}")
        else:
            seen[tid] = snapshot


def check_parallelism(core):
    snap = core.capacity_snapshot()
    assert snap["K_feasible_star"] >= core.current_concurrency() >= 0
    assert snap["P_productive_star"] + 1e-9 >= core.current_productive_value() >= 0
    active = list(core.teams.values())
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            assert not set(a.members) & set(b.members)


def check_holding_uniqueness(core):
    seen = set()
    for row in core.robots:
        for r in row:
            if r.holding is not None:
                assert r.holding not in seen
                seen.add(r.holding)
                assert r.holding[0] in core.products
                p = core.products[r.holding[0]]
                assert p.token_positions.get(r.holding[1]) == (r.row, r.col)
    for p in core.products.values():
        for token, rc in p.token_positions.items():
            assert core._robot(rc).holding == (p.id, token)


def check_support_vs_ownership(core):
    """Point 6: multi-robot support must be legal WITHOUT duplicating
    material. Ownership stays single-valued (that is what conservation
    means); the support set may be larger, must always contain the owner,
    must never contain a robot twice, and must only contain robots that are
    genuinely engaged with that token (team members or a handoff partner).
    """
    for p in core.products.values():
        for token, rc in p.token_positions.items():
            owner = core.token_owner(p.id, token)
            assert owner == rc, "ownership must be single-valued and match token_positions"
            supporters = core.token_supporters(p.id, token)
            assert len(set(supporters)) == len(supporters), "a robot cannot support twice"
            assert owner in supporters, "the owner always supports its own token"
            # support never implies a second copy of the material exists
            owning_robots = [q for row in core.robots for q in row
                             if q.holding == (p.id, token)]
            assert len(owning_robots) == 1, "multi-support must not duplicate material"
            if len(supporters) > 1:
                load = core.support_load(p.id, token)
                assert load <= p.mass + 1e-9, "shared load cannot exceed total mass"


def check_recipe_state(core):
    for p in core.products.values():
        rec = core.recipe(p)
        assert not (p.completed_ops & p.in_progress_ops)
        for op_id in p.completed_ops:
            op = rec.op_map[op_id]
            assert all(pred in p.completed_ops for pred in op.predecessors)
        for op_id in p.in_progress_ops:
            op = rec.op_map[op_id]
            assert all(pred in p.completed_ops for pred in op.predecessors)
        if p.delivered:
            assert p.complete_time is not None
            assert rec.final_token not in p.token_positions


def check_shelves(core):
    for shelf in core.shelves.values():
        for stock in shelf["inventory"].values():
            assert 0 <= stock <= core.cfg.shelf_cap


def check_wip_consistency(core):
    assert core.wip() == sum(not p.delivered for p in core.products.values())


def check_action_mask_respected(core):
    cands = core.candidate_operations()
    for row in core.robots:
        for r in row:
            mask = core.action_mask(r.row, r.col, cands)
            assert len(mask) > 0 and mask[0]
            if not core._free(r):
                assert sum(mask) == 1


def check_proposals(core):
    for cid, prop in core.proposals.items():
        assert cid == prop.candidate_id
        assert prop.votes <= set(prop.members)
        for rc in prop.votes:
            assert core._robot(rc).proposal_id == cid
    for row in core.robots:
        for r in row:
            if r.proposal_id is not None:
                assert r.proposal_id in core.proposals
                assert r.coord in core.proposals[r.proposal_id].votes


def check_reservations(core):
    for rid, reservation in core.active_reservations.items():
        assert reservation.id == rid
        assert reservation.until > core.t
        assert len(set(reservation.members)) == len(reservation.members)


def check_monotonic_counts(core, prev_delivered, prev_spawned):
    assert core.delivered_count >= prev_delivered and core.spawned_count >= prev_spawned


def check_all(core):
    check_coalitions(core)
    check_parallelism(core)
    check_holding_uniqueness(core)
    check_support_vs_ownership(core)
    check_recipe_state(core)
    check_shelves(core)
    check_wip_consistency(core)
    check_action_mask_respected(core)
    check_proposals(core)
    check_reservations(core)
