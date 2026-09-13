"""Baselines must be genuinely distinct, or comparisons between them are void."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import EnvConfig
from policies import Tier1Policy, _destination_congestion


def test_parallel_aware_is_not_identical_to_greedy():
    """Regression: the congestion term was computed once per ROBOT and reused
    for every candidate direction, making it a constant column in the sort key
    that could never reorder candidates. parallel_aware therefore emitted
    byte-identical actions to greedy, and any comparison between the two was
    meaningless. Congestion must be evaluated per candidate DESTINATION."""
    differing = total = 0
    for seed in (0, 1, 2):
        cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=10,
                        motion_planning=False, seed=seed)
        pa = Tier1Policy("parallel_aware", seed=seed)
        gr = Tier1Policy("greedy", seed=seed)
        c1, c2 = pa.make_core(cfg), gr.make_core(cfg)
        o1, o2 = c1.reset(), c2.reset()
        for _ in range(150):
            a1, a2 = pa.act(c1, o1), gr.act(c2, o2)
            total += 1
            if a1 != a2:
                differing += 1
            o1, _r1, t1, u1, _ = c1.step(a1)
            o2, _r2, t2, u2, _ = c2.step(a2)
            if t1 or u1:
                break
    assert total > 0
    assert differing > 0, "parallel_aware is emitting identical actions to greedy"
    # a genuinely different routing rule should diverge on a substantial
    # fraction of ticks, not just once by luck
    assert differing / total > 0.05


def test_destination_congestion_varies_by_destination():
    """The whole point: the score must depend on WHICH cell is scored."""
    cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=10, motion_planning=False, seed=1)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    obs = core.reset()
    seen = set()
    for _ in range(120):
        for rc, o in obs.items():
            for nb in core.neighbors(rc):
                seen.add(_destination_congestion(core, o, nb))
        if len(seen) > 1:
            break
        obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
        if term or trunc:
            break
    assert len(seen) > 1, "congestion is constant across destinations; it cannot rank routes"


def test_destination_congestion_uses_only_locally_visible_cells():
    """It must not consult cells the acting robot cannot observe, or the
    heuristic becomes a privileged-information leak."""
    cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=10, motion_planning=False, seed=0)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    obs = core.reset()
    rc = (2, 3)
    o = obs[rc]
    far = (5, 7)  # deliberately outside this robot's neighbourhood
    assert far not in o.neighbors
    # scoring a far cell must not raise and must ignore unobservable neighbours
    score = _destination_congestion(core, o, far)
    visible = sum(1 for nb in core.neighbors(far) if nb in o.neighbors)
    assert 0 <= score <= 2 * visible


def test_staging_targets_are_adjacent_under_every_admissible_topology():
    """Regression for the AG-Architecture confound: the Tier-1 staging rule
    used to place a product's tokens on a fixed 2x2 patch, i.e. diagonally
    offset cells. Those are adjacent under Moore but NOT under von Neumann or
    line, so coalitions could never form there and the architecture
    comparison measured the heuristic's Moore assumption rather than the
    topology. Staging cells must now be pairwise adjacent under whatever
    topology is configured."""
    from policies import _staging_clique

    for topo, recipe in (("moore", "standard_ab"),
                         ("von_neumann", "pair_assembly"),
                         ("line", "pair_assembly")):
        cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=10,
                        motion_planning=False, topology=topo,
                        recipe_ids=(recipe,), seed=1)
        policy = Tier1Policy("parallel_aware", seed=0)
        core = policy.make_core(cfg)
        core.reset()
        cells = _staging_clique(core, (0, 5), 2)
        assert len(cells) == 2, f"{topo}: could not build a 2-cell staging set"
        a, b = cells[0], cells[1]
        assert b in core.adjacency.get(a, ()), (
            f"{topo}: staging cells {a} and {b} are not adjacent, so a "
            f"two-robot coalition can never form there")


def test_staging_route_distance_is_topology_matched():
    """The staging anchor must not be systematically further away under one
    topology than another, or route length alone would drive the comparison."""
    from policies import _route_distance

    dists = {}
    for topo in ("moore", "von_neumann", "line"):
        cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=10,
                        motion_planning=False, topology=topo,
                        recipe_ids=("pair_assembly",), seed=1)
        policy = Tier1Policy("parallel_aware", seed=0)
        core = policy.make_core(cfg)
        core.reset()
        dists[topo] = _route_distance(core, (0, 0), (0, 5))
    assert len(set(dists.values())) == 1, (
        f"staging anchor is not equidistant across topologies: {dists}")
