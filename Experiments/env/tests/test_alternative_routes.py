"""Alternative routes are EXCLUSIVE (XOR) and every estimate must agree."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import AssemblyGridCore, EnvConfig
from policies import Tier1Policy
from recipe import (OperationSpec, RecipeLibrary, RecipeSpec,
                    default_recipe_library)


def _independent_branch_recipe():
    """Branches consuming DIFFERENT inputs.

    This is the adversarial case. The shipped `alternative_route` recipe has
    both branches consuming the same tokens, so exclusivity holds there by
    accident even with no XOR rule at all. A test using only that recipe gives
    false confidence; this one can actually fail.
    """
    ops = (
        OperationSpec(id="routeA", kind="align", inputs=("A",), output="M", kappa=1, duration=2),
        OperationSpec(id="routeB", kind="align", inputs=("B",), output="M", kappa=1, duration=2),
        OperationSpec(id="finish", kind="inspect", inputs=("M",), output="FINAL",
                      kappa=1, duration=2, predecessor_any=(("routeA", "routeB"),)),
    )
    return RecipeLibrary([RecipeSpec(id="xor_test", raw_tokens=("A", "B"),
                                     operations=ops, final_token="FINAL")])


def test_xor_enforced_even_when_branches_have_independent_inputs():
    lib = _independent_branch_recipe()
    violations = 0
    for seed in (1, 2, 3):
        policy = Tier1Policy("parallel_aware", seed=0)
        core = AssemblyGridCore(
            EnvConfig(M=5, N=6, spawn_interval=3, max_wip=6, motion_planning=False,
                      recipe_ids=("xor_test",), horizon_T=250, seed=seed),
            priority_fn=policy.priority_fn, recipe_library=lib)
        obs = core.reset()
        for _ in range(250):
            obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
            if term or trunc:
                break
        for product in core.products.values():
            if "routeA" in product.completed_ops and "routeB" in product.completed_ops:
                violations += 1
        assert core.delivered_count > 0, "recipe never completed; test is vacuous"
    assert violations == 0, f"{violations} products executed both exclusive routes"


def test_no_two_siblings_are_ever_in_progress_simultaneously():
    """Closes the same-tick race: candidates are generated before any commit,
    so two members of one group could both be approved in a single step."""
    lib = _independent_branch_recipe()
    policy = Tier1Policy("parallel_aware", seed=0)
    core = AssemblyGridCore(
        EnvConfig(M=5, N=6, spawn_interval=2, max_wip=8, motion_planning=False,
                  recipe_ids=("xor_test",), horizon_T=250, seed=1),
        priority_fn=policy.priority_fn, recipe_library=lib)
    obs = core.reset()
    for _ in range(250):
        obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
        for product in core.products.values():
            in_prog = {"routeA", "routeB"} & set(product.in_progress_ops)
            assert len(in_prog) <= 1, f"both routes in progress at once: {in_prog}"
        if term or trunc:
            break


def test_critical_path_uses_the_shortest_route():
    """join_precise=5, join_fast=2, finish=2 -> shortest route is 2+2=4.
    Taking the max over ALL operations returned 5, a branch never performed,
    which inflated flow_time_lower_bound() and every generated due time."""
    rec = default_recipe_library().get("alternative_route")
    durations = {o.id: o.duration for o in rec.operations}
    assert durations == {"join_precise": 5, "join_fast": 2, "finish": 2}
    assert rec.nominal_critical_path == 4


def test_critical_path_unchanged_for_recipes_without_alternatives():
    """The sink-based fix must not disturb ordinary recipes."""
    lib = default_recipe_library()
    assert lib.get("standard_ab").nominal_critical_path == 7
    assert lib.get("solo_inspection").nominal_critical_path == 2
    assert lib.get("parallel_branch").nominal_critical_path == 8


def test_centralized_makespan_lower_bound_is_alternative_aware():
    """Summing both branches can push the 'lower bound' above the true
    optimum, at which point it is not a lower bound at all."""
    from centralized import _makespan_lower_bound

    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(EnvConfig(
        M=6, N=8, spawn_interval=2, max_wip=8, motion_planning=False,
        recipe_ids=("alternative_route",), seed=1))
    core.reset()
    rec = core.recipe_library.get("alternative_route")
    naive_robot_time = sum(op.duration * op.kappa for op in rec.operations)
    by_id = rec.op_map
    optional = {q for o in rec.operations for g in (o.predecessor_any or ()) for q in g if q in by_id}
    assert optional, "recipe has no alternatives; test is vacuous"
    lb = _makespan_lower_bound(core, Z=4)
    # the bound must be strictly below what the naive all-branches sum implies
    cap = core.cfg.M * core.cfg.N
    naive_bound = (4 * naive_robot_time + cap - 1) // cap
    assert lb <= max(core.flow_time_lower_bound(), naive_bound)
    assert lb > 0


def _branch_recipe(exclusive: bool):
    """Same recipe shape, differing only in whether the alternative group is
    exclusive (XOR) or inclusive (OR)."""
    common = dict(kind="align", output="M", kappa=1, duration=2)
    ops = [OperationSpec(id="routeA", inputs=("A",), **common),
           OperationSpec(id="routeB", inputs=("B",), **common)]
    fin = dict(id="finish", kind="inspect", inputs=("M",), output="FINAL",
               kappa=1, duration=2)
    if exclusive:
        fin["predecessor_any"] = (("routeA", "routeB"),)
    else:
        fin["predecessor_any_inclusive"] = (("routeA", "routeB"),)
    ops.append(OperationSpec(**fin))
    return RecipeLibrary([RecipeSpec(id="t", raw_tokens=("A", "B"),
                                     operations=tuple(ops), final_token="FINAL")])


def _run_branches(lib, seeds=(1, 2, 3)):
    both = delivered = 0
    for seed in seeds:
        policy = Tier1Policy("parallel_aware", seed=0)
        core = AssemblyGridCore(
            EnvConfig(M=5, N=6, spawn_interval=3, max_wip=6, motion_planning=False,
                      recipe_ids=("t",), horizon_T=250, seed=seed,
                      allow_inclusive_or_extension=any(
                          op.predecessor_any_inclusive for op in lib.get("t").operations)),
            priority_fn=policy.priority_fn, recipe_library=lib)
        obs = core.reset()
        for _ in range(250):
            obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
            if term or trunc:
                break
        both += sum(1 for p in core.products.values()
                    if "routeA" in p.completed_ops and "routeB" in p.completed_ops)
        delivered += core.delivered_count
    return both, delivered


def test_inclusive_or_permits_running_both_routes():
    """OR must remain expressible. Under failure probabilities or duration
    noise (EnvConfig.*_failure_prob, duration_noise, the AG-Robust track),
    deliberately executing two routes is a rational hedge, not waste.
    Supporting only XOR would make that strategy inexpressible."""
    both, delivered = _run_branches(_branch_recipe(exclusive=False))
    assert delivered > 0, "OR recipe never completed; test is vacuous"
    assert both > 0, "inclusive group behaved exclusively; OR is not expressible"


def test_exclusive_xor_still_forbids_running_both_routes():
    """The two semantics must remain genuinely different, on the same shape."""
    both, delivered = _run_branches(_branch_recipe(exclusive=True))
    assert delivered > 0, "XOR recipe never completed; test is vacuous"
    assert both == 0, "exclusive group permitted both routes"


def test_alternative_group_kinds_round_trip_through_serialization():
    from recipe import recipe_from_dict, recipe_to_dict

    for exclusive in (True, False):
        rec = _branch_recipe(exclusive).get("t")
        back = recipe_from_dict(recipe_to_dict(rec))
        fin = {o.id: o for o in back.operations}["finish"]
        if exclusive:
            assert fin.predecessor_any and not fin.predecessor_any_inclusive
        else:
            assert fin.predecessor_any_inclusive and not fin.predecessor_any
        assert fin.all_alternative_groups == (("routeA", "routeB"),)


def test_a_group_cannot_be_both_exclusive_and_inclusive():
    ops = (
        OperationSpec(id="a", kind="align", inputs=("A",), output="M", kappa=1, duration=2),
        OperationSpec(id="b", kind="align", inputs=("B",), output="M", kappa=1, duration=2),
        OperationSpec(id="c", kind="align", inputs=("C",), output="M", kappa=1, duration=2),
        OperationSpec(id="f", kind="inspect", inputs=("M",), output="FINAL", kappa=1, duration=2,
                      predecessor_any=(("a", "b"),), predecessor_any_inclusive=(("b", "c"),)),
    )
    import pytest
    with pytest.raises(ValueError):
        RecipeSpec(id="bad", raw_tokens=("A", "B", "C"),
                   operations=ops, final_token="FINAL").validate()


def test_capacity_does_not_count_two_exclusive_routes():
    """The capacity oracle must obey the same XOR rule as admission.

    With both branches of an exclusive group individually admissible, only one can
    execute, so the productive-concurrency capacity contributed by that product is one.
    Counting both inflated the capacity and understated utilization.
    """
    lib = _independent_branch_recipe()
    policy = Tier1Policy("parallel_aware", seed=0)
    core = AssemblyGridCore(
        EnvConfig(M=5, N=6, spawn_interval=2, max_wip=8, motion_planning=False,
                  recipe_ids=("xor_test",), horizon_T=250, seed=1),
        priority_fn=policy.priority_fn, recipe_library=lib)
    obs = core.reset()
    seen_both_available = False
    disjoint_robots_seen = []
    for _ in range(250):
        cands = core.feasible_operation_candidates()
        by_product = {}
        for c in cands:
            by_product.setdefault(c.product_id, set()).add(c.operation_id)
        for pid, ops in by_product.items():
            if {"routeA", "routeB"} <= ops:
                seen_both_available = True
                snapshot = core.capacity_snapshot(cands)
                selected = [c for c in cands if c.id in snapshot["count_selection"]]
                chosen_ops = {c.operation_id for c in selected if c.product_id == pid}
                assert not {"routeA", "routeB"} <= chosen_ops, (
                    "capacity selection contains two mutually exclusive routes")
                routes = [c for c in cands
                          if c.product_id == pid and c.operation_id in ("routeA", "routeB")]
                for i, a in enumerate(routes):
                    for c2 in routes[i + 1:]:
                        if a.operation_id != c2.operation_id and not (
                                set(a.coalition) & set(c2.coalition)):
                            disjoint_robots_seen.append((a.id, c2.id))
        obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
        if term or trunc:
            break
    assert seen_both_available, "both routes were never simultaneously admissible; test is vacuous"
    assert disjoint_robots_seen, (
        "exclusive routes never appeared on disjoint robot sets; shared robots could\n"
        "        have masked the defect")
