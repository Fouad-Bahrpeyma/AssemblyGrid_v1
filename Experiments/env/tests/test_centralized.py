"""Tests for the centralized dispatch baseline and TAMP oracle (Sec. 10, items 2, 4)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invariants
from centralized import CentralizedDispatchPolicy, run_motion_aware_reference
from core import AssemblyGridCore, EnvConfig


def test_centralized_dispatch_runs_and_respects_invariants():
    cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=12, motion_planning=False, seed=2)
    policy = CentralizedDispatchPolicy(seed=0)
    core = policy.make_core(cfg)
    obs = core.reset()
    for _ in range(300):
        obs, _reward, terminated, truncated, _ = core.step(policy.act(core, obs))
        invariants.check_coalitions(core)
        invariants.check_parallelism(core)
        if terminated or truncated:
            break
    assert core.delivered_count > 0
    assert core.unmasked_invalid_attempts == 0


def test_centralized_dispatch_forms_cooperative_teams():
    """It must actually commit some kappa>=2 teams (up to the physical cap
    of four), not silently degrade to solo operations only."""
    cfg = EnvConfig(M=7, N=10, spawn_interval=2, max_wip=16, motion_planning=False, seed=1)
    policy = CentralizedDispatchPolicy(seed=0)
    core = policy.make_core(cfg)
    obs = core.reset()
    max_team_size_seen = 0
    for _ in range(300):
        obs, _reward, terminated, truncated, _ = core.step(policy.act(core, obs))
        for team in core.teams.values():
            max_team_size_seen = max(max_team_size_seen, len(team.members))
        if terminated or truncated:
            break
    assert max_team_size_seen >= 2


def test_tamp_oracle_reaches_batch_target_on_small_instance():
    cfg = EnvConfig(M=5, N=6, spawn_interval=2, max_wip=8)
    result = run_motion_aware_reference(cfg, Z=4, seed=0, max_ticks=3000)
    assert result["reached"] is True
    assert result["delivered"] >= 4
    # motion planning was forced on regardless of the passed config
    assert result["makespan"] >= result["makespan_lower_bound"]
    assert result["unmasked_invalid_attempts"] == 0


def test_tamp_oracle_forces_motion_planning_even_if_disabled_in_config():
    cfg = EnvConfig(M=5, N=6, spawn_interval=2, max_wip=8, motion_planning=False,
                    trajectory_conflicts=False)
    result = run_motion_aware_reference(cfg, Z=3, seed=0, max_ticks=3000)
    assert result["reached"] is True


def test_committed_coalition_membership_is_immutable_mid_operation():
    """Point 1 (coalition lifetime): once a team commits, its members and
    roles must not change until a terminal event. Tracks every committed
    team across a long run and fails if any membership/role set mutates."""
    import invariants
    from policies import Tier1Policy

    cfg = EnvConfig(M=7, N=10, spawn_interval=2, max_wip=16, motion_planning=False, seed=3)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    obs = core.reset()
    seen = {}
    multi_tick_teams_observed = 0
    for _ in range(400):
        obs, _r, terminated, truncated, _ = core.step(policy.act(core, obs))
        for tid, team in core.teams.items():
            if team.status != "committed":
                continue
            snap = (tuple(team.members), tuple(sorted(team.roles.items())), team.operation_id)
            if tid in seen:
                assert seen[tid] == snap, f"coalition {tid} mutated mid-operation"
                multi_tick_teams_observed += 1
            else:
                seen[tid] = snap
        invariants.check_coalition_lifetime(core, _seen={})
        if terminated or truncated:
            break
    # the assertion above is only meaningful if teams actually persisted
    # across ticks, so make sure the run really exercised multi-tick teams
    assert multi_tick_teams_observed > 0


def test_candidate_reports_declared_minimum_and_realized_size_separately():
    """Point 2 (notation): k_min and |C| are distinct fields. With recipes
    that declare no support roles (kappa_max defaults to kappa), arity is
    EXACT, so size must equal kappa_min for every candidate."""
    cfg = EnvConfig(M=7, N=10, spawn_interval=2, max_wip=16, motion_planning=False, seed=1)
    policy = CentralizedDispatchPolicy(seed=0)
    core = policy.make_core(cfg)
    obs = core.reset()
    checked = 0
    for _ in range(200):
        for cand in core.candidate_operations():
            spec = core.operation_spec(core.products[cand.product_id], cand.operation_id)
            assert cand.kappa_min == spec.kappa
            assert cand.size == len(cand.coalition)
            assert spec.kappa <= cand.size <= spec.arity_max
            checked += 1
        obs, _r, terminated, truncated, _ = core.step(policy.act(core, obs))
        if terminated or truncated:
            break
    assert checked > 0, "no candidate was ever inspected; test is vacuous"


def test_operation_admits_extra_members_only_when_recipe_declares_them():
    """Point 3: an operation with kappa_max == kappa admits EXACT arity only.
    Declaring kappa_max > kappa (with roles for the extra members) is the
    only way a larger coalition becomes admissible, and even then it must
    not change the operation's duration."""
    from recipe import OperationSpec, RecipeSpec, RecipeLibrary

    strict = OperationSpec(id="join", kind="assemble", inputs=("A", "B"),
                           output="FINAL", kappa=2, duration=3)
    assert strict.arity_max == 2

    supported = OperationSpec(id="join", kind="assemble", inputs=("A", "B"),
                              output="FINAL", kappa=2, kappa_max=3, duration=3,
                              roles=("holder", "inserter", "stabilizer"))
    assert supported.arity_max == 3
    supported.validate()

    lib = RecipeLibrary([RecipeSpec(id="supported", raw_tokens=("A", "B"),
                                    operations=(supported,), final_token="FINAL")])
    cfg = EnvConfig(M=5, N=6, spawn_interval=2, max_wip=8, motion_planning=False,
                    recipe_ids=("supported",), seed=0)
    from policies import Tier1Policy
    policy = Tier1Policy("parallel_aware", seed=0)
    core = AssemblyGridCore(cfg, priority_fn=policy.priority_fn, recipe_library=lib)
    obs = core.reset()
    sizes = set()
    for _ in range(200):
        for cand in core.candidate_operations():
            sizes.add(cand.size)
            # duration must be the declared one regardless of team size
            spec = core.operation_spec(core.products[cand.product_id], cand.operation_id)
            assert core._operation_duration(spec, cand.size) == spec.duration
        obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
        if term or trunc:
            break
    assert sizes, "no candidates generated; test is vacuous"
    assert sizes <= {2, 3}, f"arity outside declared interval: {sizes}"


def test_inputs_spread_over_more_robots_than_k_min_is_still_feasible():
    """Regression: when a recipe declares k_max > k_min, the operation's
    input tokens may already sit on MORE distinct robots than k_min. That is
    legal as long as it fits within k_max, and must not be rejected. A guard
    comparing the holder count against k_min instead of k_max made such
    recipes silently undeliverable (zero candidates ever generated)."""
    from recipe import OperationSpec, RecipeLibrary, RecipeSpec
    from policies import Tier1Policy

    op = OperationSpec(id="join3", kind="assemble", inputs=("A", "B", "C"),
                       output="FINAL", kappa=2, kappa_max=4, duration=3,
                       roles=("holder", "inserter", "s1", "s2"))
    op.validate()
    lib = RecipeLibrary([RecipeSpec(id="three_in", raw_tokens=("A", "B", "C"),
                                    operations=(op,), final_token="FINAL")])
    cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=8, motion_planning=False,
                    recipe_ids=("three_in",), seed=0)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = AssemblyGridCore(cfg, priority_fn=policy.priority_fn, recipe_library=lib)
    obs = core.reset()
    sizes = set()
    for _ in range(250):
        for cand in core.candidate_operations():
            sizes.add(cand.size)
            assert cand.kappa_min <= cand.size <= op.arity_max
        obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
        if term or trunc:
            break
    assert sizes, "no candidate generated: the k_min/k_max guard has regressed"
    assert max(sizes) > op.kappa, "expected coalitions larger than k_min to be admissible"
    assert core.delivered_count > 0, "recipe never completed end to end"
