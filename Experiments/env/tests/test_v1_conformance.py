"""AssGrDims#1 -> AssGrDims#2 canonical v1 conformance checks."""
from __future__ import annotations

import os
import sys
from itertools import combinations

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core as core_module
from core import (
    ACTION_IDLE, ACTION_OPERATION_BASE, AssemblyGridCore, EnvConfig, Product,
    TOPOLOGY_MAX_ARITY,
)
from motion import common_target, plan_motion
from profiles import DECLARED_VARIANTS, GEOMETRY_PROFILES, apply_geometry_profile
from recipe import OperationSpec, RecipeLibrary, RecipeSpec


def _max_clique(core: AssemblyGridCore) -> int:
    nodes = list(core.adjacency)
    for k in range(min(4, len(nodes)), 0, -1):
        for group in combinations(nodes, k):
            if all(b in core.adjacency[a] for a, b in combinations(group, 2)):
                return k
    return 1


def test_topology_arity_cap_equals_manipulation_graph_clique_number():
    recipes = {"moore": ("standard_ab",), "von_neumann": ("pair_assembly",), "line": ("pair_assembly",)}
    for topology, recipe_ids in recipes.items():
        c = AssemblyGridCore(EnvConfig(M=3, N=3, topology=topology, recipe_ids=recipe_ids,
                                       motion_planning=False, trajectory_conflicts=False))
        assert _max_clique(c) == TOPOLOGY_MAX_ARITY[topology]
        assert c.obs_neighbors((1, 1)) == c.handoff_neighbors((1, 1)) == c.manip_neighbors((1, 1))


def test_canonical_interface_has_no_environment_communication_action_or_message():
    assert not hasattr(core_module, "ACTION_COMMUNICATE")
    c = AssemblyGridCore(EnvConfig(M=2, N=2, recipe_ids=("pair_assembly",)))
    o = c.observation(0, 0)
    assert all("message" not in info for info in o.neighbors.values())
    assert not hasattr(c._robot((0, 0)), "last_message")
    assert ACTION_IDLE == 0


def test_profile_ownership_and_noncanonical_demo_guard():
    base = EnvConfig(M=2, N=2, recipe_ids=("standard_ab",))
    canonical = apply_geometry_profile(base, "abstract-v1", profile_enforced=True)
    canonical.validate()
    with pytest.raises(ValueError, match="profile-owned fields"):
        EnvConfig(**{**canonical.__dict__, "cell_spacing": 9.0}).validate()

    demo = apply_geometry_profile(base, "ur10-demo-v1", profile_enforced=True)
    demo.validate()
    with pytest.raises(ValueError, match="non-canonical geometry profile"):
        EnvConfig(**{**demo.__dict__, "official_result": True}).validate()

    # official_result itself makes profile ownership mandatory; callers cannot
    # bypass it by leaving profile_enforced=False.
    with pytest.raises(ValueError, match="profile-owned fields"):
        EnvConfig(M=2, N=2, recipe_ids=("standard_ab",), official_result=True,
                  profile_enforced=False, cell_spacing=9.0).validate()
    with pytest.raises(ValueError, match="profile-owned fields"):
        EnvConfig(M=2, N=2, recipe_ids=("standard_ab",), official_result=True,
                  profile_enforced=False, motion_planning=False,
                  trajectory_conflicts=False).validate()


def test_each_declared_profile_variant_admits_nominal_four_robot_common_target():
    cases = [(name, None) for name in GEOMETRY_PROFILES]
    cases += [(name, variant) for name, variants in DECLARED_VARIANTS.items() for variant in variants]
    for name, variant in cases:
        base = EnvConfig(M=2, N=2, recipe_ids=("standard_ab",))
        cfg = apply_geometry_profile(base, name, variant, profile_enforced=True)
        c = AssemblyGridCore(cfg)
        coalition = ((0, 0), (0, 1), (1, 0), (1, 1))
        assert c._coalition_local(coalition)
        target = common_target(coalition, cfg.cell_spacing)
        plan = plan_motion(c._robot_map(), coalition, target, cfg.cell_spacing, process_duration=1,
                           retreat_fraction=cfg.retreat_fraction)
        assert plan.min_reach_margin >= 0.0


def _ready_pair_core(*, support_visible: bool) -> tuple[AssemblyGridCore, str, tuple]:
    c = AssemblyGridCore(EnvConfig(M=2, N=2, recipe_ids=("pair_assembly",), spawn_interval=999,
                                   proposal_support_visible=support_visible,
                                   motion_planning=False, trajectory_conflicts=False))
    c.products.clear(); c.spawned_count = 1
    p = Product(0, "pair_assembly", 0, 999, source_rows={"A": 0, "B": 1})
    p.raw_picked.update(("A", "B")); p.token_positions.update({"A": (0, 0), "B": (0, 1)})
    c.products[0] = p
    c._robot((0, 0)).holding = (0, "A"); c._robot((0, 1)).holding = (0, "B")
    c.next_product_id = 1; c._invalidate()
    cand = c.candidate_operations()[0]
    return c, cand.id, cand.coalition


def test_proposal_support_is_aggregate_and_supporter_identity_is_not_exposed():
    c, cid, coalition = _ready_pair_core(support_visible=True)
    proposer = coalition[0]
    obs = c._all_observations()
    slot = next(i for i, x in enumerate(obs[proposer].operation_options) if x.candidate_id == cid)
    c.step({proposer: ACTION_OPERATION_BASE + slot})
    other = next(rc for rc in coalition if rc != proposer)
    o = c.observation(*other)
    view = next(x for x in o.operation_options if x.candidate_id == cid)
    assert view.proposal_support == pytest.approx(1.0 / len(coalition))
    assert not hasattr(view, "votes") and not hasattr(view, "supporters")
    assert "votes" not in o.neighbors.get(proposer, {}) and "proposal_id" not in o.neighbors.get(proposer, {})

    c2, cid2, coalition2 = _ready_pair_core(support_visible=False)
    proposer2 = coalition2[0]; obs2 = c2._all_observations()
    slot2 = next(i for i, x in enumerate(obs2[proposer2].operation_options) if x.candidate_id == cid2)
    c2.step({proposer2: ACTION_OPERATION_BASE + slot2})
    other2 = next(rc for rc in coalition2 if rc != proposer2)
    view2 = next(x for x in c2.observation(*other2).operation_options if x.candidate_id == cid2)
    assert view2.proposal_support == 0.0



def test_canonical_observation_has_time_proposal_and_recipe_identity_but_no_urgency():
    from dataclasses import replace
    from encoding import observation_to_array

    c, cid, coalition = _ready_pair_core(support_visible=True)
    rc = coalition[0]
    o = c.observation(*rc)
    assert o.operation_options
    assert all(hasattr(x, "recipe_id") and not hasattr(x, "urgency") for x in o.operation_options)
    assert all(hasattr(x, "recipe_id") and not hasattr(x, "urgency") for x in o.pick_options)

    base = observation_to_array(o, c.cfg.M, c.cfg.N)
    # Public synchronized time is part of the tensor contract.
    assert not (base == observation_to_array(replace(o, t=o.t + 1), c.cfg.M, c.cfg.N)).all()
    # Current proposal identity, not merely a boolean, is represented.
    assert not (base == observation_to_array(replace(o, proposal_id="different:candidate"), c.cfg.M, c.cfg.N)).all()
    # Recipe identity attached to a candidate is represented as well.
    changed_op = replace(o.operation_options[0], recipe_id="different_recipe")
    changed = replace(o, operation_options=(changed_op,) + o.operation_options[1:])
    assert not (base == observation_to_array(changed, c.cfg.M, c.cfg.N)).all()

    # Due-date/value extension attributes do not affect the canonical local
    # observation or its deterministic ordering.
    p = c.products[o.operation_options[0].product_id]
    before = observation_to_array(c.observation(*rc), c.cfg.M, c.cfg.N)
    p.due_time += 10_000; p.value += 100.0; c._invalidate()
    after = observation_to_array(c.observation(*rc), c.cfg.M, c.cfg.N)
    assert (before == after).all()

def test_action_capacity_overflow_is_loud_not_silent():
    c = AssemblyGridCore(EnvConfig(M=1, N=2, recipe_ids=("solo_inspection",), spawn_interval=999,
                                   shelf_cap=100, max_wip=100, motion_planning=False,
                                   trajectory_conflicts=False))
    c.products.clear(); c.spawned_count = 0
    # Make more simultaneously eligible pick identities than the fixed MARL
    # adapter can encode. Canonical code must fail loudly rather than slice.
    c.shelves[0]["inventory"]["A"] = 100
    for pid in range(core_module.MAX_PICK_OPTIONS + 1):
        c.products[pid] = Product(pid, "solo_inspection", 0, 999, source_rows={"A": 0})
    c._invalidate()
    with pytest.raises(RuntimeError, match="forbids silent truncation"):
        c._pick_options_for((0, 0))


def test_inclusive_or_is_extension_only_not_canonical_v1():
    rec = RecipeSpec(
        id="inclusive_ext", raw_tokens=("A", "B"), final_token="FINAL",
        operations=(
            OperationSpec("a", "inspect", ("A",), "M", kappa=1),
            OperationSpec("b", "inspect", ("B",), "M", kappa=1),
            OperationSpec("finish", "inspect", ("M",), "FINAL", kappa=1,
                          predecessor_any_inclusive=(("a", "b"),)),
        ),
    )
    lib = RecipeLibrary((rec,))
    with pytest.raises(ValueError, match="inclusive-OR"):
        AssemblyGridCore(EnvConfig(M=2, N=2, recipe_ids=("inclusive_ext",)), recipe_library=lib)
    AssemblyGridCore(EnvConfig(M=2, N=2, recipe_ids=("inclusive_ext",),
                               allow_inclusive_or_extension=True), recipe_library=lib)


def test_achievement_state_excludes_enabling_material_state():
    c = AssemblyGridCore(EnvConfig(M=1, N=2, recipe_ids=("solo_inspection",), spawn_interval=999,
                                   motion_planning=False, trajectory_conflicts=False))
    pid = next(iter(c.products))
    before = c.achievement_state(pid)
    # Moving/holding a raw token is physical/enabling state, not achievement.
    p = c.products[pid]
    p.raw_picked.add("A"); p.token_positions["A"] = (0, 0); c._robot((0, 0)).holding = (pid, "A")
    after = c.achievement_state(pid)
    assert before == after


def test_metrics_expose_unweighted_productive_concurrency_as_canonical_alias():
    c = AssemblyGridCore(EnvConfig(M=2, N=2, recipe_ids=("pair_assembly",), spawn_interval=999))
    m = c.metrics()
    assert m["productive_concurrency_capacity_mean"] == m["K_feasible_star_mean"]
    assert m["productive_concurrency_realized_mean"] == m["K_realized_mean"]
    assert m["productive_concurrency_utilization"] == m["parallelism_utilization"]
    assert m["weighted_productive_parallelism_utilization"] == m["productive_parallelism_utilization"]
    assert c.productive_parallelism() == c.feasible_parallelism()


def test_generation_seed_is_independent_of_execution_seed_for_initial_realization():
    from core import URModel
    models = (URModel("a", reach_radius=1.5), URModel("b", reach_radius=1.6))
    common = dict(M=3, N=3, recipe_ids=("solo_inspection",), robot_models=models,
                  robot_model_assignment="random", generation_seed=123, seed=None)
    a = AssemblyGridCore(EnvConfig(**common, execution_seed=1))
    b = AssemblyGridCore(EnvConfig(**common, execution_seed=999))
    sig_a = (
        tuple(r.model.name for row in a.robots for r in row),
        tuple(sorted((k, tuple(sorted(v))) for k, v in a.shelf_reach.items())),
        tuple((p.recipe_id, p.due_time, tuple(sorted(p.source_rows.items())), round(p.mass, 8))
              for p in a.products.values()),
    )
    sig_b = (
        tuple(r.model.name for row in b.robots for r in row),
        tuple(sorted((k, tuple(sorted(v))) for k, v in b.shelf_reach.items())),
        tuple((p.recipe_id, p.due_time, tuple(sorted(p.source_rows.items())), round(p.mass, 8))
              for p in b.products.values()),
    )
    assert sig_a == sig_b


def test_training_reward_is_algorithm_layer_not_env_config():
    from training_reward import TrainingRewardConfig

    cfg = EnvConfig(M=1, N=2, recipe_ids=("solo_inspection",), spawn_interval=999,
                    horizon_T=1, motion_planning=False, trajectory_conflicts=False)
    for old_field in ("reward_alpha", "reward_beta", "reward_eta", "reward_zeta", "product_value"):
        assert not hasattr(cfg, old_field)

    tr = TrainingRewardConfig(profile="test-profile", tardiness_penalty=0.25, wip_penalty=0.0)
    c = AssemblyGridCore(cfg, training_reward=tr)
    _, _reward, _terminated, _truncated, info = c.step({})
    assert info["training_reward_profile"] == "test-profile"


def test_tier1_result_reports_benchmark_metrics_not_total_reward():
    from policies import Tier1Policy

    cfg = EnvConfig(M=1, N=2, recipe_ids=("solo_inspection",), spawn_interval=999,
                    horizon_T=2, motion_planning=False, trajectory_conflicts=False)
    policy = Tier1Policy("greedy", seed=0)
    result = policy.run_episode(policy.make_core(cfg), max_steps=2)
    assert "total_reward" not in result
    assert "throughput" in result and "completion_rate" in result
