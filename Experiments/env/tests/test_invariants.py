import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core import *
from motion import MotionPlan
from training_reward import TrainingRewardConfig
from recipe import default_recipe_library
import invariants as inv


def inject_tokens(core, pid, recipe_id, token_positions, due=999):
    rec = core.recipe_library.get(recipe_id)
    source_rows = {t: token_positions[t][0] if t in token_positions else 0 for t in rec.raw_tokens}
    p = Product(pid, recipe_id, core.t, due, source_rows=source_rows)
    p.raw_picked.update(t for t in rec.raw_tokens if t in token_positions)
    p.token_positions.update(token_positions)
    core.products[pid] = p
    for token, rc in token_positions.items():
        core._robot(rc).holding = (pid, token)
    core.next_product_id = max(core.next_product_id, pid + 1)
    core._invalidate()
    return p


def dummy_plan():
    return MotionPlan((0.0, 0.0), tuple(), 0, 1, 0, 1, 1.0, 0.0)


def test_moore_neighborhood_and_pairwise_coalition_locality():
    c = AssemblyGridCore(EnvConfig(M=3, N=3, motion_planning=False, trajectory_conflicts=False))
    assert c._local((1, 1), (2, 2))
    assert not c._local((0, 0), (0, 2))
    assert c._coalition_local(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert not c._coalition_local(((1, 1), (0, 0), (0, 1), (0, 2)))


def test_non_2x2_four_robot_shape_is_not_generated_but_2x2_is():
    c = AssemblyGridCore(EnvConfig(M=3, N=3, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99))
    c.products.clear()
    p = Product(0, "standard_ab", 0, 99, source_rows={"A": 0, "B": 1})
    p.completed_ops.add("align_ab"); p.token_positions["AB"] = (1, 1)
    c.products[0] = p; c._robot((1, 1)).holding = (0, "AB"); c._invalidate()
    coalitions = {frozenset(x.coalition) for x in c.candidate_operations() if x.operation_id == "final_assembly"}
    assert frozenset({(1, 1), (0, 0), (0, 1), (1, 0)}) in coalitions
    assert frozenset({(1, 1), (0, 0), (0, 1), (0, 2)}) not in coalitions


def test_no_exclusive_2x2_capacity_assumption():
    c = AssemblyGridCore(EnvConfig(M=2, N=3, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99))
    c.products.clear()
    inject_tokens(c, 0, "standard_ab", {"A": (0, 0), "B": (1, 0)})
    inject_tokens(c, 1, "standard_ab", {"A": (0, 2), "B": (1, 2)})
    assert c.capacity_snapshot()["K_feasible_star"] == 2


def test_parallel_dag_branches_of_same_product_can_run_concurrently():
    c = AssemblyGridCore(EnvConfig(M=2, N=4, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("parallel_branch",), spawn_interval=99))
    c.products.clear()
    inject_tokens(c, 0, "parallel_branch", {"A": (0, 0), "B": (1, 0), "C": (0, 3), "D": (1, 3)})
    ids = {x.operation_id for x in c.candidate_operations()}
    assert {"align_ab", "align_cd"} <= ids
    assert c.capacity_snapshot()["K_feasible_star"] == 2


def test_exact_capacity_solver_beats_greedy_counterexample():
    c = AssemblyGridCore(EnvConfig(M=1, N=3, motion_planning=False, trajectory_conflicts=False, spawn_interval=99))
    p = dummy_plan()
    # A conflicts with B and C by unit resources, B and C are compatible.
    A = OperationCandidate("A", 0, "a", "x", ((0, 0),), (((0, 0), "r"),), frozenset(), frozenset({"x", "y"}), p, 1.0, 1, 1)
    B = OperationCandidate("B", 1, "b", "x", ((0, 1),), (((0, 1), "r"),), frozenset(), frozenset({"x"}), p, 1.0, 1, 1)
    C = OperationCandidate("C", 2, "c", "x", ((0, 2),), (((0, 2), "r"),), frozenset(), frozenset({"y"}), p, 1.0, 1, 1)
    chosen, score, exact = c._optimal_select([A, B, C], weighted=False)
    assert exact and score == 2 and {x.id for x in chosen} == {"B", "C"}


def test_productive_utilization_is_realized_over_available_not_capacity_over_capacity():
    c = AssemblyGridCore(EnvConfig(M=2, N=3, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99))
    c.products.clear(); inject_tokens(c, 0, "standard_ab", {"A": (0, 0), "B": (1, 0)})
    c.step({})  # idle despite a ready productive operation
    assert c.p_opportunity_samples == 1
    assert c.productive_utilization() == 0.0


def test_agents_choose_exact_candidate_and_team_forms_only_on_matching_votes():
    c = AssemblyGridCore(EnvConfig(M=2, N=3, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99, formation_timeout=3))
    c.products.clear(); inject_tokens(c, 0, "standard_ab", {"A": (0, 0), "B": (1, 0)})
    obs = c._all_observations(); cand = c.candidate_operations()[0]
    acts = {}
    for rc in cand.coalition:
        slot = next(k for k, x in enumerate(obs[rc].operation_options) if x.candidate_id == cand.id)
        acts[rc] = ACTION_OPERATION_BASE + slot
    c.step(acts)
    assert c.team_formation_count == 1
    assert any(t.candidate_id == cand.id for t in c.teams.values())
    inv.check_all(c)


def test_pick_reserves_stock_and_respects_persistent_duration():
    c = AssemblyGridCore(EnvConfig(M=3, N=4, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99, pick_duration=2))
    obs = c._all_observations()
    rc = next(rc for rc, o in obs.items() if o.pick_options)
    opt = obs[rc].pick_options[0]; before = c.shelves[opt.shelf_row]["inventory"][opt.token]
    c.step({rc: ACTION_PICK_BASE})
    assert c.shelves[opt.shelf_row]["inventory"][opt.token] == before - 1
    assert c._robot(rc).holding is None and c._robot(rc).busy_until > c.t
    while c._robot(rc).busy_until > c.t:
        c.step({})
    assert c._robot(rc).holding is not None
    inv.check_all(c)


def test_bilateral_handoff_is_persistent_and_unilateral_request_does_not_transfer():
    c = AssemblyGridCore(EnvConfig(M=2, N=3, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99, handoff_duration=2))
    c.products.clear(); inject_tokens(c, 0, "standard_ab", {"A": (0, 0)})
    east = DIRECTIONS.index((0, 1)); west = DIRECTIONS.index((0, -1))
    c.step({(0, 0): ACTION_HANDOFF_BASE + east})
    assert c._robot((0, 0)).holding == (0, "A")
    c.step({(0, 0): ACTION_HANDOFF_BASE + east, (0, 1): ACTION_RECEIVE_BASE + west})
    assert c._robot((0, 0)).holding == (0, "A")  # transfer is in flight
    while c._robot((0, 0)).busy_until > c.t:
        c.step({})
    assert c._robot((0, 1)).holding == (0, "A")
    inv.check_all(c)


def test_heterogeneous_reach_is_checked_for_both_handoff_partners():
    short = URModel("short", reach_radius=0.40)
    long = URModel("long", reach_radius=2.0)
    c = AssemblyGridCore(EnvConfig(M=1, N=2, cell_spacing=1.0, robot_models=(short, long),
                                   robot_model_assignment="round_robin", spawn_interval=99))
    assert not c._handoff_static_feasible((0, 0), (0, 1))
    assert not c._handoff_static_feasible((0, 1), (0, 0))


def test_topology_variants_are_real_not_labels():
    moore = AssemblyGridCore(EnvConfig(M=3, N=3, topology="moore", spawn_interval=99))
    # von_neumann and line admit cliques of at most two, so they can only host
    # recipes whose declared arities fit; standard_ab (kappa=4) cannot run here.
    von = AssemblyGridCore(EnvConfig(M=3, N=3, topology="von_neumann", spawn_interval=99,
                                     recipe_ids=("pair_assembly",)))
    line = AssemblyGridCore(EnvConfig(M=3, N=3, topology="line", spawn_interval=99,
                                      recipe_ids=("pair_assembly",)))
    assert moore._local((1, 1), (0, 0))
    assert not von._local((1, 1), (0, 0))
    assert max(len(line.neighbors((i, j))) for i in range(3) for j in range(3)) <= 2


def test_skill_requirement_and_reference_learning_update():
    c = AssemblyGridCore(EnvConfig(M=2, N=3, skill_mode="required", skill_initial_level=0.2,
                                   skill_threshold=0.5, recipe_ids=("standard_ab",), spawn_interval=99,
                                   motion_planning=False, trajectory_conflicts=False))
    c.products.clear(); inject_tokens(c, 0, "standard_ab", {"A": (0, 0), "B": (1, 0)})
    assert not c.candidate_operations(), "low skill should gate the align operation"
    c._robot((0, 0)).skill_levels["align"] = 1.0; c._robot((1, 0)).skill_levels["align"] = 1.0; c._invalidate()
    assert c.candidate_operations()


def test_random_masked_rollout_invariants():
    c = AssemblyGridCore(EnvConfig(M=4, N=6, seed=4, horizon_T=120, spawn_interval=3,
                                   motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab", "solo_inspection")))
    obs = c._all_observations(); rng = random.Random(4)
    prev_d = c.delivered_count; prev_s = c.spawned_count
    for _ in range(120):
        acts = {rc: rng.choice([i for i, x in enumerate(o.action_mask) if x]) for rc, o in obs.items()}
        obs, _, _, trunc, _ = c.step(acts)
        inv.check_all(c); inv.check_monotonic_counts(c, prev_d, prev_s)
        prev_d, prev_s = c.delivered_count, c.spawned_count
        if trunc:
            break


def test_resource_capacity_greater_than_one_is_enforced_in_exact_selection():
    c = AssemblyGridCore(EnvConfig(M=1, N=3, motion_planning=False, trajectory_conflicts=False,
                                   resource_capacities={"fixture": 2}, spawn_interval=99))
    p = dummy_plan()
    cs = [
        OperationCandidate(f"c{k}", k, f"o{k}", "x", ((0, k),), (((0, k), "r"),),
                           frozenset(), frozenset({"fixture"}), p, 1.0, 1, 1)
        for k in range(3)
    ]
    chosen, score, exact = c._optimal_select(cs, weighted=False)
    assert exact and score == 2 and len(chosen) == 2


def test_tool_and_payload_are_operation_feasibility_constraints():
    from recipe import OperationSpec, RecipeSpec, RecipeLibrary
    rec = RecipeSpec(
        id="weld_one", raw_tokens=("A",), final_token="FINAL",
        operations=(OperationSpec(
            id="weld", kind="weld", inputs=("A",), output="FINAL", kappa=1,
            roles=("welder",), role_inputs={"welder": "A"},
            role_tools={"welder": ("welder",)}, duration=1,
        ),),
    )
    lib = RecipeLibrary((rec,))
    no_tool = URModel("no_tool", reach_radius=2.0, payload=10.0, tools=("gripper",))
    c = AssemblyGridCore(EnvConfig(M=1, N=2, robot_models=(no_tool,), recipe_ids=("weld_one",),
                                   motion_planning=False, trajectory_conflicts=False, spawn_interval=99),
                         recipe_library=lib)
    c.products.clear(); p = inject_tokens(c, 0, "weld_one", {"A": (0, 0)}); p.mass = 2.0
    assert not c.candidate_operations(), "missing required tool must make the operation infeasible"

    low_payload = URModel("welder_low", reach_radius=2.0, payload=1.0, tools=("welder",))
    c2 = AssemblyGridCore(EnvConfig(M=1, N=2, robot_models=(low_payload,), recipe_ids=("weld_one",),
                                    motion_planning=False, trajectory_conflicts=False, spawn_interval=99),
                          recipe_library=lib)
    c2.products.clear(); p2 = inject_tokens(c2, 0, "weld_one", {"A": (0, 0)}); p2.mass = 2.0
    assert not c2.candidate_operations(), "payload must constrain a material-holding role"

    capable = URModel("welder_ok", reach_radius=2.0, payload=3.0, tools=("welder",))
    c3 = AssemblyGridCore(EnvConfig(M=1, N=2, robot_models=(capable,), recipe_ids=("weld_one",),
                                    motion_planning=False, trajectory_conflicts=False, spawn_interval=99),
                          recipe_library=lib)
    c3.products.clear(); p3 = inject_tokens(c3, 0, "weld_one", {"A": (0, 0)}); p3.mass = 2.0
    assert c3.candidate_operations()


def test_tardiness_reward_weight_is_live_not_dead_configuration():
    def one_step(alpha):
        cfg = EnvConfig(M=1, N=2, recipe_ids=("solo_inspection",), spawn_interval=99,
                        motion_planning=False, trajectory_conflicts=False, deliver_duration=1)
        reward_cfg = TrainingRewardConfig(
            profile="test-tardiness", delivery_value_scale=1.0,
            tardiness_penalty=alpha, motion_energy_penalty=0.0,
            safety_penalty=0.0, wip_penalty=0.0,
        )
        c = AssemblyGridCore(cfg, training_reward=reward_cfg)
        c.products.clear(); c.spawned_count = 1
        p = Product(0, "solo_inspection", 0, 0, source_rows={"A": 0})
        p.completed_ops.add("inspect"); p.token_positions["FINAL"] = (0, 1)
        c.products[0] = p; c._robot((0, 1)).holding = (0, "FINAL"); c.next_product_id = 1; c._invalidate()
        _, reward, _, _, _ = c.step({(0, 1): ACTION_DELIVER})
        assert c.delivered_count == 1 and c.sum_tardiness == 1
        return reward
    assert one_step(0.5) == pytest.approx(one_step(0.0) - 0.5)


def test_local_operation_options_do_not_leak_global_active_conflicts():
    c = AssemblyGridCore(EnvConfig(M=2, N=4, motion_planning=False, trajectory_conflicts=False,
                                   recipe_ids=("standard_ab",), spawn_interval=99))
    c.products.clear(); inject_tokens(c, 0, "standard_ab", {"A": (0, 0), "B": (1, 0)})
    cand = c.candidate_operations()[0]
    # Occupy one of the candidate's exclusive resources with a reservation.
    rid = "remote_fixture_use"
    c.active_reservations[rid] = ActivityReservation(
        rid, "other", ((0, 3),), frozenset({next(iter(cand.resources))}), dummy_plan(), c.t + 5
    )
    c._invalidate()
    # The local team proposal remains visible: action availability is not a
    # global feasibility oracle.  The capacity diagnostic, however, excludes it.
    assert any(x.id == cand.id for x in c.candidate_operations())
    assert c.capacity_snapshot()["K_feasible_star"] == 0


def test_commitment_horizon_prevents_immediate_team_switching():
    c = AssemblyGridCore(EnvConfig(M=3, N=3, recipe_ids=("standard_ab",), spawn_interval=99,
                                   motion_planning=False, trajectory_conflicts=False, commitment_horizon=2))
    c.products.clear()
    p = Product(0, "standard_ab", 0, 99, source_rows={"A": 0, "B": 1})
    p.completed_ops.add("align_ab"); p.token_positions["AB"] = (1, 1)
    c.products[0] = p; c._robot((1, 1)).holding = (0, "AB"); c._invalidate()
    opts = c._operation_options_for((1, 1))
    assert len(opts) >= 2
    r = c._robot((1, 1)); r.proposal_id = opts[0].candidate_id; r.proposal_lock_until = c.t + 2
    mask_locked = c.action_mask(1, 1)
    enabled_locked = [k for k in range(MAX_OPERATION_OPTIONS) if mask_locked[ACTION_OPERATION_BASE + k]]
    assert len(enabled_locked) == 1
    c.t += 2
    mask_free = c.action_mask(1, 1)
    enabled_free = [k for k in range(MAX_OPERATION_OPTIONS) if mask_free[ACTION_OPERATION_BASE + k]]
    assert len(enabled_free) >= 2


def test_alternative_recipe_operations_share_inputs_and_are_mutually_exclusive():
    c = AssemblyGridCore(EnvConfig(M=2, N=3, recipe_ids=("alternative_route",), spawn_interval=99,
                                   motion_planning=False, trajectory_conflicts=False))
    c.products.clear(); inject_tokens(c, 0, "alternative_route", {"A": (0, 0), "B": (1, 0)})
    ops = c.candidate_operations()
    assert {x.operation_id for x in ops} >= {"join_precise", "join_fast"}
    # Both alternatives consume the same material tokens, so a capacity solver
    # must never schedule both for the same product at once.
    chosen, _, _ = c._optimal_select(ops, weighted=False)
    chosen_ids = {x.operation_id for x in chosen}
    assert not ({"join_precise", "join_fast"} <= chosen_ids)


def test_or_predecessor_group_unlocks_after_either_alternative():
    c = AssemblyGridCore(EnvConfig(M=1, N=3, recipe_ids=("alternative_route",), spawn_interval=99,
                                   motion_planning=False, trajectory_conflicts=False))
    c.products.clear()
    p = Product(0, "alternative_route", 0, 99, source_rows={"A": 0, "B": 0})
    p.completed_ops.add("join_fast"); p.token_positions["AB"] = (0, 1)
    c.products[0] = p; c._robot((0, 1)).holding = (0, "AB"); c._invalidate()
    assert any(x.operation_id == "finish" for x in c.candidate_operations())


def test_robot_joint_speed_changes_motion_duration_proxy():
    slow = URModel("slow", reach_radius=2.0, speed=10.0, joint_speed=0.1)
    fast = URModel("fast", reach_radius=2.0, speed=10.0, joint_speed=10.0)
    cs = AssemblyGridCore(EnvConfig(M=1, N=2, robot_models=(slow,), cell_spacing=1.0, spawn_interval=99))
    cf = AssemblyGridCore(EnvConfig(M=1, N=2, robot_models=(fast,), cell_spacing=1.0, spawn_interval=99))
    ps = cs._make_plan(((0, 0),), (1.0, 0.0), 1)
    pf = cf._make_plan(((0, 0),), (1.0, 0.0), 1)
    assert ps.total_duration > pf.total_duration


def test_support_set_can_exceed_owner_without_duplicating_material():
    """Point 6: during a committed cooperative operation the workpiece is
    supported by every team member, while ownership stays with exactly one
    robot. Verifies both that multi-support genuinely occurs and that it
    never creates a second copy of the material."""
    from policies import Tier1Policy
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(EnvConfig(M=7, N=10, spawn_interval=2, max_wip=16,
                                      motion_planning=False, seed=1))
    obs = core.reset()
    saw_multi_support = False
    for _ in range(300):
        obs, _r, term, trunc, _ = core.step(policy.act(core, obs))
        inv.check_support_vs_ownership(core)
        for p in core.products.values():
            for token in list(p.token_positions):
                if len(core.token_supporters(p.id, token)) > 1:
                    saw_multi_support = True
        if term or trunc:
            break
    assert saw_multi_support, "no multi-robot support ever occurred; test is vacuous"
