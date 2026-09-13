# AssemblyGrid v1 implementation conformance

This file maps the frozen AssGrDims#1 rules to the authoritative executable
implementation and automated evidence. It describes **canonical v1**. Richer
internal state and extension interfaces may exist without changing the
canonical decentralized interface.

## Authority and interfaces

- Authoritative transition/feasibility core: `core.py`
- Recipe schema/validation: `recipe.py`
- Canonical local tensor encoding: `encoding.py::observation_to_array`
- Privileged/CTDE tensor encoding: `encoding.py::global_state_to_array`
- Geometry/motion profile registry: `profiles.py`
- Abstract motion proxy: `motion.py`
- Decentralized MARL wrapper: `pettingzoo_env.py`
- Privileged centralized wrapper: `gym_env.py`
- AssGrDims#2 mechanism audits: `audits.py`

The canonical decentralized observation is not the same thing as all state the
software can store. Robot joint proxies, model/capability details, skills and
full product/global state remain available internally and/or through the
explicit privileged interface for centralized training, CTDE, diagnostics and
future extensions.

## #1 rule -> code -> test

| AssGrDims#1 rule | Authoritative implementation | Automated evidence |
|---|---|---|
| Local visibility, handoff and manipulation are distinct relations | `core.py`: `obs_neighbors`, `handoff_neighbors`, `manip_neighbors` | `test_v1_conformance.py::test_topology_arity_cap_equals_manipulation_graph_clique_number` |
| Coalition locality is pairwise adjacency in the manipulation graph; max arity is its clique bound | `core.py`: `_coalition_local`, `TOPOLOGY_MAX_ARITY` | `test_invariants.py::test_moore_neighborhood_and_pairwise_coalition_locality`; `test_v1_conformance.py::test_topology_arity_cap_equals_manipulation_graph_clique_number` |
| The four-agent Moore case is not an independent hard-coded 2x2 resource rule | `core.py`: clique/locality candidate generation | `test_invariants.py::test_non_2x2_four_robot_shape_is_not_generated_but_2x2_is`; `test_no_exclusive_2x2_capacity_assumption` |
| Recipes are data-driven and canonical v1 uses sequence/AND/XOR; inclusive-OR is extension-only | `recipe.py`; `core.py::_op_ready`, XOR route lock; constructor extension guard | `test_alternative_routes.py`; `test_v1_conformance.py::test_inclusive_or_is_extension_only_not_canonical_v1` |
| A product has a task-level achievement state distinct from material/physical state | `core.py::ProductAchievementState`, `achievement_state` | `test_v1_conformance.py::test_achievement_state_excludes_enabling_material_state` |
| Pickup/transport/handoff are enabling, while productive concurrency counts recipe-operation progress | `core.py`: operation candidates/teams and `K_*` concurrency metrics | `test_parallel_dag_branches_of_same_product_can_run_concurrently`; `test_v1_conformance.py::test_metrics_expose_unweighted_productive_concurrency_as_canonical_alias` |
| Agents choose exact coalition candidates; the environment checks/arbitrates feasibility but does not assign teammates | `core.py::_operation_options_for`, `_team_proposals` | `test_invariants.py::test_agents_choose_exact_candidate_and_team_forms_only_on_matching_votes` |
| Coalition membership is fixed during a committed operation | `core.py::Team`, `_start_team`; invariant checks | `test_centralized.py::test_committed_coalition_membership_is_immutable_mid_operation` |
| Proposal support is an aggregate local intent signal and does not reveal supporter identities | `core.py::_operation_options_for` | `test_v1_conformance.py::test_proposal_support_is_aggregate_and_supporter_identity_is_not_exposed` |
| Canonical v1 has no explicit environment communication action/message | action layout and `Observation` in `core.py` | `test_v1_conformance.py::test_canonical_interface_has_no_environment_communication_action_or_message` |
| Canonical decentralized observation excludes raw joint state and global state | `core.py::Observation`; `encoding.py::observation_to_array` | `test_encoding_and_benchmark.py::test_canonical_local_encoding_excludes_raw_joint_state_and_has_fixed_shape` |
| Canonical decentralized observation includes synchronized time, current proposal identity, product/recipe identity and excludes due-date urgency/value | `core.py::Observation`, `PickOption`, `OperationOptionView`; `encoding.py::observation_to_array` | `test_v1_conformance.py::test_canonical_observation_has_time_proposal_and_recipe_identity_but_no_urgency` |
| Privileged global state is a separate interface, not concatenated local observations | `encoding.py::global_state_to_array`; `pettingzoo_env.py::state`; `gym_env.py` | `test_encoding_and_benchmark.py::test_privileged_global_state_is_not_just_concatenated_local_observation`; wrapper tests when optional deps are installed |
| Every locally eligible candidate is represented; fixed MARL adapters may never silently truncate | `_pick_options_for`, `_operation_options_for` raise on capacity overflow | `test_v1_conformance.py::test_action_capacity_overflow_is_loud_not_silent`; `audits.py::addressability_audit` |
| Handoff is bilateral and persistent | `core.py::_handoffs`, completion events | `test_invariants.py::test_bilateral_handoff_is_persistent_and_unilateral_request_does_not_transfer` |
| Picks/operations/delivery are persistent state transitions and material ownership is conserved | `core.py` event scheduling/completion; `invariants.py` | `test_invariants.py::test_pick_reserves_stock_and_respects_persistent_duration`; random invariant rollout |
| Compatible recipe branches can execute productively in parallel | candidate/conflict selection in `core.py` | `test_invariants.py::test_parallel_dag_branches_of_same_product_can_run_concurrently` |
| Local candidate exposure is not a global-conflict oracle | `candidate_operations` vs `feasible_operation_candidates` | `test_invariants.py::test_local_operation_options_do_not_leak_global_active_conflicts` |
| Canonical motion awareness uses a named nominal profile, separate from visualization | `profiles.py`, `motion.py`; demos select profiles | `test_v1_conformance.py::test_profile_ownership_and_noncanonical_demo_guard`; `test_each_declared_profile_variant_admits_nominal_four_robot_common_target` |
| The UR10-labelled demo profile cannot be used silently as an official v1 result profile | `profiles.py`; `EnvConfig.validate` official-result guard | `test_v1_conformance.py::test_profile_ownership_and_noncanonical_demo_guard` |
| Flow-time conditional means retain censoring context | `core.py::metrics`, backlog/quantile helpers | `test_metrics_audit.py::test_conditional_means_are_reported_with_their_censoring_context`; related metric tests |
| Training reward is not a benchmark parameter or KPI | `training_reward.py::TrainingRewardConfig`; `AssemblyGridCore(..., training_reward=...)`; reward fields are absent from `EnvConfig`; benchmark metrics are reported separately | `test_v1_conformance.py::test_training_reward_is_algorithm_layer_not_env_config`; `test_invariants.py::test_tardiness_reward_weight_is_live_not_dead_configuration` |
| Instance generation and execution stochasticity can be seeded independently | `EnvConfig.generation_seed`, `execution_seed`; separate RNG streams in `core.py` | `test_v1_conformance.py::test_generation_seed_is_independent_of_execution_seed_for_initial_realization` |

## Canonical action adapter

The fixed-size MARL adapter uses 114 discrete actions:

- `0`: idle
- `1`: deliver
- `2..33`: pick option slots (32)
- `34..41`: handoff direction slots (8)
- `42..49`: receive direction slots (8)
- `50..113`: operation/candidate slots (64)

Every slot maps to an explicit identity-bearing option in the same decision
state. Exact IDs live on the structured observation/option objects; the fixed-size neural tensor uses deterministic compressed categorical features for string labels and is not itself the normative identity store. If a state exceeds the fixed adapter capacity, the core raises a clear
addressability error; it never slices away valid choices silently. Official
release realizations must pass `audits.py::addressability_audit`.

## Canonical decentralized observation

The reference canonical-local tensor adapter (`OBS_DIM = 1566`) contains task-level self state,
the synchronized public clock, current-proposal identity information, local
neighbour task state, local pick opportunities with product/recipe identity,
identity-bearing operation candidates with product/recipe/operation/candidate
identity, proposal support, candidate abstract-motion consequences, and the
action-mask/position context needed by the wrapper. Due-date urgency and
product value are not canonical observation fields or candidate-ordering keys.
It excludes environment messages, skills, raw own/neighbour joint
configurations, raw neighbour model vectors and global state.

The larger internal state is deliberately retained for the privileged global
encoder and future named extensions.

## Parallelism metric boundary

Canonical v1 productive parallelism is the **unweighted concurrency of recipe-operation achievement transitions**. The canonical metric names are `productive_concurrency_capacity_mean`, `productive_concurrency_realized_mean`, and `productive_concurrency_utilization`. Weighted `P_*` quantities remain optional research diagnostics; the older `productive_parallelism_utilization` key is retained only as a backwards-compatible weighted alias.

## Motion/profile boundary

`abstract-v1` is canonical. It uses nominal geometric units and abstract reach,
travel and straight-line conflict checks. It does not establish URDF/FK/IK,
link-level/self collision, dynamics, contact, grasp stability or physical
multi-arm manipulability.

`ur10-demo-v1` is an optional non-canonical compatibility profile.
The registry rejects it when `official_result=True`. Canonical benchmark
presets enforce their profile-owned fields, and `official_result=True` forces
that validation even if a caller attempts to disable profile enforcement.
Controlled geometry sweeps use declared profile variants instead of silent
free-field mutation.


## Algorithm-support reward interface

Reward is deliberately outside the canonical benchmark configuration.
`EnvConfig` contains no reward weights. RL/MARL code may pass a separate
`TrainingRewardConfig` to the core or wrappers; the active profile is reported
in step info for reproducibility. `legacy-shaped-v1` preserves the historical
environment training signal, while `sparse_delivery_reward()` provides a
convenience sparse signal. Neither is a benchmark KPI or part of instance
identity, and #3.3 must explicitly select/report the training signal it uses.

Tier-1 heuristic and centralized baseline result dictionaries no longer expose
`total_reward`; they report benchmark KPIs instead.

## Algorithm Support Interface (non-canonical)

AssemblyGrid v1 additionally exposes `AssemblyGridCore.algorithm_support_snapshot()` for
algorithm adapters. This is a **read-only structured superset**, not an expansion of the
canonical decentralized observation. The snapshot groups are:

- `metadata`: time, instance/profile identity, mode, seeds and detached config values;
- `canonical`: the exact local observations, masks and canonical fixed-size vectors;
- `privileged`: the existing CTDE/global-state vector;
- `entities`: full robot and product state, including optional joint/model/skill and due/value fields;
- `topology`: observation, handoff and manipulation edges plus candidate-membership edges;
- `recipes`: recipe/operation structures and typed precedence edges;
- `candidates`: operation candidates, proposals and committed teams;
- `motion`: abstract motion settings/plans and active reservations;
- `resources`: capacities, active use and shelf state;
- `diagnostics`: benchmark metrics, scheduled events and diagnostic pulses;
- `extensions`: explicit flags identifying optional/non-canonical mechanisms.

Wrappers may select any subset via `snapshot.to_dict(groups=[...])`. The snapshot contains
copied primitive/tuple data and does not expose mutable references to core state. Querying it
must not alter the environment trajectory; `test_algorithm_support.py` verifies this by
running matched seeded environments with and without support-view queries.

Algorithm learning state (network hidden state, replay buffers, advantages, value targets,
optimizer state, gradients, exploration schedules, etc.) is intentionally not exposed by the
environment and remains owned by the algorithm.
