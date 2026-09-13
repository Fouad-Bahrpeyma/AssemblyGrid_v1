# Changelog

All notable changes to AssemblyGrid are recorded here. Versions follow the benchmark
version tag in `VERSION` and `benchmark_metadata.json`.

## [1.0.0] - 2026-09

First public release of the AssemblyGrid v1 benchmark.

### Benchmark definition

- Task definition for continuing multi-robot production on an `M x N` lattice of
  fixed-base manipulators, with one task-level agent per robot.
- Four represented mechanisms: recipe and material progression, temporary coalition
  coordination, abstract task-level geometric feasibility, and productive concurrency.
- Official suite of nine scenarios: the `flow`, `coalition`, and `concurrency` workload
  families at easy, medium, and hard settings (`standards.official_instance_suite`).
- Maximum coalition arity is derived from the manipulation topology
  (`TOPOLOGY_MAX_ARITY`), not configured independently: four on the Moore grids used by
  the official scenarios.
- Canonical geometry profile `abstract-v1`; recipe schema `assemblygrid-recipe-v1`.
- Benchmark outcomes and mechanism diagnostics are defined independently of any training
  reward. Reward configuration lives in `training_reward.py` and is an algorithm-layer
  concern that does not change benchmark instance identity.

### Interfaces

- `AssemblyGridCore` for direct simulation, `AssemblyGridParallelEnv` (PettingZoo) for the
  parallel interface, and a dependency-free adapter in `marl/core_parallel_env.py`.
- Local observations expose the aggregate candidate-support fraction only; supporter
  identities, global feasibility, and messages are not exposed.
- Action masks cover 32 pick slots and 64 operation slots, with an explicit error rather
  than silent truncation if a state exceeds them.
- Separate generation, execution, and algorithm seeds.

### Reference controllers

- Non-learning references: Random masked (`random_feasible`), Greedy, Parallel-aware, and
  a privileged Centralized dispatch reference.
- Learned references: IPPO, MAPPO, and QMIX sharing the same local execution information
  and action masks (`Experiments/env/marl/`, interface-demonstration implementations).
- `reference_training/`: the training code of the reported 270-run campaign, with the
  `StructuredActor` policy network, the value network, the PPO update and the four campaign
  runners, at the hyperparameters given in the paper's MARL configuration table.

### Data and verification

- 45 official realizations over suite indices 0-4 in `official/official_realizations.json`
  (`assemblygrid-official-realizations-v4`), verified by `tools/verify_instances.py`.
- Conformance, invariant, and interface test suite covering the eleven benchmark
  invariants (`python -m pytest Experiments/env/tests -q`).
- Recipe and result JSON schemas in `schemas/`.

### Pre-release corrections

- The productive-concurrency capacity search now treats mutually exclusive alternative
  routes (XOR) of one product as incompatible, matching the exclusivity that admission
  already enforced. Previously the search could count two exclusive routes together and
  report the state as exact, overstating capacity and understating utilization for such
  recipes. No official recipe declares alternative routes, so no reported value changes.
  Covered by a regression test in `Experiments/env/tests/test_alternative_routes.py`.
- Operation admissions blocked by conflict with a committed activity now increment the
  workspace-conflict counter, as the pick, handoff and delivery paths already did. The
  matched conflict-radius study was regenerated for this change; no other metric of that
  study moved, and no learned-reference value depends on the counter.

- Blocked activities are now counted by cause: the aggregate keeps its previous meaning
  and value, and the cause-specific blocking counters `geometry_block_count`,
  `resource_block_count` and `robot_block_count` were added, with `blocked_activity_count`
  as a semantic alias of the legacy aggregate. A replay of the nine official scenarios
  under three controllers before and after the corrections changed no metric other than the
  blocking counter and wall-clock timing.

- Blocking classification: resource exhaustion now outranks geometric overlap, so an
  activity blocked by both is reported as resource-blocked rather than geometry-only.
  Causes are ranked robot, then resource, then geometry.

### Earlier development (pre-release)

These changes were made during development and are recorded for traceability. They are
part of this first release, not of any earlier published version.

- The geometry workload family used in earlier revision candidates was moved out of the
  official suite. It varied release and work-in-progress pressure together with the
  workspace conflict radius, so it could not isolate geometry as a single factor. It is
  retained as a declared historical extension in `historical_instance_suite`, excluded
  from every official aggregate, and replaced for causal analysis by the
  conflict-radius-only `matched_geometry_control_suite`.
- The Parallel-aware reference evaluates congestion at each candidate destination cell.
  Scoring congestion around the acting robot instead produced the same value for every
  candidate direction, which made Parallel-aware behave identically to Greedy.
- The material-transfer rule for the structured references accepts a transfer only when it
  reduces the distance to the deterministic staging or output target, which removes
  ping-pong transfers between neighbors.
- Coalition roles bound to declared material inputs are assigned first to the robots
  holding those inputs; remaining members take the remaining roles in row-major order.

## Unreleased

No changes.
