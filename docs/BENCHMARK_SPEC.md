# AssemblyGrid normative benchmark specification

Status: normative for the AssemblyGrid v1.0.0 release.

## Scope

AssemblyGrid is a dimensionless, task-level benchmark for decentralized
multi-robot production. Its canonical contribution couples three questions:

1. **Production semantics:** can controllers complete recipe-driven material
   transformations while preserving precedence, ownership, conservation, and
   delivery invariants?
2. **Temporary coalitions:** can locally informed robots form and execute
   operation-specific coalitions with declared roles and arity?
3. **Geometry-constrained productive coordination:** how do normalized reach,
   approach cost, workspace interference, and simultaneous productive
   opportunities affect coalition choice and production performance?

Canonical v1 does not claim continuous control, physical safety, contact or
grasp simulation, collision-free trajectory planning, perception, a free-form/addressable message action, failures/repair, skill learning,
zero-shot coordination, or
large-scale generalization.

## Dimensionless geometry

The canonical profile declares a positive reference reach and reference time
in dimensionless benchmark coordinates (the frozen `abstract-v1` values are
1.5 and 1.0 respectively). Comparable ratios are derived as:

`distance_norm = distance_physical / reference_reach`

`time_norm = time_physical / reference_operation_time`

`payload_norm = payload_required / payload_capacity`

Canonical ticks remain dimensionless. A physical profile may report seconds
in a separate validation record, but it must not silently redefine a tick.

Each robot has a fixed normalized base, reach, speed, clearance, tool set,
and payload capacity. An operation has an abstract operation centre and a
role-specific approach target. Geometry represents task-level reach,
approach/retreat cost, workspace occupancy, and interference; it does not
represent joint configurations or guaranteed collision-free trajectories.

## Recipe validity levels

Every official recipe passes three separate gates:

1. **Schema validity:** fields, data types, identifiers, and ranges conform to
   `schemas/recipe-v1.schema.json`.
2. **Semantic validity:** the operation graph is acyclic; tokens,
   predecessors, alternatives, roles, arities, and outputs are internally
   consistent; material outputs are not ambiguously produced.
3. **Instance realizability:** required arities fit the active manipulation
   topology and at least one reference solution exists for every official
   instance advertised as solvable.

Canonical alternatives are exclusive XOR. Inclusive OR remains an explicitly
named extension and is not used by official revision-candidate instances.

## Information contract

Canonical decentralized execution exposes the robot's own task state,
locally observable neighbour task state, locally available material actions,
locally certifiable operation/coalition candidates, public synchronized time,
and an action mask. The mask is not a global-feasibility oracle.

`proposal_support_visible` is environment-mediated local intention evidence.
For each locally exposed identity-bearing candidate, the reference observation
contains the normalized aggregate support fraction `len(votes)/candidate_size`;
it does **not** expose a list of supporting robot identities or a neighbour
`proposal_id`. It is therefore not a free-form/addressable message action and
must be reported as part of the information condition. In very small candidate
sets, an agent may sometimes infer another member's support logically from the
candidate membership, its own action, and the aggregate value; canonical v1
therefore makes no information-theoretic anonymity claim. Global state, remote
resource state, and exact physical feasibility remain privileged.


## Reference neural adapter boundary

The canonical structured observation/action semantics are not identical to the
fixed-dimensional neural adapter shipped for the first-paper reference methods.
That adapter uses finite candidate-slot capacities and compact numeric identity
encodings. Overflow is rejected explicitly rather than silently truncated, but
the adapter therefore has a finite addressability ceiling even when the benchmark
semantics admit a larger local candidate set. Compact hashing/modulo encodings can
also alias distinct symbolic identities. These are limitations of the supplied
reference representation, not changes to recipe/material semantics or the
canonical identity-bearing action definition. Larger-scale studies should use a
set/attention/graph or categorical-embedding adapter and report that interface.

Candidate slots are deterministically ordered, with proposal support participating
in the operation-option ordering. This gives the feed-forward reference policy an
inductive bias: slot position can correlate with local coalition interest. The
identity-bearing candidate remains authoritative, and candidate-permutation
robustness is an optional implementation audit rather than a canonical invariant.

## Simultaneous actions and rejection taxonomy

All robot actions at tick `t` are collected before the transition is applied.
Resolution must be deterministic for fixed instance and execution seeds.
Competing commitments use a documented identity-based ordering that is
independent of container iteration order. This is a reproducibility convention,
not a claim of fairness or a physical factory priority rule. A conformance test
checks invariance to joint-action mapping order; an identity-relabeling audit is
recommended when studying fairness or symmetry.

Outcomes are distinct:

- `locally_invalid`: violates the local action contract or mask;
- `unmatched_proposal`: required coalition members did not agree in time;
- `resource_blocked`: a declared exclusive capacity is unavailable;
- `geometry_infeasible`: normalized reach or abstract geometry rejects it;
- `geometry_conflict`: it conflicts with an already selected simultaneous
  activity;
- `execution_failure`: an admitted action fails only under a named extension.

## Episode regimes

`batch` is finite production. It terminates when the declared batch is fully
delivered. If the safety horizon is reached first, the episode is truncated
and makespan is undefined.

`horizon` is continuing production. It is truncated at exactly `horizon_T`;
throughput, delivered count, completion fraction, and residual backlog are
reported. Ordinary batch makespan is not a ranking metric in this regime.

Deadlock is diagnostic, not an automatic terminal state in canonical v1.
Reports distinguish temporary inactivity, absence of productive opportunity,
policy deadlock, and an invalid/unrealizable instance.

## Official instance families

Families are properties of problems, never of algorithms:

- `flow`: recipe/material semantics with low coalition and geometry pressure;
- `coalition`: operation arity, roles, proposal matching, and local contention;
- `concurrency`: parallel recipe branches and multi-product contention.

The official suite therefore contains nine scenarios: three families at easy, medium and hard.

## Historical geometry family, not official

- `geometry`: alternative structurally valid coalitions under a predeclared
  geometry-stress setting. It is excluded from the official suite and from every
  official aggregate. In the frozen historical geometry suite, the
  three geometry levels also differ in WIP/release pressure, so they are **not**
  a conflict-radius-only causal control. The separate
  `matched_geometry_control_suite()` holds recipe, grid, WIP, release process,
  seeds, and all other problem settings fixed while varying only the declared
  conflict-radius variant.

Each family has `easy`, `medium`, and `hard` levels assigned before training
and ordered only within that family. The catalogue records recipe depth/width,
active WIP, arity distribution, coalition fraction, spacing/reach and
conflict-radius/spacing descriptors. Algorithm outcomes never define level,
and no global scalar pretends that difficulty is comparable across mechanisms.

## Required metrics

Every headline result reports five mandatory groups:

1. delivered count, completion rate, and unfinished count;
2. throughput for continuing production;
3. restricted mean flow time in every mode, plus batch makespan only when the declared batch completes;
4. coalition commitment success/failure and formation latency;
5. geometry rejection/conflict and geometry-induced delay.

Optional diagnostics are productive concurrency with opportunity counts,
unfinished-product age, handoffs/route burden, workload balance, and compute
cost. Training return is never a benchmark KPI.

## Statistical unit and failure handling

The primary independent unit for learned methods is the algorithm-training
seed. Each trained policy is evaluated on the same frozen generation/execution
pairs as every comparator. Aggregate within a training seed before computing
across-seed uncertainty. Zero-delivery, timeout, unavailable, and failed runs
remain in the record. Undefined conditional metrics are `null`, never zero.

The experiment protocol freezes methods, budgets, checkpoints, instance IDs,
seed roles, and aggregation before the main campaign. Evidence is judged by
semantic correctness, controlled mechanism effects, task validity,
discriminative coverage, statistical reliability, and reproducibility—not by
whether a particular learning method wins.

## Fidelity labels

Every run declares one of:

- `abstract-v1`: canonical normalized geometry;
- `geometry-off-control`: matched diagnostic control only;
- `ur10-case-v1`: non-canonical UR10-derived parameter case;
- `ur10-kinematic-validation-v1`: offline kinematic/capsule comparison.

Only `abstract-v1` determines canonical trajectories and official scores.
Higher-fidelity validation quantifies abstraction error and cannot silently
modify a canonical trajectory.
