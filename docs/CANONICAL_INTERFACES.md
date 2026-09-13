# Canonical interfaces and information boundaries

## Canonical decentralized interface

AssemblyGrid v1 defines a local, task-level decentralized observation and a discrete identity-bound action interface. The canonical observation includes the agent's task state, locally observable neighboring task state, local material/pick opportunities, eligible operation/coalition candidates, candidate attributes including proposal support, synchronized public time, and the action mask.

It deliberately excludes explicit environment messages, skills, raw own/neighbor joint configurations, unnecessary capability vectors, remote state, and global resource information.

Exact semantic identities live on the structured observation/candidate objects. The fixed neural tensor in `encoding.py` uses deterministic compressed categorical features; those scalars are not the authoritative identity mechanism and may collide.

## Canonical actions

The executable action layout contains idle, delivery, bounded pick slots, directional handoff/receive actions, and bounded operation-candidate slots. Every theoretically eligible candidate must be addressable; overflow is an error rather than silent truncation. See the constants in `Experiments/env/core.py` and the action-mask construction for the authoritative implementation.

## Privileged/global interface

`global_state_to_array()` and wrapper `.state()` methods expose non-canonical privileged state for centralized methods and CTDE critics. Access to privileged state must be reported by an algorithm using it.

## Algorithm Support Interface (#2.5)

`AssemblyGridCore.algorithm_support_snapshot()` returns a detached, read-only structured superset grouped as:

- `metadata`
- `canonical`
- `privileged`
- `entities`
- `topology`
- `recipes`
- `candidates`
- `motion`
- `resources`
- `diagnostics`
- `extensions`

Thin selectors are available under `Experiments/env/marl/support_adapters.py` for canonical, CTDE, graph, motion, and full-research views.

**Invariant:** querying any richer information interface must not change the environment trajectory for fixed seeds and actions. This is covered by the Algorithm Support Interface tests.

## Rewards

Reward is algorithm-layer support, not part of AssemblyGrid theory or instance identity. `TrainingRewardConfig` may be supplied to wrappers/trainers for convenience. Benchmark comparisons use benchmark KPIs, not training return.
