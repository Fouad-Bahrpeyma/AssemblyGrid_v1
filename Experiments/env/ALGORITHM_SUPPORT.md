# AssemblyGrid Algorithm Support Interface

`AssemblyGridCore.algorithm_support_snapshot()` exposes a rich, read-only snapshot for
algorithm wrappers without changing canonical AssemblyGrid v1 semantics.

```python
snapshot = core.algorithm_support_snapshot()
ctde = snapshot.to_dict(["canonical", "privileged"])
graph = snapshot.to_dict(["entities", "topology", "recipes", "candidates"])
motion = snapshot.to_dict(["canonical", "motion", "entities"])
```

Typical use:

- **IPPO:** canonical observations and action masks only.
- **MAPPO:** canonical actor observations plus `privileged` critic state.
- **QMIX:** per-agent canonical observations/actions plus privileged state.
- **Graph MARL:** `entities` + `topology` + `recipes` + `candidates`.
- **Motion-aware extension:** canonical task observation plus explicit `motion`/robot entity features.

The rich snapshot is non-canonical. A paper may only call extra groups part of its agent
observation if it explicitly declares the corresponding extension/interface. Querying rich
views never changes state transitions.
