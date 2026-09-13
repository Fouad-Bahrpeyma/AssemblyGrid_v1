# Configuration, profiles, seeds, and official instances

## Instance versus runtime configuration

An AssemblyGrid **instance** is the problem definition: parameters that can change feasibility, state/achievement dynamics, the observation/information structure, or task success. Runtime labels, diagnostic exactness limits, execution seeds, and algorithm seeds do not redefine instance identity.

`benchmark.instance_definition(cfg)` produces the normative hashable instance representation.

## Geometry profiles

- `abstract-v1` — canonical dimensionless geometry profile. It stores benchmark coordinates plus declared reach/time references and compares spacing/reach, clearance/reach and normalized duration; it is not physical validation.
- `ur10-demo-v1` — optional non-canonical compatibility profile, excluded from official v1 results.
- `ur10-case-v1` — non-canonical dimensional interpretation that maps normalized ratios to a declared 1.3 m reference reach. It is not IK, collision, or hardware validation.

The fidelity label `ur10-kinematic-validation-v1` is reserved for an extension that actually performs robot-model IK, joint-limit and synchronized link-collision checks.

Declared variants such as `abstract-v1/conflict-low` are explicit controlled variants. Official results enforce profile ownership; direct silent overrides are rejected.

## Three seed roles

Keep these separate:

1. **Generation seed** — selects the instance realization.
2. **Execution seed** — selects stochastic execution trajectory where applicable.
3. **Algorithm seed** — controls method initialization/training randomness.

The reported campaign uses five matched held generation/execution pairs across every method and ten independent training seeds for each MARL method/cell.

## Official realizations

Catalogue:

`release/assemblygrid-v1.0.0/official/official_realizations.json`

Each record contains:

- stable `instance_id`
- normative `instance_definition`
- `instance_sha256`
- reference execution seed
- family-relative structural difficulty descriptors
- reference execution seed and complete reference run configuration

Use `official.load_official_config()` to verify an ID/hash and reconstruct the reference configuration.

## Track configs

The official AssemblyGrid v1 suite has three mechanism families (`flow`, `coalition`, `concurrency`) with easy, medium and hard cells: 9 scenarios. Difficulty is ordinal within a family and is frozen from structural descriptors before rollouts. The pre-v1 `geometry` family is retained as a declared historical extension in `historical_realizations.json`, excluded from every official aggregate; the conflict-radius-only causal check is `standards.matched_geometry_control_suite`. Other named configs remain extension or legacy study scaffolding; for submission reproduction, use the official catalogue and `tools/verify_instances.py`.
