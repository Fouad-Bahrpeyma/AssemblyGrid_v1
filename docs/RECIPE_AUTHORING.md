# Authoring AssemblyGrid recipes

Recipes are explicit directed acyclic transformation graphs. Reference JSON recipes live in `Experiments/recipes/`.

## Required top-level fields

- `id`
- `schema` with value `assemblygrid-recipe-v1`
- `raw_tokens`
- `operations`
- `final_token`

`product_value` may exist as an optional product-type/extension attribute; it is not an RL reward coefficient.

## Operation fields

A canonical operation specifies:

- `id`, `kind`
- `inputs`, `output`
- `predecessors`
- `predecessor_any` for exclusive alternative readiness where used by the canonical XOR semantics
- `kappa` (required simultaneous participants)
- `roles`
- optional `role_inputs`, `role_skills`, `role_tools`
- `resources`
- `duration`
- optional diagnostic `productive_weight`

Canonical v1 uses sequence/AND/XOR semantics; inclusive-OR is not part of canonical v1.

## Coalition arity

A recipe may request `kappa` up to the capacity admitted by the active manipulation topology. The canonical maximum under Moore radius-1 is derived from the manipulation-graph clique number, not from a hard-coded 2x2 reservation rule.

## Validation

Load one recipe with `recipe.load_recipe_json(path)` or a library with `recipe.load_recipe_library(paths)`. Validation is deliberately layered:

1. Schema validity rejects unknown fields, implicit type coercion, missing fields and invalid ranges.
2. Structural/semantic validity rejects cycles, unknown predecessors/tokens, invalid AND/XOR groups, invalid arity, duplicate IDs, and inconsistent role/material bindings.
3. Instance executability checks topology clique capacity, sources, handoff paths, resources, and the selected normalized geometry.

A recipe can be well formed at levels 1--2 while a particular instance using it is not executable; official realizations must satisfy all applicable levels.

The normative JSON Schema is `schemas/recipe-v1.schema.json`. Use existing recipes such as `flow_serial_3.json`, `parallel_branch_light.json`, `supported_insert.json`, and `alternative_route.json` as templates.
