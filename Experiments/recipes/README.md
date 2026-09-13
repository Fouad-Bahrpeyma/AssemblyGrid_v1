# AssemblyGrid recipe files

The canonical recipe engine accepts arbitrary directed acyclic process graphs with operation arity `kappa` from 1 to 4.  The JSON files in this folder are the reference recipes shipped with AAB_009 and are round-trip equivalents of the built-in defaults.

Each recipe declares:

- `raw_tokens`: material tokens that must be picked from the input interface;
- `operations`: transformation nodes;
- `inputs` / `output`: material-token consumption and production;
- `predecessors`: explicit process precedence;
- `kappa`: number of robots required simultaneously for that operation;
- `roles`, `role_inputs`, `role_skills`, `role_tools`: team-role and capability requirements;
- `resources`: fixtures/tools/zones reserved by the operation;
- `duration`: nominal process duration;
- `productive_weight`: optional value used by the productive-parallelism diagnostic;
- `final_token`: token that makes the product eligible for delivery.

Load a recipe with `recipe.load_recipe_json(path)` or a set with `recipe.load_recipe_library(paths)`.  Invalid DAGs, unknown predecessors/tokens, duplicate ids, invalid arity, or role inconsistencies fail validation before a run begins.
