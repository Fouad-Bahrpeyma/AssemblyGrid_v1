"""Recipe DAG primitives for AssemblyGrid.

The reference layer keeps recipes data-driven.  Picks and final delivery are
logistics actions; every transformation between them is an OperationSpec in a
DAG.  Operation arity is independent of topology and may be 1..4.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Optional


RECIPE_SCHEMA_VERSION = "assemblygrid-recipe-v1"


def validate_recipe_document(data: Mapping) -> None:
    """Validate the public JSON contract before coercing values.

    The runtime intentionally does not depend on a JSON-Schema package; the
    normative machine-readable schema lives at ``schemas/recipe-v1.schema.json``
    and this function enforces its safety-critical subset at load time.
    Rejecting before ``str``/``int`` coercion prevents malformed documents from
    becoming apparently valid recipes by accident.
    """
    if not isinstance(data, Mapping):
        raise ValueError("recipe document must be a JSON object")
    allowed = {"schema", "id", "raw_tokens", "operations", "final_token", "product_value"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"recipe document has unknown fields: {unknown}")
    for key in ("id", "raw_tokens", "operations", "final_token"):
        if key not in data:
            raise ValueError(f"recipe document is missing required field {key!r}")
    if "schema" in data and data["schema"] != RECIPE_SCHEMA_VERSION:
        raise ValueError(f"unsupported recipe schema {data['schema']!r}")
    if not isinstance(data["id"], str) or not data["id"]:
        raise ValueError("recipe id must be a non-empty string")
    if not isinstance(data["final_token"], str) or not data["final_token"]:
        raise ValueError("final_token must be a non-empty string")
    if not isinstance(data["raw_tokens"], list) or not data["raw_tokens"]:
        raise ValueError("raw_tokens must be a non-empty JSON array")
    if not all(isinstance(x, str) and x for x in data["raw_tokens"]):
        raise ValueError("raw_tokens must contain non-empty strings")
    if not isinstance(data["operations"], list) or not data["operations"]:
        raise ValueError("operations must be a non-empty JSON array")
    allowed_op = {
        "id", "kind", "inputs", "output", "predecessors", "predecessor_any",
        "predecessor_any_inclusive", "kappa", "kappa_max", "roles",
        "role_inputs", "role_skills", "role_tools", "resources", "duration",
        "productive_weight",
    }
    for index, op in enumerate(data["operations"]):
        if not isinstance(op, Mapping):
            raise ValueError(f"operation {index} must be a JSON object")
        unknown_op = sorted(set(op) - allowed_op)
        if unknown_op:
            raise ValueError(f"operation {index} has unknown fields: {unknown_op}")
        for key in ("id", "kind", "inputs", "output", "kappa", "duration"):
            if key not in op:
                raise ValueError(f"operation {index} is missing required field {key!r}")
        if not isinstance(op["id"], str) or not op["id"]:
            raise ValueError(f"operation {index} id must be a non-empty string")
        if not isinstance(op["kind"], str) or not op["kind"]:
            raise ValueError(f"operation {op['id']!r} kind must be a non-empty string")
        if not isinstance(op["inputs"], list) or not all(isinstance(x, str) and x for x in op["inputs"]):
            raise ValueError(f"operation {op['id']!r} inputs must be an array of non-empty strings")
        if not isinstance(op["output"], str) or not op["output"]:
            raise ValueError(f"operation {op['id']!r} output must be a non-empty string")
        if type(op["kappa"]) is not int or not 1 <= op["kappa"] <= 4:
            raise ValueError(f"operation {op['id']!r} kappa must be an integer in 1..4")
        if type(op["duration"]) is not int or op["duration"] < 1:
            raise ValueError(f"operation {op['id']!r} duration must be a positive integer")


@dataclass(frozen=True)
class OperationSpec:
    id: str
    kind: str
    inputs: Tuple[str, ...]
    output: str
    predecessors: Tuple[str, ...] = ()
    # EXCLUSIVE alternative routes (XOR): exactly one member of each group is
    # performed. This is the process-planning reading, a product takes one
    # route through the plant, and running two would be duplicated work.
    predecessor_any: Tuple[Tuple[str, ...], ...] = ()
    # INCLUSIVE alternative routes (OR): at least one member must complete,
    # and running more than one is PERMITTED. This is not the same thing as
    # XOR and is not redundant with it: under operation/handoff failure
    # probabilities or duration noise (see EnvConfig.*_failure_prob,
    # duration_noise, and the AG-Robust track), deliberately executing two
    # routes is a rational hedge rather than waste. Supporting only XOR would
    # make that strategy inexpressible.
    predecessor_any_inclusive: Tuple[Tuple[str, ...], ...] = ()
    # Admissible arity interval [kappa, kappa_max]. kappa is the MINIMUM the
    # operation requires; kappa_max is the largest team that has a physically
    # meaningful job to do. kappa_max defaults to kappa, i.e. EXACT arity: a
    # recipe must opt in explicitly before extra robots are admissible, and
    # must give them declared roles. Extra robots never accelerate anything
    # (see core._operation_duration): "one more arm" is not a general
    # mechanism for making a task faster, so overstaffing is only ever
    # allowed where the recipe author says a support role genuinely exists.
    kappa: int = 1
    kappa_max: Optional[int] = None
    roles: Tuple[str, ...] = ()
    role_inputs: Mapping[str, str] = field(default_factory=dict)
    role_skills: Mapping[str, str] = field(default_factory=dict)
    role_tools: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    resources: Tuple[str, ...] = ()
    duration: int = 1
    productive_weight: float = 1.0

    @property
    def all_alternative_groups(self) -> Tuple[Tuple[str, ...], ...]:
        """Every alternative group, exclusive and inclusive alike. Use this
        for READINESS (both kinds mean "at least one must be done"); use
        predecessor_any alone for EXCLUSIVITY."""
        return tuple(self.predecessor_any) + tuple(self.predecessor_any_inclusive)

    @property
    def arity_max(self) -> int:
        """Largest admissible team; defaults to exact arity (== kappa)."""
        return self.kappa if self.kappa_max is None else self.kappa_max

    def validate(self) -> None:
        if not self.id:
            raise ValueError("operation id may not be empty")
        if not (1 <= self.kappa <= 4):
            raise ValueError(f"operation {self.id!r}: kappa must be in 1..4")
        if not (self.kappa <= self.arity_max <= 4):
            raise ValueError(
                f"operation {self.id!r}: kappa_max must satisfy kappa <= kappa_max <= 4 "
                f"(got kappa={self.kappa}, kappa_max={self.arity_max})")
        if len(self.roles) not in (0, self.arity_max):
            raise ValueError(
                f"operation {self.id!r}: roles must be empty or have length kappa_max "
                f"({self.arity_max}), so every admissible member has a declared role")
        if self.duration < 1:
            raise ValueError(f"operation {self.id!r}: duration must be >=1")
        if not self.output:
            raise ValueError(f"operation {self.id!r}: output token may not be empty")
        for role, token in self.role_inputs.items():
            if self.roles and role not in self.roles:
                raise ValueError(f"operation {self.id!r}: role_inputs contains unknown role {role!r}")
            if token not in self.inputs:
                raise ValueError(f"operation {self.id!r}: role {role!r} refers to non-input token {token!r}")


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    raw_tokens: Tuple[str, ...]
    operations: Tuple[OperationSpec, ...]
    final_token: str
    product_value: float = 1.0

    def validate(self) -> None:
        if not self.id:
            raise ValueError("recipe id may not be empty")
        if not self.raw_tokens:
            raise ValueError(f"recipe {self.id!r}: at least one raw token is required")
        if not self.operations:
            raise ValueError(f"recipe {self.id!r}: at least one operation is required")
        if len(set(self.raw_tokens)) != len(self.raw_tokens):
            raise ValueError(f"recipe {self.id!r}: raw token ids must be unique")
        ids = [op.id for op in self.operations]
        if len(set(ids)) != len(ids):
            raise ValueError(f"recipe {self.id!r}: operation ids must be unique")
        op_map = {op.id: op for op in self.operations}
        available = set(self.raw_tokens)
        for op in self.operations:
            op.validate()
            for pred in op.predecessors:
                if pred not in op_map:
                    raise ValueError(f"recipe {self.id!r}: operation {op.id!r} has unknown predecessor {pred!r}")
            for group in op.all_alternative_groups:
                if not group:
                    raise ValueError(f"recipe {self.id!r}: operation {op.id!r} has an empty alternative group")
            xor_members = {q for g in op.predecessor_any for q in g}
            or_members = {q for g in op.predecessor_any_inclusive for q in g}
            if xor_members & or_members:
                raise ValueError(
                    f"recipe {self.id!r}: operation {op.id!r} lists "
                    f"{sorted(xor_members & or_members)} in both an exclusive and an "
                    f"inclusive alternative group; the semantics would be ambiguous")
            for group in op.all_alternative_groups:
                for pred in group:
                    if pred not in op_map:
                        raise ValueError(
                            f"recipe {self.id!r}: operation {op.id!r} has unknown alternative predecessor {pred!r}")
            for token in op.inputs:
                # Inputs may be produced later in tuple order; full DAG validation below.
                if token not in available and token not in {x.output for x in self.operations}:
                    raise ValueError(f"recipe {self.id!r}: operation {op.id!r} has unknown input token {token!r}")
            available.add(op.output)
        if self.final_token not in available:
            raise ValueError(f"recipe {self.id!r}: final_token {self.final_token!r} is never produced")
        if self.final_token in self.raw_tokens:
            raise ValueError(f"recipe {self.id!r}: final_token may not be a raw token")
        consumers = {token for op in self.operations for token in op.inputs}
        if self.final_token in consumers:
            raise ValueError(f"recipe {self.id!r}: final_token must be terminal and may not be consumed")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        op_map = {op.id: op for op in self.operations}
        state: Dict[str, int] = {}

        def visit(node: str) -> None:
            mark = state.get(node, 0)
            if mark == 1:
                raise ValueError(f"recipe {self.id!r}: predecessor graph contains a cycle")
            if mark == 2:
                return
            state[node] = 1
            for pred in op_map[node].predecessors:
                visit(pred)
            for group in op_map[node].all_alternative_groups:
                for pred in group:
                    visit(pred)
            state[node] = 2

        for op_id in op_map:
            visit(op_id)

    @property
    def op_map(self) -> Dict[str, OperationSpec]:
        return {op.id: op for op in self.operations}

    def topological_operations(self) -> Tuple[OperationSpec, ...]:
        op_map = self.op_map
        done = set()
        out = []
        while len(done) < len(op_map):
            progressed = False
            for op in self.operations:
                if op.id in done:
                    continue
                if all(p in done for p in op.predecessors) and all(any(p in done for p in group) for group in op.all_alternative_groups):
                    out.append(op)
                    done.add(op.id)
                    progressed = True
            if not progressed:
                raise ValueError(f"recipe {self.id!r}: could not topologically order operations")
        return tuple(out)

    def downstream_critical_duration(self, op_id: str) -> int:
        """Nominal critical-path work from op_id through its descendants."""
        op_map = self.op_map
        succ: Dict[str, list[str]] = {k: [] for k in op_map}
        for op in self.operations:
            for p in op.predecessors:
                succ[p].append(op.id)
            for group in op.all_alternative_groups:
                for p in group:
                    succ[p].append(op.id)
        memo: Dict[str, int] = {}

        def rec(x: str) -> int:
            if x in memo:
                return memo[x]
            tail = max((rec(y) for y in succ[x]), default=0)
            memo[x] = op_map[x].duration + tail
            return memo[x]

        return rec(op_id)

    @property
    def nominal_critical_path(self) -> int:
        if not self.operations:
            return 0
        op_map = self.op_map
        memo: Dict[str, int] = {}

        def finish(op_id: str) -> int:
            if op_id in memo:
                return memo[op_id]
            op = op_map[op_id]
            required = [finish(p) for p in op.predecessors]
            # An OR-predecessor group becomes ready after its earliest viable
            # alternative.  This gives a nominal lower-bound critical path.
            required += [min(finish(p) for p in group) for group in op.all_alternative_groups]
            memo[op_id] = op.duration + max(required, default=0)
            return memo[op_id]

        # The critical path is the completion time of the TERMINAL
        # operations, not the maximum over every operation. Taking the max
        # over all of them let an unselected alternative dominate: for
        # alternative_route (join_precise=5, join_fast=2, finish=2) the
        # shortest valid route is join_fast + finish = 4, but max() returned
        # finish(join_precise) = 5, a branch the product never performs. That
        # inflated flow_time_lower_bound() and every generated due time.
        consumed = set()
        for op in self.operations:
            consumed.update(op.predecessors)
            for group in op.all_alternative_groups:
                consumed.update(group)
        sinks = [op.id for op in self.operations if op.id not in consumed]
        if not sinks:  # pathological/cyclic guard: fall back to all operations
            sinks = [op.id for op in self.operations]
        return max(finish(op_id) for op_id in sinks)


class RecipeLibrary:
    def __init__(self, recipes: Iterable[RecipeSpec] = ()):
        self._recipes: Dict[str, RecipeSpec] = {}
        for recipe in recipes:
            self.add(recipe)

    def add(self, recipe: RecipeSpec) -> None:
        recipe.validate()
        self._recipes[recipe.id] = recipe

    def get(self, recipe_id: str) -> RecipeSpec:
        try:
            return self._recipes[recipe_id]
        except KeyError as exc:
            raise KeyError(f"unknown recipe {recipe_id!r}; available={sorted(self._recipes)}") from exc

    def __contains__(self, recipe_id: str) -> bool:
        return recipe_id in self._recipes

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._recipes)

    def values(self) -> Tuple[RecipeSpec, ...]:
        return tuple(self._recipes.values())


# ---------------------------------------------------------------------------
# Built-in reference recipes.  These are intentionally small but structurally
# distinct so AG-Recipes and held-out-composition tests work without external
# files.  Users can construct additional RecipeSpec objects programmatically.
# ---------------------------------------------------------------------------

STANDARD_AB = RecipeSpec(
    id="standard_ab",
    raw_tokens=("A", "B"),
    operations=(
        OperationSpec(
            id="align_ab", kind="align", inputs=("A", "B"), output="AB", kappa=2,
            roles=("holder_A", "holder_B"),
            role_inputs={"holder_A": "A", "holder_B": "B"},
            role_skills={"holder_A": "align", "holder_B": "align"},
            duration=3, productive_weight=1.0,
        ),
        OperationSpec(
            id="final_assembly", kind="assemble", inputs=("AB",), output="FINAL",
            predecessors=("align_ab",), kappa=4,
            roles=("holder_AB", "operator", "stabilizer", "support"),
            role_inputs={"holder_AB": "AB"},
            role_skills={
                "holder_AB": "assemble", "operator": "assemble",
                "stabilizer": "assemble", "support": "assemble",
            },
            duration=4, productive_weight=1.2,
        ),
    ),
    final_token="FINAL",
)

STANDARD_ABC = RecipeSpec(
    id="standard_abc",
    raw_tokens=("A", "B", "C"),
    operations=(
        OperationSpec(
            id="align_ab", kind="align", inputs=("A", "B"), output="AB", kappa=2,
            roles=("holder_A", "holder_B"),
            role_inputs={"holder_A": "A", "holder_B": "B"},
            role_skills={"holder_A": "align", "holder_B": "align"},
            duration=3, productive_weight=1.0,
        ),
        OperationSpec(
            id="stabilize_insert", kind="assemble", inputs=("AB", "C"), output="FINAL",
            predecessors=("align_ab",), kappa=4,
            roles=("holder_AB", "holder_C", "operator", "stabilizer"),
            role_inputs={"holder_AB": "AB", "holder_C": "C"},
            role_skills={
                "holder_AB": "assemble", "holder_C": "assemble",
                "operator": "assemble", "stabilizer": "assemble",
            },
            duration=5, productive_weight=1.5,
        ),
    ),
    final_token="FINAL",
)

PARALLEL_BRANCH = RecipeSpec(
    id="parallel_branch",
    raw_tokens=("A", "B", "C", "D"),
    operations=(
        OperationSpec(
            id="align_ab", kind="align", inputs=("A", "B"), output="AB", kappa=2,
            roles=("holder_A", "holder_B"),
            role_inputs={"holder_A": "A", "holder_B": "B"},
            duration=3, productive_weight=1.0,
        ),
        OperationSpec(
            id="align_cd", kind="align", inputs=("C", "D"), output="CD", kappa=2,
            roles=("holder_C", "holder_D"),
            role_inputs={"holder_C": "C", "holder_D": "D"},
            duration=3, productive_weight=1.0,
        ),
        OperationSpec(
            id="join_branches", kind="assemble", inputs=("AB", "CD"), output="FINAL",
            predecessors=("align_ab", "align_cd"), kappa=4,
            roles=("holder_AB", "holder_CD", "operator", "stabilizer"),
            role_inputs={"holder_AB": "AB", "holder_CD": "CD"},
            duration=5, productive_weight=1.8,
        ),
    ),
    final_token="FINAL",
)





ALTERNATIVE_ROUTE = RecipeSpec(
    id="alternative_route",
    raw_tokens=("A", "B"),
    operations=(
        # Two alternative transformations consume the same input tokens and
        # produce the same intermediate.  Token exclusivity ensures only one
        # can execute; the policy chooses between a smaller/slower team and a
        # larger/faster team without a special recipe-language primitive.
        OperationSpec(
            id="join_precise", kind="align", inputs=("A", "B"), output="AB", kappa=2,
            roles=("holder_A", "holder_B"), role_inputs={"holder_A": "A", "holder_B": "B"},
            duration=5, productive_weight=1.0,
        ),
        OperationSpec(
            id="join_fast", kind="assemble", inputs=("A", "B"), output="AB", kappa=3,
            roles=("holder_A", "holder_B", "operator"),
            role_inputs={"holder_A": "A", "holder_B": "B"},
            duration=2, productive_weight=1.15,
        ),
        OperationSpec(
            id="finish", kind="inspect", inputs=("AB",), output="FINAL",
            predecessor_any=(("join_precise", "join_fast"),), kappa=1,
            roles=("inspector",), role_inputs={"inspector": "AB"}, duration=2,
        ),
    ),
    final_token="FINAL",
)

# The only shipped recipe with kappa_max > kappa. It exists so the
# "extra members are admissible where the recipe declares support roles"
# path is actually exercised by a default preset: without it, every shipped
# recipe had kappa_max == kappa, that code path had no realistic coverage
# (a latent bug hid there once already), and the interactive demo taught a
# "k_min=2, |C|=4" example that no shipped recipe could ever display.
# Extra members do NOT shorten the operation; they occupy declared support
# roles (a second stabilizer, an inspector).
SUPPORTED_INSERT = RecipeSpec(
    id="supported_insert",
    raw_tokens=("A", "B"),
    operations=(
        OperationSpec(
            id="align_ab", kind="align", inputs=("A", "B"), output="AB", kappa=2,
            roles=("holder_A", "holder_B"),
            role_inputs={"holder_A": "A", "holder_B": "B"},
            duration=3,
        ),
        OperationSpec(
            id="supported_insert", kind="assemble", inputs=("AB",), output="FINAL",
            predecessors=("align_ab",),
            kappa=2, kappa_max=4,
            roles=("holder", "inserter", "stabilizer", "inspector"),
            role_inputs={"holder": "AB"},
            duration=4, productive_weight=1.3,
        ),
    ),
    final_token="FINAL",
)

PAIR_ASSEMBLY = RecipeSpec(
    id="pair_assembly",
    raw_tokens=("A", "B"),
    operations=(
        OperationSpec(
            id="pair_join", kind="assemble", inputs=("A", "B"), output="FINAL", kappa=2,
            roles=("holder_A", "holder_B"),
            role_inputs={"holder_A": "A", "holder_B": "B"},
            role_skills={"holder_A": "assemble", "holder_B": "assemble"},
            duration=4, productive_weight=1.2,
        ),
    ),
    final_token="FINAL",
)

SOLO_INSPECTION = RecipeSpec(
    id="solo_inspection",
    raw_tokens=("A",),
    operations=(
        OperationSpec(
            id="inspect", kind="inspect", inputs=("A",), output="FINAL", kappa=1,
            roles=("inspector",), role_inputs={"inspector": "A"},
            role_skills={"inspector": "inspect"}, duration=2, productive_weight=1.0,
        ),
    ),
    final_token="FINAL",
)

# A minimally confounded concurrency control: two independent single-robot
# transformations may overlap, followed by one pairwise join.  The older
# four-input/four-robot recipe remains available as a coalition-heavy stress
# test, but is not used to define the official concurrency family.
PARALLEL_BRANCH_LIGHT = RecipeSpec(
    id="parallel_branch_light",
    raw_tokens=("A", "B"),
    operations=(
        OperationSpec(
            id="prepare_a", kind="prepare", inputs=("A",), output="PA",
            kappa=1, roles=("operator_a",), role_inputs={"operator_a": "A"},
            duration=3, productive_weight=1.0,
        ),
        OperationSpec(
            id="prepare_b", kind="prepare", inputs=("B",), output="PB",
            kappa=1, roles=("operator_b",), role_inputs={"operator_b": "B"},
            duration=3, productive_weight=1.0,
        ),
        OperationSpec(
            id="join_branches", kind="assemble", inputs=("PA", "PB"),
            output="FINAL", predecessors=("prepare_a", "prepare_b"),
            kappa=2, roles=("holder_PA", "holder_PB"),
            role_inputs={"holder_PA": "PA", "holder_PB": "PB"},
            duration=4, productive_weight=1.5,
        ),
    ),
    final_token="FINAL",
)


FLOW_SERIAL_3 = RecipeSpec(
    id="flow_serial_3",
    raw_tokens=("A",),
    operations=(
        OperationSpec(id="prepare", kind="prepare", inputs=("A",), output="P1", kappa=1,
                      roles=("operator",), role_inputs={"operator": "A"}, duration=2),
        OperationSpec(id="process", kind="process", inputs=("P1",), output="P2",
                      predecessors=("prepare",), kappa=1, roles=("operator",),
                      role_inputs={"operator": "P1"}, duration=3),
        OperationSpec(id="inspect", kind="inspect", inputs=("P2",), output="FINAL",
                      predecessors=("process",), kappa=1, roles=("inspector",),
                      role_inputs={"inspector": "P2"}, duration=2),
    ),
    final_token="FINAL",
)


FLOW_SERIAL_5 = RecipeSpec(
    id="flow_serial_5",
    raw_tokens=("A",),
    operations=(
        OperationSpec(id="prepare", kind="prepare", inputs=("A",), output="P1", kappa=1,
                      roles=("operator",), role_inputs={"operator": "A"}, duration=2),
        OperationSpec(id="process_1", kind="process", inputs=("P1",), output="P2",
                      predecessors=("prepare",), kappa=1, roles=("operator",),
                      role_inputs={"operator": "P1"}, duration=3),
        OperationSpec(id="process_2", kind="process", inputs=("P2",), output="P3",
                      predecessors=("process_1",), kappa=1, roles=("operator",),
                      role_inputs={"operator": "P2"}, duration=3),
        OperationSpec(id="finish", kind="finish", inputs=("P3",), output="P4",
                      predecessors=("process_2",), kappa=1, roles=("operator",),
                      role_inputs={"operator": "P3"}, duration=2),
        OperationSpec(id="inspect", kind="inspect", inputs=("P4",), output="FINAL",
                      predecessors=("finish",), kappa=1, roles=("inspector",),
                      role_inputs={"inspector": "P4"}, duration=2),
    ),
    final_token="FINAL",
)


def default_recipe_library() -> RecipeLibrary:
    return RecipeLibrary((
        STANDARD_AB, STANDARD_ABC, PARALLEL_BRANCH, PARALLEL_BRANCH_LIGHT, SOLO_INSPECTION,
        FLOW_SERIAL_3, FLOW_SERIAL_5, PAIR_ASSEMBLY, ALTERNATIVE_ROUTE,
        SUPPORTED_INSERT,
    ))


def recipe_required_raw_types(recipe_ids: Sequence[str], library: RecipeLibrary) -> Tuple[str, ...]:
    out = []
    for rid in recipe_ids:
        for token in library.get(rid).raw_tokens:
            if token not in out:
                out.append(token)
    return tuple(out)


# ---------------------------------------------------------------------------
# External JSON I/O.  This keeps the canonical engine generic while allowing
# experiment manifests to version recipes independently of Python source.
# ---------------------------------------------------------------------------

def operation_from_dict(data: Mapping) -> OperationSpec:
    role_tools = {str(k): tuple(v) for k, v in dict(data.get("role_tools", {})).items()}
    return OperationSpec(
        id=str(data["id"]),
        kind=str(data.get("kind", data["id"])),
        inputs=tuple(data.get("inputs", ())),
        output=str(data["output"]),
        predecessors=tuple(data.get("predecessors", ())),
        predecessor_any=tuple(tuple(g) for g in data.get("predecessor_any", ())),
        predecessor_any_inclusive=tuple(tuple(g) for g in data.get("predecessor_any_inclusive", ())),
        kappa=int(data.get("kappa", 1)),
        # kappa_max must round-trip: without it, a saved operation that
        # declares support roles is rebuilt with kappa_max defaulting back to
        # kappa, and then REJECTED at load because it has more role names than
        # the (shrunken) admissible team size. Saved recipes became unopenable.
        kappa_max=(None if data.get("kappa_max") is None else int(data["kappa_max"])),
        roles=tuple(data.get("roles", ())),
        role_inputs=dict(data.get("role_inputs", {})),
        role_skills=dict(data.get("role_skills", {})),
        role_tools=role_tools,
        resources=tuple(data.get("resources", ())),
        duration=int(data.get("duration", 1)),
        productive_weight=float(data.get("productive_weight", 1.0)),
    )


def recipe_from_dict(data: Mapping) -> RecipeSpec:
    validate_recipe_document(data)
    recipe = RecipeSpec(
        id=str(data["id"]),
        raw_tokens=tuple(data.get("raw_tokens", ())),
        operations=tuple(operation_from_dict(x) for x in data.get("operations", ())),
        final_token=str(data["final_token"]),
        product_value=float(data.get("product_value", 1.0)),
    )
    recipe.validate()
    return recipe


def recipe_to_dict(recipe: RecipeSpec) -> dict:
    """Return a JSON-safe canonical recipe representation."""
    return {
        "schema": RECIPE_SCHEMA_VERSION,
        "id": recipe.id,
        "raw_tokens": list(recipe.raw_tokens),
        "operations": [
            {
                "id": op.id,
                "kind": op.kind,
                "inputs": list(op.inputs),
                "output": op.output,
                "predecessors": list(op.predecessors),
                "predecessor_any": [list(g) for g in op.predecessor_any],
                "predecessor_any_inclusive": [list(g) for g in op.predecessor_any_inclusive],
                "kappa": op.kappa,
                "kappa_max": op.kappa_max,
                "roles": list(op.roles),
                "role_inputs": dict(op.role_inputs),
                "role_skills": dict(op.role_skills),
                "role_tools": {k: list(v) for k, v in op.role_tools.items()},
                "resources": list(op.resources),
                "duration": op.duration,
                "productive_weight": op.productive_weight,
            }
            for op in recipe.operations
        ],
        "final_token": recipe.final_token,
        "product_value": recipe.product_value,
    }


def load_recipe_json(path) -> RecipeSpec:
    return recipe_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_recipe_json(recipe: RecipeSpec, path) -> None:
    Path(path).write_text(json.dumps(recipe_to_dict(recipe), indent=2), encoding="utf-8")


def load_recipe_library(paths: Iterable) -> RecipeLibrary:
    return RecipeLibrary(load_recipe_json(path) for path in paths)
