"""Canonical AssemblyGrid discrete-event reference environment.

This module implements the revised benchmark semantics:

* direct interaction follows an explicit local topology (canonical: 3x3
  Moore neighborhood); every direct-manipulation coalition is pairwise local;
* an operation has an explicit recipe-defined arity kappa in 1..4; a 2x2
  footprint is never an exclusive resource by itself;
* robots choose concrete local pick options and concrete operation/team
  proposals; the environment does not silently choose a team after a generic
  "participate" action;
* recipes are data-driven DAGs rather than a hard-coded A+B state machine;
* bilateral handoffs, picks, transformations and delivery have persistent
  durations and lightweight kinematic motion plans;
* heterogeneous robot reach, payload, tool, speed and clearance may affect
  feasibility through optional/privileged capabilities;
* parallelism is state dependent and solved exactly on small diagnostic
  instances (deterministic approximation on larger instances); canonical
  productive concurrency is unweighted, while weighted productivity remains
  an optional diagnostic;
* extension configurations may vary architecture, recipes, failures, skills and scale
  without changing canonical v1 semantics.

The motion layer here is an abstract nominal geometric/kinematic proxy, not rigid-body
URDF/physics validation.  See motion.py and IMPLEMENTATION_CONFORMANCE.md.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from motion import MotionInfeasible, MotionPlan, base_point, common_target, plan_motion, plans_conflict, role_targets
from profiles import get_geometry_profile, profile_values
from recipe import OperationSpec, RecipeLibrary, RecipeSpec, default_recipe_library, recipe_required_raw_types
from skills import DEFAULT_SKILLS, initial_skill_levels, update_proficiency
from training_reward import TrainingRewardConfig

Coord = Tuple[int, int]
DIRECTIONS: List[Coord] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Parameterized local action slots.  The numerical wrapper remains fixed-size,
# but every pick/operation slot is bound to an explicit local option included in
# the observation returned for the same decision state.
# Fixed-size canonical MARL slots.  Capacity is deliberately generous for
# v1 and, critically, overflow is an error rather than silent truncation.
MAX_PICK_OPTIONS = 32
MAX_OPERATION_OPTIONS = 64
ACTION_IDLE = 0
ACTION_DELIVER = 1
ACTION_PICK_BASE = 2
ACTION_HANDOFF_BASE = ACTION_PICK_BASE + MAX_PICK_OPTIONS
ACTION_RECEIVE_BASE = ACTION_HANDOFF_BASE + 8
ACTION_OPERATION_BASE = ACTION_RECEIVE_BASE + 8
NUM_ACTIONS = ACTION_OPERATION_BASE + MAX_OPERATION_OPTIONS

# Compatibility aliases used by older external scripts.  They point to the
# first parameterized slot; canonical policies should use the *_BASE constants.
ACTION_PICK = ACTION_PICK_BASE
ACTION_PARTICIPATE = ACTION_OPERATION_BASE


@dataclass(frozen=True)
class URModel:
    name: str = "generic_ur"
    reach_radius: float = 1.5
    payload: float = 10.0
    speed: float = 1.0
    clearance: float = 0.10
    tools: Tuple[str, ...] = ("gripper",)
    joint_speed: float = 1.0


@dataclass
class Robot:
    row: int
    col: int
    model: URModel = field(default_factory=URModel)
    holding: Optional[Tuple[int, str]] = None  # (product_id, token_id)
    busy_until: int = 0
    failed_until: int = 0
    role: str = ""
    op: str = ""
    q: Tuple[float, ...] = (0.0,) * 6
    ee_pos: Optional[Tuple[float, float]] = None
    skill_levels: Dict[str, float] = field(default_factory=lambda: initial_skill_levels(DEFAULT_SKILLS, 1.0))
    skill_uses: Dict[str, int] = field(default_factory=dict)
    proposal_id: Optional[str] = None
    proposal_until: int = 0
    proposal_lock_until: int = 0
    incoming_transfer: Optional[Tuple[int, str]] = None
    role_switches: int = 0
    team_uses: int = 0

    @property
    def coord(self) -> Coord:
        return (self.row, self.col)


@dataclass(frozen=True)
class ProductAchievementState:
    """Algorithm-independent recipe/completion progress for one product.

    Physical/material state (holders, positions, reservations) is deliberately
    excluded. XOR choices are represented by the selected member of each
    exclusive alternative group once that choice has been resolved.
    """
    completed_operations: frozenset[str]
    resolved_xor_choices: Tuple[Tuple[Tuple[str, ...], str], ...]
    delivered: bool


@dataclass
class Product:
    id: int
    recipe_id: str
    spawn_time: int
    due_time: int
    source_rows: Dict[str, int]
    mass: float = 1.0
    value: float = 1.0
    raw_picked: Set[str] = field(default_factory=set)
    reserved_raw: Dict[str, Coord] = field(default_factory=dict)
    token_positions: Dict[str, Coord] = field(default_factory=dict)
    completed_ops: Set[str] = field(default_factory=set)
    in_progress_ops: Set[str] = field(default_factory=set)
    delivered: bool = False
    complete_time: Optional[int] = None


@dataclass(frozen=True)
class PickOption:
    id: str
    product_id: int
    recipe_id: str
    token: str
    shelf_row: int
    stock_fraction: float


@dataclass(frozen=True)
class OperationCandidate:
    id: str
    product_id: int
    operation_id: str
    kind: str
    coalition: Tuple[Coord, ...]
    roles: Tuple[Tuple[Coord, str], ...]
    workspace: frozenset
    resources: frozenset
    motion_plan: MotionPlan
    productive_value: float
    # Point 2 (notation): kappa_min is the arity the OPERATION declares as its
    # minimum (paper's kappa_tau / k_min); size is the REALIZED |C| this
    # particular candidate would commit. They are different quantities and
    # must never be conflated: k_min is a requirement, size is a decision.
    kappa_min: int
    size: int

    @property
    def role_map(self) -> Dict[Coord, str]:
        return dict(self.roles)


@dataclass
class TeamProposal:
    candidate_id: str
    product_id: int
    operation_id: str
    members: Tuple[Coord, ...]
    created_at: int
    expires_at: int
    votes: Set[Coord] = field(default_factory=set)


@dataclass
class Team:
    id: int
    candidate_id: str
    product_id: int
    operation_id: str
    operation: str
    members: Tuple[Coord, ...]
    roles: Dict[Coord, str]
    kappa_min: int          # declared minimum arity of the operation (k_min)
    formed_at: int
    first_proposed_at: int
    committed_until: int
    productive_value: float
    motion_plan: MotionPlan
    resources: frozenset
    status: str = "committed"


@dataclass
class ActivityReservation:
    id: str
    kind: str
    members: Tuple[Coord, ...]
    resources: frozenset
    motion_plan: MotionPlan
    until: int


@dataclass(frozen=True)
class OperationOptionView:
    candidate_id: str
    product_id: int
    recipe_id: str
    operation_id: str
    kind: str
    kappa_min: int          # operation's declared minimum arity (k_min)
    size: int               # realized |C| of this candidate coalition
    role: str
    members: Tuple[Coord, ...]
    motion_duration: int
    min_clearance: float
    proposal_support: float = 0.0


@dataclass
class Observation:
    robot_id: Coord
    t: int
    holding_kind: Optional[str]
    holding_pid: Optional[int]
    busy: bool
    failed: bool
    neighbors: Dict[Coord, dict]
    pending_pick_here: bool
    pick_options: Tuple[PickOption, ...]
    operation_options: Tuple[OperationOptionView, ...]
    action_mask: List[bool]
    proposal_id: Optional[str] = None


# Maximum pairwise-adjacent set size per admissible topology.  Coalition
# members must be pairwise adjacent, so this is exactly the largest operation
# arity the topology can realise: a clique of 4 exists only when diagonals do.
# kappa_max is therefore DERIVED from the topology, never an independent knob.
TOPOLOGY_MAX_ARITY = {"moore": 4, "von_neumann": 2, "line": 2}


@dataclass
class EnvConfig:
    M: int = 5
    N: int = 8
    cooperation_radius: int = 1
    # Admissible interaction topologies must be vertex-transitive up to
    # boundary effects, so that the maximum pairwise-adjacent set size (and
    # hence the maximum realisable operation arity) is a constant of the cell
    # rather than a function of position or seed.  Randomly sparsified
    # variants are therefore not admissible here: on a uniform grid of
    # identical arms, whether two neighbours can reach a common workpiece is
    # fixed by geometry, so a random dropout has no physical referent.
    # Stochastic degradation belongs to optional disturbance extensions.
    topology: str = "moore"  # moore | von_neumann | line
    # Geometry/motion profile identity is normative for v1.  The numeric
    # fields below remain materialized for efficient core use, but must match
    # the selected profile/declared variant when profile_enforced=True.
    geometry_profile: str = "abstract-v1"
    geometry_variant: Optional[str] = None
    profile_enforced: bool = False
    official_result: bool = False
    cell_spacing: float = 1.0
    workspace_conflict_radius: float = 0.25
    reference_reach: float = 1.5
    reference_time: float = 1.0
    physical_reference_reach_m: Optional[float] = None
    role_target_offset: float = 0.12

    robot_models: Tuple[URModel, ...] = (URModel(),)
    robot_model_assignment: str = "homogeneous"  # homogeneous | round_robin | random

    pick_duration: int = 2
    # Recipe-operation durations live in OperationSpec.  Keeping one source of
    # truth avoids silently overriding heterogeneous/alternative recipes.
    deliver_duration: int = 2
    handoff_duration: int = 1
    retreat_fraction: float = 0.25

    motion_planning: bool = True
    trajectory_conflicts: bool = True
    require_bilateral_handoff: bool = True

    formation_timeout: int = 3
    commitment_horizon: int = 1
    spawn_interval: int = 4
    max_wip: int = 6
    shelf_cap: int = 6
    shelf_replenish_interval: int = 5
    due_slack: float = 2.2
    due_slack_jitter: float = 0.5
    reach_extra_prob: float = 0.30

    recipe_ids: Tuple[str, ...] = ("standard_ab",)
    product_mass_min: float = 0.5
    product_mass_max: float = 2.0
    resource_capacities: Dict[str, int] = field(default_factory=dict)

    episode_mode: str = "horizon"
    horizon_T: int = 400
    batch_Z: int = 30

    skill_mode: str = "disabled"  # disabled | required | learning
    skill_threshold: float = 0.5
    skill_initial_level: float = 0.75
    skill_learning_rate: float = 0.05

    # Robustness / disturbance factors.  Zero preserves deterministic AG-Core.
    robot_failure_prob: float = 0.0
    repair_duration: int = 8
    operation_failure_prob: float = 0.0
    handoff_failure_prob: float = 0.0
    duration_noise: float = 0.0
    stockout_prob: float = 0.0

    parallelism_exact_candidate_limit: int = 64
    parallelism_exact_product_limit: int = 6
    parallelism_exact_combination_limit: int = 100_000

    # Optional/non-canonical extension switches. They preserve software
    # capacity without changing the canonical v1 interface.
    allow_inclusive_or_extension: bool = False
    proposal_support_visible: bool = True

    held_out_axes: Tuple[str, ...] = ()
    benchmark_track: str = "AG-Core"
    # Three seed roles are separated in the v1 implementation. ``seed`` is a
    # backwards-compatible fallback used only when the specific seed is None.
    generation_seed: Optional[int] = None
    execution_seed: Optional[int] = None
    seed: Optional[int] = None

    @property
    def max_coalition_size(self) -> int:
        """Derived manipulation-coalition cap for the active topology.

        This is intentionally not a configurable dataclass field: canonical
        theory fixes kappa_max = omega(G_manip).  The property is retained as
        a compatibility/readability alias for algorithms that need the derived
        value.
        """
        return TOPOLOGY_MAX_ARITY[self.topology]

    def validate(self) -> None:
        if self.cooperation_radius != 1:
            raise ValueError("canonical direct interaction radius is one grid step (3x3 Moore when topology='moore')")
        if self.topology not in TOPOLOGY_MAX_ARITY:
            raise ValueError(
                "unsupported topology; admissible topologies are %s (an interaction "
                "topology must be vertex-transitive up to boundary effects)"
                % sorted(TOPOLOGY_MAX_ARITY)
            )
        if self.robot_model_assignment not in {"homogeneous", "round_robin", "random"}:
            raise ValueError("robot_model_assignment must be homogeneous, round_robin, or random")
        if not self.robot_models:
            raise ValueError("at least one robot model is required")
        # Canonical benchmark presets explicitly enforce profile ownership.
        # Research extensions may opt out, but an official v1 result may never
        # bypass ownership even if profile_enforced was set false.
        expected = profile_values(self.geometry_profile, self.geometry_variant)
        if self.profile_enforced or self.official_result:
            exp_model = URModel(**expected["robot_model"])
            mismatches = []
            for key in (
                "cell_spacing", "workspace_conflict_radius", "reference_reach",
                "reference_time", "physical_reference_reach_m", "role_target_offset",
                "retreat_fraction", "motion_planning", "trajectory_conflicts",
            ):
                if getattr(self, key) != expected[key]:
                    mismatches.append(f"{key}={getattr(self,key)!r} (profile requires {expected[key]!r})")
            if self.robot_model_assignment != "homogeneous" or tuple(self.robot_models) != (exp_model,):
                mismatches.append("robot_models/assignment do not match the selected profile")
            if mismatches:
                raise ValueError("profile-owned fields were overridden outside a declared variant: " + "; ".join(mismatches))
        if self.official_result and not get_geometry_profile(self.geometry_profile).canonical:
            raise ValueError(f"non-canonical geometry profile {self.geometry_profile!r} cannot be used for official v1 results")
        if self.episode_mode not in {"horizon", "batch"}:
            raise ValueError("episode_mode must be horizon or batch")
        if self.skill_mode not in {"disabled", "required", "learning"}:
            raise ValueError("skill_mode must be disabled, required, or learning")
        if not self.recipe_ids:
            raise ValueError("recipe_ids may not be empty")
        if self.M < 1 or self.N < 1:
            raise ValueError("grid dimensions M and N must be positive")
        if self.horizon_T < 1 or self.batch_Z < 1:
            raise ValueError("horizon_T and batch_Z must be positive")
        if self.reference_reach <= 0 or self.reference_time <= 0:
            raise ValueError("reference reach and time must be positive")
        if self.physical_reference_reach_m is not None and self.physical_reference_reach_m <= 0:
            raise ValueError("physical_reference_reach_m must be positive when declared")
        if self.role_target_offset < 0:
            raise ValueError("role_target_offset must be non-negative")

        if self.product_mass_min <= 0 or self.product_mass_max < self.product_mass_min:
            raise ValueError("invalid product mass range")
        for name in ("robot_failure_prob", "operation_failure_prob", "handoff_failure_prob", "stockout_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.duration_noise < 0:
            raise ValueError("duration_noise must be non-negative")
        if self.parallelism_exact_candidate_limit < 1 or self.parallelism_exact_product_limit < 1 or self.parallelism_exact_combination_limit < 1:
            raise ValueError("parallelism exact-search limits must be positive")


class AssemblyGridCore:
    def __init__(self, cfg: EnvConfig, priority_fn=None, recipe_library: Optional[RecipeLibrary] = None,
                 training_reward: Optional[TrainingRewardConfig] = None):
        cfg.validate()
        self.cfg = cfg
        # Reward is an algorithm-layer convenience, deliberately separate from
        # EnvConfig so changing a training signal does not change benchmark
        # instance identity.  The default is non-normative; a training
        # algorithm must explicitly report the reward it uses.
        self.training_reward = training_reward or TrainingRewardConfig()
        self.training_reward.validate()
        self.recipe_library = recipe_library or default_recipe_library()
        # Topology and coalition arity are not independent: members of a
        # coalition must be pairwise adjacent, so a topology whose largest
        # clique is 2 can never host an operation that REQUIRES 3 or 4.
        #
        # Only the MINIMUM is a feasibility question. k_max is an upper option,
        # not an obligation, so the effective ceiling is simply clamped:
        #
        #     k_min <= |C| <= min(k_max, topology capacity)
        #
        # Rejecting a recipe because its OPTIONAL maximum exceeds the topology
        # capacity was wrong: an operation needing at least 2 robots and
        # allowing up to 4 runs perfectly well with 2 on a clique-2 topology,
        # but was refused outright at construction.
        arity_cap = cfg.max_coalition_size
        for rid in cfg.recipe_ids:
            spec = self.recipe_library.get(rid)
            if not cfg.allow_inclusive_or_extension and any(op.predecessor_any_inclusive for op in spec.operations):
                raise ValueError(
                    f"recipe {rid!r} uses inclusive-OR, which is outside canonical AssemblyGrid v1; "
                    "set allow_inclusive_or_extension=True only for an explicitly named extension"
                )
            for op in spec.operations:
                if op.kappa > arity_cap:
                    raise ValueError(
                        "recipe %r operation %r REQUIRES at least %d robots "
                        "(k_min), but topology %r admits coalitions of at most "
                        "%d pairwise-adjacent robots, so the operation can "
                        "never be performed"
                        % (rid, op.id, op.kappa, cfg.topology, arity_cap)
                    )
        self.priority_fn = priority_fn or (lambda p, t, core: p.spawn_time)
        self.generation_rng = random.Random(cfg.generation_seed if cfg.generation_seed is not None else cfg.seed)
        self.execution_rng = random.Random(cfg.execution_seed if cfg.execution_seed is not None else cfg.seed)
        # Backwards-compatible alias for external code; environment internals
        # use the specific streams below.
        self.rng = self.execution_rng
        self.reset()

    # ------------------------------------------------------------------
    # reset / topology / physical helpers
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            # Wrapper-style reset(seed=...) intentionally reproduces both the
            # generated realization and its stochastic execution.
            self.generation_rng = random.Random(seed)
            self.execution_rng = random.Random(seed)
            self.rng = self.execution_rng
        c = self.cfg
        self.t = 0
        self.products: Dict[int, Product] = {}
        self.next_product_id = 0
        self.scheduled: List[dict] = []
        self.teams: Dict[int, Team] = {}
        # product id -> operation ids closed off by an exclusive route choice
        self._route_locked: Dict[int, Set[str]] = {}
        self.proposals: Dict[str, TeamProposal] = {}
        self.next_team_id = 0
        self.active_reservations: Dict[str, ActivityReservation] = {}

        self.robots: List[List[Robot]] = []
        for i in range(c.M):
            row = []
            for j in range(c.N):
                if c.robot_model_assignment == "homogeneous":
                    model = c.robot_models[0]
                elif c.robot_model_assignment == "round_robin":
                    model = c.robot_models[(i * c.N + j) % len(c.robot_models)]
                else:
                    model = self.generation_rng.choice(c.robot_models)
                skills = initial_skill_levels(DEFAULT_SKILLS, 1.0 if c.skill_mode == "disabled" else c.skill_initial_level)
                r = Robot(i, j, model=model, skill_levels=skills)
                r.ee_pos = base_point((i, j), c.cell_spacing)
                row.append(r)
            self.robots.append(row)

        self._build_adjacency()
        self._build_shelves()

        # Counters and running statistics.
        self.spawned_count = self.delivered_count = self.handoff_count = 0
        self.delivered_value = 0.0
        self.invalid_attempts = self.unmasked_invalid_attempts = 0
        # Point 17: distinct failure categories, counted separately, because
        # "the action was illegal here" and "the team never assembled" and
        # "the motion was infeasible" are different phenomena with different
        # diagnoses. Each already costs the agent the tick it was attempted
        # in (a rejected action is coerced to idle), so probing is not free;
        # these counters make the wasted coordination effort measurable.
        self.reject_locally_invalid = 0      # action not permitted by own mask
        self.reject_unmatched_proposal = 0   # locally legal, teammates never agreed
        # A locally plausible (product, operation, coalition) that has no admissible
        # task-level motion plan (reach margin < 0 or unresolvable role-target
        # clearance) and is therefore never exposed as a candidate. Counted once
        # per such triple per decision epoch; surfaced as geometry_rejection_count.
        self.reject_motion_infeasible = 0
        self._geo_reject_counted_tick = -1
        self.reject_execution_failure = 0    # committed, then failed during execution
        self.sum_flow_time = self.sum_tardiness = 0.0
        self.delivered_flow_times: List[int] = []
        self.sum_wip = 0.0
        self.sum_k_feasible_star = self.sum_p_productive_star = 0.0
        self.sum_k_realized = self.sum_p_realized = 0.0
        self.sum_k_util = self.sum_p_util = 0.0
        self.sum_k_util_approx = self.sum_p_util_approx = 0.0
        self.k_opportunity_samples_approx = self.p_opportunity_samples_approx = 0
        self.parallelism_samples = self.parallelism_nonzero_samples = 0
        self.k_opportunity_samples = self.p_opportunity_samples = 0
        self.capacity_exact_samples = self.capacity_approx_samples = 0
        self.trajectory_conflict_count = self.team_formation_failures = self.team_proposal_churn = 0
        self.geometry_block_count = self.resource_block_count = self.robot_block_count = 0
        self.team_formation_latency_sum = self.team_formation_count = 0
        self.team_size_sum = 0
        self.team_abort_count = 0
        self.ready_task_count_sum = self.ready_task_with_team_sum = 0
        self.sum_robot_occupancy = self.sum_k_coop_realized = 0.0
        self.motion_path_length = self.motion_duration_total = self.motion_planning_seconds = 0.0
        # Geometry delay excludes declared process duration: it is the
        # approach/retreat overhead introduced by the normalized geometry.
        self.geometry_delay_ticks = 0.0
        self.min_motion_clearance = float("inf")
        self.energy_proxy = 0.0
        self.safety_violations = 0
        self.operation_failures = self.handoff_failures = self.robot_failures = 0
        self.deadlock_ticks = 0
        self.progress_events = 0
        self.last_step_energy = 0.0
        self.last_step_safety = 0
        self.last_pulses: List[dict] = []

        self._candidate_cache: Optional[Tuple[OperationCandidate, ...]] = None
        self._pick_cache: Dict[Coord, Tuple[PickOption, ...]] = {}

        # A product exists at t=0, avoiding an empty first decision state.
        self._spawn(force=True)
        return self._all_observations()

    def _robot(self, rc: Coord) -> Robot:
        return self.robots[rc[0]][rc[1]]

    def _robot_map(self) -> Dict[Coord, Robot]:
        return {(i, j): self.robots[i][j] for i in range(self.cfg.M) for j in range(self.cfg.N)}

    def _invalidate(self) -> None:
        self._candidate_cache = None
        self._pick_cache = {}

    def _in_bounds(self, rc: Coord) -> bool:
        return 0 <= rc[0] < self.cfg.M and 0 <= rc[1] < self.cfg.N

    def _build_adjacency(self) -> None:
        c = self.cfg
        adj: Dict[Coord, Set[Coord]] = {(i, j): set() for i in range(c.M) for j in range(c.N)}
        if c.topology == "line":
            # Serpentine chain preserves robot count and physical grid geometry.
            chain: List[Coord] = []
            for i in range(c.M):
                cols = range(c.N) if i % 2 == 0 else range(c.N - 1, -1, -1)
                chain.extend((i, j) for j in cols)
            for a, b in zip(chain, chain[1:]):
                adj[a].add(b); adj[b].add(a)
        else:
            dirs = DIRECTIONS if c.topology == "moore" else [(-1, 0), (0, 1), (1, 0), (0, -1)]
            all_edges: List[Tuple[Coord, Coord]] = []
            for i in range(c.M):
                for j in range(c.N):
                    a = (i, j)
                    for di, dj in dirs:
                        b = (i + di, j + dj)
                        if self._in_bounds(b) and a < b:
                            all_edges.append((a, b))
            for a, b in all_edges:
                adj[a].add(b); adj[b].add(a)
        self.adjacency = {k: frozenset(v) for k, v in adj.items()}

    def neighbors(self, rc: Coord) -> Tuple[Coord, ...]:
        """Compatibility alias for the canonical observation neighborhood."""
        return self.obs_neighbors(rc)

    def obs_neighbors(self, rc: Coord) -> Tuple[Coord, ...]:
        """G_obs: agents whose task-level state is locally observable."""
        return tuple(sorted(self.adjacency.get(rc, ())))

    def handoff_neighbors(self, rc: Coord) -> Tuple[Coord, ...]:
        """G_handoff: agents eligible for bilateral material transfer."""
        return tuple(sorted(self.adjacency.get(rc, ())))

    def manip_neighbors(self, rc: Coord) -> Tuple[Coord, ...]:
        """G_manip pairwise projection used for direct joint manipulation."""
        return tuple(sorted(self.adjacency.get(rc, ())))

    def _manip_local(self, a: Coord, b: Coord) -> bool:
        return a == b or b in self.manip_neighbors(a)

    def _local(self, a: Coord, b: Coord) -> bool:
        """Deprecated compatibility alias for manipulation locality."""
        return self._manip_local(a, b)

    def _coalition_local(self, coalition: Sequence[Coord]) -> bool:
        return all(self._manip_local(a, b) for a, b in combinations(coalition, 2))

    def _free(self, r: Robot) -> bool:
        return r.busy_until <= self.t and r.failed_until <= self.t and r.incoming_transfer is None

    def _failed(self, r: Robot) -> bool:
        return r.failed_until > self.t

    def _skill(self, r: Robot, skill: str) -> bool:
        return self.cfg.skill_mode == "disabled" or r.skill_levels.get(skill, 0.0) >= self.cfg.skill_threshold

    def _can_reach_point(self, rc: Coord, point: Tuple[float, float]) -> bool:
        base = base_point(rc, self.cfg.cell_spacing)
        return math.hypot(point[0] - base[0], point[1] - base[1]) <= self._robot(rc).model.reach_radius + 1e-9

    def _handoff_target(self, a: Coord, b: Coord) -> Tuple[float, float]:
        pa = base_point(a, self.cfg.cell_spacing); pb = base_point(b, self.cfg.cell_spacing)
        return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)

    def _handoff_static_feasible(self, a: Coord, b: Coord) -> bool:
        if b not in self.handoff_neighbors(a):
            return False
        target = self._handoff_target(a, b)
        return self._can_reach_point(a, target) and self._can_reach_point(b, target)

    def _shelf_target(self, shelf_row: int) -> Tuple[float, float]:
        return (-0.5 * self.cfg.cell_spacing, shelf_row * self.cfg.cell_spacing)

    def _output_target(self, row: int) -> Tuple[float, float]:
        return (self.cfg.N * self.cfg.cell_spacing, row * self.cfg.cell_spacing)

    def _build_shelves(self) -> None:
        raw_types = recipe_required_raw_types(self.cfg.recipe_ids, self.recipe_library)
        self.raw_types = raw_types
        self.shelves: Dict[int, dict] = {
            i: {"inventory": {}, "cap": self.cfg.shelf_cap, "outage_until": 0} for i in range(self.cfg.M)
        }
        # One primary stock type per row, plus extra types when M < number of
        # raw types.  This preserves heterogeneous entry interfaces while
        # guaranteeing every configured recipe can spawn.
        for i in range(self.cfg.M):
            token = raw_types[i % len(raw_types)]
            self.shelves[i]["inventory"][token] = self.cfg.shelf_cap
        for k, token in enumerate(raw_types):
            row = k % self.cfg.M
            self.shelves[row]["inventory"].setdefault(token, self.cfg.shelf_cap)

        self.shelf_reach: Dict[int, Set[int]] = {}
        for srow in range(self.cfg.M):
            target = self._shelf_target(srow)
            reachable: Set[int] = set()
            for rrow in range(self.cfg.M):
                rc = (rrow, 0)
                if not self._can_reach_point(rc, target):
                    continue
                if rrow == srow or self.generation_rng.random() < self.cfg.reach_extra_prob:
                    reachable.add(rrow)
            if srow not in reachable and self._can_reach_point((srow, 0), target):
                reachable.add(srow)
            self.shelf_reach[srow] = reachable

    # ------------------------------------------------------------------
    # recipe / product helpers
    # ------------------------------------------------------------------
    def recipe(self, p: Product) -> RecipeSpec:
        return self.recipe_library.get(p.recipe_id)

    def operation_spec(self, p: Product, op_id: str) -> OperationSpec:
        return self.recipe(p).op_map[op_id]

    def _operation_duration(self, op: OperationSpec, team_size: Optional[int] = None) -> int:
        """Duration of operation op. INDEPENDENT of the realized team size.

        Design decision (reverses an earlier idealized-linear-speedup model):
        adding a robot to an operation does NOT make it finish faster. A
        generic "extra arm" is not a mechanism for accelerating a physical
        manipulation task; real speedup would require the task to decompose
        into genuinely parallel sub-motions, which is rare and is not
        something this abstraction can claim in general. Modelling it as a
        universal law let a policy learn the artificial strategy of piling
        spare robots onto every operation, which dominated the intended
        cooperation/routing/scheduling trade-off.

        Extra members beyond op.kappa are therefore admissible only where the
        recipe explicitly declares kappa_max > kappa AND gives them roles;
        they contribute a declared physical function, never a speed multiplier.
        team_size is accepted (and ignored) so callers need not special-case.
        """
        return op.duration

    def token_owner(self, pid: int, token: str) -> Optional[Coord]:
        """Logical OWNERSHIP: the single robot accountable for this token.

        Point 6: ownership and physical support are different relations.
        Exactly one robot owns a token at a time (this is what conservation
        and the one-item-per-gripper rule are about), but several robots may
        physically support/contact the same workpiece while a cooperative
        operation runs, and during a bilateral handoff both sender and
        receiver are briefly in contact with it. Use token_supporters() for
        the physical relation; never infer contact from ownership.
        """
        return self._token_holder(pid, token)

    def token_supporters(self, pid: int, token: str) -> Tuple[Coord, ...]:
        """Physical SUPPORT set: every robot currently in contact with this
        token. Superset of {owner}, and never duplicates the material.

        Three cases:
          * idle/carried: just the owner;
          * cooperative operation: every member of the committed team
            operating on this token, since they are jointly manipulating the
            same workpiece (holding, stabilizing, aligning, inserting);
          * bilateral handoff in progress: sender and receiver together, the
            shared-support phase of sender-only -> shared -> receiver-only.
        """
        owner = self._token_holder(pid, token)
        if owner is None:
            return ()
        supporters = {owner}
        for team in self.teams.values():
            if team.status != "committed" or team.committed_until <= self.t:
                continue
            if team.product_id != pid:
                continue
            op = self.operation_spec(self.products[pid], team.operation_id)
            if token in op.inputs and owner in team.members:
                supporters.update(team.members)
        # shared-support phase of a bilateral handoff
        r_owner = self._robot(owner)
        if r_owner.op == "handoff_send":
            for nb in self.handoff_neighbors(owner):
                q = self._robot(nb)
                if q.op == "handoff_receive" and q.incoming_transfer == (pid, token):
                    supporters.add(nb)
        return tuple(sorted(supporters))

    def support_load(self, pid: int, token: str) -> float:
        """Mass borne per supporter under the equal-sharing abstraction.

        Stated explicitly because payload feasibility is otherwise ambiguous
        once several robots support one object: at this fidelity the load is
        distributed equally over the support set. A motion/physics layer that
        models grasp geometry should override this rather than inherit it.
        """
        supporters = self.token_supporters(pid, token)
        if not supporters:
            return 0.0
        return self.products[pid].mass / len(supporters)

    def _token_holder(self, pid: int, token: str) -> Optional[Coord]:
        for i in range(self.cfg.M):
            for j in range(self.cfg.N):
                if self.robots[i][j].holding == (pid, token):
                    return (i, j)
        return None

    def _xor_siblings(self, rec, op_id: str) -> Set[str]:
        """Other members of any alternative-route group containing op_id.

        Alternative routes are EXCLUSIVE (XOR): a product takes ONE route.
        The groups are declared on the consuming operation
        (`finish.predecessor_any = (('routeA','routeB'),)`), so an operation
        is exclusive with every other member of every group it appears in.
        """
        siblings: Set[str] = set()
        for other in rec.operations:
            # ONLY predecessor_any (exclusive/XOR) closes off alternatives.
            # predecessor_any_inclusive (OR) deliberately permits running more
            # than one route, e.g. hedging against operation failure, so its
            # members must NOT exclude each other.
            for group in (other.predecessor_any or ()):
                if op_id in group:
                    siblings.update(q for q in group if q != op_id)
        return siblings

    def _route_excluded(self, p: Product, op: OperationSpec) -> bool:
        """True if a competing alternative route has already been taken.

        Without this the simulator only asked "has at least one predecessor
        finished", i.e. OR. With the shipped recipe both branches consume the
        same inputs so exclusivity held by accident; a recipe whose branches
        consume DIFFERENT inputs would run both, duplicating work that the
        CP-SAT model, the critical-path bound and the remaining-work estimate
        all assume is never performed.
        """
        rec = self.recipe(p)
        for sib in self._xor_siblings(rec, op.id):
            if sib in p.completed_ops or sib in p.in_progress_ops:
                return True
        return False

    def _op_ready(self, p: Product, op: OperationSpec) -> bool:
        return (
            not p.delivered
            and op.id not in p.completed_ops
            and op.id not in p.in_progress_ops
            and not self._route_excluded(p, op)
            and op.id not in self._route_locked.get(p.id, ())
            and all(pred in p.completed_ops for pred in op.predecessors)
            and all(any(pred in p.completed_ops for pred in group)
                    for group in op.all_alternative_groups)
            and all(token in p.token_positions for token in op.inputs)
            and all(self._token_holder(p.id, token) is not None for token in op.inputs)
        )

    def logically_ready_operation_keys(self) -> Set[Tuple[int, str]]:
        """Recipe-ready operation keys before coalition/physical feasibility."""
        ready: Set[Tuple[int, str]] = set()
        for p in self.products.values():
            if p.delivered:
                continue
            for op in self.recipe(p).operations:
                if op.id in p.completed_ops or op.id in p.in_progress_ops:
                    continue
                if not all(pred in p.completed_ops for pred in op.predecessors):
                    continue
                if not all(any(pred in p.completed_ops for pred in group)
                           for group in op.all_alternative_groups):
                    continue
                if not all(token in p.token_positions for token in op.inputs):
                    continue
                ready.add((p.id, op.id))
        return ready

    def _remaining_nominal_work(self, p: Product) -> int:
        """Nominal work still to do for product p.

        Alternative routes are mutually exclusive, so only ONE branch of each
        OR group is ever performed. Summing every remaining operation counted
        work the product will never do, inflating the estimate whenever a
        recipe has alternatives. That estimate feeds the `spt` (shortest
        processing time) and `critical_ratio` dispatch rules directly, so the
        inflation silently biased two shipped baselines. Charge the cheapest
        still-available alternative per group instead.
        """
        rec = self.recipe(p)
        by_id = rec.op_map
        # Members of an EXCLUSIVE group are optional: only one of them runs.
        exclusive = {q for op in rec.operations for grp in (op.predecessor_any or ())
                     for q in grp if q in by_id}
        # Members of an INCLUSIVE group may all run, so they are charged
        # normally (below) rather than collapsed to their cheapest member.
        total = sum(self._operation_duration(op) for op in rec.operations
                    if op.id not in p.completed_ops and op.id not in exclusive)
        for op in rec.operations:
            for grp in (op.predecessor_any or ()):
                members = [by_id[q] for q in grp if q in by_id]
                if not members:
                    continue
                if any(m.id in p.completed_ops for m in members):
                    continue  # an exclusive route was already taken
                total += min(self._operation_duration(m) for m in members)
        return total

    def _urgency(self, p: Product) -> float:
        remaining = max(1.0, float(self._remaining_nominal_work(p) + self.cfg.deliver_duration))
        slack_ratio = (p.due_time - self.t) / remaining
        # 1 for comfortable slack, increasing smoothly up to 3 when overdue.
        return min(3.0, max(1.0, 2.0 - slack_ratio))

    #: Optional weighted-productivity diagnostic profile.  Canonical v1
    #: productive concurrency is unweighted; this named profile is retained
    #: only for explicitly requested richer diagnostics/extension studies.
    #: Any result using it must report the profile identifier.
    PRODUCTIVE_WEIGHT_PROFILE = "critical-urgency-value/v1"

    def _productive_weight(self, p: Product, op: OperationSpec) -> float:
        """Optional production-relevance weight w_t(e), profile v1:

            w = productive_weight(op)
                * (1 + downstream_critical_duration(op) / nominal_critical_path)
                * urgency(p)
                * value(p)

        It combines downstream critical work, due-date urgency and product
        value.  These are optional/privileged extension attributes, not part
        of the canonical decentralized observation and not the definition of
        productive activity.  The canonical v1 productive-concurrency metric
        is the unweighted count of achievement-advancing operations.
        """
        rec = self.recipe(p)
        critical = rec.downstream_critical_duration(op.id) / max(1, rec.nominal_critical_path)
        return max(1e-6, op.productive_weight * (1.0 + critical) * self._urgency(p) * p.value)

    @property
    def recipe_critical_path(self) -> int:
        return max((self.recipe_library.get(r).nominal_critical_path for r in self.cfg.recipe_ids), default=0) + self.cfg.pick_duration + self.cfg.deliver_duration

    @property
    def min_transport_hops(self) -> int:
        return max(0, self.cfg.N - 1)

    def flow_time_lower_bound(self, recipe_id: Optional[str] = None) -> int:
        rec = self.recipe_library.get(recipe_id or self.cfg.recipe_ids[0])
        # Conservative: recipe critical path and minimum eastward relay count
        # are not blindly added because some motion/processing can overlap.
        return max(self.cfg.pick_duration + rec.nominal_critical_path + self.cfg.deliver_duration,
                   self.min_transport_hops)

    # ------------------------------------------------------------------
    # production statistics
    # ------------------------------------------------------------------
    def wip(self) -> int:
        return sum(not p.delivered for p in self.products.values())

    def throughput(self) -> float:
        return self.delivered_count / max(1, self.t)

    def completion_rate(self) -> float:
        return self.delivered_count / max(1, self.spawned_count)

    def mean_flow_time(self) -> float:
        """CONDITIONAL mean over delivered products only.

        Point 13: this is right-censored. A policy that finishes easy jobs
        and abandons hard ones scores well here while serving worse overall,
        so this number is only interpretable next to completion_rate(),
        unfinished_count() and the residual-age statistics below. Never rank
        policies on this alone.
        """
        return self.sum_flow_time / max(1, self.delivered_count)

    def restricted_mean_flow_time(self) -> Optional[float]:
        """Mean observed time in system with unfinished products censored at t.

        This is a finite-horizon descriptive statistic, not an estimator of
        eventual completion time. Unlike the delivered-only mean it retains
        every released product and therefore cannot reward abandoning old WIP.
        """
        if not self.products:
            return None
        observed = [
            (p.complete_time if p.complete_time is not None else self.t) - p.spawn_time
            for p in self.products.values()
        ]
        return float(sum(observed) / len(observed))

    def mean_tardiness(self) -> float:
        """CONDITIONAL mean over delivered products only; see mean_flow_time
        for the censoring caveat."""
        return self.sum_tardiness / max(1, self.delivered_count)

    def flow_time_quantiles(self, qs: Sequence[float] = (0.5, 0.9)) -> Dict[str, float]:
        """Quantiles of delivered flow time. Means hide the tail that
        distinguishes a policy that is uniformly mediocre from one that is
        usually fast but occasionally starves a product."""
        samples = sorted(self.delivered_flow_times)
        out = {}
        for q in qs:
            if not samples:
                out[f"p{int(q * 100)}"] = 0.0
                continue
            k = min(len(samples) - 1, max(0, int(round(q * (len(samples) - 1)))))
            out[f"p{int(q * 100)}"] = float(samples[k])
        return out

    def unfinished_count(self) -> int:
        """Released but not delivered: the backlog the conditional means omit."""
        return sum(1 for p in self.products.values() if not p.delivered)

    def unfinished_age_stats(self) -> Dict[str, float]:
        """Age and lateness of the products still in the system. These are
        the censored observations: each one is a lower bound on a flow time
        that mean_flow_time() never counts."""
        ages = [self.t - p.spawn_time for p in self.products.values() if not p.delivered]
        overdue = [self.t - p.due_time for p in self.products.values()
                   if not p.delivered and self.t > p.due_time]
        return {
            "unfinished": float(len(ages)),
            "mean_age": float(sum(ages) / len(ages)) if ages else 0.0,
            "max_age": float(max(ages)) if ages else 0.0,
            "overdue_unfinished": float(len(overdue)),
            "mean_overdue_by": float(sum(overdue) / len(overdue)) if overdue else 0.0,
        }

    def makespan(self) -> float:
        completed = [p.complete_time for p in self.products.values() if p.complete_time is not None]
        if not completed:
            return 0.0
        first = min((p.spawn_time for p in self.products.values()), default=0)
        return float(max(completed) - first)

    def batch_makespan(self) -> Optional[float]:
        if self.cfg.episode_mode != "batch" or self.delivered_count < self.cfg.batch_Z:
            return None
        return self.makespan()

    # ------------------------------------------------------------------
    # motion / candidate feasibility
    # ------------------------------------------------------------------
    def _duration_with_noise(self, base_duration: int) -> int:
        if self.cfg.duration_noise <= 0:
            return max(1, int(base_duration))
        factor = max(0.1, 1.0 + self.execution_rng.gauss(0.0, self.cfg.duration_noise))
        return max(1, int(round(base_duration * factor)))

    def _make_plan(self, coalition: Sequence[Coord], target: Tuple[float, float], process_duration: int,
                   targets_by_coord: Optional[Mapping[Coord, Tuple[float, float]]] = None) -> MotionPlan:
        plan = plan_motion(self._robot_map(), coalition, target, self.cfg.cell_spacing,
                           self._duration_with_noise(process_duration), self.cfg.retreat_fraction,
                           targets_by_coord=targets_by_coord)
        if not self.cfg.motion_planning:
            plan = replace(plan, approach_duration=0, retreat_duration=0,
                           total_duration=max(1, self._duration_with_noise(process_duration)))
        return plan

    def _roles_for(self, p: Product, op: OperationSpec, coalition: Tuple[Coord, ...], holders: Mapping[str, Coord]) -> Optional[Dict[Coord, str]]:
        roles = list(op.roles or tuple(f"role_{k}" for k in range(op.arity_max)))
        assigned: Dict[Coord, str] = {}
        used_roles: Set[str] = set()
        for role, token in op.role_inputs.items():
            rc = holders.get(token)
            if rc is None or rc not in coalition or rc in assigned:
                return None
            assigned[rc] = role; used_roles.add(role)
        remaining_roles = [x for x in roles if x not in used_roles]
        extra = 0
        for rc in coalition:
            if rc not in assigned:
                if remaining_roles:
                    assigned[rc] = remaining_roles.pop(0)
                else:
                    # coalition is larger than op.kappa (the declared MINIMUM
                    # team size): this robot is a recruited extra helper
                    # beyond the declared role list, not a required role.
                    assigned[rc] = f"helper_extra_{extra}"; extra += 1
        return assigned

    def _role_capable(self, p: Product, op: OperationSpec, rc: Coord, role: str) -> bool:
        r = self._robot(rc)
        skill = op.role_skills.get(role, op.kind)
        if not self._skill(r, skill):
            return False
        required_tools = op.role_tools.get(role, ())
        if any(tool not in r.model.tools for tool in required_tools):
            return False
        if role in op.role_inputs:
            holder_roles = max(1, len(op.role_inputs))
            if p.mass / holder_roles > r.model.payload + 1e-9:
                return False
        return True

    def _resource_capacity(self, resource: str) -> int:
        return max(1, int(self.cfg.resource_capacities.get(resource, 1)))

    def _active_resource_count(self, resource: str) -> int:
        return sum(resource in r.resources and r.until > self.t for r in self.active_reservations.values())

    def _resources_available(self, resources: Iterable[str], chosen: Sequence[OperationCandidate] = ()) -> bool:
        for resource in resources:
            count = self._active_resource_count(resource) + sum(resource in c.resources for c in chosen)
            if count >= self._resource_capacity(resource):
                return False
        return True

    def _activity_conflict(self, members: Sequence[Coord], resources: Iterable[str], plan: MotionPlan,
                           reservation: ActivityReservation) -> bool:
        if set(members) & set(reservation.members):
            return True
        if any(self._resource_capacity(x) <= 1 for x in set(resources) & set(reservation.resources)):
            return True
        if self.cfg.trajectory_conflicts:
            clearance = self.cfg.workspace_conflict_radius + max(
                [self._robot(x).model.clearance for x in members] +
                [self._robot(x).model.clearance for x in reservation.members]
            )
            if plans_conflict(plan, reservation.motion_plan, clearance):
                return True
        return False

    def _blocking_reason(self, members: Sequence[Coord], resources: Iterable[str],
                         plan: MotionPlan) -> Optional[str]:
        """Why a prospective activity cannot start against the currently committed activities.

        Returns ``"robot"`` for a shared robot, ``"resource"`` for a shared exclusive resource or
        an exhausted finite-capacity resource, ``"geometry"`` when abstract workspace overlap is
        the only cause, and ``None`` when no committed activity blocks it. Causes are ranked
        robot, then resource, then geometry, so geometry is reported only when nothing else
        blocks the activity. The ``None`` case is exactly the negation of
        :meth:`_activity_conflict` taken over the active reservations together with the resource
        availability check, so classifying a blocking event never changes whether it is one.
        """
        resources = set(resources)
        geometric = False
        exhausted = not self._resources_available(resources)
        for reservation in self.active_reservations.values():
            if reservation.until <= self.t:
                continue
            if set(members) & set(reservation.members):
                return "robot"
            if any(self._resource_capacity(x) <= 1 for x in resources & set(reservation.resources)):
                return "resource"
            if self.cfg.trajectory_conflicts:
                clearance = self.cfg.workspace_conflict_radius + max(
                    [self._robot(x).model.clearance for x in members] +
                    [self._robot(x).model.clearance for x in reservation.members]
                )
                if plans_conflict(plan, reservation.motion_plan, clearance):
                    geometric = True
        if exhausted:
            return "resource"
        return "geometry" if geometric else None

    def _count_blocking(self, reason: str) -> None:
        """Record one blocked activity in the aggregate counter and in its cause-specific counter."""
        self.trajectory_conflict_count += 1
        if reason == "geometry":
            self.geometry_block_count += 1
        elif reason == "resource":
            self.resource_block_count += 1
        else:
            self.robot_block_count += 1

    def _candidate_conflicts_active(self, cand: OperationCandidate) -> bool:
        if not self._resources_available(cand.resources):
            return True
        return any(self._activity_conflict(cand.coalition, cand.resources, cand.motion_plan, r)
                   for r in self.active_reservations.values() if r.until > self.t)

    def candidate_operations(self) -> Tuple[OperationCandidate, ...]:
        if self._candidate_cache is not None:
            return self._candidate_cache
        count_geo = self._geo_reject_counted_tick != self.t
        out: List[OperationCandidate] = []
        for p in self.products.values():
            if p.delivered:
                continue
            rec = self.recipe(p)
            for op in rec.topological_operations():
                if not self._op_ready(p, op):
                    continue
                holders = {token: self._token_holder(p.id, token) for token in op.inputs}
                if any(v is None for v in holders.values()):
                    continue
                holder_coords = tuple(sorted(set(holders.values())))  # type: ignore[arg-type]
                # The input tokens may already be spread over more robots than
                # the declared MINIMUM arity. That is only infeasible if it
                # exceeds the operation's admissible MAXIMUM (or the global
                # physical cap): comparing against op.kappa here would reject
                # a perfectly legal coalition whenever k_max > k_min, which
                # silently made such recipes undeliverable.
                arity_cap = min(op.arity_max, self.cfg.max_coalition_size)
                if len(holder_coords) > arity_cap:
                    continue
                if any(not self._free(self._robot(rc)) for rc in holder_coords):
                    continue
                # op.kappa is the MINIMUM the operation requires and
                # op.arity_max the largest admissible team (defaulting to
                # kappa, i.e. exact arity). Extra members do NOT accelerate
                # the operation; they are admissible only where the recipe
                # declares support roles for them. Every admissible team SIZE
                # in [kappa, arity_max] is exposed as its own candidate so
                # team formation can choose, but choosing more buys no speed.
                helpers_min = max(0, op.kappa - len(holder_coords))
                helpers_cap = arity_cap - len(holder_coords)
                if len(holder_coords) > 1 and not self._coalition_local(holder_coords):
                    continue

                # Locality-aware helper generation.  The previous prototype
                # formed combinations from every free robot in the grid and
                # rejected nonlocal teams afterwards, which becomes
                # combinatorial on large grids.  Any valid helper must be a
                # direct neighbor of every existing holder, so restrict first.
                if holder_coords:
                    helper_pool = []
                    for i in range(self.cfg.M):
                        for j in range(self.cfg.N):
                            rc = (i, j); r = self._robot(rc)
                            if rc in holder_coords or not self._free(r) or r.holding is not None:
                                continue
                            if all(self._manip_local(rc, h) for h in holder_coords):
                                helper_pool.append(rc)
                    helper_sets = [
                        hs for size in range(helpers_min, helpers_cap + 1)
                        for hs in combinations(helper_pool, size)
                    ]
                else:
                    # Zero-input transformations are allowed by the recipe
                    # schema.  Enumerate local cliques around anchors rather
                    # than combinations over the entire robot population.
                    seen_helpers = set()
                    local_sets = []
                    for i in range(self.cfg.M):
                        for j in range(self.cfg.N):
                            anchor = (i, j); ar = self._robot(anchor)
                            if not self._free(ar) or ar.holding is not None:
                                continue
                            pool = [anchor] + [q for q in self.neighbors(anchor)
                                               if self._free(self._robot(q)) and self._robot(q).holding is None]
                            for size in range(helpers_min, helpers_cap + 1):
                                for hs in combinations(sorted(set(pool)), size):
                                    key = tuple(sorted(hs))
                                    if key in seen_helpers or not self._coalition_local(key):
                                        continue
                                    seen_helpers.add(key); local_sets.append(key)
                    helper_sets = local_sets

                for hs in helper_sets:
                    coalition = tuple(sorted(holder_coords + tuple(hs)))
                    if not (op.kappa <= len(coalition) <= arity_cap):
                        continue
                    if not self._coalition_local(coalition):
                        continue
                    roles = self._roles_for(p, op, coalition, holders)  # type: ignore[arg-type]
                    if roles is None or any(not self._role_capable(p, op, rc, role) for rc, role in roles.items()):
                        continue
                    target = common_target(holder_coords if holder_coords else coalition, self.cfg.cell_spacing)
                    per_robot_targets = role_targets(
                        target,
                        roles,
                        self.cfg.role_target_offset,
                        bases={rc: base_point(rc, self.cfg.cell_spacing) for rc in coalition},
                    )
                    duration = self._operation_duration(op)
                    try:
                        plan = self._make_plan(coalition, target, duration, per_robot_targets)
                    except MotionInfeasible:
                        if count_geo:
                            self.reject_motion_infeasible += 1
                        continue
                    resources = frozenset({f"operation:{p.id}:{op.id}", *(f"token:{p.id}:{tok}" for tok in op.inputs), *op.resources})
                    cid = f"op:{p.id}:{op.id}:" + ";".join(f"{i},{j}" for i, j in coalition)
                    cand = OperationCandidate(
                        id=cid, product_id=p.id, operation_id=op.id, kind=op.kind,
                        coalition=coalition, roles=tuple(sorted(roles.items())),
                        workspace=frozenset(coalition), resources=resources,
                        motion_plan=plan, productive_value=self._productive_weight(p, op),
                        kappa_min=op.kappa, size=len(coalition),
                    )
                    # Candidate exposure is deliberately LOCAL/STRUCTURAL.
                    # We do not remove it because of a remote active trajectory
                    # or globally contended resource: doing so would turn the
                    # option list/action mask into a hidden global-feasibility
                    # oracle.  Global physical conflicts are checked only when
                    # capacity is diagnosed or a fully agreed team commits.
                    out.append(cand)
        out.sort(key=lambda x: (x.product_id, x.operation_id, x.coalition))
        if count_geo:
            self._geo_reject_counted_tick = self.t
        self._candidate_cache = tuple(out)
        return self._candidate_cache

    def operations_conflict(self, a: OperationCandidate, b: OperationCandidate) -> bool:
        if set(a.coalition) & set(b.coalition):
            return True
        if any(self._resource_capacity(x) <= 1 for x in a.resources & b.resources):
            return True
        if self.cfg.trajectory_conflicts:
            clearance = self.cfg.workspace_conflict_radius + max(
                [self._robot(x).model.clearance for x in a.coalition] +
                [self._robot(x).model.clearance for x in b.coalition]
            )
            if plans_conflict(a.motion_plan, b.motion_plan, clearance):
                return True
        return False

    def _routes_mutually_exclusive(self, a: OperationCandidate, b: OperationCandidate) -> bool:
        """True when two candidates realize mutually exclusive alternative routes.

        Alternative routes of one product are exclusive (XOR) and admission enforces this
        through :meth:`_route_excluded`, so a capacity selection must not contain both.
        """
        if a.product_id != b.product_id or a.operation_id == b.operation_id:
            return False
        p = self.products.get(a.product_id)
        if p is None:
            return False
        return b.operation_id in self._xor_siblings(self.recipe(p), a.operation_id)

    def _selection_feasible(self, cand: OperationCandidate, chosen: Sequence[OperationCandidate]) -> bool:
        if any(self.operations_conflict(cand, x) for x in chosen):
            return False
        if any(self._routes_mutually_exclusive(cand, x) for x in chosen):
            return False
        return self._resources_available(cand.resources, chosen)

    def _greedy_select(self, cands: Sequence[OperationCandidate], weighted: bool) -> Tuple[List[OperationCandidate], float]:
        if weighted:
            order = sorted(cands, key=lambda x: (-x.productive_value, len(x.coalition), -x.motion_plan.min_reach_margin, x.id))
        else:
            order = sorted(cands, key=lambda x: (len(x.coalition), x.motion_plan.total_duration, -x.motion_plan.min_reach_margin, x.id))
        chosen: List[OperationCandidate] = []
        for c in order:
            if self._selection_feasible(c, chosen):
                chosen.append(c)
        score = sum(x.productive_value for x in chosen) if weighted else float(len(chosen))
        return chosen, score

    def _optimal_select(self, cands: Sequence[OperationCandidate], weighted: bool = False) -> Tuple[List[OperationCandidate], float, bool]:
        """Exact group-aware independent-set search on diagnostic-sized states.

        Alternative realizations of one product-operation share an operation/token
        resource, so the search branches by (product, operation) and has an
        upper bound of one realization per group.  Distinct parallel DAG
        branches of the same product may therefore execute concurrently.  Larger states fall back to deterministic greedy selection
        and report exact=False; the metric therefore never silently pretends a
        heuristic value is an exact optimum.
        """
        cands = list(cands)
        groups: Dict[Tuple[int, str], List[OperationCandidate]] = {}
        for c in cands:
            groups.setdefault((c.product_id, c.operation_id), []).append(c)
        # Exactness is controlled by both raw candidate count and the actual
        # grouped branch count.  A state with only a few operation groups can
        # still have dozens of alternative coalitions per group, so a simple
        # candidate threshold is not enough to prevent exponential stalls.
        combination_bound = 1
        for g in groups.values():
            combination_bound *= (len(g) + 1)  # one realization or skip group
            if combination_bound > self.cfg.parallelism_exact_combination_limit:
                break
        exact = (
            len(cands) <= self.cfg.parallelism_exact_candidate_limit
            and len(groups) <= self.cfg.parallelism_exact_product_limit
            and combination_bound <= self.cfg.parallelism_exact_combination_limit
        )
        if not exact:
            chosen, score = self._greedy_select(cands, weighted)
            return chosen, score, False

        grouped = sorted(groups.values(), key=lambda g: (len(g), g[0].product_id, g[0].operation_id))
        best: List[OperationCandidate] = []
        best_score = -1.0
        max_weight_remaining = [0.0] * (len(grouped) + 1)
        if weighted:
            for k in range(len(grouped) - 1, -1, -1):
                max_weight_remaining[k] = max_weight_remaining[k + 1] + max((c.productive_value for c in grouped[k]), default=0.0)

        def dfs(k: int, chosen: List[OperationCandidate], score: float) -> None:
            nonlocal best, best_score
            if k == len(grouped):
                if score > best_score + 1e-12:
                    best_score = score; best = list(chosen)
                return
            if weighted:
                if score + max_weight_remaining[k] <= best_score + 1e-12:
                    return
            else:
                if score + (len(grouped) - k) <= best_score + 1e-12:
                    return
            # Try feasible candidates first, ordered by objective quality.
            options = grouped[k]
            options = sorted(options, key=(lambda x: -x.productive_value) if weighted else (lambda x: (len(x.coalition), x.motion_plan.total_duration)))
            for cand in options:
                if self._selection_feasible(cand, chosen):
                    chosen.append(cand)
                    dfs(k + 1, chosen, score + (cand.productive_value if weighted else 1.0))
                    chosen.pop()
            dfs(k + 1, chosen, score)  # skip this product

        # Seed with a greedy lower bound to improve pruning.
        seed, seed_score = self._greedy_select(cands, weighted)
        best, best_score = seed, seed_score
        dfs(0, [], 0.0)
        return best, max(0.0, best_score), True

    def feasible_operation_candidates(self, cands: Optional[Sequence[OperationCandidate]] = None) -> Tuple[OperationCandidate, ...]:
        """Globally/physically feasible operation realizations for diagnostics.

        This is intentionally separate from :meth:`candidate_operations`,
        which is what local policies see.  The separation prevents global
        reservation/trajectory state from leaking through action availability.
        """
        cands = tuple(cands if cands is not None else self.candidate_operations())
        return tuple(c for c in cands if not self._candidate_conflicts_active(c))

    def capacity_snapshot(self, cands: Optional[Sequence[OperationCandidate]] = None) -> dict:
        cands = self.feasible_operation_candidates(cands)
        count_sel, count_score, exact_count = self._optimal_select(cands, weighted=False)
        prod_sel, prod_score, exact_prod = self._optimal_select(cands, weighted=True)
        active = [t for t in self.teams.values() if t.status == "committed" and t.committed_until > self.t]
        active_k = len(active)
        active_p = sum(t.productive_value for t in active)
        # Point (audit): when the selection is only approximate, K_feasible_star
        # is a LOWER bound on the true feasible maximum, so K_t / K_feasible_star
        # is not a utilization and can exceed 1. Callers must branch on "exact"
        # and report a bounded gap instead of a ratio when it is False; the
        # bound fields below say explicitly which side the estimate errs on.
        return {
            "K_additional": int(round(count_score)),
            "K_feasible_star": active_k + int(round(count_score)),
            "P_additional": float(prod_score),
            "P_productive_star": active_p + float(prod_score),
            "count_selection": tuple(x.id for x in count_sel),
            "productive_selection": tuple(x.id for x in prod_sel),
            "exact": bool(exact_count and exact_prod),
            # explicit bound direction, so a consumer never has to guess
            "K_feasible_bound": "exact" if exact_count else "lower",
            "P_productive_bound": "exact" if exact_prod else "lower",
        }

    def parallelism_utilization_report(self) -> dict:
        """Safe reporting of K_t against feasible capacity.

        Only returns a ratio when the capacity oracle solved the selection
        EXACTLY. When it fell back to the greedy approximation the denominator
        is a lower bound on the true feasible maximum, so K_t/K*_feas is not a
        utilization at all and can exceed 1; in that case a bounded gap is
        reported instead, and "utilization" is None. Consumers (GUI, metrics,
        papers) should use this rather than dividing the raw fields.
        """
        snap = self.capacity_snapshot()
        k_t = self.current_concurrency()
        k_star = snap["K_feasible_star"]
        if snap["exact"]:
            return {
                "exact": True,
                "K_t": k_t,
                "K_feasible_star": k_star,
                "utilization": (k_t / k_star) if k_star > 0 else None,
                "unused_capacity": max(0, k_star - k_t),
            }
        return {
            "exact": False,
            "K_t": k_t,
            "K_feasible_star_lower_bound": k_star,
            "utilization": None,  # undefined against a lower-bound denominator
            "utilization_upper_bound": (k_t / k_star) if k_star > 0 else None,
            "note": "approximate selection: denominator is a lower bound, "
                    "so the ratio is an upper bound on true utilization, not a utilization",
        }

    def feasible_parallelism(self) -> int:
        return self.capacity_snapshot()["K_feasible_star"]

    def productive_parallelism(self) -> int:
        """Canonical v1 productive-operation concurrency capacity.

        Every operation candidate represents an achievement-state transition on
        successful completion, so the unweighted K* concurrency is the base v1
        productive-parallelism notion. Logistics/enabling activities are not
        counted here.
        """
        return self.capacity_snapshot()["K_feasible_star"]

    def weighted_productive_parallelism(self) -> float:
        """Optional weighted productive-concurrency diagnostic (legacy P*)."""
        return self.capacity_snapshot()["P_productive_star"]

    def current_concurrency(self) -> int:
        return sum(t.status == "committed" and t.committed_until > self.t for t in self.teams.values())

    def current_productive_value(self) -> float:
        return sum(t.productive_value for t in self.teams.values() if t.status == "committed" and t.committed_until > self.t)

    def concurrency_utilization(self) -> float:
        """Mean realized/available operation concurrency, over EXACT-capacity
        opportunity states only. States where the capacity oracle was only
        approximate are excluded (see concurrency_utilization_approx_bound);
        including them would mix a true ratio with an upper bound. Returns
        None when no exact opportunity state was observed, so a caller can
        never mistake "not measurable" for "zero"."""
        if self.k_opportunity_samples == 0:
            return None
        return self.sum_k_util / self.k_opportunity_samples

    def concurrency_utilization_approx_bound(self) -> float:
        """Mean of k_real/K*_feas over APPROXIMATE-capacity states. The
        denominator is a lower bound there, so this is an UPPER BOUND on true
        utilization, not a utilization. Reported separately and named as a
        bound so it cannot be quoted as saturation."""
        if self.k_opportunity_samples_approx == 0:
            return None
        return self.sum_k_util_approx / self.k_opportunity_samples_approx

    def productive_concurrency_utilization(self) -> float:
        """Canonical v1 utilization of productive-operation concurrency.

        This is the same exact-opportunity-conditioned K_realized/K*_feasible
        quantity as :meth:`concurrency_utilization`; the explicit name makes
        clear that recipe-operation concurrency is the base productive notion.
        """
        return self.concurrency_utilization()

    def productive_concurrency_utilization_approx_bound(self) -> float:
        """Approximate-capacity upper bound counterpart of the base metric."""
        return self.concurrency_utilization_approx_bound()

    def weighted_productive_utilization(self) -> float:
        """Optional weighted P_realized/P* diagnostic over exact states."""
        if self.p_opportunity_samples == 0:
            return None
        return self.sum_p_util / self.p_opportunity_samples

    def weighted_productive_utilization_approx_bound(self) -> float:
        """Upper bound counterpart of :meth:`weighted_productive_utilization`."""
        if self.p_opportunity_samples_approx == 0:
            return None
        return self.sum_p_util_approx / self.p_opportunity_samples_approx

    # Backwards-compatible aliases from the pre-AssGrDims#2 implementation.
    # They remain weighted diagnostics and are deliberately not the canonical
    # v1 productive-concurrency KPI names.
    def productive_utilization(self) -> float:
        return self.weighted_productive_utilization()

    def productive_utilization_approx_bound(self) -> float:
        return self.weighted_productive_utilization_approx_bound()

    # ------------------------------------------------------------------
    # local option/action construction
    # ------------------------------------------------------------------
    def _pick_options_for(self, rc: Coord) -> Tuple[PickOption, ...]:
        if rc in self._pick_cache:
            return self._pick_cache[rc]
        i, j = rc; r = self._robot(rc)
        if j != 0 or not self._free(r) or r.holding is not None or r.proposal_id is not None or not self._skill(r, "pick"):
            self._pick_cache[rc] = tuple(); return tuple()
        opts: List[Tuple[float, PickOption]] = []
        for p in self.products.values():
            if p.delivered:
                continue
            rec = self.recipe(p)
            for token in rec.raw_tokens:
                if token in p.raw_picked or token in p.reserved_raw:
                    continue
                srow = p.source_rows[token]
                if i not in self.shelf_reach.get(srow, set()):
                    continue
                stock = self.shelves[srow]["inventory"].get(token, 0)
                if stock <= 0 or self.shelves[srow]["outage_until"] > self.t:
                    continue
                target = self._shelf_target(srow)
                if not self._can_reach_point(rc, target):
                    continue
                opt = PickOption(
                    f"pick:{p.id}:{token}:{srow}", p.id, p.recipe_id, token, srow,
                    stock / max(1, self.cfg.shelf_cap),
                )
                opts.append((self.priority_fn(p, self.t, self), opt))
        # The environment does not inject due-date urgency into the canonical
        # ordering.  A baseline may supply its own priority_fn, while the
        # identity-bearing option set remains unchanged and fully addressable.
        opts.sort(key=lambda x: (x[0], x[1].id))
        result = tuple(x[1] for x in opts)
        if len(result) > MAX_PICK_OPTIONS:
            raise RuntimeError(
                f"action-addressability overflow at {rc}: {len(result)} eligible pick choices "
                f"exceed MAX_PICK_OPTIONS={MAX_PICK_OPTIONS}; canonical v1 forbids silent truncation"
            )
        self._pick_cache[rc] = result
        return result

    def _operation_options_for(self, rc: Coord, cands: Optional[Sequence[OperationCandidate]] = None) -> Tuple[OperationOptionView, ...]:
        cands = cands if cands is not None else self.candidate_operations()
        pviews = []
        for c in cands:
            if rc not in c.coalition:
                continue
            p = self.products[c.product_id]
            prop = self.proposals.get(c.id)
            support = len(prop.votes) / max(1, c.size) if prop is not None else 0.0
            if not self.cfg.proposal_support_visible:
                support = 0.0
            # Weighted productivity and due-date urgency stay internal or in
            # explicitly named extensions. Canonical local candidates expose
            # task/recipe identity, motion consequences and aggregate proposal
            # support, but no urgency/value score.
            pviews.append(OperationOptionView(
                candidate_id=c.id, product_id=c.product_id, recipe_id=p.recipe_id,
                operation_id=c.operation_id, kind=c.kind, kappa_min=c.kappa_min,
                size=c.size, role=c.role_map[rc], members=c.coalition,
                motion_duration=c.motion_plan.total_duration,
                min_clearance=c.motion_plan.min_reach_margin, proposal_support=support,
            ))
        # Support is an aggregate local intent signal; supporter identities are
        # never exposed. Fallback ordering is deterministic and does not use
        # privileged weighted-productivity information.
        # Candidate-slot order must not encode the preferred geometric choice.
        # Support remains a declared local intention signal; otherwise semantic
        # identity determines the stable order. Geometry-aware policies must
        # explicitly use the exposed duration/margin fields.
        pviews.sort(key=lambda x: (-x.proposal_support, x.candidate_id))
        if len(pviews) > MAX_OPERATION_OPTIONS:
            raise RuntimeError(
                f"action-addressability overflow at {rc}: {len(pviews)} eligible operation choices "
                f"exceed MAX_OPERATION_OPTIONS={MAX_OPERATION_OPTIONS}; canonical v1 forbids silent truncation"
            )
        return tuple(pviews)

    def action_mask(self, i: int, j: int, cands: Optional[Sequence[OperationCandidate]] = None) -> List[bool]:
        rc = (i, j); r = self._robot(rc)
        m = [False] * NUM_ACTIONS
        m[ACTION_IDLE] = True
        if self._failed(r) or not self._free(r):
            return m
        # A pending proposal is a local commitment. The robot may wait or
        # select an explicit operation candidate, but may not simultaneously
        # pick/transfer/deliver.
        op_opts = self._operation_options_for(rc, cands)
        if r.proposal_id is not None and self.t < r.proposal_lock_until:
            # During the short proposal-commitment horizon the robot may wait
            # or repeat the same proposal, but cannot opportunistically switch
            # teams. After the lock expires it may explicitly switch.
            for k, opt in enumerate(op_opts):
                if opt.candidate_id == r.proposal_id:
                    m[ACTION_OPERATION_BASE + k] = True
            return m
        for k in range(len(op_opts)):
            m[ACTION_OPERATION_BASE + k] = True
        if r.proposal_id is not None:
            return m

        if r.holding is None:
            for k in range(len(self._pick_options_for(rc))):
                m[ACTION_PICK_BASE + k] = True
            for d, (di, dj) in enumerate(DIRECTIONS):
                q = (i + di, j + dj)
                if self._in_bounds(q) and q in self.handoff_neighbors(rc) and self._handoff_static_feasible(rc, q):
                    # Receiving is statically possible; whether the neighbor
                    # requests it is deliberately not leaked through the mask.
                    m[ACTION_RECEIVE_BASE + d] = True
        else:
            for d, (di, dj) in enumerate(DIRECTIONS):
                q = (i + di, j + dj)
                if self._in_bounds(q) and q in self.handoff_neighbors(rc) and self._handoff_static_feasible(rc, q):
                    m[ACTION_HANDOFF_BASE + d] = True
            pid, token = r.holding
            p = self.products.get(pid)
            if p is not None and token == self.recipe(p).final_token and j == self.cfg.N - 1 and self._can_reach_point(rc, self._output_target(i)):
                m[ACTION_DELIVER] = True
        return m

    def _direction_neighbors(self, rc: Coord) -> Dict[int, Coord]:
        out = {}
        for d, (di, dj) in enumerate(DIRECTIONS):
            q = (rc[0] + di, rc[1] + dj)
            if q in self.adjacency.get(rc, ()):
                out[d] = q
        return out

    def observation(self, i: int, j: int, cands: Optional[Sequence[OperationCandidate]] = None) -> Observation:
        rc = (i, j); r = self._robot(rc)
        cands = tuple(cands if cands is not None else self.candidate_operations())
        ns = {}
        for qrc in self.obs_neighbors(rc):
            q = self._robot(qrc)
            ns[qrc] = {
                "holding_kind": q.holding[1] if q.holding else None,
                "holding_pid": q.holding[0] if q.holding else None,
                "busy": not self._free(q),
                "failed": self._failed(q),
                "op": q.op,
                "role": q.role,
            }
        picks = self._pick_options_for(rc)
        ops = self._operation_options_for(rc, cands)
        return Observation(
            robot_id=rc, t=self.t,
            holding_kind=r.holding[1] if r.holding else None,
            holding_pid=r.holding[0] if r.holding else None,
            busy=not self._free(r), failed=self._failed(r), neighbors=ns,
            pending_pick_here=bool(picks), pick_options=picks, operation_options=ops,
            action_mask=self.action_mask(i, j, cands), proposal_id=r.proposal_id,
        )

    def _all_observations(self) -> Dict[Coord, Observation]:
        cands = self.candidate_operations()
        return {(i, j): self.observation(i, j, cands) for i in range(self.cfg.M) for j in range(self.cfg.N)}

    # ------------------------------------------------------------------
    # step and action execution
    # ------------------------------------------------------------------
    def step(self, joint: Mapping[Coord, int]):
        # All action slot decoding is based on this pre-action state.
        cands = self.candidate_operations()
        cand_by_id = {c.id: c for c in cands}
        obs = self._all_observations()
        capacity = self.capacity_snapshot(cands)
        ready_keys = self.logically_ready_operation_keys()
        feasible_keys = {(c.product_id, c.operation_id) for c in self.feasible_operation_candidates(cands)}
        self.ready_task_count_sum += len(ready_keys)
        self.ready_task_with_team_sum += len(ready_keys & feasible_keys)
        if capacity["exact"]:
            self.capacity_exact_samples += 1
        else:
            self.capacity_approx_samples += 1

        actions: Dict[Coord, int] = {}
        for rc, o in obs.items():
            a = int(joint.get(rc, ACTION_IDLE))
            if not 0 <= a < NUM_ACTIONS or not o.action_mask[a]:
                if a != ACTION_IDLE:
                    self.unmasked_invalid_attempts += 1
                self.invalid_attempts += 1
                self.reject_locally_invalid += 1
                a = ACTION_IDLE
            actions[rc] = a

        self.last_step_energy = 0.0
        self.last_step_safety = 0
        progress_before = self.progress_events

        self._picks(actions, obs)
        self._handoffs(actions)
        self._team_proposals(actions, obs, cand_by_id)
        self._deliver(actions)

        # Realized productive concurrency after this decision, compared with
        # the optimum available in the state in which the decision was made.
        k_real = self.current_concurrency()
        p_real = self.current_productive_value()
        k_coop = sum(1 for t in self.teams.values() if t.status == "committed" and t.committed_until > self.t and len(t.members) >= 2)
        busy_robots = sum(not self._free(r) for row in self.robots for r in row)
        self.sum_k_coop_realized += k_coop
        self.sum_robot_occupancy += busy_robots / max(1, self.cfg.M * self.cfg.N)
        k_star = max(0, int(capacity["K_feasible_star"]))
        p_star = max(0.0, float(capacity["P_productive_star"]))
        k_util = 1.0 if k_star == 0 else min(1.0, k_real / k_star)
        p_util = 1.0 if p_star <= 1e-12 else min(1.0, p_real / p_star)
        self.sum_k_feasible_star += k_star; self.sum_p_productive_star += p_star
        self.sum_k_realized += k_real; self.sum_p_realized += p_real
        self.parallelism_samples += 1
        # Utilization is only a UTILIZATION when the capacity oracle solved
        # the selection exactly. When it fell back to the greedy
        # approximation, K*_feas is a LOWER bound, so k_real/k_star is an
        # upper bound on the true ratio and can exceed 1. Accumulating both
        # kinds into one average produced the reported failure mode: a
        # headline utilization of 1.0 while roughly half the underlying
        # capacity samples were approximations. Ticks with no opportunity at
        # all (k_star == 0) are excluded rather than scored as "perfect",
        # since having nothing to do is not full utilization.
        exact = bool(capacity.get("exact"))
        if k_star > 0:
            ratio = k_real / k_star
            if exact:
                self.sum_k_util += ratio
                self.k_opportunity_samples += 1
            else:
                self.sum_k_util_approx += ratio
                self.k_opportunity_samples_approx += 1
        if p_star > 1e-12:
            ratio_p = p_real / p_star
            if exact:
                self.sum_p_util += ratio_p
                self.p_opportunity_samples += 1
            else:
                self.sum_p_util_approx += ratio_p
                self.p_opportunity_samples_approx += 1
        if k_star > 0 or p_star > 0:
            self.parallelism_nonzero_samples += 1

        # Optional shared RL training signal. It is intentionally separate
        # from the benchmark KPIs and is not part of AssemblyGrid's theory.
        delivered_before = self.delivered_count
        delivered_value_before = self.delivered_value
        tardiness_before = self.sum_tardiness

        # Advance one simulation tick after decisions, then process events and
        # exogenous arrivals/failures.  Returned observations therefore match
        # the next decision state.
        self.t += 1
        self._complete()
        self._replenish_and_stockouts()
        self._fail_robots()
        self._spawn()
        self._expire_proposals(cand_by_id=None)
        self._cleanup_reservations()
        self.sum_wip += self.wip()

        deliveries = self.delivered_count - delivered_before
        delivery_value_inc = self.delivered_value - delivered_value_before
        tardiness_inc = self.sum_tardiness - tardiness_before
        tr = self.training_reward
        reward = (
            delivery_value_inc * tr.delivery_value_scale
            - tr.tardiness_penalty * tardiness_inc
            - tr.motion_energy_penalty * self.last_step_energy
            - tr.safety_penalty * self.last_step_safety
            - tr.wip_penalty * self.wip()
        )

        if self.wip() > 0 and self.progress_events == progress_before and self.current_concurrency() == 0:
            self.deadlock_ticks += 1

        # A batch is complete only when every one of the Z released products
        # has been delivered, so makespan is the time until the LAST of them
        # completes, matching the reference solvers' definition.
        terminated = (
            self.cfg.episode_mode == "batch"
            and self.spawned_count >= self.cfg.batch_Z
            and self.delivered_count >= self.cfg.batch_Z
        )
        # horizon_T is the declared evaluation/safety horizon in both modes.
        # A finite batch that has not completed by then is truncated and has
        # undefined canonical makespan rather than an arbitrarily long hidden
        # runner timeout.
        truncated = self.t >= self.cfg.horizon_T and not terminated
        self._invalidate()
        next_obs = self._all_observations()
        info = {
            "t": self.t,
            "K_realized": k_real,
            "P_realized": p_real,
            "K_feasible_star": k_star,
            "P_productive_star": p_star,
            "K_utilization": k_util,
            "productive_concurrency_utilization": k_util,
            "weighted_productive_parallelism_utilization": p_util,
            # Legacy weighted alias retained for older scripts.
            "productive_parallelism_utilization": p_util,
            "capacity_exact": capacity["exact"],
            "wip": self.wip(),
            "delivered": self.delivered_count,
            "spawned": self.spawned_count,
            "throughput": self.throughput(),
            "active_teams": len(self.teams),
            "trajectory_conflicts": self.trajectory_conflict_count,
            "geometry_block_count": self.geometry_block_count,
            "resource_block_count": self.resource_block_count,
            "robot_block_count": self.robot_block_count,
            "unmasked_invalid_attempts": self.unmasked_invalid_attempts,
            "track": self.cfg.benchmark_track,
            "topology": self.cfg.topology,
            "geometry_profile": self.cfg.geometry_profile,
            "geometry_variant": self.cfg.geometry_variant,
            "training_reward_profile": self.training_reward.profile,
        }
        return next_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # event starters
    # ------------------------------------------------------------------
    def _record_motion(self, plan: MotionPlan) -> None:
        self.motion_path_length += plan.path_length
        self.motion_duration_total += plan.total_duration
        self.geometry_delay_ticks += plan.approach_duration + plan.retreat_duration
        self.motion_planning_seconds += plan.planning_seconds
        self.min_motion_clearance = min(self.min_motion_clearance, plan.min_reach_margin)
        energy = plan.path_length
        self.energy_proxy += energy; self.last_step_energy += energy
        if plan.min_reach_margin < 0:
            self.safety_violations += 1; self.last_step_safety += 1

    def _reserve(self, rid: str, kind: str, members: Sequence[Coord], resources: Iterable[str], plan: MotionPlan, until: int) -> None:
        self.active_reservations[rid] = ActivityReservation(rid, kind, tuple(members), frozenset(resources), plan, until)
        self._record_motion(plan)

    def _picks(self, actions: Mapping[Coord, int], observations: Mapping[Coord, Observation]) -> None:
        for rc in sorted(actions):
            a = actions[rc]
            if not ACTION_PICK_BASE <= a < ACTION_PICK_BASE + MAX_PICK_OPTIONS:
                continue
            slot = a - ACTION_PICK_BASE
            opts = observations[rc].pick_options
            if slot >= len(opts):
                continue
            opt = opts[slot]; p = self.products.get(opt.product_id); r = self._robot(rc)
            if p is None or opt.token in p.raw_picked or opt.token in p.reserved_raw or r.holding is not None or not self._free(r):
                continue
            inv = self.shelves[opt.shelf_row]["inventory"]
            if inv.get(opt.token, 0) <= 0:
                continue
            target = self._shelf_target(opt.shelf_row)
            try:
                plan = self._make_plan((rc,), target, self.cfg.pick_duration)
            except MotionInfeasible:
                continue
            rid = f"pick:{p.id}:{opt.token}:{rc}:{self.t}"
            block = self._blocking_reason((rc,), {f"shelf:{opt.shelf_row}:{opt.token}"}, plan)
            if block is not None:
                self._count_blocking(block); continue
            inv[opt.token] -= 1
            p.reserved_raw[opt.token] = rc
            r.busy_until = self.t + plan.total_duration; r.op = "pick"
            success = self.execution_rng.random() >= self.cfg.operation_failure_prob
            self.scheduled.append({"at": r.busy_until, "kind": "pick_complete", "pid": p.id,
                                   "token": opt.token, "robot": rc, "shelf_row": opt.shelf_row,
                                   "reservation": rid, "plan": plan, "success": success})
            self._reserve(rid, "pick", (rc,), {f"shelf:{opt.shelf_row}:{opt.token}"}, plan, r.busy_until)
            self._invalidate()

    def _handoffs(self, actions: Mapping[Coord, int]) -> None:
        accepted_members: Set[Coord] = set()
        for s in sorted(actions):
            act = actions[s]
            if not ACTION_HANDOFF_BASE <= act < ACTION_HANDOFF_BASE + 8:
                continue
            d = act - ACTION_HANDOFF_BASE; di, dj = DIRECTIONS[d]; q = (s[0] + di, s[1] + dj)
            if not self._in_bounds(q) or q not in self.handoff_neighbors(s):
                continue
            reverse = DIRECTIONS.index((-di, -dj))
            matched = actions.get(q, ACTION_IDLE) == ACTION_RECEIVE_BASE + reverse
            if self.cfg.require_bilateral_handoff and not matched:
                continue
            rs, rq = self._robot(s), self._robot(q)
            if not self._free(rs) or not self._free(rq) or rs.holding is None or rq.holding is not None:
                continue
            if s in accepted_members or q in accepted_members:
                continue
            pid, token = rs.holding
            p = self.products.get(pid)
            if p is None or p.mass > rs.model.payload + 1e-9 or p.mass > rq.model.payload + 1e-9:
                continue
            if not self._skill(rs, "handoff") or not self._skill(rq, "handoff"):
                continue
            target = self._handoff_target(s, q)
            try:
                plan = self._make_plan((s, q), target, self.cfg.handoff_duration)
            except MotionInfeasible:
                continue
            rid = f"handoff:{pid}:{token}:{s}:{q}:{self.t}"
            resources = {f"token:{pid}:{token}"}
            block = self._blocking_reason((s, q), resources, plan)
            if block is not None:
                self._count_blocking(block); continue
            rs.busy_until = rq.busy_until = self.t + plan.total_duration
            rs.op = "handoff_send"; rq.op = "handoff_receive"; rq.incoming_transfer = (pid, token)
            success = self.execution_rng.random() >= self.cfg.handoff_failure_prob
            self.scheduled.append({"at": rs.busy_until, "kind": "handoff_complete", "pid": pid, "token": token,
                                   "sender": s, "receiver": q, "reservation": rid, "plan": plan, "success": success})
            self._reserve(rid, "handoff", (s, q), resources, plan, rs.busy_until)
            accepted_members.update((s, q)); self._invalidate()

    def _clear_robot_proposal(self, rc: Coord, count_churn: bool = False) -> None:
        r = self._robot(rc); old = r.proposal_id
        if old is None:
            return
        prop = self.proposals.get(old)
        if prop is not None:
            prop.votes.discard(rc)
        r.proposal_id = None; r.proposal_until = 0; r.proposal_lock_until = 0
        if count_churn:
            self.team_proposal_churn += 1

    def _team_proposals(self, actions: Mapping[Coord, int], observations: Mapping[Coord, Observation], cand_by_id: Mapping[str, OperationCandidate]) -> None:
        # Apply explicit proposal/switch actions.
        for rc in sorted(actions):
            a = actions[rc]; r = self._robot(rc)
            if ACTION_OPERATION_BASE <= a < ACTION_OPERATION_BASE + MAX_OPERATION_OPTIONS:
                slot = a - ACTION_OPERATION_BASE
                opts = observations[rc].operation_options
                if slot >= len(opts):
                    continue
                cid = opts[slot].candidate_id
                cand = cand_by_id.get(cid)
                if cand is None or rc not in cand.coalition:
                    continue
                if r.proposal_id is not None and r.proposal_id != cid:
                    self._clear_robot_proposal(rc, count_churn=True)
                if cid not in self.proposals:
                    self.proposals[cid] = TeamProposal(cid, cand.product_id, cand.operation_id, cand.coalition,
                                                       self.t, self.t + self.cfg.formation_timeout)
                prop = self.proposals[cid]
                if r.proposal_id != cid:
                    r.proposal_lock_until = self.t + self.cfg.commitment_horizon
                prop.votes.add(rc); r.proposal_id = cid; r.proposal_until = prop.expires_at
            elif r.proposal_id is not None and a != ACTION_IDLE:
                self._clear_robot_proposal(rc, count_churn=True)

        # Start all fully agreed, still-feasible proposals using a deterministic
        # conflict-free subset.  The environment resolves simultaneous physical
        # conflicts, but never chooses teammates: every member has named the
        # exact same candidate id through its own local action slot.
        ready: List[Tuple[TeamProposal, OperationCandidate]] = []
        for cid, prop in list(self.proposals.items()):
            cand = cand_by_id.get(cid)
            if cand is None:
                continue
            if set(prop.votes) == set(cand.coalition) and all(self._robot(rc).proposal_id == cid for rc in cand.coalition):
                ready.append((prop, cand))
        started: List[OperationCandidate] = []
        # Neutral deterministic arbitration: when fully agreed proposals are
        # mutually incompatible, the environment resolves the simultaneous
        # conflict by candidate identity only. It must not inject a hidden
        # policy objective such as weighted productivity.
        for prop, cand in sorted(ready, key=lambda x: x[1].id):
            if any(self.operations_conflict(cand, x) for x in started):
                continue
            if self._candidate_conflicts_active(cand):
                # An individually admissible operation blocked by a committed activity counts as
                # blocking, exactly as for blocked logistics activities. An exhausted resource of
                # capacity above one is not a pairwise conflict, so it is classified separately.
                block = self._blocking_reason(cand.coalition, cand.resources, cand.motion_plan)
                self._count_blocking(block or "resource")  # reason is never None here
                continue
            if not all(self._free(self._robot(rc)) for rc in cand.coalition):
                continue
            # XOR routes: candidates were generated before ANY commit this
            # tick, so two members of one alternative group can both reach
            # this loop. Re-check against live state (and against what has
            # already been started in this same loop) so exactly one route is
            # ever taken per product.
            product = self.products.get(cand.product_id)
            if product is not None:
                op_spec = self.operation_spec(product, cand.operation_id)
                if self._route_excluded(product, op_spec) or                    cand.operation_id in self._route_locked.get(product.id, ()):
                    continue
                if any(x.product_id == cand.product_id
                       and x.operation_id in self._xor_siblings(self.recipe(product), cand.operation_id)
                       for x in started):
                    continue
            self._start_team(prop, cand); started.append(cand)
        self._expire_proposals(cand_by_id)

    def _start_team(self, prop: TeamProposal, cand: OperationCandidate) -> None:
        """Commit a coalition to an operation.

        Point 1 (coalition lifetime), canonical track: once committed, a
        team's membership is IMMUTABLE for the whole operation:
        C_t = C_{t+1} = ... = C_{t+D-1}. Membership may only change through a
        terminal event (successful completion, explicit abort, or declared
        failure), never by mid-operation re-decision. Nothing in this engine
        mutates Team.members after construction, and
        invariants.check_coalition_lifetime() enforces it across ticks.
        Mid-operation replacement/reconfiguration, if ever wanted, belongs in
        a separate advanced track with its own work-progress, re-sync,
        role-transfer and penalty semantics.
        """
        p = self.products[cand.product_id]; op = self.operation_spec(p, cand.operation_id)
        tid = self.next_team_id; self.next_team_id += 1
        until = self.t + cand.motion_plan.total_duration
        team = Team(
            id=tid, candidate_id=cand.id, product_id=p.id, operation_id=op.id, operation=op.kind,
            members=cand.coalition, roles=cand.role_map, kappa_min=cand.kappa_min,
            formed_at=self.t, first_proposed_at=prop.created_at,
            committed_until=until,
            productive_value=cand.productive_value, motion_plan=cand.motion_plan,
            resources=cand.resources,
        )
        self.teams[tid] = team; p.in_progress_ops.add(op.id)
        # Same-tick guard: candidates are generated once per step, so two
        # members of one XOR group could otherwise both commit before either
        # appeared in in_progress_ops. Committing one closes the others.
        self._route_locked.setdefault(p.id, set()).update(self._xor_siblings(self.recipe(p), op.id))
        for rc in cand.coalition:
            r = self._robot(rc)
            new_role = cand.role_map[rc]
            if r.role and r.role != new_role:
                r.role_switches += 1
            r.role = new_role; r.op = op.kind; r.busy_until = until; r.team_uses += 1
            self._clear_robot_proposal(rc)
        success = self.execution_rng.random() >= self.cfg.operation_failure_prob
        rid = f"team:{tid}"
        self.scheduled.append({"at": until, "kind": "operation_complete", "pid": p.id, "op_id": op.id,
                               "team": tid, "reservation": rid, "plan": cand.motion_plan,
                               "success": success, "candidate_id": cand.id})
        self._reserve(rid, op.kind, cand.coalition, cand.resources, cand.motion_plan, until)
        latency = self.t - prop.created_at
        self.team_formation_latency_sum += latency; self.team_formation_count += 1
        self.team_size_sum += len(cand.coalition)
        self.proposals.pop(cand.id, None)
        self.progress_events += 1; self._invalidate()

    def _expire_proposals(self, cand_by_id: Optional[Mapping[str, OperationCandidate]]) -> None:
        for cid, prop in list(self.proposals.items()):
            invalid = cand_by_id is not None and cid not in cand_by_id
            expired = self.t > prop.expires_at
            if not (invalid or expired):
                continue
            if prop.votes:
                self.team_formation_failures += 1
                self.reject_unmatched_proposal += 1
            for rc in list(prop.votes):
                if self._robot(rc).proposal_id == cid:
                    self._robot(rc).proposal_id = None; self._robot(rc).proposal_until = 0
            self.proposals.pop(cid, None)

    def _deliver(self, actions: Mapping[Coord, int]) -> None:
        for rc in sorted(actions):
            if actions[rc] != ACTION_DELIVER:
                continue
            r = self._robot(rc)
            if not self._free(r) or r.holding is None or rc[1] != self.cfg.N - 1:
                continue
            pid, token = r.holding; p = self.products.get(pid)
            if p is None or token != self.recipe(p).final_token:
                continue
            target = self._output_target(rc[0])
            try:
                plan = self._make_plan((rc,), target, self.cfg.deliver_duration)
            except MotionInfeasible:
                continue
            rid = f"deliver:{pid}:{rc}:{self.t}"
            block = self._blocking_reason((rc,), {f"product:{pid}"}, plan)
            if block is not None:
                self._count_blocking(block); continue
            r.busy_until = self.t + plan.total_duration; r.op = "deliver"
            success = self.execution_rng.random() >= self.cfg.operation_failure_prob
            self.scheduled.append({"at": r.busy_until, "kind": "deliver_complete", "pid": pid, "robot": rc,
                                   "token": token, "reservation": rid, "plan": plan, "success": success})
            self._reserve(rid, "deliver", (rc,), {f"product:{pid}"}, plan, r.busy_until)
            self._invalidate()

    # ------------------------------------------------------------------
    # event completion / exogenous dynamics
    # ------------------------------------------------------------------
    def _apply_motion_end(self, plan: MotionPlan) -> None:
        for rm in plan.robot_motions:
            r = self._robot(rm.robot)
            r.q = rm.q_target; r.ee_pos = rm.target

    def _learn_skill(self, rc: Coord, skill: str, success: bool) -> None:
        r = self._robot(rc); r.skill_uses[skill] = r.skill_uses.get(skill, 0) + 1
        if self.cfg.skill_mode == "learning":
            r.skill_levels[skill] = update_proficiency(r.skill_levels.get(skill, self.cfg.skill_initial_level), success, self.cfg.skill_learning_rate)

    def _complete(self) -> None:
        due = [e for e in self.scheduled if e["at"] <= self.t]
        self.scheduled = [e for e in self.scheduled if e["at"] > self.t]
        for e in due:
            kind = e["kind"]; success = bool(e.get("success", True)); plan = e.get("plan")
            if plan is not None:
                self._apply_motion_end(plan)
            if kind == "pick_complete":
                p = self.products[e["pid"]]; rc = e["robot"]; token = e["token"]; r = self._robot(rc)
                p.reserved_raw.pop(token, None)
                if success and r.holding is None:
                    r.holding = (p.id, token); p.raw_picked.add(token); p.token_positions[token] = rc
                    self._learn_skill(rc, "pick", True); self.progress_events += 1
                else:
                    self.shelves[e["shelf_row"]]["inventory"][token] = min(self.cfg.shelf_cap, self.shelves[e["shelf_row"]]["inventory"].get(token, 0) + 1)
                    self.operation_failures += 1; self._learn_skill(rc, "pick", False)
            elif kind == "handoff_complete":
                s, q = e["sender"], e["receiver"]; rs, rq = self._robot(s), self._robot(q)
                rq.incoming_transfer = None
                if success and rs.holding == (e["pid"], e["token"]) and rq.holding is None:
                    rq.holding = rs.holding; rs.holding = None
                    p = self.products[e["pid"]]; p.token_positions[e["token"]] = q
                    self.handoff_count += 1; self.progress_events += 1
                    self._learn_skill(s, "handoff", True); self._learn_skill(q, "handoff", True)
                else:
                    self.handoff_failures += 1
                    self._learn_skill(s, "handoff", False); self._learn_skill(q, "handoff", False)
            elif kind == "operation_complete":
                p = self.products[e["pid"]]; op = self.operation_spec(p, e["op_id"]); team = self.teams.get(e["team"])
                p.in_progress_ops.discard(op.id)
                if success and team is not None:
                    # Input ownership is consumed atomically; the first role
                    # associated with an input, otherwise rightmost member,
                    # becomes the output holder.
                    holders = {token: self._token_holder(p.id, token) for token in op.inputs}
                    if all(rc is not None for rc in holders.values()):
                        for token, rc in holders.items():
                            rr = self._robot(rc)  # type: ignore[arg-type]
                            if rr.holding == (p.id, token):
                                rr.holding = None
                            p.token_positions.pop(token, None)
                        role_map = team.roles
                        input_role_members = [rc for rc, role in role_map.items() if role in op.role_inputs]
                        output_rc = max(input_role_members or list(team.members), key=lambda x: (x[1], -x[0]))
                        self._robot(output_rc).holding = (p.id, op.output)
                        p.token_positions[op.output] = output_rc; p.completed_ops.add(op.id)
                        self.progress_events += 1
                        for rc, role in role_map.items():
                            self._learn_skill(rc, op.role_skills.get(role, op.kind), True)
                    else:
                        success = False
                if not success:
                    self.operation_failures += 1
                    if team is not None:
                        self.team_abort_count += 1
                        self.reject_execution_failure += 1
                        for rc, role in team.roles.items():
                            self._learn_skill(rc, op.role_skills.get(role, op.kind), False)
                if team is not None:
                    team.status = "completed" if success else "aborted"
                    self.teams.pop(team.id, None)
            elif kind == "deliver_complete":
                p = self.products[e["pid"]]; rc = e["robot"]; r = self._robot(rc)
                if success and r.holding == (p.id, e["token"]):
                    r.holding = None; p.token_positions.pop(e["token"], None)
                    p.delivered = True; p.complete_time = self.t
                    self.delivered_count += 1; self.delivered_value += p.value
                    _flow = self.t - p.spawn_time
                    self.sum_flow_time += _flow; self.delivered_flow_times.append(_flow)
                    self.sum_tardiness += max(0, self.t - p.due_time); self.progress_events += 1
                    self._learn_skill(rc, "deliver", True)
                else:
                    self.operation_failures += 1; self._learn_skill(rc, "deliver", False)
            self.active_reservations.pop(e.get("reservation", ""), None)

        for row in self.robots:
            for r in row:
                if self._free(r):
                    r.role = ""; r.op = ""
        if due:
            self._invalidate()

    def _cleanup_reservations(self) -> None:
        for rid in [rid for rid, r in self.active_reservations.items() if r.until <= self.t]:
            self.active_reservations.pop(rid, None)

    def _replenish_and_stockouts(self) -> None:
        if self.cfg.shelf_replenish_interval > 0 and self.t % self.cfg.shelf_replenish_interval == 0:
            for shelf in self.shelves.values():
                if shelf["outage_until"] > self.t:
                    continue
                for token in list(shelf["inventory"]):
                    shelf["inventory"][token] = min(self.cfg.shelf_cap, shelf["inventory"][token] + 1)
        if self.cfg.stockout_prob > 0:
            for shelf in self.shelves.values():
                for token in list(shelf["inventory"]):
                    if shelf["inventory"][token] > 0 and self.execution_rng.random() < self.cfg.stockout_prob:
                        shelf["inventory"][token] -= 1

    def _fail_robots(self) -> None:
        if self.cfg.robot_failure_prob <= 0:
            return
        for i in range(self.cfg.M):
            for j in range(self.cfg.N):
                r = self.robots[i][j]
                if self._free(r) and r.proposal_id is None and self.execution_rng.random() < self.cfg.robot_failure_prob:
                    r.failed_until = self.t + self.cfg.repair_duration; self.robot_failures += 1
                    self._clear_robot_proposal((i, j), count_churn=True); self._invalidate()

    def _spawn(self, force: bool = False) -> None:
        # Batch mode releases a FIXED population of exactly batch_Z products
        # and never replenishes. Previously it kept spawning and terminated at
        # the Z-th delivery, which measures "time until the Z-th completion in
        # a continuously replenished system", not "makespan of a batch of Z
        # products". Those differ: with Z=3 a run could spawn 12 and abandon 9,
        # so any makespan or optimality-gap claim against a MILP/TAMP oracle
        # (which schedules exactly Z jobs) was comparing unlike quantities.
        if self.cfg.episode_mode == "batch" and self.spawned_count >= self.cfg.batch_Z:
            return
        if not force:
            if self.cfg.spawn_interval <= 0 or self.t % self.cfg.spawn_interval != 0:
                return
        if self.wip() >= self.cfg.max_wip:
            return
        rid = self.generation_rng.choice(self.cfg.recipe_ids); rec = self.recipe_library.get(rid)
        source_rows: Dict[str, int] = {}
        for token in rec.raw_tokens:
            rows = [row for row, shelf in self.shelves.items() if token in shelf["inventory"]]
            if not rows:
                return
            source_rows[token] = self.generation_rng.choice(rows)
        mass = self.generation_rng.uniform(self.cfg.product_mass_min, self.cfg.product_mass_max)
        slack = self.generation_rng.uniform(max(1.0, self.cfg.due_slack - self.cfg.due_slack_jitter), self.cfg.due_slack + self.cfg.due_slack_jitter)
        lb = self.flow_time_lower_bound(rid)
        p = Product(
            id=self.next_product_id, recipe_id=rid, spawn_time=self.t,
            due_time=self.t + int(math.ceil(slack * lb)), source_rows=source_rows,
            mass=mass, value=rec.product_value,
        )
        self.products[p.id] = p; self.next_product_id += 1; self.spawned_count += 1
        self._invalidate()

    def achievement_state(self, pid: int) -> ProductAchievementState:
        """Return the canonical task-level achievement state for product ``pid``.

        This is separate from material/physical state: pickup, transport and
        handoff do not change it. A resolved XOR route remains resolved even if
        its selected operation later fails, matching canonical v1's no-rework/
        no-rollback assumption.
        """
        p = self.products[pid]
        rec = self.recipe(p)
        locked = self._route_locked.get(pid, set())
        choices = []
        seen = set()
        for op in rec.operations:
            for group in op.predecessor_any or ():
                group_key = tuple(group)
                if group_key in seen:
                    continue
                seen.add(group_key)
                selected = [x for x in group if x in p.completed_ops or x in p.in_progress_ops]
                if not selected and locked:
                    remaining = [x for x in group if x not in locked]
                    if len(remaining) == 1 and any(x in locked for x in group):
                        selected = remaining
                if selected:
                    choices.append((group_key, sorted(selected)[0]))
        return ProductAchievementState(
            completed_operations=frozenset(p.completed_ops),
            resolved_xor_choices=tuple(sorted(choices)),
            delivered=bool(p.delivered),
        )

    # ------------------------------------------------------------------
    # reporting / debug state
    # ------------------------------------------------------------------
    def algorithm_support_snapshot(self):
        """Return a detached read-only superset view for algorithm wrappers.

        This is deliberately non-canonical: querying richer state does not
        expand any agent's AssemblyGrid v1 observation or action space.
        """
        from algorithm_support import build_algorithm_support_snapshot
        return build_algorithm_support_snapshot(self)

    def local_option_tables(self) -> Dict[Coord, dict]:
        cands = self.candidate_operations()
        return {
            rc: {"picks": self._pick_options_for(rc), "operations": self._operation_options_for(rc, cands)}
            for rc in ((i, j) for i in range(self.cfg.M) for j in range(self.cfg.N))
        }

    def metrics(self) -> Dict[str, float]:
        delivered = max(1, self.delivered_count)
        team_count = max(1, self.team_formation_count)
        coalition_attempts = self.team_formation_count + self.team_formation_failures
        min_clear = 0.0 if self.min_motion_clearance == float("inf") else self.min_motion_clearance
        team_loads = [r.team_uses for row in self.robots for r in row]
        mean_team_load = sum(team_loads) / max(1, len(team_loads))
        team_load_cv = 0.0
        if mean_team_load > 1e-12:
            variance = sum((x - mean_team_load) ** 2 for x in team_loads) / max(1, len(team_loads))
            team_load_cv = math.sqrt(variance) / mean_team_load
        ready_total = max(1, self.ready_task_count_sum)
        return {
            "throughput": self.throughput(),
            "value_throughput": self.delivered_value / max(1, self.t),
            "completion_rate": self.completion_rate(),
            # Point 13: conditional means must travel with their censoring
            # context, so backlog and quantiles are reported alongside them.
            "unfinished_count": float(self.unfinished_count()),
            **{f"unfinished_{k}": v for k, v in self.unfinished_age_stats().items()},
            **{f"flow_time_{k}": v for k, v in self.flow_time_quantiles().items()},
            # Optional weighted diagnostic profile; canonical v1 productive
            # concurrency is the unweighted K-based measure below.
            "productive_weight_profile": self.PRODUCTIVE_WEIGHT_PROFILE,
            # Point 17: rejection taxonomy.
            "reject_locally_invalid": float(self.reject_locally_invalid),
            "reject_unmatched_proposal": float(self.reject_unmatched_proposal),
            "reject_motion_infeasible": float(self.reject_motion_infeasible),
            "reject_execution_failure": float(self.reject_execution_failure),
            "mean_flow_time": self.mean_flow_time(),
            "restricted_mean_flow_time": self.restricted_mean_flow_time(),
            "mean_tardiness": self.mean_tardiness(),
            "makespan": self.makespan(),
            "batch_makespan": self.batch_makespan(),
            "mean_wip": self.sum_wip / max(1, self.t),
            "current_wip": float(self.wip()),
            "robot_occupancy": self.sum_robot_occupancy / max(1, self.parallelism_samples),
            "cooperative_concurrency_mean": self.sum_k_coop_realized / max(1, self.parallelism_samples),
            "blocked_ready_task_rate": (self.ready_task_count_sum - self.ready_task_with_team_sum) / ready_total,
            "ready_task_feasible_team_fraction": self.ready_task_with_team_sum / ready_total,
            "handoffs_per_delivered_product": self.handoff_count / delivered,
            "K_feasible_star_mean": self.sum_k_feasible_star / max(1, self.parallelism_samples),
            "K_realized_mean": self.sum_k_realized / max(1, self.parallelism_samples),
            # Canonical unweighted productive-operation concurrency. Every
            # operation candidate advances recipe achievement on successful
            # completion; enabling logistics are not counted in K.
            "productive_concurrency_capacity_mean": self.sum_k_feasible_star / max(1, self.parallelism_samples),
            "productive_concurrency_realized_mean": self.sum_k_realized / max(1, self.parallelism_samples),
            # Optional weighted diagnostic retained for research extensions.
            "P_productive_star_mean": self.sum_p_productive_star / max(1, self.parallelism_samples),
            "P_realized_mean": self.sum_p_realized / max(1, self.parallelism_samples),
            # exact-only utilizations (None when not measurable), plus the
            # separately-named upper bounds from approximate-capacity states
            "utilization_semantics": (
                "productive_concurrency_utilization (and its parallelism_utilization alias) "
                "is averaged over EXACT-capacity opportunity states only. States with no "
                "operation opportunity are excluded. Approximate-capacity states are "
                "reported separately as *_approx_upper_bound. Weighted P-based quantities "
                "are optional diagnostics, not the canonical v1 productive-parallelism KPI."
            ),
            "parallelism_utilization": self.concurrency_utilization(),
            "productive_concurrency_utilization": self.productive_concurrency_utilization(),
            "parallelism_utilization_approx_upper_bound": self.concurrency_utilization_approx_bound(),
            "productive_concurrency_utilization_approx_upper_bound": self.productive_concurrency_utilization_approx_bound(),
            # Optional weighted research diagnostics. Keep the old ambiguous
            # names as compatibility aliases, but do not treat them as canonical.
            "weighted_productive_parallelism_utilization": self.weighted_productive_utilization(),
            "weighted_productive_parallelism_utilization_approx_upper_bound": self.weighted_productive_utilization_approx_bound(),
            "productive_parallelism_utilization": self.weighted_productive_utilization(),
            "productive_parallelism_utilization_approx_upper_bound": self.weighted_productive_utilization_approx_bound(),
            "parallelism_exact_opportunity_samples": float(self.k_opportunity_samples),
            "parallelism_approx_opportunity_samples": float(self.k_opportunity_samples_approx),
            "capacity_exact_fraction": self.capacity_exact_samples / max(1, self.capacity_exact_samples + self.capacity_approx_samples),
            "team_formation_latency": self.team_formation_latency_sum / team_count,
            "coalition_formation_latency": (
                self.team_formation_latency_sum / self.team_formation_count
                if self.team_formation_count else None
            ),
            "coalition_success_rate": (
                self.team_formation_count / coalition_attempts if coalition_attempts else None
            ),
            "team_formation_failures": float(self.team_formation_failures),
            "team_abort_rate": self.team_abort_count / team_count,
            "team_proposal_churn": float(self.team_proposal_churn),
            "mean_team_size": self.team_size_sum / team_count,
            "team_load_cv": team_load_cv,
            "role_switches": float(sum(r.role_switches for row in self.robots for r in row)),
            "trajectory_conflicts": float(self.trajectory_conflict_count),
            "geometry_rejection_count": int(self.reject_motion_infeasible),
            "geometry_conflict_count": int(self.trajectory_conflict_count),
            "blocked_activity_count": int(self.trajectory_conflict_count),
            "geometry_block_count": int(self.geometry_block_count),
            "resource_block_count": int(self.resource_block_count),
            "robot_block_count": int(self.robot_block_count),
            "geometry_delay_ticks": self.geometry_delay_ticks,
            "motion_path_length": self.motion_path_length,
            "motion_duration_total": self.motion_duration_total,
            "motion_planning_seconds": self.motion_planning_seconds,
            "minimum_reach_margin": min_clear,
            "energy_proxy": self.energy_proxy,
            "safety_violations": float(self.safety_violations),
            "operation_failures": float(self.operation_failures),
            "handoff_failures": float(self.handoff_failures),
            "robot_failures": float(self.robot_failures),
            "deadlock_ticks": float(self.deadlock_ticks),
            "invalid_action_attempts": float(self.invalid_attempts),
        }
