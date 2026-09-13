"""Tier-1 decentralized rule/dispatching baselines for the revised interface."""
from __future__ import annotations

import random
from typing import Callable, Dict, List

from core import (
    ACTION_DELIVER, ACTION_HANDOFF_BASE, ACTION_IDLE, ACTION_OPERATION_BASE,
    ACTION_PICK_BASE, ACTION_RECEIVE_BASE, DIRECTIONS, AssemblyGridCore, Coord,
    Observation, Product,
)

PolicyFn = Callable[[AssemblyGridCore, Coord, Observation, random.Random], int]
PriorityFn = Callable[[Product, int, AssemblyGridCore], float]


def priority_fifo(p: Product, t: int, core: AssemblyGridCore) -> float:
    return p.spawn_time


def priority_edd(p: Product, t: int, core: AssemblyGridCore) -> float:
    return p.due_time


def priority_spt(p: Product, t: int, core: AssemblyGridCore) -> float:
    return core._remaining_nominal_work(p)


def priority_critical_ratio(p: Product, t: int, core: AssemblyGridCore) -> float:
    return (p.due_time - t) / max(1, core._remaining_nominal_work(p))


def priority_bottleneck_first(p: Product, t: int, core: AssemblyGridCore) -> float:
    rec = core.recipe(p)
    remaining = [op for op in rec.operations if op.id not in p.completed_ops]
    if not remaining:
        return 999.0
    # Highest arity / longest downstream work receives highest priority.
    score = max(op.kappa * rec.downstream_critical_duration(op.id) for op in remaining)
    return -float(score)


def priority_queue_balancing(p: Product, t: int, core: AssemblyGridCore) -> float:
    stocks = []
    for token, row in p.source_rows.items():
        if token not in p.raw_picked:
            stocks.append(core.shelves[row]["inventory"].get(token, 0))
    return min(stocks) if stocks else 10**9


PRIORITIES: Dict[str, PriorityFn] = {
    "fifo": priority_fifo,
    "edd": priority_edd,
    "spt": priority_spt,
    "critical_ratio": priority_critical_ratio,
    "bottleneck_first": priority_bottleneck_first,
    "queue_balancing": priority_queue_balancing,
}


def _valid_send_dirs(obs: Observation) -> list[int]:
    return [d for d in range(8) if obs.action_mask[ACTION_HANDOFF_BASE + d]]


def _stable_slot(text: str) -> int:
    return sum((k + 1) * ord(ch) for k, ch in enumerate(str(text))) % 4


def _staging_clique(core: AssemblyGridCore, anchor: Coord, size: int) -> List[Coord]:
    """A deterministic set of `size` cells that are PAIRWISE ADJACENT under the
    configured topology, containing `anchor` where possible.

    Topology-neutrality matters here. The previous staging rule placed the
    tokens of one product on a fixed 2x2 patch, i.e. two cells offset by one
    row AND one column. Those are adjacent under Moore but NOT under von
    Neumann or line, so a coalition could never form on non-Moore topologies
    and the architecture comparison measured "does the heuristic assume
    diagonals" rather than "is this topology better". Building the staging set
    out of the actual adjacency graph removes that specific bias.

    Uses only the static, public interaction graph; no per-robot or runtime
    state, so this remains a legitimate public-geometry heuristic.
    """
    if size <= 1:
        return [anchor]
    chosen = [anchor]
    # deterministic order; prefer candidates that keep the set a clique
    for cand in sorted(core.neighbors(anchor)):
        if len(chosen) >= size:
            break
        if all(c == cand or cand in core.adjacency.get(c, ()) for c in chosen):
            chosen.append(cand)
    if len(chosen) < size:
        # topology cannot host a clique this large anywhere near the anchor;
        # fall back to adjacency-ordered cells so behaviour stays defined.
        for cand in sorted(core.neighbors(anchor)):
            if len(chosen) >= size:
                break
            if cand not in chosen:
                chosen.append(cand)
    return chosen


def _material_target(core: AssemblyGridCore, pid: int, token: str, rc: Coord) -> Coord:
    """Deterministic local staging target used by Tier-1 rules.

    All material of one product is routed toward a small staging set near the
    output side, so recipe inputs co-locate and coalition operations become
    possible. The staging set is derived from the CONFIGURED TOPOLOGY (see
    _staging_clique), not from an assumed diagonal 2x2 patch, so the rule does
    not silently favour Moore adjacency. It uses only product/token identity
    and public geometry.
    """
    if str(token).upper().startswith("FINAL"):
        return (rc[0], core.cfg.N - 1)
    if core.cfg.M <= 1:
        base_row = 0
    else:
        base_row = int(pid) % (core.cfg.M - 1)
    left = max(0, core.cfg.N - 3)
    anchor = (min(base_row, core.cfg.M - 1), min(left, core.cfg.N - 1))
    slot = _stable_slot(token)
    cells = _staging_clique(core, anchor, 4)
    return cells[slot % len(cells)]


def _grid_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _route_distance(core: AssemblyGridCore, a: Coord, b: Coord) -> int:
    """Shortest-path distance on the declared public interaction topology."""
    if a == b:
        return 0
    frontier = [a]; seen = {a}; depth = 0
    while frontier:
        depth += 1; nxt = []
        for u in frontier:
            for v in core.adjacency.get(u, ()):
                if v == b:
                    return depth
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt
    return 10**9


def _destination_congestion(core: AssemblyGridCore, obs: Observation, q: Coord) -> int:
    """Occupancy around candidate destination cell q, using only locally
    knowable information.

    Counts, over q's own topology neighbours, how many are busy or already
    carrying material. Both facts are visible to the acting robot for cells it
    can see; for any neighbour of q that this robot cannot observe, the cell is
    skipped rather than guessed, so the score never depends on privileged
    global state. Lower is better, so a route into a quieter region is
    preferred among candidates that make equal progress toward the target.
    """
    score = 0
    for nb in core.neighbors(q):
        info = obs.neighbors.get(nb)
        if info is None:
            continue  # not locally observable: contribute nothing
        if info.get("busy", False):
            score += 1
        if info.get("holding_kind") is not None:
            score += 1
    return score


def _greedy_action(core: AssemblyGridCore, rc: Coord, obs: Observation, rng: random.Random,
                   parallel_aware: bool) -> int:
    i, j = rc

    # Continue a pending local team proposal while it is locked.  Once the
    # commitment horizon has elapsed the mask exposes alternatives again; the
    # baseline then joins a candidate with stronger visible teammate support.
    if obs.proposal_id:
        same = None
        enabled = []
        for k, op in enumerate(obs.operation_options):
            if obs.action_mask[ACTION_OPERATION_BASE + k]:
                enabled.append((k, op))
            if op.candidate_id == obs.proposal_id and obs.action_mask[ACTION_OPERATION_BASE + k]:
                same = (k, op)
        if len(enabled) > 1:
            best = max(enabled, key=lambda ko: (ko[1].proposal_support, -ko[1].motion_duration, ko[1].candidate_id))
            if same is None or best[1].proposal_support > same[1].proposal_support + 1e-12:
                return ACTION_OPERATION_BASE + best[0]
        if same is not None:
            return ACTION_OPERATION_BASE + same[0]
        return ACTION_IDLE

    # A specific ready team candidate is an explicit action choice.  Option
    # order prioritizes existing locally visible proposal support.
    if obs.operation_options:
        return ACTION_OPERATION_BASE

    if obs.holding_kind is None:
        if obs.pick_options:
            return ACTION_PICK_BASE
        # Accept a transfer only when this cell makes progress toward the
        # sender material's deterministic local staging/output target.  This
        # avoids ping-pong transfers while using only locally visible
        # product/token identity.
        candidates = []
        for nc, info in obs.neighbors.items():
            token = info.get("holding_kind"); pid = info.get("holding_pid")
            if token is None or pid is None or info.get("busy"):
                continue
            delta = (nc[0] - i, nc[1] - j)
            if delta in DIRECTIONS:
                d = DIRECTIONS.index(delta)
                if obs.action_mask[ACTION_RECEIVE_BASE + d]:
                    target = _material_target(core, int(pid), str(token), nc)
                    gain = _route_distance(core, nc, target) - _route_distance(core, rc, target)
                    if gain > 0:
                        candidates.append((-gain, _route_distance(core, rc, target), d))
        if candidates:
            candidates.sort()
            return ACTION_RECEIVE_BASE + candidates[0][2]
        return ACTION_IDLE

    # Final-token classification is represented in the action mask, not by a
    # hard-coded token string.
    if obs.action_mask[ACTION_DELIVER]:
        return ACTION_DELIVER

    dirs = _valid_send_dirs(obs)
    if not dirs:
        return ACTION_IDLE

    pid = int(obs.holding_pid if obs.holding_pid is not None else 0)
    target = _material_target(core, pid, str(obs.holding_kind), rc)
    if rc == target:
        return ACTION_IDLE

    # Move only when the handoff reduces distance to the public staging target.
    # For the parallel-aware variant, congestion is a secondary tie-breaker and
    # must be evaluated PER CANDIDATE DESTINATION: it is the occupancy around
    # the cell the material would actually move INTO, which is what "prefer a
    # less congested route" has to mean.  A score taken from the acting robot's
    # own neighbourhood would be the same scalar for every candidate direction
    # and could therefore never reorder them.
    current_dist = _route_distance(core, rc, target)
    scored = []
    for d in dirs:
        di, dj = DIRECTIONS[d]; q = (i + di, j + dj)
        info = obs.neighbors.get(q, {})
        if info.get("holding_kind") is not None or info.get("busy", False):
            continue
        new_dist = _route_distance(core, q, target)
        if new_dist >= current_dist:
            continue
        congestion = _destination_congestion(core, obs, q) if parallel_aware else 0
        scored.append((new_dist, congestion, -dj, abs(di), d))
    if not scored:
        return ACTION_IDLE
    scored.sort()
    return ACTION_HANDOFF_BASE + scored[0][-1]


def policy_greedy(core: AssemblyGridCore, rc: Coord, obs: Observation, rng: random.Random) -> int:
    return _greedy_action(core, rc, obs, rng, False)


def policy_parallel_aware(core: AssemblyGridCore, rc: Coord, obs: Observation, rng: random.Random) -> int:
    return _greedy_action(core, rc, obs, rng, True)


def policy_geometry_aware(core: AssemblyGridCore, rc: Coord, obs: Observation, rng: random.Random) -> int:
    """Local task policy that explicitly selects lower-cost geometry.

    Logistics behavior matches ``parallel_aware``. For operation proposals it
    chooses by visible support, normalized motion duration, reach margin, and
    semantic identity. It never queries privileged global feasibility.
    """
    enabled = [
        (index, option)
        for index, option in enumerate(obs.operation_options)
        if obs.action_mask[ACTION_OPERATION_BASE + index]
    ]
    if enabled:
        index, _option = min(
            enabled,
            key=lambda item: (
                -item[1].proposal_support,
                item[1].motion_duration,
                -item[1].min_clearance,
                item[1].candidate_id,
            ),
        )
        return ACTION_OPERATION_BASE + index
    return _greedy_action(core, rc, obs, rng, True)


ACTION_POLICIES: Dict[str, PolicyFn] = {
    "greedy": policy_greedy,
    "parallel_aware": policy_parallel_aware,
    "geometry_aware": policy_geometry_aware,
}

TIER1_BASELINES: Dict[str, Dict[str, object]] = {
    "greedy": {"action": "greedy", "priority": "fifo"},
    "parallel_aware": {"action": "parallel_aware", "priority": "fifo"},
    "geometry_aware": {"action": "geometry_aware", "priority": "fifo"},
    "edd": {"action": "greedy", "priority": "edd"},
    "spt": {"action": "greedy", "priority": "spt"},
    "critical_ratio": {"action": "greedy", "priority": "critical_ratio"},
    "bottleneck_first": {"action": "greedy", "priority": "bottleneck_first"},
    "queue_balancing": {"action": "greedy", "priority": "queue_balancing"},
}


class Tier1Policy:
    def __init__(self, name: str, seed: int = 0):
        if name not in TIER1_BASELINES:
            raise ValueError(f"Unknown Tier-1 baseline {name!r}; choose from {sorted(TIER1_BASELINES)}")
        spec = TIER1_BASELINES[name]
        self.name = name
        self.action_fn = ACTION_POLICIES[spec["action"]]
        self.priority_fn = PRIORITIES[spec["priority"]]
        self.rng = random.Random(seed)

    def act(self, core: AssemblyGridCore, observations: Dict[Coord, Observation]) -> Dict[Coord, int]:
        return {rc: self.action_fn(core, rc, obs, self.rng) for rc, obs in observations.items()}

    def make_core(self, cfg) -> AssemblyGridCore:
        return AssemblyGridCore(cfg, priority_fn=self.priority_fn)

    def run_episode(self, core: AssemblyGridCore, max_steps: int = 100_000) -> dict:
        assert core.priority_fn is self.priority_fn, "use policy.make_core(cfg) so the dispatch priority is actually applied"
        obs = core.reset()
        for _ in range(max_steps):
            obs, _reward, terminated, truncated, _ = core.step(self.act(core, obs))
            if terminated or truncated:
                break
        result = {"baseline": self.name, "t": core.t,
                  "delivered": core.delivered_count, "spawned": core.spawned_count}
        result.update(core.metrics())
        result["unmasked_invalid_attempts"] = core.unmasked_invalid_attempts
        return result
