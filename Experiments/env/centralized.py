"""Centralized dispatching baseline and centralized motion-aware reference
(paper Sec. 10, items 2 and 4).

Terminology: the motion-aware reference below is NOT a task-and-motion
planning oracle. It performs centralized allocation with motion feasibility
CHECKING, but it does not jointly optimise task assignment and motion, and it
offers no optimality guarantee. Calling it a "TAMP oracle" would overstate
it. A genuine TAMP oracle would require an integrated task-and-motion solver.

Both reuse the environment's own privileged-state machinery
(``feasible_operation_candidates`` / ``_optimal_select``), which already
computes a global-conflict-checked candidate set and an exact or greedy
max-count/max-value independent-set selection over it for the
ConcurrencyUtilization diagnostic. That is exactly what a "global greedy
allocation with privileged state" baseline needs: this module turns that
one-shot diagnostic into a running POLICY that decides, every tick, which
operation coalitions the whole system will attempt to form.

Design choice, stated explicitly so the comparison stays fair: only the
COALITION-FORMATION decision is centralized. Picks, handoffs, and delivery
still fall back to the same local greedy rule the decentralized baselines
use (``policies._greedy_action``), and every committed team still goes
through the environment's normal proposal/vote/commit mechanics. This
isolates exactly the claimed difference between centralized and
decentralized coordination (the paper's stated purpose, "to quantify the
decentralization gap") rather than conflating it with a different pick/
handoff heuristic.

The motion-aware reference (item 4) is this same policy run with
motion_planning and trajectory_conflicts enabled, on small instances, in
batch (fixed-Z) mode so it reports feasibility and makespan rather than
throughput -- matching the paper's description: "small motion-aware instances using
centralized allocation plus motion planning, providing feasibility and
makespan references rather than a scalable competitor."
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, Optional

from core import (
    ACTION_IDLE, ACTION_OPERATION_BASE, AssemblyGridCore, Coord, EnvConfig,
    Observation,
)
from policies import PriorityFn, _greedy_action, priority_fifo


class CentralizedDispatchPolicy:
    """Global greedy allocation with privileged state (Sec. 10, item 2).

    Coalition formation is decided once per tick from the environment's own
    full-state feasible-candidate set, via a weighted max-independent-set
    selection (``_optimal_select``, exact on small states, falls back to a
    greedy approximation beyond the configured exactness limits -- the same
    fallback the paper's own ConcurrencyUtilization diagnostic uses). Every
    other action (pick / handoff / deliver / idle) uses the same local
    greedy rule as the decentralized Tier-1 baselines, so the only thing
    that differs from a decentralized baseline is who decides which
    coalitions form.
    """

    name = "centralized_greedy"

    def __init__(self, seed: int = 0, weighted: bool = True,
                 priority_fn: PriorityFn = priority_fifo):
        self.rng = random.Random(seed)
        self.weighted = weighted
        self.priority_fn = priority_fn

    def make_core(self, cfg: EnvConfig) -> AssemblyGridCore:
        return AssemblyGridCore(cfg, priority_fn=self.priority_fn)

    def act(self, core: AssemblyGridCore, observations: Dict[Coord, Observation]) -> Dict[Coord, int]:
        cands = core.feasible_operation_candidates()
        selected, _score, _exact = core._optimal_select(cands, weighted=self.weighted)
        selected_ids = {c.id for c in selected}

        actions: Dict[Coord, int] = {}
        for rc, obs in observations.items():
            action = None
            # A robot that already voted into a coalition proposal must keep
            # voting for that SAME candidate while the vote is still open:
            # commitment requires every member to agree within one lock
            # window (core.py's set(prop.votes) == set(cand.coalition)
            # check), and _optimal_select recomputes from scratch every
            # tick, so without this the centralized choice can flip a
            # member's vote before the rest of its coalition ever catches
            # up, and no cooperative team ever actually commits.
            if obs.proposal_id:
                for k, opt in enumerate(obs.operation_options):
                    if opt.candidate_id == obs.proposal_id and obs.action_mask[ACTION_OPERATION_BASE + k]:
                        action = ACTION_OPERATION_BASE + k
                        break
            if action is None and selected_ids:
                for k, opt in enumerate(obs.operation_options):
                    if opt.candidate_id in selected_ids and obs.action_mask[ACTION_OPERATION_BASE + k]:
                        action = ACTION_OPERATION_BASE + k
                        break
            if action is None:
                # Not part of a centrally chosen coalition this tick: fall
                # back to the same local greedy rule the decentralized
                # baselines use for picks / handoffs / delivery / idling.
                action = _greedy_action(core, rc, obs, self.rng, parallel_aware=True)
            actions[rc] = action
        return actions

    def run_episode(self, core: AssemblyGridCore, max_steps: int = 100_000) -> dict:
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


def run_motion_aware_reference(cfg: EnvConfig, Z: int, seed: int = 0, max_ticks: int = 20_000) -> dict:
    """Centralized motion-aware reference (Sec. 10, item 4): centralized
    allocation plus motion feasibility checking on a small fixed-Z batch
    instance, reporting feasibility and makespan.

    NOT an optimising task-and-motion planner: it provides a reference
    trajectory and makespan, not a bound. Named accordingly.

    Not a scalable competitor -- intended for small instances only, exactly
    as the paper specifies. Forces motion_planning / trajectory_conflicts on
    regardless of the passed config, since a motion-aware reference without
    motion feasibility checking is just the centralized dispatcher (item 2).
    """
    tamp_cfg = replace(cfg, motion_planning=True, trajectory_conflicts=True,
                       episode_mode="batch", batch_Z=Z, horizon_T=max_ticks, seed=seed)
    policy = CentralizedDispatchPolicy(seed=seed)
    core = policy.make_core(tamp_cfg)
    obs = core.reset()
    reached = False
    for _ in range(max_ticks):
        obs, _reward, terminated, truncated, _ = core.step(policy.act(core, obs))
        if core.delivered_count >= Z:
            reached = True
            break
        if terminated or truncated:
            break
    return {
        "baseline": "centralized_motion_aware_reference", "Z": Z, "reached": reached, "makespan": core.t,
        "makespan_lower_bound": _makespan_lower_bound(core, Z),
        "delivered": core.delivered_count, "spawned": core.spawned_count,
        "unmasked_invalid_attempts": core.unmasked_invalid_attempts,
    }


def _makespan_lower_bound(core: AssemblyGridCore, Z: int) -> int:
    """Same conservative bound the paper defines: max(structural per-product
    critical path, batch robot-time lower bound), independent of the TAMP
    run itself, so it is a fair thing to compare the oracle's makespan
    against.
    """
    rec = core.recipe_library.get(core.cfg.recipe_ids[0])
    # Alternative routes are exclusive, so only ONE branch of each group is
    # ever performed. Summing every operation counts work the product never
    # does, which can push this above the true optimum and stop it being a
    # valid lower bound at all, so only the mandatory branch of each
    # alternative group is counted below.
    by_id = {o.id: o for o in rec.operations}
    optional = {q for o in rec.operations for grp in (o.predecessor_any or ())
                for q in grp if q in by_id}
    mandatory_time = sum(op.duration * op.kappa for op in rec.operations
                         if op.id not in optional)
    cheapest_alt = 0
    for o in rec.operations:
        for grp in (o.predecessor_any or ()):
            members = [by_id[q] for q in grp if q in by_id]
            if members:
                cheapest_alt += min(m.duration * m.kappa for m in members)
    robot_time = (
        len(rec.raw_tokens) * core.cfg.pick_duration
        + mandatory_time + cheapest_alt
        + core.cfg.deliver_duration
    )
    cap = max(1, core.cfg.M * core.cfg.N)
    return int(max(core.flow_time_lower_bound(), (Z * robot_time + cap - 1) // cap))


# Backwards-compatible alias. The old name overstated what this computes;
# prefer run_motion_aware_reference in new code.
run_tamp_oracle = run_motion_aware_reference
