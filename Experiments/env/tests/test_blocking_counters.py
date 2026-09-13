"""Blocked-activity counters: the aggregate splits exactly by cause."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policies import Tier1Policy
from standards import official_instance_suite

from dataclasses import replace


def _run(cfg, ticks=200):
    pol = Tier1Policy("greedy", seed=0)
    core = pol.make_core(cfg)
    core.reset()
    obs = core._all_observations() if hasattr(core, "_all_observations") else None
    mm = pol.run_episode(core, max_steps=ticks)
    return core, mm


def _cell(family, difficulty):
    for cell in official_instance_suite(0):
        if cell["family"] == family and cell["difficulty"] == difficulty:
            return cell["config"]
    raise AssertionError("scenario not found")


def test_aggregate_equals_sum_of_causes():
    cfg = _cell("concurrency", "hard")
    core, mm = _run(cfg)
    assert core.trajectory_conflict_count == (
        core.geometry_block_count + core.resource_block_count + core.robot_block_count)
    assert mm["blocked_activity_count"] == mm["geometry_conflict_count"], (
        "the legacy aggregate key must keep reporting the aggregate")
    assert core.trajectory_conflict_count > 0, "no blocking observed; test is vacuous"


def test_geometry_cause_is_zero_when_geometry_is_disabled():
    """With workspace conflicts disabled, no blocked activity may be attributed to geometry.

    The aggregate may still be positive, because a committed activity can hold a robot or
    exhaust a resource; that blocking is not geometric and must not be counted as such.
    """
    cfg = replace(_cell("concurrency", "hard"), trajectory_conflicts=False,
                  profile_enforced=False, official_result=False)
    core, _mm = _run(cfg)
    assert core.geometry_block_count == 0, "geometric cause reported with geometry disabled"
    assert core.trajectory_conflict_count == (
        core.resource_block_count + core.robot_block_count)


def test_resource_exhaustion_outranks_geometry():
    """A blocked activity that is also short of resource capacity is not geometry-only.

    Exhaustion of a resource whose capacity exceeds one is not a pairwise conflict, so it is
    detected by the availability check rather than by the reservation scan; the classifier must
    still prefer it over the geometric cause.
    """
    cfg = _cell("coalition", "medium")
    pol = Tier1Policy("greedy", seed=0)
    core = pol.make_core(cfg)
    obs = core.reset()
    cands = ()
    for _ in range(200):
        cands = core.candidate_operations()
        if cands:
            break
        obs, _r, term, trunc, _ = core.step(pol.act(core, obs))
        if term or trunc:
            break
    assert cands, "no operation candidate ever appeared; test is vacuous"
    cand = cands[0]
    core._resources_available = lambda *a, **k: False
    reason = core._blocking_reason(cand.coalition, cand.resources, cand.motion_plan)
    assert reason == "resource", f"expected resource precedence, got {reason!r}"
