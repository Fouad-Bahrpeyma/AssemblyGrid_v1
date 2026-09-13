"""Tests for audit Points 13 (censoring), 14 (U_prod weight), 17 (rejections)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import AssemblyGridCore, EnvConfig
from policies import Tier1Policy


def _run(steps=150, **kw):
    cfg = EnvConfig(M=6, N=8, spawn_interval=2, max_wip=10, motion_planning=False,
                    horizon_T=steps, seed=1, **kw)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    return core, policy.run_episode(core, max_steps=steps)


def test_conditional_means_are_reported_with_their_censoring_context():
    """Point 13: mean flow time alone is right-censored, so completion rate,
    backlog size and residual ages must be reported with it."""
    core, res = _run()
    for key in ("completion_rate", "unfinished_count", "unfinished_mean_age",
                "unfinished_max_age", "unfinished_overdue_unfinished",
                "flow_time_p50", "flow_time_p90"):
        assert key in res, f"missing censoring context: {key}"
    assert 0.0 <= res["completion_rate"] <= 1.0
    assert res["unfinished_count"] == core.unfinished_count()


def test_backlog_can_be_worse_than_the_conditional_mean_suggests():
    """The bias must be observable under SATURATION, which is the regime it
    matters in: there the abandoned products are far older than the delivered
    ones' mean flow time, so reporting the conditional mean alone would flatter
    the policy. Deliberately overloaded (spawn every tick, high WIP cap) so the
    test measures the metric's censoring behaviour rather than how good the
    current routing heuristic happens to be."""
    cfg = EnvConfig(M=6, N=8, spawn_interval=1, max_wip=24, motion_planning=False,
                    horizon_T=200, seed=1)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    res = policy.run_episode(core, max_steps=200)
    assert res["unfinished_count"] > 0, "run not saturated; cannot show censoring"
    assert res["completion_rate"] < 0.9, "run not saturated enough"
    assert res["unfinished_mean_age"] > res["mean_flow_time"], (
        "under saturation the residual backlog should be older than the "
        "delivered mean, which is precisely what the conditional mean hides")


def test_flow_time_quantiles_are_ordered_and_consistent():
    core, _res = _run()
    q = core.flow_time_quantiles((0.5, 0.9))
    assert q["p50"] <= q["p90"]
    if core.delivered_flow_times:
        assert min(core.delivered_flow_times) <= q["p50"] <= max(core.delivered_flow_times)


def test_productive_weight_profile_is_named_and_reported():
    """Optional weighted diagnostics are reproducible only when their
    non-canonical profile identifier is emitted."""
    core, res = _run()
    assert res["productive_weight_profile"] == core.PRODUCTIVE_WEIGHT_PROFILE
    assert core.PRODUCTIVE_WEIGHT_PROFILE.endswith("/v1")


def test_productive_weight_is_deterministic_for_identical_state():
    """The optional weighted diagnostic must still be deterministic for an
    identical state when it is explicitly requested."""
    core, _ = _run(steps=60)
    for p in core.products.values():
        rec = core.recipe(p)
        for op in rec.operations:
            a = core._productive_weight(p, op)
            b = core._productive_weight(p, op)
            assert a == b and a > 0.0
            return


def test_rejection_categories_are_counted_separately():
    """Point 17: a failed coordination attempt must be attributable to a
    category, not lumped into one opaque counter."""
    core, res = _run()
    for key in ("reject_locally_invalid", "reject_unmatched_proposal",
                "reject_motion_infeasible", "reject_execution_failure"):
        assert key in res and res[key] >= 0.0
    # a loaded decentralized run should show real wasted coordination effort
    assert res["reject_unmatched_proposal"] > 0, "expected some unmatched proposals"


def test_invalid_actions_cost_the_tick_and_are_counted():
    """Point 17: probing must not be free. An illegal action is coerced to
    idle, so the agent loses that tick AND the attempt is recorded."""
    cfg = EnvConfig(M=4, N=5, spawn_interval=3, max_wip=6, motion_planning=False, seed=0)
    core = AssemblyGridCore(cfg)
    obs = core.reset()
    rc = (0, 0)
    mask = obs[rc].action_mask
    illegal = next(a for a in range(len(mask)) if not mask[a])
    before_t, before_count = core.t, core.reject_locally_invalid
    core.step({rc: illegal})
    assert core.reject_locally_invalid == before_count + 1
    assert core.t == before_t + 1, "the tick must still elapse; probing is not free"


def test_utilization_separates_exact_from_approximate_capacity():
    """Regression: utilization was averaged over ALL states, mixing true
    ratios with ones whose denominator was only a lower bound, so a run could
    report utilization 1.0 while a large share of capacity samples were
    approximations. Exact and approximate must be reported separately, and the
    approximate one must be named as a bound."""
    cfg = EnvConfig(M=7, N=10, spawn_interval=1, max_wip=20, motion_planning=False,
                    horizon_T=150, seed=1,
                    parallelism_exact_candidate_limit=1,
                    parallelism_exact_product_limit=1,
                    parallelism_exact_combination_limit=1)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    res = policy.run_episode(core, max_steps=150)

    assert res["capacity_exact_fraction"] < 1.0, "did not reach the approximate regime"
    assert res["parallelism_approx_opportunity_samples"] > 0
    # the bound is reported under a name that cannot be mistaken for saturation
    assert "parallelism_utilization_approx_upper_bound" in res
    assert res["parallelism_utilization_approx_upper_bound"] is not None


def test_no_opportunity_states_are_excluded_from_the_utilization_average():
    """Having nothing available to do is not full utilization. Ticks where
    K*_feas == 0 must not enter the average at all (they used to be scored
    as 1.0). Verified structurally: in a nearly idle run the number of
    opportunity samples must be far smaller than the number of ticks, i.e.
    the no-opportunity ticks were skipped rather than counted as perfect."""
    ticks = 40
    cfg = EnvConfig(M=4, N=5, spawn_interval=999, max_wip=2,
                    motion_planning=False, horizon_T=ticks, seed=0)
    policy = Tier1Policy("parallel_aware", seed=0)
    core = policy.make_core(cfg)
    res = policy.run_episode(core, max_steps=ticks)

    sampled = (res["parallelism_exact_opportunity_samples"]
               + res["parallelism_approx_opportunity_samples"])
    assert sampled > 0, "no opportunity state at all; test cannot discriminate"
    assert sampled < ticks, (
        f"{sampled} opportunity samples over {ticks} ticks: no-opportunity "
        f"ticks appear to be counted instead of skipped")
    # and whatever it reports must be a real ratio, never a placeholder
    util = res["parallelism_utilization"]
    assert util is None or 0.0 <= util <= 1.0
