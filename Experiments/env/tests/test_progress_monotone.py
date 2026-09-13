"""Invariant I-Progress: a product's completed operations never shrink across transitions."""
from __future__ import annotations

from policies import Tier1Policy
from standards import official_instance_suite


def test_progress_is_monotone():
    checked = 0
    for cell in official_instance_suite(0):
        if cell["difficulty"] != "medium":
            continue
        pol = Tier1Policy("parallel_aware", seed=0)
        core = pol.make_core(cell["config"])
        obs = core.reset()
        seen = {}
        for _ in range(250):
            obs, _r, term, trunc, _ = core.step(pol.act(core, obs))
            for pid, product in core.products.items():
                done = set(product.completed_ops)
                assert seen.get(pid, set()) <= done, (
                    f"{cell['family']}: product {pid} lost completed operations")
                seen[pid] = done
                checked += len(done)
            if term or trunc:
                break
    assert checked > 0, "no operation ever completed; test is vacuous"
