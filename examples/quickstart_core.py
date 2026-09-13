#!/usr/bin/env python3
"""Minimal canonical-core smoke example."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "Experiments" / "env"
sys.path.insert(0, str(ENV))

from benchmark import track_config
from core import ACTION_IDLE, AssemblyGridCore

cfg = track_config("AG-Core", M=2, N=2, horizon_T=20, max_wip=2)
core = AssemblyGridCore(cfg)
obs = core.reset(seed=0)
print("agents:", len(obs), "actions per agent:", len(next(iter(obs.values())).action_mask))
for _ in range(3):
    joint = {rc: ACTION_IDLE for rc in obs}
    obs, reward, terminated, truncated, info = core.step(joint)
    print("t=", core.t, "delivered=", core.delivered_count, "terminated=", terminated, "truncated=", truncated)
print("delivered/throughput:", core.delivered_count, core.metrics()["throughput"])
