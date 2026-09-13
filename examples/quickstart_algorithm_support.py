#!/usr/bin/env python3
"""Inspect the detached non-canonical Algorithm Support Interface."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "Experiments" / "env"
sys.path.insert(0, str(ENV))

from benchmark import track_config
from core import AssemblyGridCore

core = AssemblyGridCore(track_config("AG-Core", M=2, N=2, horizon_T=20, max_wip=2))
core.reset(seed=0)
snapshot = core.algorithm_support_snapshot()
print("groups:", snapshot.available_groups)
print("canonical keys:", snapshot.to_dict(["canonical"]).keys())
print("privileged keys:", snapshot.to_dict(["privileged"]).keys())
