#!/usr/bin/env python3
"""Load and verify one official AssemblyGrid v1 realization."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "Experiments" / "env"
sys.path.insert(0, str(ENV))

from official import load_official_config

catalogue = ROOT / "official" / "official_realizations.json"
payload = json.loads(catalogue.read_text(encoding="utf-8"))
instance_id = payload["realizations"][0]["instance_id"]
cfg = load_official_config(catalogue, instance_id)
print("verified:", instance_id)
print("profile:", cfg.geometry_profile, "generation_seed:", cfg.generation_seed, "execution_seed:", cfg.execution_seed)
