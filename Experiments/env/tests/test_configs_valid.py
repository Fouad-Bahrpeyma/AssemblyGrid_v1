"""Every shipped experiment config must actually load and be runnable.

Regression: all 33 configs carried `topology_edge_keep`, a field removed with
the retired sparse_moore topology, so every documented `--config` run crashed
with "unexpected keyword argument". They were advertised as reproducible
presets while none of them loaded. This test loads every config on every run
so a schema change can never silently orphan them again.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ENV))

from benchmark import load_config
from core import AssemblyGridCore, TOPOLOGY_MAX_ARITY
from recipe import default_recipe_library

CONFIG_DIR = _ENV.parent / "configs"
CONFIG_FILES = sorted(glob.glob(str(CONFIG_DIR / "*.json")))


def test_config_directory_is_not_empty():
    assert CONFIG_FILES, f"no configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=[Path(p).name for p in CONFIG_FILES])
def test_config_loads(path):
    cfg = load_config(Path(path))
    cfg.validate()


@pytest.mark.parametrize("path", CONFIG_FILES, ids=[Path(p).name for p in CONFIG_FILES])
def test_config_topology_can_host_its_recipe_arities(path):
    """Only the REQUIRED arity has to be realizable. k_max is an optional
    ceiling that the runtime clamps to the topology capacity, so a recipe
    allowing up to 4 robots may legitimately run with 2 on a clique-2
    topology. Asserting `arity_max <= cap` encoded the older, stricter
    semantics and would reject a perfectly valid config."""
    cfg = load_config(Path(path))
    lib = default_recipe_library()
    cap = cfg.max_coalition_size
    for rid in cfg.recipe_ids:
        for op in lib.get(rid).operations:
            assert op.kappa <= cap, (
                f"{Path(path).name}: recipe {rid!r} operation {op.id!r} REQUIRES "
                f"at least {op.kappa} robots, but topology {cfg.topology!r} "
                f"admits at most {cap}, so it can never be performed")


@pytest.mark.parametrize("path", CONFIG_FILES[:6], ids=[Path(p).name for p in CONFIG_FILES[:6]])
def test_config_actually_constructs_an_environment(path):
    """Loading is necessary but not sufficient; the config must build a core."""
    cfg = load_config(Path(path))
    core = AssemblyGridCore(cfg)
    core.reset()
    assert core.cfg.M > 0 and core.cfg.N > 0


def test_no_config_references_a_retired_topology():
    import json
    for path in CONFIG_FILES:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        assert raw.get("topology") in TOPOLOGY_MAX_ARITY, (
            f"{Path(path).name}: topology {raw.get('topology')!r} is not admissible")
        assert "topology_edge_keep" not in raw, (
            f"{Path(path).name}: still carries the retired topology_edge_keep field")
