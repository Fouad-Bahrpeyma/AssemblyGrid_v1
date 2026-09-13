from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from benchmark import instance_definition, load_config
from core import EnvConfig, TOPOLOGY_MAX_ARITY
from official import get_official_record, instance_sha256, load_official_config


def test_coalition_capacity_is_derived_not_serialized():
    cfg = EnvConfig(topology="von_neumann")
    assert cfg.max_coalition_size == TOPOLOGY_MAX_ARITY["von_neumann"] == 2
    assert "max_coalition_size" not in asdict(cfg)


def test_instance_definition_excludes_run_only_fields_and_includes_information_structure():
    a = EnvConfig(generation_seed=7, execution_seed=9, proposal_support_visible=True)
    b = replace(a, execution_seed=99, benchmark_track="anything", official_result=False)
    assert instance_definition(a) == instance_definition(b)
    c = replace(a, proposal_support_visible=False)
    assert instance_definition(a) != instance_definition(c)


def test_legacy_config_cap_is_accepted_only_as_redundant_four(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"topology": "von_neumann", "max_coalition_size": 4}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.max_coalition_size == 2
    p.write_text(json.dumps({"max_coalition_size": 3}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_official_catalogue_hash_and_loader_if_release_exists():
    root = Path(__file__).resolve().parents[3]
    cat = root / "official" / "official_realizations.json"
    if not cat.exists():
        pytest.skip("release catalogue not built in this source tree")
    payload = json.loads(cat.read_text(encoding="utf-8"))
    record = payload["realizations"][0]
    assert record["instance_sha256"] == instance_sha256(record["instance_definition"])
    checked = get_official_record(cat, record["instance_id"])
    cfg = load_official_config(cat, checked["instance_id"])
    assert cfg.official_result is True
    assert instance_definition(cfg) == checked["instance_definition"]
