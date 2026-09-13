from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ENV = Path(__file__).resolve().parents[1]
ROOT = ENV.parents[1]
sys.path.insert(0, str(ENV))

from core import AssemblyGridCore, EnvConfig
from motion import role_targets
from policies import Tier1Policy
from profiles import apply_geometry_profile
from recipe import load_recipe_json
from standards import (
    MANDATORY_METRIC_GROUPS,
    build_result_record,
    historical_instance_suite,
    matched_geometry_control_suite,
    official_instance_suite,
    official_suite_catalogue,
)


def test_all_public_recipe_documents_pass_schema_and_semantic_validation():
    paths = sorted((ROOT / "Experiments" / "recipes").glob("*.json"))
    assert paths
    for path in paths:
        recipe = load_recipe_json(path)
        assert recipe.id


def test_recipe_loader_rejects_type_coercion_and_unknown_fields(tmp_path):
    bad = {
        "id": "bad", "raw_tokens": ["A"], "final_token": "FINAL",
        "operations": [{"id": "x", "kind": "inspect", "inputs": ["A"],
                        "output": "FINAL", "kappa": "1", "duration": 1,
                        "mystery": True}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recipe_json(path)


def test_role_targets_are_distinct_and_do_not_increase_center_reach():
    center = (0.5, 0.5)
    roles = {(0, 0): "holder", (0, 1): "operator", (1, 0): "support", (1, 1): "inspector"}
    bases = {rc: (float(rc[1]), float(rc[0])) for rc in roles}
    targets = role_targets(center, roles, 0.1, bases)
    assert len(set(targets.values())) == 4
    for rc, target in targets.items():
        base = bases[rc]
        centre_distance = ((center[0] - base[0]) ** 2 + (center[1] - base[1]) ** 2) ** 0.5
        target_distance = ((target[0] - base[0]) ** 2 + (target[1] - base[1]) ** 2) ** 0.5
        assert target_distance < centre_distance


def test_official_suite_has_three_families_three_levels_and_constructs():
    cells = official_instance_suite(0)
    assert len(cells) == 9
    assert {(x["family"], x["difficulty"]) for x in cells} == {
        (family, level)
        for family in ("flow", "coalition", "concurrency")
        for level in ("easy", "medium", "hard")
    }
    for cell in cells:
        core = AssemblyGridCore(cell["config"])
        core.reset()


def test_historical_geometry_suite_still_constructs_but_is_separate():
    cells = historical_instance_suite(0)
    assert len(cells) == 3
    assert {x["family"] for x in cells} == {"geometry"}
    assert {x["family"] for x in official_instance_suite(0)}.isdisjoint({"geometry"})
    for cell in cells:
        AssemblyGridCore(cell["config"]).reset()


def test_official_catalogue_hashes_are_stable():
    a = official_suite_catalogue(range(2))
    b = official_suite_catalogue(range(2))
    assert a == b
    assert len(a["realizations"]) == 18
    assert not any("-geometry-" in x["instance_id"] for x in a["realizations"])
    assert all(len(x["instance_sha256"]) == 64 for x in a["realizations"])


def test_core_emits_all_mandatory_metric_fields():
    cfg = EnvConfig(M=4, N=5, horizon_T=20, max_wip=2, spawn_interval=8, seed=0)
    policy = Tier1Policy("greedy", seed=0)
    result = policy.run_episode(policy.make_core(cfg), max_steps=20)
    required = {name for group in MANDATORY_METRIC_GROUPS.values() for name in group}
    # canonical result name `delivered` is runner metadata, the others are core metrics
    assert required - {"delivered"} <= set(result)
    assert "delivered" in result


def test_result_record_contains_information_and_seed_provenance():
    cell = official_instance_suite(0)[0]
    cfg = cell["config"]
    policy = Tier1Policy("greedy", seed=7)
    metrics = policy.run_episode(policy.make_core(cfg), max_steps=cfg.horizon_T)
    record_metrics = dict(metrics)
    record_metrics["coalition_formation_latency"] = metrics["coalition_formation_latency"]
    record = build_result_record(
        benchmark_version="1.0.0", cfg=cfg, instance_id=cell["instance_id"],
        family=cell["family"], difficulty=cell["difficulty"], method_name="greedy",
        execution_information="local", training_information="not_applicable",
        algorithm_seed=7, environment_steps=0, evaluation_episodes=1,
        fidelity="abstract-v1", status="complete", metrics=record_metrics,
    )
    assert record["seeds"] == {"generation": 10000, "execution": 20000, "algorithm": 7}
    assert record["method"]["execution_information"] == "local"


def test_catalogue_difficulty_is_predeclared_within_family_not_rollout_scored():
    catalogue = official_suite_catalogue(range(1))
    for row in catalogue["realizations"]:
        difficulty = row["structural_difficulty"]
        assert difficulty["level"] == row["difficulty"]
        assert difficulty["comparison_scope"] == "within_family"
        assert "basis" in difficulty
        assert "score" not in difficulty


def test_ur10_case_is_a_dimensionless_nonofficial_mapping():
    cfg = apply_geometry_profile(EnvConfig(), "ur10-case-v1")
    assert cfg.reference_reach == 1.0
    assert cfg.physical_reference_reach_m == 1.3
    assert cfg.cell_spacing == pytest.approx(1.2 / 1.3)
    with pytest.raises(ValueError, match="non-canonical"):
        AssemblyGridCore(EnvConfig(**{**cfg.__dict__, "official_result": True}))


def test_published_json_schemas_have_stable_urn_ids_and_recipe_discriminator():
    recipe_schema = json.loads((ROOT / "schemas" / "recipe-v1.schema.json").read_text(encoding="utf-8"))
    result_schema = json.loads((ROOT / "schemas" / "result-v1.schema.json").read_text(encoding="utf-8"))
    assert recipe_schema["$id"] == "urn:assemblygrid:schema:recipe:v1"
    assert result_schema["$id"] == "urn:assemblygrid:schema:result:v1"
    assert "schema" in recipe_schema["required"]
    assert recipe_schema["properties"]["schema"]["const"] == "assemblygrid-recipe-v1"
    assert result_schema["additionalProperties"] is False


def test_geometry_aware_choice_is_explicit_not_slot_ordering():
    cfg = EnvConfig(M=5, N=6, recipe_ids=("supported_insert",), horizon_T=80, seed=3)
    core = AssemblyGridCore(cfg)
    obs = core.reset()
    # Stable slot order is semantic identity wherever proposal support ties.
    for view in obs.values():
        opts = list(view.operation_options)
        tied = [x for x in opts if x.proposal_support == 0]
        assert [x.candidate_id for x in tied] == sorted(x.candidate_id for x in tied)


def test_frozen_geometry_family_is_composite_not_mislabelled_as_radius_only():
    cells = {c["difficulty"]: c["config"] for c in historical_instance_suite(0) if c["family"] == "geometry"}
    assert {cells[d].workspace_conflict_radius for d in cells} == {0.12, 0.25, 0.40}
    # Preserve the historical/frozen RC cells, but make the confound explicit.
    assert len({(cells[d].max_wip, cells[d].spawn_interval) for d in cells}) > 1


def test_matched_geometry_control_changes_only_declared_conflict_variant():
    from dataclasses import asdict
    cells = matched_geometry_control_suite(0)
    assert [c["difficulty"] for c in cells] == ["easy", "medium", "hard"]
    configs = [asdict(c["config"]) for c in cells]
    radii = [cfg.pop("workspace_conflict_radius") for cfg in configs]
    variants = [cfg.pop("geometry_variant") for cfg in configs]
    assert radii == [0.12, 0.25, 0.40]
    assert variants == ["conflict-low", "conflict-medium", "conflict-high"]
    assert configs[0] == configs[1] == configs[2]
    assert all(c["config"].official_result is False for c in cells)
