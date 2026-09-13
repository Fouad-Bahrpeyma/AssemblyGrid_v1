from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_version_and_machine_readable_metadata_agree():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    meta = json.loads((ROOT / "benchmark_metadata.json").read_text(encoding="utf-8"))
    assert version == "1.0.0"
    assert meta["version"] == version
    assert meta["canonical_version_tag"] == f"v{version}"
    assert meta["canonical_geometry_profile"] == "abstract-v1"
    assert [a["name"] for a in meta["authors"]] == [
        "Fouad Bahrpeyma", "David Heik", "Dirk Reichelt"
    ]


def test_public_documentation_and_examples_are_present():
    required = [
        "README.md",
        "CITATION.cff",
        "examples/quickstart_core.py",
        "examples/quickstart_official.py",
        "examples/quickstart_algorithm_support.py",
        "requirements-v1-lock.txt",
        "schemas/recipe-v1.schema.json",
        "schemas/result-v1.schema.json",
        "official/official_realizations.json",
    ]
    missing = [x for x in required if not (ROOT / x).is_file()]
    assert not missing, missing
    for rel in [x for x in required if x.endswith(".py")]:
        compile((ROOT / rel).read_text(encoding="utf-8"), rel, "exec")


def test_exact_dependency_lock_declares_the_reproduction_stack():
    lock = (ROOT / "requirements-v1-lock.txt").read_text(encoding="utf-8")
    for name in ("numpy", "torch", "gymnasium", "pettingzoo", "pytest"):
        assert any(line.lower().startswith(name + "==") for line in lock.splitlines())

