"""Official AssemblyGrid realization catalogue helpers.

An official realization ID denotes an instance definition (family realization)
plus a SHA-256 digest.  Reference execution seeds and other run settings are
kept separately so they do not silently redefine what an instance is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark import env_config_from_dict, instance_definition


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def instance_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_catalogue(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_historical(instance_id: str) -> bool:
    # v4: the geometry family is a historical extension, not an official realization.
    return "-geometry-" in instance_id


def get_official_record(path, instance_id: str, *, allow_historical: bool = False) -> dict:
    if _is_historical(instance_id) and not allow_historical:
        raise KeyError(
            f"{instance_id!r} is a historical (geometry) realization, not part of the "
            f"AssemblyGrid v1 official suite; pass allow_historical=True together with the "
            f"historical_realizations.json catalogue to load it"
        )
    catalogue = load_catalogue(path)
    matches = [x for x in catalogue.get("realizations", []) if x.get("instance_id") == instance_id]
    if len(matches) != 1:
        raise KeyError(f"official realization {instance_id!r} not found uniquely")
    record = matches[0]
    expected = record.get("instance_sha256")
    actual = instance_sha256(record.get("instance_definition"))
    if expected != actual:
        raise ValueError(f"official realization {instance_id!r} failed instance SHA-256 verification")
    return record


def load_official_config(path, instance_id: str, *, allow_historical: bool = False):
    """Load and verify the catalogue's reference run configuration.

    The returned config is marked ``official_result=True`` and validates its
    canonical geometry profile.  The instance-definition digest is checked
    again against the reconstructed configuration before use.
    """
    record = get_official_record(path, instance_id, allow_historical=allow_historical)
    data = dict(record["reference_run_configuration"])
    data["official_result"] = True
    data["profile_enforced"] = True
    cfg = env_config_from_dict(data)
    cfg.validate()
    reconstructed = instance_definition(cfg)
    if reconstructed != record["instance_definition"]:
        raise ValueError(f"official realization {instance_id!r} run configuration does not match its instance definition")
    return cfg
