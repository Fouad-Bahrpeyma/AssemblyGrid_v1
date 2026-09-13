"""Named geometry/motion profiles for AssemblyGrid v1.

Profiles own the feasibility-affecting geometry values.  ``abstract-v1`` is
canonical and uses dimensionless reference coordinates; it is not a physical robot calibration.
``ur10-demo-v1`` is an optional compatibility profile and is explicitly
non-canonical for benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class GeometryProfile:
    name: str
    canonical: bool
    description: str
    cell_spacing: float
    workspace_conflict_radius: float
    reference_reach: float
    reference_time: float
    physical_reference_reach_m: Optional[float]
    role_target_offset: float
    robot_model: Mapping[str, object]
    motion_planning: bool = True
    trajectory_conflicts: bool = True
    retreat_fraction: float = 0.25


GEOMETRY_PROFILES: Dict[str, GeometryProfile] = {
    "abstract-v1": GeometryProfile(
        name="abstract-v1",
        canonical=True,
        description=(
            "Canonical AssemblyGrid v1 dimensionless geometric feasibility profile; "
            "not a URDF/IK/link-level physical validation."
        ),
        cell_spacing=1.0,
        workspace_conflict_radius=0.25,
        reference_reach=1.5,
        reference_time=1.0,
        physical_reference_reach_m=None,
        role_target_offset=0.12,
        robot_model={
            "name": "generic_ur",
            "reach_radius": 1.5,
            "payload": 10.0,
            "speed": 1.0,
            "clearance": 0.10,
            "tools": ("gripper",),
            "joint_speed": 1.0,
        },
    ),
    "ur10-demo-v1": GeometryProfile(
        name="ur10-demo-v1",
        canonical=False,
        description=(
            "Non-canonical UR10-labelled compatibility profile. These values are "
            "not an authoritative robot calibration."
        ),
        cell_spacing=1.2,
        workspace_conflict_radius=0.30,
        reference_reach=1.3,
        reference_time=1.0,
        physical_reference_reach_m=None,
        role_target_offset=0.12,
        robot_model={
            "name": "ur10",
            "reach_radius": 1.3,
            "payload": 10.0,
            "speed": 1.0,
            "clearance": 0.15,
            "tools": ("gripper",),
            "joint_speed": 2.09,
        },
    ),
    "ur10-case-v1": GeometryProfile(
        name="ur10-case-v1",
        canonical=False,
        description=(
            "Non-canonical UR10-derived case profile. Core values are "
            "dimensionless and physical_reference_reach_m declares the mapping."
        ),
        cell_spacing=1.2 / 1.3,
        workspace_conflict_radius=0.30 / 1.3,
        reference_reach=1.0,
        reference_time=1.0,
        physical_reference_reach_m=1.3,
        role_target_offset=0.16 / 1.3,
        robot_model={
            "name": "ur10_case",
            "reach_radius": 1.0,
            "payload": 1.0,
            "speed": 0.75,
            "clearance": 0.15 / 1.3,
            "tools": ("gripper",),
            "joint_speed": 2.09,
        },
    ),
}

# Controlled variants supported by the v1 implementation.  Variants are
# intentionally narrow: they make geometry sweeps explicit rather than
# allowing silent mutation of profile-owned fields.
DECLARED_VARIANTS: Dict[str, Dict[str, Mapping[str, object]]] = {
    "abstract-v1": {
        "spacing-0.75": {"cell_spacing": 0.75},
        "spacing-1.00": {"cell_spacing": 1.00},
        "spacing-1.25": {"cell_spacing": 1.25},
        "conflict-low": {"workspace_conflict_radius": 0.12},
        "conflict-medium": {"workspace_conflict_radius": 0.25},
        "conflict-high": {"workspace_conflict_radius": 0.40},
    },
    "ur10-demo-v1": {
        "motion-off": {"motion_planning": False, "trajectory_conflicts": False},
    },
    "ur10-case-v1": {
        "conflict-low": {"workspace_conflict_radius": 0.12 / 1.3},
        "conflict-high": {"workspace_conflict_radius": 0.40 / 1.3},
    },
}


def get_geometry_profile(name: str) -> GeometryProfile:
    try:
        return GEOMETRY_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown geometry profile {name!r}; choose from {sorted(GEOMETRY_PROFILES)}") from exc


def profile_values(name: str, variant: Optional[str] = None) -> dict:
    p = get_geometry_profile(name)
    values = {
        "cell_spacing": p.cell_spacing,
        "workspace_conflict_radius": p.workspace_conflict_radius,
        "reference_reach": p.reference_reach,
        "reference_time": p.reference_time,
        "physical_reference_reach_m": p.physical_reference_reach_m,
        "role_target_offset": p.role_target_offset,
        "robot_model": dict(p.robot_model),
        "motion_planning": p.motion_planning,
        "trajectory_conflicts": p.trajectory_conflicts,
        "retreat_fraction": p.retreat_fraction,
    }
    if variant:
        try:
            values.update(DECLARED_VARIANTS.get(name, {})[variant])
        except KeyError as exc:
            raise ValueError(
                f"undeclared geometry variant {variant!r} for {name!r}; "
                f"choose from {sorted(DECLARED_VARIANTS.get(name, {}))}"
            ) from exc
    return values


def is_canonical_profile(name: str) -> bool:
    return get_geometry_profile(name).canonical


def apply_geometry_profile(cfg, name: str, variant: Optional[str] = None, *, profile_enforced: bool = True):
    """Return ``cfg`` with all profile-owned fields materialized consistently."""
    from dataclasses import replace
    from core import URModel  # lazy import avoids a module-import cycle

    values = profile_values(name, variant)
    model = URModel(**values.pop("robot_model"))
    return replace(
        cfg,
        geometry_profile=name,
        geometry_variant=variant,
        profile_enforced=profile_enforced,
        robot_models=(model,),
        robot_model_assignment="homogeneous",
        **values,
    )
