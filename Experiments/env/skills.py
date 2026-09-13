"""Reusable manipulation/cooperation skill descriptors and bookkeeping.

The benchmark does not hard-code a particular option-learning algorithm.  It
provides option-like skill metadata, per-robot proficiency, execution counts,
and a simple reference adaptation rule so skill transfer/generalization can be
measured consistently by external learning code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class SkillSpec:
    id: str
    arity: int
    initiation: str
    termination: str
    applicability: str
    required_tools: Tuple[str, ...] = ()


DEFAULT_SKILLS: Dict[str, SkillSpec] = {
    "pick": SkillSpec("pick", 1, "free gripper and reachable shelf", "token acquired", "raw-part acquisition", ("gripper",)),
    "handoff": SkillSpec("handoff", 2, "neighboring compatible robots", "ownership transferred", "bilateral transfer", ("gripper",)),
    "align": SkillSpec("align", 2, "ready two-input operation", "aligned output produced", "paired alignment", ("gripper",)),
    "assemble": SkillSpec("assemble", 4, "ready coalition operation", "subassembly output produced", "multi-arm hold/insert/stabilize", ("gripper",)),
    "inspect": SkillSpec("inspect", 1, "inspectable token", "inspection output produced", "single-arm inspection"),
    "deliver": SkillSpec("deliver", 1, "final token at output boundary", "product delivered", "conveyor placement", ("gripper",)),
}


def initial_skill_levels(skill_ids: Iterable[str] = DEFAULT_SKILLS, value: float = 1.0) -> Dict[str, float]:
    return {sid: float(value) for sid in skill_ids}


def update_proficiency(level: float, success: bool, learning_rate: float) -> float:
    """Small deterministic reference adaptation rule, bounded to [0,1]."""
    target = 1.0 if success else 0.0
    return min(1.0, max(0.0, level + learning_rate * (target - level)))
