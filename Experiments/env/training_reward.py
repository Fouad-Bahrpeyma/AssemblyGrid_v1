"""Optional training-reward utilities for AssemblyGrid algorithms.

AssemblyGrid's canonical benchmark definition is reward-independent.  This
module exists only to make RL/MARL implementations convenient: a training
algorithm may attach one of these signals to the environment without making
that signal part of the benchmark instance, success condition, or KPI set.

The default profile preserves the pre-AssGrDims#2 environment reward exactly
for backwards-compatible experimentation.  It is explicitly non-normative;
reference MARL studies must name and report the training reward they use.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrainingRewardConfig:
    """Algorithm-layer reward settings; never part of ``EnvConfig``.

    ``profile`` is descriptive metadata for reproducibility.  Coefficients are
    intentionally given semantic names rather than the old alpha/beta/eta/zeta
    symbols, which were easy to mistake for benchmark parameters.
    """

    profile: str = "legacy-shaped-v1"
    delivery_value_scale: float = 1.0
    tardiness_penalty: float = 0.1
    motion_energy_penalty: float = 0.0
    safety_penalty: float = 0.0
    wip_penalty: float = 0.01

    def validate(self) -> None:
        for name in (
            "delivery_value_scale",
            "tardiness_penalty",
            "motion_energy_penalty",
            "safety_penalty",
            "wip_penalty",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


def sparse_delivery_reward(**overrides) -> TrainingRewardConfig:
    """Convenience sparse delivery signal for RL experiments.

    This helper is not part of the canonical benchmark definition; it is an
    algorithm-support option that a training algorithm may choose or replace
    explicitly.
    """

    base = TrainingRewardConfig(
        profile="sparse-delivery-v1",
        delivery_value_scale=1.0,
        tardiness_penalty=0.0,
        motion_energy_penalty=0.0,
        safety_penalty=0.0,
        wip_penalty=0.0,
    )
    return replace(base, **overrides)


def load_training_reward_config(path) -> TrainingRewardConfig:
    """Load an algorithm-layer reward profile from JSON."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = TrainingRewardConfig(**data)
    cfg.validate()
    return cfg


def save_training_reward_config(cfg: TrainingRewardConfig, path) -> None:
    """Save an algorithm-layer reward profile without touching EnvConfig."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    cfg.validate()
    Path(path).write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
