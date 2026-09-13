"""Thin selectors over AssemblyGrid's non-canonical algorithm-support snapshot.

These helpers do not create new environment semantics; they merely select
read-only information groups commonly used by algorithm families.
"""
from __future__ import annotations


def canonical_view(core) -> dict:
    """IPPO-style: canonical decentralized information only."""
    return core.algorithm_support_snapshot().to_dict(["metadata", "canonical"])


def ctde_view(core) -> dict:
    """MAPPO/QMIX-style: canonical actor inputs plus privileged critic/mixer state."""
    return core.algorithm_support_snapshot().to_dict(["metadata", "canonical", "privileged"])


def graph_view(core) -> dict:
    """GNN-style structured entities, relations, process graph and candidates."""
    return core.algorithm_support_snapshot().to_dict(
        ["metadata", "entities", "topology", "recipes", "candidates", "resources"]
    )


def motion_view(core) -> dict:
    """Explicit motion-aware extension view; not part of canonical v1 observation."""
    return core.algorithm_support_snapshot().to_dict(
        ["metadata", "canonical", "entities", "motion", "resources"]
    )


def full_research_view(core) -> dict:
    """All environment-derived groups for analysis/new wrappers."""
    return core.algorithm_support_snapshot().to_dict()
