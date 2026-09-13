"""Fixed-size local encoding and privileged global-state encoding.

Local observations implement the canonical AssemblyGrid v1 decentralized
contract: task-level self/neighbour state, local material choices, explicit
operation-candidate attributes, deterministic categorical identity features, proposal support and action masks.
Raw joint configuration, detailed robot-model features and skills remain in
the internal/privileged state, not in the canonical decentralized interface.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from core import (
    ACTION_OPERATION_BASE, DIRECTIONS, MAX_OPERATION_OPTIONS, MAX_PICK_OPTIONS,
    NUM_ACTIONS, Observation,
)
_TOKEN_CLASSES = ("none", "raw", "intermediate", "final")

# 12 self + 8*10 neighbours + 2 option counts + pick descriptors (6 each)
# + operation descriptors (20 each). Time, current proposal identity, recipe,
# product/candidate strings are represented by deterministic compressed categorical
# features in this neural adapter. Exact semantic IDs remain on Observation and
# option objects and are the normative identity-bearing interface.
OBS_DIM = 12 + 8 * 10 + 2 + MAX_PICK_OPTIONS * 6 + MAX_OPERATION_OPTIONS * 20


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _token_class(token) -> str:
    if token is None:
        return "none"
    text = str(token)
    if text.upper().startswith("FINAL"):
        return "final"
    if len(text) == 1 and text.isalpha():
        return "raw"
    return "intermediate"


def _one_hot_token(token) -> List[float]:
    cls = _token_class(token)
    return [1.0 if x == cls else 0.0 for x in _TOKEN_CLASSES]


def _hash01(text: str) -> float:
    # Stable deterministic *compressed* categorical scalar for reference tensors.
    # Collisions are possible; environment identity and action binding never rely on it.
    acc = 0
    for ch in text:
        acc = (acc * 131 + ord(ch)) % 1009
    return acc / 1008.0


def _q01(q: Sequence[float]) -> List[float]:
    import math
    return [_clip01((float(x) + math.pi) / (2 * math.pi)) for x in q[:6]] + [0.5] * max(0, 6 - len(q))


def _model_features(features) -> List[float]:
    values = list(features)
    while len(values) < 5:
        values.append(1.0 if len(values) == 4 else 0.0)
    reach, payload, speed, clearance, joint_speed = values[:5]
    return [
        _clip01(reach / 3.0),
        _clip01(payload / 25.0),
        _clip01(speed / 3.0),
        _clip01(clearance / 0.5),
        _clip01(joint_speed / 3.0),
    ]


def observation_to_array(obs: Observation, M: int, N: int) -> np.ndarray:
    out: List[float] = []

    # Canonical self block: token4 + product id1 + busy/failed2 +
    # proposal-present1 + proposal-identity1 + grid position2 + public time1 = 12.
    out += _one_hot_token(obs.holding_kind)
    out.append(0.0 if obs.holding_pid is None else (obs.holding_pid % 256) / 255.0)
    out += [float(obs.busy), float(obs.failed)]
    out.append(1.0 if obs.proposal_id else 0.0)
    out.append(0.0 if obs.proposal_id is None else _hash01(obs.proposal_id))
    i, j = obs.robot_id
    out += [i / max(1, M - 1), j / max(1, N - 1)]
    # Bounded monotone encoding of the synchronized public tick; no episode
    # horizon is needed and exact t remains available on the Observation object.
    out.append(float(obs.t) / (1.0 + float(obs.t)))

    # Fixed Moore-direction slots. Missing topology edge/boundary = all zero.
    # Neighbour block: token4 + pid1 + busy/failed2 + op1 + role1 + exists1.
    for di, dj in DIRECTIONS:
        nc = (i + di, j + dj)
        info = obs.neighbors.get(nc)
        if info is None:
            out += [0.0] * 10
        else:
            out += _one_hot_token(info.get("holding_kind"))
            pid = info.get("holding_pid")
            out.append(0.0 if pid is None else (int(pid) % 256) / 255.0)
            out += [float(info.get("busy", False)), float(info.get("failed", False))]
            out += [_hash01(str(info.get("op", ""))), _hash01(str(info.get("role", ""))), 1.0]

    out += [len(obs.pick_options) / MAX_PICK_OPTIONS,
            len(obs.operation_options) / MAX_OPERATION_OPTIONS]

    for k in range(MAX_PICK_OPTIONS):
        if k >= len(obs.pick_options):
            out += [0.0] * 6
            continue
        pick = obs.pick_options[k]
        out += [
            1.0,
            _hash01(pick.token),
            _hash01(pick.recipe_id),
            pick.shelf_row / max(1, M - 1),
            _clip01(pick.stock_fraction),
            (pick.product_id % 256) / 255.0,
        ]

    for k in range(MAX_OPERATION_OPTIONS):
        if k >= len(obs.operation_options):
            out += [0.0] * 20
            continue
        op = obs.operation_options[k]
        feats = [
            1.0,
            op.kappa_min / 4.0,
            op.size / 4.0,
            _hash01(op.kind),
            _hash01(op.role),
            (op.product_id % 256) / 255.0,
            _hash01(op.operation_id),
            _hash01(op.candidate_id),
            _hash01(op.recipe_id),
            _clip01(op.motion_duration / 30.0),
            _clip01((op.min_clearance + 0.5) / 3.5),
            _clip01(op.proposal_support),
        ]
        members = list(op.members)[:4]
        for m in range(4):
            if m >= len(members):
                feats += [0.5, 0.5]
            else:
                di = members[m][0] - i; dj = members[m][1] - j
                feats += [_clip01((di + 1.0) / 2.0), _clip01((dj + 1.0) / 2.0)]
        out += feats

    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape == (OBS_DIM,), f"encoding drifted: got {arr.shape}, expected ({OBS_DIM},)"
    return arr

def action_mask_to_array(mask: List[bool]) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.int8)
    assert arr.shape == (NUM_ACTIONS,)
    return arr


def _configured_tokens(core) -> List[str]:
    tokens = set()
    for rid in core.cfg.recipe_ids:
        rec = core.recipe_library.get(rid)
        tokens.update(rec.raw_tokens)
        tokens.update(op.output for op in rec.operations)
    return sorted(tokens)


def _configured_ops(core) -> List[str]:
    return sorted({op.id for rid in core.cfg.recipe_ids for op in core.recipe_library.get(rid).operations})


def global_state_dim(core) -> int:
    # robot feature: pos2 + holding-present/pid/token 3 + busy/failed 2 + q6 + model5 + proposal1 = 19
    robot_dim = core.cfg.M * core.cfg.N * 19
    tokens = _configured_tokens(core); ops = _configured_ops(core)
    # canonical privileged product block: active, recipe, age, mass,
    # raw-progress, delivered = 6. Due date and economic value are optional
    # extension attributes and are not injected into the default CTDE state.
    # + per-op status + per-token [present,row,col] = 6 + |ops| + 3|tokens|
    product_dim = core.cfg.max_wip * (6 + len(ops) + 3 * len(tokens))
    # global time/load counters: 6
    return robot_dim + product_dim + 6


def global_state_to_array(core) -> np.ndarray:
    out: List[float] = []
    tokens = _configured_tokens(core); ops = _configured_ops(core)
    for i in range(core.cfg.M):
        for j in range(core.cfg.N):
            r = core.robots[i][j]
            out += [i / max(1, core.cfg.M - 1), j / max(1, core.cfg.N - 1)]
            if r.holding is None:
                out += [0.0, 0.0, 0.0]
            else:
                out += [1.0, (r.holding[0] % 32) / 31.0, _hash01(r.holding[1])]
            out += [float(not core._free(r)), float(core._failed(r))]
            out += _q01(r.q)
            out += _model_features((r.model.reach_radius, r.model.payload, r.model.speed, r.model.clearance, r.model.joint_speed))
            out.append(1.0 if r.proposal_id else 0.0)

    active = [p for p in core.products.values() if not p.delivered]
    active.sort(key=lambda p: (p.spawn_time, p.id))
    recipe_ids = list(core.cfg.recipe_ids)
    for slot in range(core.cfg.max_wip):
        if slot >= len(active):
            out += [0.0] * (6 + len(ops) + 3 * len(tokens))
            continue
        p = active[slot]; rec = core.recipe(p)
        recipe_idx = recipe_ids.index(p.recipe_id) if p.recipe_id in recipe_ids else 0
        raw_progress = len(p.raw_picked) / max(1, len(rec.raw_tokens))
        out += [
            1.0,
            recipe_idx / max(1, len(recipe_ids) - 1),
            _clip01((core.t - p.spawn_time) / max(1, core.cfg.horizon_T)),
            _clip01(p.mass / 25.0),
            raw_progress,
            float(p.delivered),
        ]
        for op_id in ops:
            if op_id in p.completed_ops:
                out.append(1.0)
            elif op_id in p.in_progress_ops:
                out.append(0.5)
            else:
                out.append(0.0)
        for token in tokens:
            loc = p.token_positions.get(token)
            if loc is None:
                out += [0.0, 0.0, 0.0]
            else:
                out += [1.0, loc[0] / max(1, core.cfg.M - 1), loc[1] / max(1, core.cfg.N - 1)]

    out += [
        _clip01(core.t / max(1, core.cfg.horizon_T)),
        _clip01(core.wip() / max(1, core.cfg.max_wip)),
        _clip01(core.delivered_count / max(1, core.cfg.batch_Z)),
        _clip01(core.spawned_count / max(1, core.cfg.batch_Z + core.cfg.max_wip)),
        _clip01(core.current_concurrency() / max(1, core.cfg.M * core.cfg.N)),
        _clip01(len(core.proposals) / max(1, core.cfg.M * core.cfg.N)),
    ]
    arr = np.asarray(out, dtype=np.float32)
    expected = global_state_dim(core)
    assert arr.shape == (expected,), f"global-state encoding drifted: got {arr.shape}, expected ({expected},)"
    return arr
