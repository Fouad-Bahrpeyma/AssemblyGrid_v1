"""Tests for the QMIX controller."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "marl"))

import numpy as np
import pytest
import torch

from core import EnvConfig, NUM_ACTIONS, ACTION_PICK_BASE
from encoding import global_state_dim
from qmix import (
    JointReplayBuffer, QMixMixer, QNetwork, _train_step,
    epsilon_greedy_actions, make_env, train_value_decomposition,
)


def _tiny_cfg():
    return EnvConfig(M=2, N=3, spawn_interval=3, max_wip=3, motion_planning=False,
                     horizon_T=60, recipe_ids=("solo_inspection",), seed=0)




def test_qmix_mixer_is_monotonic_in_each_agent_q_value():
    """The defining structural property of QMIX: increasing any single
    agent's Q-value must never decrease Q_tot, for any global state."""
    torch.manual_seed(0)
    n_agents, gdim = 4, 12
    mixer = QMixMixer(n_agents, gdim)
    g = torch.randn(5, gdim)
    base_q = torch.randn(5, n_agents)
    base_out = mixer(base_q, g)
    for i in range(n_agents):
        bumped = base_q.clone()
        bumped[:, i] += 1.0
        bumped_out = mixer(bumped, g)
        assert torch.all(bumped_out >= base_out - 1e-4)


def test_normalized_qmix_mixer_is_monotonic_and_scale_stable():
    torch.manual_seed(0)
    mixer = QMixMixer(48, 12, normalize_weights=True)
    agent_q = torch.ones(4, 48, requires_grad=True)
    out = mixer(agent_q, torch.zeros(4, 12))
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 10.0
    assert torch.all(agent_q.grad >= -1e-7)


def test_epsilon_greedy_respects_action_mask():
    torch.manual_seed(0)
    qnet = QNetwork()
    obs = {"a0": np.random.randn(qnet.net[0].in_features).astype(np.float32)}
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8); mask[0] = 1; mask[ACTION_PICK_BASE] = 1
    for eps in (0.0, 0.5, 1.0):
        actions = epsilon_greedy_actions(qnet, obs, {"a0": mask}, eps, torch.device("cpu"), ["a0"])
        assert actions["a0"] in (0, ACTION_PICK_BASE)




def test_train_qmix_end_to_end_smoke():
    cfg = _tiny_cfg()
    qnet, mixer, history = train_value_decomposition("qmix", cfg, total_steps=300, warmup_steps=50, seed=0)
    assert len(history) > 0
    for h in history:
        assert np.isfinite(h["td_loss"])


def test_qmix_behavior_anchor_is_opt_in_and_finite():
    cfg = _tiny_cfg()
    _, _, history = train_value_decomposition(
        "qmix", cfg, total_steps=300, warmup_steps=50, seed=0,
        bc_warmstart_steps=100, bc_anchor_coef=0.5,
        bc_parameter_anchor_coef=0.1,
        bc_demo_seed_count=2,
        normalize_mixer_weights=True, double_q=True, huber_loss=True,
    )
    anchored = [row for row in history if "bc_anchor_loss" in row]
    assert anchored
    assert all(np.isfinite(row["bc_anchor_loss"]) for row in anchored)
    assert all(np.isfinite(row["bc_parameter_anchor_loss"]) for row in anchored)




def _zero_module(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_qmix_bellman_target_masks_illegal_next_actions_and_stops_at_terminal():
    """Golden numerical check of the exact `_train_step` target path.

    The target Q network assigns 1 to legal action 0 and 100 to illegal action
    1. If masking is correct, the bootstrap uses 1 for every agent. Zeroed
    QMIX hypernetworks make the expected monotonic mix analytically tractable.
    """
    n_agents, gdim, mix_hidden = 2, 3, 32
    qnet = QNetwork(); target_qnet = QNetwork()
    mixer = QMixMixer(n_agents, gdim, mix_hidden=mix_hidden)
    target_mixer = QMixMixer(n_agents, gdim, mix_hidden=mix_hidden)
    _zero_module(qnet); _zero_module(target_qnet); _zero_module(mixer); _zero_module(target_mixer)
    with torch.no_grad():
        # Last layer of build_mlp is index 4.
        target_qnet.net[4].bias[0] = 1.0
        target_qnet.net[4].bias[1] = 100.0

    obs_dim = qnet.net[0].in_features
    obs = np.zeros((n_agents, obs_dim), dtype=np.float32)
    mask = np.zeros((n_agents, NUM_ACTIONS), dtype=np.int8); mask[:, 0] = 1
    actions = np.zeros(n_agents, dtype=np.int64)
    g = np.zeros(gdim, dtype=np.float32)
    opt = torch.optim.Adam(list(qnet.parameters()) + list(mixer.parameters()), lr=0.0)

    buf = JointReplayBuffer(2, n_agents)
    buf.add(obs, mask, actions, 2.0, obs, mask, 0.0, g, g)
    stats = _train_step(qnet, target_qnet, mixer, target_mixer, opt, buf, 1, 0.5, torch.device("cpu"))
    ln2 = np.log(2.0)
    expected_next_qtot = mix_hidden * n_agents * (ln2 ** 2)
    expected_target = 2.0 + 0.5 * expected_next_qtot
    assert stats["mean_q_tot"] == pytest.approx(0.0, abs=1e-7)
    assert stats["td_loss"] == pytest.approx(expected_target ** 2, rel=1e-5)

    terminal = JointReplayBuffer(2, n_agents)
    terminal.add(obs, mask, actions, 2.0, obs, mask, 1.0, g, g)
    terminal_stats = _train_step(qnet, target_qnet, mixer, target_mixer, opt, terminal, 1, 0.5, torch.device("cpu"))
    assert terminal_stats["td_loss"] == pytest.approx(4.0, rel=1e-6)
