from __future__ import annotations
import sys
from pathlib import Path
ENV = Path(__file__).resolve().parents[1]
if str(ENV) not in sys.path:
    sys.path.insert(0, str(ENV))

import numpy as np
from core import EnvConfig, ACTION_IDLE
from encoding import observation_to_array
from marl.core_parallel_env import CoreParallelEnv


def test_core_parallel_adapter_delegates_without_semantic_change():
    cfg = EnvConfig(M=2, N=2, episode_mode="horizon", horizon_T=8,
                    generation_seed=123, execution_seed=456, seed=123,
                    official_result=False)
    env = CoreParallelEnv(cfg)
    obs, infos = env.reset()
    for a in env.agents:
        rc = env._agent_to_coord[a]
        assert np.array_equal(obs[a], observation_to_array(env.core._all_observations()[rc], cfg.M, cfg.N))
        assert infos[a]["action_mask"][ACTION_IDLE] == 1
    actions = {a: ACTION_IDLE for a in env.agents}
    env.step(actions)
    assert env.core.t == 1


def test_core_parallel_adapter_snapshot_is_read_only_observational_query():
    cfg = EnvConfig(M=2, N=2, episode_mode="horizon", horizon_T=5,
                    generation_seed=7, execution_seed=11, seed=7,
                    official_result=False)
    a = CoreParallelEnv(cfg); b = CoreParallelEnv(cfg)
    oa, ia = a.reset(); ob, ib = b.reset()
    for _ in range(4):
        _ = a.algorithm_support_snapshot()
        aa = {x: ACTION_IDLE for x in a.agents}
        ab = {x: ACTION_IDLE for x in b.agents}
        oa, ra, ta, tra, ia = a.step(aa)
        ob, rb, tb, trb, ib = b.step(ab)
        assert a.core.t == b.core.t
        assert a.core.delivered_count == b.core.delivered_count
        assert a.core.spawned_count == b.core.spawned_count
