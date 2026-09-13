import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmark import generalization_split, track_config, track_suite
from core import EnvConfig, URModel, AssemblyGridCore
from encoding import OBS_DIM, global_state_dim, global_state_to_array, observation_to_array


def test_canonical_local_encoding_excludes_raw_joint_state_and_has_fixed_shape():
    c = AssemblyGridCore(EnvConfig(M=3, N=4, seed=1))
    obs = c._all_observations()
    a = observation_to_array(next(iter(obs.values())), c.cfg.M, c.cfg.N)
    assert a.shape == (OBS_DIM,) and np.all((0 <= a) & (a <= 1))
    # Raw proprioceptive configuration is internal/privileged, not part of
    # canonical decentralized v1 observation.
    rc = next(iter(obs)); before = observation_to_array(obs[rc], c.cfg.M, c.cfg.N)
    c._robot(rc).q = (1.0, 0, 0, 0, 0, 0)
    after = observation_to_array(c.observation(*rc), c.cfg.M, c.cfg.N)
    assert np.array_equal(before, after)
    assert not hasattr(c.observation(*rc), "q")
    assert all("q" not in info and "model_features" not in info and "message" not in info
               for info in c.observation(*rc).neighbors.values())


def test_privileged_global_state_is_not_just_concatenated_local_observation():
    c = AssemblyGridCore(EnvConfig(M=3, N=4, seed=1))
    g = global_state_to_array(c)
    assert g.shape == (global_state_dim(c),)
    assert global_state_dim(c) != c.cfg.M * c.cfg.N * OBS_DIM

    # Due dates and economic values are optional extension attributes. They
    # remain internally available but are not silently injected into the
    # default privileged/CTDE tensor either.
    p = next(iter(c.products.values()))
    before = global_state_to_array(c).copy()
    p.due_time += 10_000; p.value += 100.0
    after = global_state_to_array(c)
    assert np.array_equal(before, after)


def test_all_tracks_have_executable_presets_and_architecture_suite_varies_topology():
    names = ("AG-Core", "AG-Team", "AG-Parallel", "AG-Motion", "AG-Recipes", "AG-Skills",
             "AG-Scale", "AG-Generalize", "AG-Robust", "AG-Architecture")
    for name in names:
        cfg = track_config(name, seed=0); cfg.validate()
        assert cfg.benchmark_track == name

    # Every v1 mechanism family keeps the canonical abstract-motion model
    # active. Skills/generalization/robustness are extension presets.
    for name in ("AG-Core", "AG-Team", "AG-Parallel", "AG-Motion", "AG-Recipes", "AG-Scale", "AG-Architecture"):
        cfg = track_config(name, seed=0)
        assert cfg.geometry_profile == "abstract-v1"
        assert cfg.profile_enforced and cfg.motion_planning and cfg.trajectory_conflicts

    tops = {x.topology for x in track_suite("AG-Architecture")}
    assert tops == {"line", "von_neumann", "moore"}


def test_generalization_split_supports_recipe_geometry_scale_faults_and_architecture():
    train_cfg = EnvConfig(M=5, N=8, recipe_ids=("standard_ab",), cell_spacing=1.0, topology="moore")
    test_cfg = EnvConfig(M=8, N=12, recipe_ids=("parallel_branch",), cell_spacing=1.25, topology="line",
                         robot_failure_prob=0.01)
    train, test = generalization_split([train_cfg, test_cfg], ("scale", "recipe", "geometry", "faults", "architecture"))
    assert train == [train_cfg] and test == [test_cfg]


def test_recipe_json_round_trip(tmp_path):
    from recipe import STANDARD_ABC, load_recipe_json, save_recipe_json, recipe_to_dict
    path = tmp_path / "recipe.json"
    save_recipe_json(STANDARD_ABC, path)
    loaded = load_recipe_json(path)
    assert recipe_to_dict(loaded) == recipe_to_dict(STANDARD_ABC)
