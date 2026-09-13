# reference_training — the MARL campaign code

This directory holds the training code that produced the reported 270-run MARL reference
campaign (IPPO, MAPPO and QMIX on the three workload families at three difficulty levels, ten
algorithm seeds each, 1,536,000 environment steps per run). The hyperparameters implemented here
are the ones listed in the MARL configuration table of the paper.

`Experiments/env/marl/` is a different thing: small reference implementations that demonstrate the
benchmark interfaces. They are intentionally simple and are not the campaign configuration.

## Contents

- `stable_ppo.py` — `StructuredActor` (LayerNorm, linear encoder to width 256, two tanh residual
  blocks, separate action-family and action-slot heads whose logits are summed, orthogonal
  initialization), `ValueNetwork` with the same trunk, and the PPO update.
- `flow_concurrency/old_ppo_runner.py`, `flow_concurrency/qmix_runner.py` — the runners used for
  the 180 Flow and Concurrency runs.
- `coalition/old_ppo_runner.py`, `coalition/qmix_runner.py` — the runners used for the 90 Coalition
  runs. The two generations are hyperparameter-identical; the later pair takes `--family` instead
  of selecting Coalition internally.

The QMIX runner imports `marl.vdn_qmix` for the Q network, the monotonic mixer and the joint
replay buffer. That module is included in the package for this reason; the VDN mixer it also
defines is not part of any reported result.

Both runners call the training step without `double_q` and without `huber_loss`, so the archived
QMIX runs use a single-network temporal-difference target with mean-squared-error loss.

## Requirements

The package requirements plus PyTorch. The campaign ran on CUDA; `--device cpu` works but is slow.

## Run one training job

From the repository root:

```sh
python reference_training/flow_concurrency/old_ppo_runner.py \
    --env-root Experiments --stable-root reference_training --output-dir runs/ippo_flow_easy_s00 \
    --method ippo --family flow --direct-easy --seed 0 --device cuda:0
```

```sh
python reference_training/flow_concurrency/qmix_runner.py \
    --env-root Experiments --stable-root reference_training --output-dir runs/qmix_flow_easy_s00 \
    --family flow --direct-easy --seed 0 --device cuda:0
```

Select the scenario with exactly one of `--direct-easy`, `--direct-medium`, `--direct-hard`, and
the family with `--family`. `--seed` is the algorithm seed; the reported campaign used 0 to 9.
`--total-steps` defaults to the reported budget of 1,536,000. Add `--smoke` to the PPO runner for
a short interface check.

Each run writes `aag042_result.json` (the per-run record the paper reports from), the final
policy weights, and a per-stage checkpoint into `--output-dir`.

## Training instances

A run does not train on shared suite instances. It takes the scenario parameters of suite indices
0 to 4 and regenerates five realizations from run-specific generation and execution seeds derived
from the algorithm seed, then splits the budget evenly across them, training on the realizations as consecutive blocks in a fixed order. Generation seeds are `80000 + family_offset + 100 * seed + 10 * stage + realization` and execution seeds `90000 + family_offset + 100 * seed + 10 * stage + realization`, with family offsets 0, 300000 and 600000 for Coalition, Flow and Concurrency. The validation instances
(indices 200 to 204) and the 50 final-test instances (indices 1000 to 1049) are the canonical suite
instances and are identical across every run, method and seed.

## Reproducing a reported number

The reported values are the final-checkpoint evaluations on the 50 final-test instances, averaged
per run and then aggregated over the ten seeds. Reproducing a full cell means ten runs of
1,536,000 steps. Training checkpoints from the original campaign are not bundled with this
repository.
