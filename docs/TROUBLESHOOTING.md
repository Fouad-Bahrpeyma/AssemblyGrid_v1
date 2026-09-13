# Troubleshooting

## `ModuleNotFoundError: gymnasium` / `pettingzoo` / `torch`

Install `requirements-v1-lock.txt`. These packages are optional to the core semantics but required for the wrapper and learned-reference tests.

## PyTorch installs a very large CUDA build

Use the CPU wheel source documented in `requirements-v1-lock.txt`.

## An action candidate is missing

Canonical v1 must not silently truncate theoretically eligible actions. Run the addressability audit in `Experiments/env/audits.py`. Fixed capacities are implementation encoding capacities; overflow is a conformance error.

## `ur10-demo-v1` rejected for an official result

Expected. `ur10-demo-v1` is a non-canonical compatibility profile. Use `abstract-v1` for official v1 results.

## Flow time looks better for a method that delivers less

Flow time is completion-conditioned. Always interpret it with delivered count, success-seed fraction, and unfinished-product age/backlog context.

## CP-SAT has a lower makespan than every AssemblyGrid controller

Expected: the shipped exact reference is an optimistic scheduling model that omits parts of AssemblyGrid transport/geometry. It is not an AssemblyGrid upper bound.

## A richer wrapper sees information not listed in the canonical observation

That can be valid only if it is explicitly a privileged/extension interface. Do not describe such an algorithm as using only the canonical decentralized observation.
