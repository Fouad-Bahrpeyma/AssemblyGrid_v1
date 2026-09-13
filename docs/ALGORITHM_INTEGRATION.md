# Algorithm integration guide

The core rule is simple: **algorithms may choose representations and training objectives, but they must not redefine AssemblyGrid transitions, feasibility, canonical observations/actions, or benchmark KPIs.**

## Shared-parameter feed-forward IPPO

The first-paper reference uses one actor and one critic shared by the homogeneous robots.
Both networks consume only each robot's canonical local observation; legal actions are
restricted by that robot's canonical action mask. Its optimization reward combines the common
`sparse-delivery-v1` and team milestone terms with an agent-specific support credit. Thus,
"local" describes actor and critic state information; no privileged global state is consumed
by either IPPO network.

## Shared-parameter feed-forward MAPPO / CTDE

The actor architecture and input are identical to IPPO: canonical local observation plus
action mask. During training only, the critic consumes the explicit privileged/global state.
At evaluation/execution the actor remains decentralized. Report this as a centralized-critic
CTDE reference, not as an exact reproduction of every optimization detail in an external
MAPPO repository.

## Shared-parameter feed-forward QMIX / value decomposition

The supplied QMIX reference uses a shared feed-forward per-agent Q network over canonical
local observations, legal-action masking for behavior and target maximization, and a
global-state-conditioned monotonic mixing network during training. Evaluation is decentralized
masked argmax from the local Q network. This reference intentionally does not add recurrence,
episode replay, or Double-Q selection; those are method-side extensions, not requirements of
the AssemblyGrid interface.

## Graph MARL

Use the read-only `graph_view()`/Algorithm Support Interface to construct agent/entity/operation graphs. Do not inject those richer graph features into the benchmark's canonical observation and then call the result canonical v1.

## Centralized optimization/planning

Centralized methods may use privileged/global state or direct instance data. They remain subject to the same benchmark feasibility and success semantics. Exact/CP-SAT references that simplify transport or geometry must be labelled as optimistic model-specific references, not AssemblyGrid upper bounds.

## Reward/training signals

Training reward lives in the algorithm layer. The reported MARL campaign used the `sparse-delivery-v1` environment reward plus a team milestone reward and a support credit that is agent-specific for IPPO and MAPPO and agent-averaged for QMIX; `reference_training/README.md` and the paper give the exact terms. This does not alter the benchmark definition. Compare methods on benchmark KPIs.

## Wrapper rule

Gymnasium/PettingZoo wrappers are convenience/API layers. A wrapper must not change action legality, state transitions, seed semantics, termination/truncation, or benchmark metrics. Wrapper compliance tests are included under `Experiments/env/tests/`.


## First-paper information contract

| Method | Actor / per-agent Q input | Critic / mixer input | Training reward | Execution |
|---|---|---|---|---|
| IPPO | canonical local observation + mask | local observation | team terms + agent-specific support credit | decentralized |
| MAPPO | canonical local observation + mask | privileged global state | team terms + agent-specific support credit | decentralized |
| QMIX | canonical local observation + mask | privileged global state in mixer | team terms + agent-mean support credit | decentralized |

`proposal_support` is part of the canonical local candidate observation when enabled. It is an
aggregate support fraction, not a free-form message payload or directly exposed supporter list.

## Fixed neural adapter limitations

The reference encoder has finite pick/operation candidate-slot capacities and compact numerical
identity encodings. Overflow fails explicitly rather than being silently truncated. Hash/modulo
identity features can theoretically alias distinct symbolic identifiers. These limitations apply
to the supplied neural adapter only; the canonical benchmark actions remain identity-bearing.
For larger-scale studies, use a declared set/attention/graph/categorical-embedding adapter.

Operation options are deterministically ordered partly by proposal support. Slot position can
therefore become a useful inductive bias for a feed-forward policy. Do not interpret slot order
as semantic identity, and use the candidate identity/features as the authoritative action meaning.
