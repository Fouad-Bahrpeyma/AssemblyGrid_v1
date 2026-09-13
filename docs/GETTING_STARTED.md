# Getting started with AssemblyGrid v1.0.0

## 1. Environment

The first-paper campaign used Python 3.11.15. For exact reproduction:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-v1-lock.txt
```

On Linux/macOS activate with `source .venv/bin/activate`.

## 2. Run the canonical core

```text
python examples/quickstart_core.py
```

The example creates the canonical `AG-Core` configuration, resets the environment, inspects local observations/action masks, executes idle actions for a few ticks, and prints benchmark metrics.

## 3. Load an official realization

```text
python examples/quickstart_official.py
```

Official IDs are verified against the catalogue SHA-256 before the reference run configuration is returned.

## 4. Inspect the rich algorithm-support interface

```text
python examples/quickstart_algorithm_support.py
```

This interface is read-only and non-canonical. It is intended for CTDE critics, centralized references, graph adapters, motion-aware research extensions, and diagnostics.

## 5. Run tests

From the repository root:

```text
python -m pytest Experiments/env/tests -q
python tools/verify_instances.py
python Experiments/env/audits.py
```

The first command runs the conformance suite, the second verifies every official catalogue entry, and the third runs the release addressability and coalition-formation audits. The PyTorch and Gymnasium/PettingZoo tests require the packages pinned in `requirements-v1-lock.txt`.

