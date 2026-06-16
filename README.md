# Contribution 1 : Add Expected Calibration Error metrics and derived
**Contribution Number:** 1
**Student:** Rishabh Padhy
**Issue:** [pytorch/ignite #1009](https://github.com/pytorch/ignite/issues/1009)
**Status:** Phase II Complete

---

## Why I Chose This Issue
I chose this issue because it sits perfectly at the intersection of my academic training and my core technical interests. As a final-semester Computer Science major at PSU with a minor in mathematics, I have completed dedicated coursework in Neural Networks and Deep Learning alongside several advanced mathematics classes. Expected Calibration Error (ECE) and Multiclass ECE are vital statistical components for measuring model uncertainty, ensuring that a neural network's confidence scores realistically align with its actual predictive accuracy. Implementing these validation metrics from scratch gives me an actionable, high-leverage way to apply my deep learning and mathematical background directly to framework engineering.This project perfectly matches my professional learning goals of moving beyond high-level model training and diving deep into the internals of open-source machine learning infrastructure. Because PyTorch-Ignite heavily relies on custom metric formalisms and utilizes Jupyter Notebooks for testing and prototyping, it allows me to fully leverage my existing toolkit while learning how to design clean, low-dependency, and production-grade library code for the broader ML community

---

## Understanding the Issue

### Problem Description

PyTorch-Ignite ships dozens of classification metrics (`Accuracy`, `Precision`, `Recall`, `Entropy`, …) but has **no metric for model calibration**. Expected Calibration Error (ECE) measures the gap between a model's *confidence* and its *actual accuracy*: a well-calibrated model that says "I'm 80% sure" should be right 80% of the time. The issue (#1009) asks for an ECE metric "as coded in [Baal](https://github.com/baal-org/baal)", plus the **derived** quantities that fall out of the same binning — chiefly **Maximum Calibration Error (MCE)**, the worst-bin gap.

ECE is defined over `M` confidence bins as:

```
ECE = Σ_{m=1..M}  (|B_m| / n) · | acc(B_m) − conf(B_m) |
MCE = max_{m=1..M}              | acc(B_m) − conf(B_m) |
```

where `B_m` is the set of predictions whose confidence falls in bin `m`, `acc(B_m)` is the empirical accuracy in that bin, and `conf(B_m)` is the mean confidence in that bin.

### Expected Behavior

A user should be able to write:

```python
from ignite.metrics import ExpectedCalibrationError, MaximumCalibrationError

ece = ExpectedCalibrationError(num_bins=10)
ece.attach(evaluator, "ece")
# ... and read state.metrics["ece"] after a run
```

and have it work for binary and multiclass inputs, on CPU/GPU, and under distributed (multi-GPU) training — consistent with every other metric in the library.

### Current Behavior

`from ignite.metrics import ExpectedCalibrationError` fails — the symbol does not exist. There is **no** calibration metric anywhere in `ignite/metrics/` (confirmed: `grep -ri "calibration" --include=*.py` returns nothing). The only prior attempt, **PR #3132**, was never merged and contains design flaws (see Analysis below).

### Affected Components

- `ignite/metrics/calibration_error.py` — **new** module housing both metrics
- `ignite/metrics/__init__.py` — import + `__all__` registration (alphabetically after `average_precision`)
- `docs/source/metrics.rst` — "Complete list of metrics" autosummary entries
- `tests/ignite/metrics/test_calibration_error.py` — **new** test module

---

## Reproduction Process

> This is a *missing-feature* issue, so "reproduction" means (A) demonstrating the
> metric is absent, and (B) reproducing the concrete flaw in the only prior
> attempt (PR #3132) so my implementation provably improves on it.

### Environment Setup

- Cloned my fork of `pytorch/ignite`/
- Installed the CPU build of PyTorch plus NumPy: `pip install torch numpy`.
- ignite is run straight from the source tree via `PYTHONPATH=.` (no editable install needed to import `ignite.metrics`).
- **Challenge:** the default PyTorch wheel index was not reachable from my environment; resolved by installing from the standard PyPI index (`pip install torch`). NumPy is an implicit runtime dependency for some torch paths — installing it silenced a `Failed to initialize NumPy` warning.
- Confirmed the working version is ignite `0.6.0` (`ignite/__init__.py`), which sets the `.. versionadded:: 0.6.0` directive my new metric will carry.

### Steps to Reproduce

1. From the repo root, run the reproduction script:
   ```bash
   pip install torch numpy
   PYTHONPATH=. python repro/repro_ece.py
   ```
2. **Part A** prints whether `ExpectedCalibrationError` exists in `ignite.metrics`.
   - **Expected (if the feature existed):** `True`
   - **Actual:** `False` — the metric is missing.
3. **Part B** runs two accumulation strategies over a growing number of batches and prints the bytes retained in each metric's internal state:
   - the **PR #3132 strategy** (`torch.cat` every `update()`), and
   - the **proposed strategy** (fixed-size bins updated with `index_add_`).
4. **Expected (a scalable metric):** retained state stays roughly constant as the dataset grows.
5. **Actual (PR #3132):** retained state grows **linearly** with the number of samples — the root of the out-of-memory (OOM) crash on large datasets.

Observed output (reproduced consistently across runs):

```
=== Part A: ExpectedCalibrationError present in ignite.metrics? ===
False

=== Part B: retained state size as dataset size N grows (batch=4096) ===
 batches          N    PR#3132 state   proposed state
      10      40960          0.20 MB      0.000120 MB
     100     409600          2.05 MB      0.000120 MB
    1000    4096000         20.48 MB      0.000120 MB
```

PR #3132's state climbs 0.20 → 2.05 → **20.48 MB** as N grows 10× then 10× again (clean O(N)), and would keep climbing until OOM on a realistically sized dataset. The proposed fixed-bin state is flat at **~120 bytes** regardless of N (O(1)).

### Reproduction Evidence

- **Working branch (fork):** 
- **Reproduction script:** [`repro/repro_ece.py`]
- **My findings:**
  - The feature genuinely does not exist (Part A → `False`).
  - PR #3132's memory cost is O(N) in the number of samples, not O(1) in the number of bins — confirmed empirically (linear growth above). This is the concrete defect my implementation must avoid.

---

## Solution Approach

### Analysis

The correct mental model for ECE is that **only the per-bin aggregates matter** — the sum of confidences, the count of correct predictions, and the count of samples *in each bin*. Individual predictions never need to be retained. PR #3132 missed this and stored every prediction:

```python
# PR #3132 — grows without bound
self.confidences = torch.cat((self.confidences, max_probs))
self.corrects    = torch.cat((self.corrects, predicted_class == y))
```

This causes three problems: (1) **O(N) memory** → OOM on large datasets (reproduced above); (2) a Python `for` loop over bins in `compute()` (slow, non-vectorized); and (3) **no distributed support** — the accumulators are never synchronized across processes, so multi-GPU runs produce wrong values.

The **Baal reference** (`baal/utils/metrics.py`) confirms the right shape of the algorithm: it bins each prediction by `floor(conf * n_bins)` and keeps only three per-bin arrays (`tp`, `samples`, `conf_agg`), then computes `ECE = Σ (samples_m / n) · |acc_m − conf_m|`. My implementation reproduces this math but as **fixed-size torch tensors updated in-place**, so it is both O(1) in memory and distributed-safe.

### Proposed Solution

Implement a shared base class `_BaseCalibrationError(Metric)` that accumulates **fixed-size per-bin aggregates** during `update()`, with two thin public subclasses — `ExpectedCalibrationError` and `MaximumCalibrationError` — that differ only in `compute()` (weighted-mean gap vs. worst-bin gap). MCE reuses the *identical* accumulators, so adding it is near-zero extra cost and directly satisfies the issue's "and derived" wording.

This follows the standard accumulator pattern already used by `ignite/metrics/entropy.py` (scalar accumulators) and `ignite/metrics/confusion_matrix.py` (vector/tensor accumulators): state is a small device tensor; `reset`/`update` decorated with `@reinit__is_reduced`; `compute` decorated with `@sync_all_reduce(...)` for distributed correctness.

**Key API decision:** `y_pred` is expected to be **probabilities** (already softmax/sigmoid-normalized). Users with raw logits apply softmax via `output_transform` — the same convention as `AveragePrecision`. This avoids the silent "double-softmax" bug that would occur if the metric softmaxed internally but the user passed already-normalized probabilities.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** PyTorch-Ignite has no calibration metric. I need to add `ExpectedCalibrationError` and `MaximumCalibrationError` that, given `(y_pred, y)`, bin predictions by confidence and return (respectively) the weighted-mean and worst-case gap between per-bin accuracy and per-bin confidence — for binary and multiclass inputs, on CPU/GPU and under distributed training — without storing every prediction.

**Match:** The codebase already has the exact patterns I need:
- `ignite/metrics/entropy.py` — a minimal accumulator metric: state initialized in `reset()` on `self._device`, accumulated in `update()`, finalized in `compute()`, with `@reinit__is_reduced` / `@sync_all_reduce(...)` and `_state_dict_all_req_keys`. It also shows the `(B, C, ...) → (N, C)` flatten via `movedim(1, -1).reshape(-1, num_classes)` that I reuse for segmentation inputs.
- `ignite/metrics/confusion_matrix.py` — shows the **vector/tensor accumulator** pattern (`torch.zeros(...)` of fixed shape, accumulated in-place, listed in `@sync_all_reduce`), which my three bin tensors follow.
- `ignite/metrics/accuracy.py` — its `argmax`/confidence-extraction conventions. Note I deliberately do **not** reuse `_BaseClassification._check_type`, because its binary path asserts `y_pred ∈ {0,1}`, which would reject probability inputs; I write explicit shape handling instead (like `entropy.py`).

**Plan:**
1. Add `ignite/metrics/calibration_error.py`:
   - `_BaseCalibrationError(Metric)` with `__init__(num_bins=10, output_transform=…, device=…, skip_unrolling=False)`; validate `num_bins` is a positive int.
   - `reset()` (`@reinit__is_reduced`): pre-allocate three fixed-size tensors on `self._device` —
     `self._bin_correct`, `self._bin_conf`, `self._bin_count`, each `torch.zeros(num_bins)`.
   - `update(output)` (`@reinit__is_reduced`): `.detach()` inputs; extract per-sample `conf` and `correct` (multiclass: `conf, pred = y_pred.max(dim=1)`, flattening `(B, C, ...)`; binary `(B,)`/`(B,1)`: `conf = max(p, 1−p)`, `pred = p ≥ 0.5`); compute bin index `clamp((conf * num_bins).long(), max=num_bins−1)`; accumulate with `index_add_` into the three tensors — **O(1) memory in N**.
   - Helper `_acc_conf()` → per-bin accuracy and confidence (masking empty bins).
   - `ExpectedCalibrationError.compute()` (`@sync_all_reduce("_bin_correct","_bin_conf","_bin_count")`): guard empty state with `NotComputableError`; return `Σ (count_m / n) · |acc_m − conf_m|` — **fully vectorized, no Python loop**.
   - `MaximumCalibrationError.compute()` (same decorator/guard): return `max_m |acc_m − conf_m|` over non-empty bins.
   - Set `_state_dict_all_req_keys = ("_bin_correct", "_bin_conf", "_bin_count")`.
   - Write complete Google-style docstrings (≤120 cols) with the math, the "expects probabilities" note, a `.. testcode::`/`.. testoutput::` doctest, and `.. versionadded:: 0.6.0`.
2. Register in `ignite/metrics/__init__.py` (import + add both names to `__all__`).
3. Add both classes to `docs/source/metrics.rst` under "Complete list of metrics".
4. Add `tests/ignite/metrics/test_calibration_error.py`.

**Implement:** *(Phase III)* — will be done on the working branch:


**Review:** Self-review against `CONTRIBUTING.md`: run `pre-commit run -a` (ruff lint + format — the project replaced flake8/black/isort) and `pyrefly check`; follow metric conventions (decorators, `NotComputableError`, `ValueError` for bad shapes, `.detach()` on inputs, `_state_dict_all_req_keys`); add a doctest example so the docs build/test (`cd docs && make html && make doctest`); write tests named `test_*`, with `cuda` in the name for GPU tests and `@pytest.mark.distributed` for distributed tests. Validate numerical results against a NumPy re-implementation of the Baal binning.

**Evaluate:** See Testing Strategy below — unit tests for correctness against hand-computed and Baal-reference values, a distributed test confirming cross-process sync, and re-running the reproduction script to confirm the new metric's state stays O(1) where PR #3132 was O(N).

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: Hand-computed small example — feed a tiny `(y_pred, y)` whose ECE/MCE I compute by hand and assert `compute()` matches within tolerance.
- [ ] Test case 2: Correctness vs. a NumPy reference re-implementing the Baal binning, on random binary and multiclass data, parametrized over `num_bins`.
- [ ] Test case 3: Edge cases — perfectly calibrated input → ECE ≈ 0; a maximally over-confident-but-wrong input → ECE ≈ 1; varying `num_bins`; `conf == 1.0` lands in the last bin.
- [ ] Test case 4: `NotComputableError` raised when `compute()` is called before any `update()` (both metrics); `ValueError` on malformed shapes.
- [ ] Test case 5: Batched vs. single-shot equivalence — splitting the same data across multiple `update()` calls yields the same result as one call (verifies online accumulation).

### Integration Tests

- [ ] Integration scenario 1: Attach to an `Engine`/`create_supervised_evaluator` and confirm `state.metrics["ece"]` is populated after a run.
- [ ] Integration scenario 2: `@pytest.mark.distributed` test (CPU gloo) confirming the metric returns the same value as the single-process reference after `@sync_all_reduce`, and that the bin tensors live on the chosen `metric_device`.

### Manual Testing

Re-run `repro/repro_ece.py` after implementation: Part A should print `True`, and the new metric's retained state should remain flat (O(1)) as N grows — the opposite of PR #3132's linear growth shown above.

---

## Implementation Notes

### Week 1 Progress (Phase II)

- Set up the environment (PyTorch CPU + NumPy), ran ignite from source (version `0.6.0`).
- Wrote and ran `repro/repro_ece.py`, confirming (A) the metric is missing and (B) PR #3132's accumulation is O(N) — measured 0.20 → 20.48 MB growth.
- Studied `entropy.py` (scalar accumulator + flatten pattern), `confusion_matrix.py` (vector accumulator), and `accuracy.py` (shape/confidence conventions) as templates, and located all registration/doc touch-points.
- Fetched the Baal reference (`baal/utils/metrics.py`) to lock down the exact binning (`floor(conf * n_bins)`, three per-bin aggregates) so my tests can validate against a faithful NumPy port.
- Confirmed scope with mentors/self: **ECE + MCE** in one module, **probabilities** as the expected input.

### Code Changes

- **Files modified (planned):** `ignite/metrics/calibration_error.py` (new), `ignite/metrics/__init__.py`, `docs/source/metrics.rst`, `tests/ignite/metrics/test_calibration_error.py` (new).
- **Files added so far (reproduction):** `repro/repro_ece.py`, `repro/README.md`.
- **Approach decisions:** fixed-size per-bin accumulators (O(1) memory) updated with `index_add_` + `@sync_all_reduce` for distributed correctness + vectorized `compute()`, with a shared base class so MCE reuses ECE's accumulators — explicitly avoiding PR #3132's `torch.cat` growth and Python bin loop. Input treated as probabilities (logits via `output_transform`) to avoid double-softmax.

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted in Phase III]

**PR Description:** [Will adapt the Analysis + Implementation Plan above]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** Not yet submitted (Phase II)

---

## Learnings & Reflections

### Technical Skills Gained

- How PyTorch-Ignite's `Metric` base class enforces distributed correctness through the `@reinit__is_reduced` / `@sync_all_reduce` decorators, and why naive accumulation breaks under multi-GPU.
- Reasoning about memory complexity (O(N) vs O(1)) of an online metric and measuring it empirically.
- Vectorized histogram-style accumulation with `index_add_` / `clamp` instead of Python loops over bins.

### Challenges Overcome

- PyTorch wheel index was unreachable; resolved by installing from standard PyPI. Documented for future contributors.

### What I'd Do Differently Next Time

[Reflection to be added during/after implementation.]

---

## Resources Used

- [Baal calibration metrics (reference implementation)](https://github.com/baal-org/baal)
- [PyTorch-Ignite — How to create a custom metric](https://docs.pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric)
- [pytorch/ignite issue #1009](https://github.com/pytorch/ignite/issues/1009) and [prior attempt PR #3132](https://github.com/pytorch/ignite/pull/3132)
- Existing in-repo metrics studied: `ignite/metrics/entropy.py`, `ignite/metrics/confusion_matrix.py`, `ignite/metrics/accuracy.py`
