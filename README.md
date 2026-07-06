# Contribution 1 : Add Expected Calibration Error metrics and derived
**Contribution Number:** 1
**Student:** Rishabh Padhy
**Issue:** [pytorch/ignite #1009](https://github.com/pytorch/ignite/issues/1009)
**Status:** Phase IV Complete

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

- Cloned my fork of `pytorch/ignite` and created the working branch `fix-issue-1009`.
- Installed the CPU build of PyTorch plus NumPy/SciPy: `pip install torch numpy scipy`.
- ignite is run straight from the source tree via `PYTHONPATH=.` (no editable install needed to import `ignite.metrics`).
- **Challenge:** the default PyTorch wheel index was not reachable from my environment; resolved by installing from the standard PyPI index (`pip install torch`). NumPy is an implicit runtime dependency for some torch paths — installing it silenced a `Failed to initialize NumPy` warning.
- Confirmed the working version is ignite `0.6.0` (`ignite/__init__.py`), which sets the `.. versionadded:: 0.6.0` directive on the new metrics.

### Steps to Reproduce

1. From the repo root, run the reproduction script:
   ```bash
   pip install torch numpy
   PYTHONPATH=. python repro/repro_ece.py
   ```
2. **Part A** prints whether `ExpectedCalibrationError` exists in `ignite.metrics`.
   - **Expected (if the feature existed):** `True`
   - **Actual (at Phase II, before my work):** `False` — the metric was missing.
3. **Part B** runs two accumulation strategies over a growing number of batches and prints the bytes retained in each metric's internal state:
   - the **PR #3132 strategy** (`torch.cat` every `update()`), and
   - the **proposed strategy** (fixed-size bins updated with `index_add_`).
4. **Expected (a scalable metric):** retained state stays roughly constant as the dataset grows.
5. **Actual (PR #3132):** retained state grows **linearly** with the number of samples — the root of the out-of-memory (OOM) crash on large datasets.

Observed output (reproduced consistently across runs):

```
=== Part A: ExpectedCalibrationError present in ignite.metrics? ===
False        # at Phase II (before implementation); now prints True on branch fix-issue-1009

=== Part B: retained state size as dataset size N grows (batch=4096) ===
 batches          N    PR#3132 state   proposed state
      10      40960          0.20 MB      0.000120 MB
     100     409600          2.05 MB      0.000120 MB
    1000    4096000         20.48 MB      0.000120 MB
```

PR #3132's state climbs 0.20 → 2.05 → **20.48 MB** as N grows 10× then 10× again (clean O(N)), and would keep climbing until OOM on a realistically sized dataset. The proposed fixed-bin state is flat at **~120 bytes** (3 × `num_bins` float32 values) regardless of N (O(1)).

### Reproduction Evidence

- **Working branch (fork):** https://github.com/rpp5501/ignite/tree/fix-issue-1009
- **Reproduction script:** [`repro/repro_ece.py`](https://github.com/rpp5501/ignite/blob/fix-issue-1009/repro/repro_ece.py)
- **My findings:**
  - The feature genuinely did not exist (Part A → `False`).
  - PR #3132's memory cost is O(N) in the number of samples, not O(1) in the number of bins — confirmed empirically (linear growth above). This is the concrete defect my implementation avoids.

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

A shared base class `_BaseCalibrationError(Metric)` accumulates **fixed-size per-bin aggregates** during `update()`, with two thin public subclasses — `ExpectedCalibrationError` and `MaximumCalibrationError` — that differ only in `compute()` (weighted-mean gap vs. worst-bin gap). MCE reuses the *identical* accumulators, so adding it is near-zero extra cost and directly satisfies the issue's "and derived" wording.

This follows the standard accumulator pattern already used by `ignite/metrics/entropy.py` (scalar accumulators) and `ignite/metrics/confusion_matrix.py` (vector/tensor accumulators): state is a small device tensor; `reset`/`update` decorated with `@reinit__is_reduced`; `compute` decorated with `@sync_all_reduce(...)` for distributed correctness.

**Key API decision:** `y_pred` is expected to be **probabilities** (already softmax/sigmoid-normalized). Users with raw logits apply softmax via `output_transform` — the same convention as `AveragePrecision`. This avoids the silent "double-softmax" bug that would occur if the metric softmaxed internally but the user passed already-normalized probabilities.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** PyTorch-Ignite has no calibration metric. I needed to add `ExpectedCalibrationError` and `MaximumCalibrationError` that, given `(y_pred, y)`, bin predictions by confidence and return (respectively) the weighted-mean and worst-case gap between per-bin accuracy and per-bin confidence — for binary and multiclass inputs, on CPU/GPU and under distributed training — without storing every prediction.

**Match:** The codebase already had the exact patterns I needed:
- `ignite/metrics/entropy.py` — a minimal accumulator metric with `@reinit__is_reduced` / `@sync_all_reduce(...)` and `_state_dict_all_req_keys`; also shows the `(B, C, ...) → (N, C)` flatten via `movedim(1, -1).reshape(-1, num_classes)` I reuse for segmentation inputs.
- `ignite/metrics/confusion_matrix.py` — the **vector/tensor accumulator** pattern (`torch.zeros(...)` of fixed shape, accumulated in-place, listed in `@sync_all_reduce`), which my three bin tensors follow.
- `ignite/metrics/accuracy.py` — its `argmax`/confidence-extraction conventions. I deliberately did **not** reuse `_BaseClassification._check_type`, because its binary path asserts `y_pred ∈ {0,1}`, which would reject probability inputs; I write explicit shape handling instead (like `entropy.py`).

**Plan:** (1) add `ignite/metrics/calibration_error.py`; (2) register in `ignite/metrics/__init__.py`; (3) document in `docs/source/metrics.rst`; (4) add `tests/ignite/metrics/test_calibration_error.py`. — **All four steps completed in Phase III (see Implementation Notes).**

**Implement:** Done on the working branch:
https://github.com/rpp5501/ignite/tree/fix-issue-1009

**Review:** Self-reviewed against `CONTRIBUTING.md`: ran `ruff check` and `ruff format` (the project replaced flake8/black/isort with ruff) — both clean; followed metric conventions (decorators, `NotComputableError`, `ValueError` for bad shapes, `.detach()` on inputs, `_state_dict_all_req_keys`); added `.. testcode::`/`.. testoutput::` doctests; tests named `test_*` with `@pytest.mark.usefixtures("distributed")` for the distributed suite. Numerical results validated against a NumPy re-implementation of the Baal binning.

**Evaluate:** See Testing Strategy below.

---

## Testing Strategy

All tests live in `tests/ignite/metrics/test_calibration_error.py` and mirror the structure of `test_entropy.py` (parametrized over the `available_device` fixture for CPU/CUDA/MPS, with a NumPy reference).

### Unit Tests — **implemented & passing (45 passed on CPU)**

- [x] **Correctness vs. NumPy reference** (`np_ece` / `np_mce` re-implementing the Baal binning) on binary, multiclass, and image-segmentation inputs, parametrized over `num_bins ∈ {5, 10, 15}`.
- [x] **Edge cases** — confidently-correct input → ECE = MCE ≈ 0; confidently-wrong input → ECE = MCE ≈ 1; `conf == 1.0` lands in the last bin.
- [x] **`NotComputableError`** raised when `compute()` is called before any `update()` (both metrics).
- [x] **`ValueError`** on a positive-integer-violating `num_bins` and on malformed input shapes.
- [x] **Batched vs. single-shot equivalence** — splitting data across multiple `update()` calls equals one call (verifies online accumulation).
- [x] **Detached state** — accumulators do not retain autograd graph (`requires_grad is False`).

### Integration / Distributed Tests — **implemented**

- [x] `TestDistributed.test_integration` — attaches each metric to an `Engine`, runs across ranks, `idist.all_gather`s the data, and asserts the result matches the single-process NumPy reference (proves `@sync_all_reduce` correctness).
- [x] `TestDistributed.test_accumulator_device` — asserts the three bin tensors live on the chosen `metric_device`.
- **Note:** the distributed suite is launched by the project's multi-process runner (`pytest --dist=each --tx ...`, requires `pytest-xdist`) and is **skipped on Windows** per `CONTRIBUTING.md`; it runs in upstream CI on Linux. Locally I confirmed the existing `test_entropy.py::TestDistributed` skips/errors identically without the runner, so this is environmental, not a defect.

### Manual Testing

- `python -m pytest tests/ignite/metrics/test_calibration_error.py` → **45 passed** (CUDA/MPS variants skipped on a CPU-only box).
- Doctests verified through a real evaluator: ECE → `0.2500000…`, MCE → `0.6999999…` (matches the `.. testoutput::` blocks).
- Re-ran `repro/repro_ece.py`: Part A now prints `True`; the new metric's retained state stays flat at ~120 bytes as N grows to 4M — the opposite of PR #3132's linear growth.

---

## Implementation Notes

### Week 1 Progress (Phase III)

**What I built:**
- Added `ignite/metrics/calibration_error.py` with `_BaseCalibrationError` (shared binning) and two public metrics, `ExpectedCalibrationError` and `MaximumCalibrationError`:
  - `reset()` pre-allocates three fixed-size tensors on `self._device`: `_bin_correct`, `_bin_conf`, `_bin_count` (each `torch.zeros(num_bins)`).
  - `update()` extracts per-sample confidence/correctness (multiclass `max(dim=1)`; binary `max(p, 1−p)`), computes `bin_idx = clamp((conf * num_bins).long(), 0, num_bins−1)`, and accumulates with `index_add_` — **O(1) memory in N**.
  - `compute()` is fully vectorized: ECE = `Σ (count_m / n)·|acc_m − conf_m|`; MCE = `max_m |acc_m − conf_m|` over non-empty bins.
  - Decorated with `@reinit__is_reduced` (reset/update) and `@sync_all_reduce(...)` (compute); set `_state_dict_all_req_keys`.
- Registered both metrics in `ignite/metrics/__init__.py` (import + `__all__`) and added them to the autosummary in `docs/source/metrics.rst`.
- Wrote `tests/ignite/metrics/test_calibration_error.py` (unit + distributed) with a NumPy reference for the Baal binning.
- Recreated `repro/repro_ece.py` so the reproduction is runnable on this branch.

**Challenges faced & how I resolved them:**
- **Probabilities vs. logits / shape handling.** I first considered subclassing `_BaseClassification` to reuse `_check_type`, but its binary branch asserts `y_pred ∈ {0,1}`, which rejects probability inputs. Resolved by writing explicit shape handling (like `entropy.py`) and documenting that `y_pred` is expected to be probabilities.
- **`conf == 1.0` overflowing the bin index.** `floor(1.0 * num_bins) = num_bins` is out of range. Baal clips confidence to `0.9999`; I instead `clamp(..., max=num_bins−1)`, which is cleaner and keeps a perfectly-confident sample in the last bin (verified by a unit test).
- **float32 vs float64 binning at bin boundaries.** To avoid flaky comparisons, the NumPy reference is computed from the same probabilities the metric sees and assertions use `pytest.approx(..., abs=1e-5)`; random softmax outputs effectively never sit exactly on a `k/num_bins` boundary.

### Code Changes

- **Working branch:** https://github.com/rpp5501/ignite/tree/fix-issue-1009
- **Files added:**
  - `ignite/metrics/calibration_error.py` (new metrics)
  - `tests/ignite/metrics/test_calibration_error.py` (new tests)
  - `repro/repro_ece.py` (reproduction aid — not intended for the upstream PR)
- **Files modified:**
  - `ignite/metrics/__init__.py` (import + `__all__`)
  - `docs/source/metrics.rst` (autosummary entries)
- **Commit (Conventional Commits):** see the **Pull Request** section for the exact message used.
- **Approach decisions:** fixed-size per-bin accumulators (O(1) memory) updated with `index_add_` + `@sync_all_reduce` for distributed correctness + vectorized `compute()`, with a shared base class so MCE reuses ECE's accumulators — explicitly avoiding PR #3132's `torch.cat` growth and Python bin loop. Input treated as probabilities (logits via `output_transform`) to avoid double-softmax.

---

## Pull Request

**PR Link:** [to be opened in Phase IV — branch `fix-issue-1009` is ready]

**Commit message (Conventional Commits v1.0.0):**

```
feat(metrics): add Expected and Maximum Calibration Error metrics

Add ExpectedCalibrationError and MaximumCalibrationError to ignite.metrics,
implementing the model-calibration metrics requested in #1009 ("as coded in
Baal").

Both metrics share a fixed-size, per-bin accumulator (sum of confidences,
correct count, and sample count per bin) updated in place with index_add_, so
the retained state is O(num_bins) regardless of dataset size. This avoids the
O(N) torch.cat growth of the prior attempt (#3132), which OOMs on large
datasets. compute() is fully vectorized (no Python loop over bins) and uses
@reinit__is_reduced / @sync_all_reduce for correct reduction under distributed
training. y_pred is expected to be probabilities; binary (B,)/(B, 1) and
multiclass (B, C)/(B, C, ...) inputs are supported.

Register both metrics in ignite.metrics, document them in
docs/source/metrics.rst, and add unit and distributed tests validated against a
NumPy reference of the Baal binning.

Closes #1009
```

PR Link: https://github.com/pytorch/ignite/pull/3799

PR Description: Added Expected and Maximum Calibration Error metrics to ignite.metrics (Fixes #1009).

Maintainer Feedback:.

Status: Awaiting review.
**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** Implementation complete on `fix-issue-1009`; PR to be opened in Phase IV.

---

## Learnings & Reflections

### Technical Skills Gained

- How PyTorch-Ignite's `Metric` base class enforces distributed correctness through the `@reinit__is_reduced` / `@sync_all_reduce` decorators, and why naive accumulation breaks under multi-GPU.
- Reasoning about memory complexity (O(N) vs O(1)) of an online metric and measuring it empirically.
- Vectorized histogram-style accumulation with `index_add_` / `clamp` instead of Python loops over bins.
- Writing device-parametrized and distributed tests against an independent NumPy reference.

### Challenges Overcome

- Reconciling the probabilities-vs-logits convention with ignite's existing shape-checking helpers (chose explicit shape handling over `_BaseClassification`).
- Handling the `conf == 1.0` bin-edge case and float32/float64 boundary sensitivity in tests.

### What I'd Do Differently Next Time

- Confirm the input convention (probabilities vs logits) with a maintainer comment on the issue *before* implementing, since it shapes the whole API.

---

## Resources Used

- [Baal calibration metrics (reference implementation)](https://github.com/baal-org/baal)
- [PyTorch-Ignite — How to create a custom metric](https://docs.pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric)
- [pytorch/ignite issue #1009](https://github.com/pytorch/ignite/issues/1009) and [prior attempt PR #3132](https://github.com/pytorch/ignite/pull/3132)
- [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
- Existing in-repo metrics studied: `ignite/metrics/entropy.py`, `ignite/metrics/confusion_matrix.py`, `ignite/metrics/accuracy.py`


# Contribution 2: [ENH] Refactor the ExhaustiveSearch algorithm / Implement Structural Intervention Distance

**Contribution Number:** 1  
**Student:** Rishabh Padhy
**Issue:** [pgmpy Issue #1766](https://github.com/pgmpy/pgmpy/issues/1766) and [Stalled PR #1927](https://github.com/pgmpy/pgmpy/pull/1927)  
**Status:** Phase I Complete

---

## Why I Chose This Issue

I chose to pick up the stalled implementation of the Structural Intervention Distance (SID) metric in `pgmpy` because it perfectly aligns with my background as a senior Computer Science student at Penn State.This issue requires implementing Judea Pearl's do-calculus and complex DAG traversal algorithms (finding valid backdoor adjustment sets) in pure Python, which is a fantastic bridge between discrete math and software engineering.

I am interested in this because:
1. I have a strong foundation in algorithmic graph theory, so I understand the math required to evaluate causal interventions.
2. The original PR (#1927) was abandoned, meaning my contribution will be to modernize the deprecated code (e.g., updating `BayesianNetwork` to `DiscreteBayesianNetwork`) and finish the remaining algorithmic logic. 
3. The maintainers are actively looking for someone to take this across the finish line. 
4. Contributing a highly respected research metric (SID) to a widely used probabilistic library will be a strong artifact for my graduate school applications.

From reading the issue thread and the stalled PR, I understand the current problem is that the SID metric computation was never fully completed or merged. My contribution will finalize this algorithm, allowing researchers to accurately measure how well a learned causal graph predicts real-world interventions compared to a true graph.

