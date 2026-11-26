# 0009. Prediction batching, ingestion fast paths and study parallelism

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P3
- **Contract/section:** §4.7, §11.3, §12

## Context

The post-ADR-0008 audit left five candidates: compact `predict_proba`, ingestion
cost in `GameTable`/I0, the Newton Hessian, bootstrap refits, and the remaining
Python passes in pattern construction. The user also asked whether Numba or
parallelism would help.

## Decisions

### Compact prediction batch

`Posterior._predict_proba_compact` builds the compact agent/deck arrays once,
computes all probabilities in one batched softmax, and evaluates the FD6
null-space residuals in chunks rather than constructing a dense `4 × P` design
per context. The dense/small-fixture path remains as the reference.

### Ingestion fast paths

- `GameTable.from_records` clones JSON-like containers directly instead of
  `copy.deepcopy(dict(record))`, and avoids the extra top-level dict copy.
- RFC 3339 parsing and canonical timestamp formatting are cached (8192
  entries); non-`played_at` field names keep their original error path.
- `_sequence4` returns the supplied sequence instead of copying it to a list.
- Required/unknown field checks use the mapping's key view.

### Study replication parallelism

`run_study(..., workers=1)` spreads independent replications over a process
pool. `workers=1` is still the default; invalid values are rejected; results
are merged in replicate order and covered by a serial/parallel parity test.
Each worker keeps its bootstrap serial. One BLAS thread per worker remains a
reference-run requirement configured before Python starts.

## Measurements

Local dev machine; 400k-row synthetic `perf_large` shape unless stated.

| Measurement | Before | After |
|---|---:|---:|
| `predict_proba`, 10k contexts | 0.534 s | 0.144 s |
| `GameTable.from_records`, 400k rows | 6.9 s | 4.4 s |
| I0 pipeline, 400k rows | 12.4 s | 10.1 s |
| End-to-end process, 400k rows | 30.8 s | 25.3 s |
| Study 10 reps, standard SC1, 1 worker | 26.4 s | — |
| Study 10 reps, standard SC1, 4 workers | — | 10.2 s |
| Bootstrap 100 refits, 1 worker | 2.14 s | — |
| Bootstrap 100 refits, 4 workers | — | 0.69 s |

The 10-rep study produced the same pooled coverage (`0.9341463414634147`) at
1 and 4 workers.

## Numba investigation (negative result)

Numba 0.68.1 installs and imports on Python 3.13. A parallel `prange` kernel for
the reduced Hessian accumulation was measured:

| Workload | NumPy | Numba warm | Numba cold |
|---|---:|---:|---:|
| Hessian, 200k patterns | 0.254 s | 0.152 s | 1.74 s (JIT) |

The warm kernel is about 1.7× faster, but the JIT compile costs about 1.7 s.
On the real 400k-row fit the Numba path made Newton slower (≈3.0 s vs ≈2.7 s)
because the one-shot fit pays the compile cost before seeing the savings. The
optional backend was removed; the core remains NumPy/SciPy and the cold-fit
target is already met. Revisit only if the reference run shows a much larger
pattern count and a warmed process.

## Remaining

- I0 is now dominated by per-row validation and canonical JSON hashing; further
  gains need a columnar `GameTable`/normalisation design, not a local loop
  rewrite.
- Newton is about 2.7 s at 400k rows and remains within the §12 cold-fit target.
- Bootstrap refits are now the natural parallelism boundary; the 16-core
  reference run remains pending.

## Verification

- `test_compact_predict_proba_matches_manual_designs` compares the batched
  probabilities, failing seats and labels with `design_row`/`aliasing_residual`.
- `test_study_worker_parity` compares serial and two-worker reports.
- `226` fast tests and `3` statistical tests pass; the SC2 `ci_smoke` miss is
  unchanged and remains recorded in ADR 0006.
- Coverage is about 91%; Ruff, strict Mypy, MkDocs strict and `uv build` pass
  on Python 3.12 and 3.13.
