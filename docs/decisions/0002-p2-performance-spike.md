# 0002. P2 performance: compact pattern representation

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P2
- **Contract/section:** §12, §4.4, §4.5

## Context

The first P2 spike stored one dense `4 × P` design matrix per compressed
pattern. At `P ≈ 199` and one million patterns that is about 6.4 GB of design
payload alone. `max_dense_bytes = 2 GiB` is a dense linear-algebra workspace
guard, not a process-RSS cap, and did not cover that allocation. The §12 cold-fit
target also excludes parsing/validation and D2/D4 identification, so the first
end-to-end timing was not a like-for-like comparison.

## Decision

Replace the owned pattern representation with compact typed arrays:

- `(M, 4)` integer agent and deck indices, with swaps stored only when η is on;
- `(M, 4)` integer winner counts;
- compact cluster-to-pattern/winner count arrays and canonical cluster keys;
- no resident dense `4 × P` matrix per pattern.

Likelihood work uses the reduced `(θ, β, seat, η)` parameterisation with
`β[d] = μ[bracket(d)] + δ[d]`. The likelihood gradient and Hessian are mapped
back with the reduced-to-full Jacobian before adding the original full prior.
Curvature is accumulated from the six seat pairs over active feature entries.
Objective, gradient, Hessian and the stable Armijo delta share one softmax pass
at accepted Newton iterates.

D2 reuses the likelihood-only Hessian at the final accepted mode. D4 accumulates
the three unweighted seat-0 differences once per pattern and maps the Gram to
full coordinates before the existing eigenvalue cutoff. SDWP/FD6 uses the null
space of the stored row-space basis, so each difference projects onto 12
null-space directions rather than 187 row-space directions in the full spike.

`max_dense_bytes` remains **2 GiB** and is still only a dense-workspace guard.

## Results

Reproduce with:

```bash
uv run --locked python benchmarks/p2_spike.py
```

Synthetic `perf_large` shape (1,000,000 games, 250,000 clusters, 50 agents,
5 brackets × 28 decks, P=199, R=194). All one million patterns are unique, so
this is the low-compression/worst-case variant.

| Measurement | Result |
|---|---:|
| Games / patterns / parameters | 1,000,000 / 1,000,000 / 199 |
| Pattern arrays | **77.5 MiB** |
| Compression | 17.00 s |
| Newton / Cholesky | 6.59 s |
| **§12 cold fit (design + compression + Newton + Cholesky)** | **23.59 s** |
| D4 row space | 0.59 s (rank 187, nullity 12) |
| D2 likelihood Hessian | 1.29 s |
| SDWP, 10,000 contexts | **0.302 s** |
| I0 pipeline | 43.38 s |
| End-to-end `GameTable → fit` components | ≈ 68.9 s |
| Peak RSS | 4.47 GiB |
| RSS after input construction | 2.22 GiB |

The peak-RSS figure includes synthetic row generation and `GameTable`
construction; the fit itself is not a 4.47 GiB dense allocation. Solver:
3 accepted steps, 0 backtracks. Environment: Python 3.13.14, NumPy 2.5.5,
SciPy 1.18.1, Linux WSL2 x86-64; BLAS thread environment variables were unset.

**Hardware caveat.** This measurement came from a smaller WSL2 machine, not the
§12 reference 16-core desktop. Treat the numbers as provisional engineering
evidence, not as §12 acceptance. The reference run still needs the named
16-core machine, five fresh processes per workload, pre-import BLAS thread
pinning, and median/min/max reporting. The local result is useful because it
confirms the compact representation has the right order of magnitude and no
obvious regression; it does not freeze the performance baseline.

## Persistence follow-up (row-free JSON baseline)

The scoped FIX2 change aggregates outcome counts during/after eligibility,
removes `likelihood_rows` and `eligible_rows` from `Posterior`, retains the
compact cluster table and custom strata, and keeps the existing JSON envelope
for this step. A loaded posterior now retains everything needed for seeded
cluster bootstrap frame parity without the original `GameTable`.

On the same provisional 1M-game synthetic workload:

| Measurement | Result |
|---|---:|
| Posterior file | **76.7 MiB** |
| Save | 4.96 s |
| Load | 5.64 s |
| `outcome_rates()` | effectively instant (pre-aggregated) |
| Save/load parity | outcome counts, cluster table, compact cluster arrays and seeded frame weights match |

The earlier 40k run had a 40.5 MiB file with row retention; the row-free 40k
baseline is 4.64 MiB. These are local numbers, subject to the hardware caveat.

The schema-2 archive is now implemented (ADR 0003): on the same 1M workload it
is 84.2 MiB, saves in 0.67 s and loads in 0.50 s, with report/cluster/strata
parity. It is slightly larger than the row-free JSON baseline but about 7–11×
faster to save/load. C.2 now documents the manifest + `.npy` member format;
field and R2 formats are unchanged.

## I0 follow-up

The scoped I0 change looks up `game_id` first and computes canonical row bytes
only for repeated IDs. The canonical JSON array is streamed directly into
SHA-256 after deduplication and before filtering, in sorted `game_id` order.

On the provisional 1M synthetic workload, I0 dropped from **43.38 s to
33.75 s**. A 100k-game profile attributes roughly 58% of I0 to row
normalisation and about 20% to canonical `data_hash`; canonical JSON no longer
dominates. A separate versioned binary fingerprint does not appear warranted
yet. Revisit if normalisation work is removed and hashing becomes the dominant
remaining cost.

## Consequences

- Both §12 numerical targets measured here pass: cold fit < 30 s and SDWP 10k
  contexts < 0.5 s.
- End-to-end fit is still dominated by I0 (Python row validation/inclusion);
  that is a separate optimization target and does not affect the cold-fit
  measurement.
- `max_dense_bytes` remains a 2 GiB dense-workspace guard. If the project wants
  a 2 GiB total-process or peak-RSS target, that must be defined separately,
  including whether input construction is counted.
- No contract, fixture or public result shape changed. Compact and dense
  reference paths are checked for objective/derivative, stable-delta, D4/D2 and
  cluster-count equivalence.

## Alternatives considered

- Raise `max_dense_bytes` and keep dense pattern designs — rejected; it hides
  the 6.4 GB allocation and leaves the Python-loop speed problem.
- Require Numba/Cython — not needed for these measurements; keep an optional
  compiled kernel as a fallback only if a later full-size profile shows
  accumulation still dominates after this refactor.
- Solve directly in a different β-model — rejected; it would change the prior,
  deck-nuisance interpretation and identification results.
