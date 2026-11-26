# 0008. Vectorised bootstrap setup and pattern compression

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P3
- **Contract/section:** §4.4, §5.3, §12

## Context

After batching the bootstrap SDWP (ADR 0007), an audit of the calculation
paths found four remaining Python-level loops that scaled with the number of
clusters, replicate modes or field contexts:

- `_support_maps` iterated every `(pattern, cluster)` entry and every seat to
  add frames to Python sets.
- `_gate` looped over every replicate weight for every reported quantity.
- `_replicate_targets` used `np.add.at` and rebuilt the flat target indices for
  every refit.
- `build_patterns` grouped rows through a Python `defaultdict` of lists and
  iterated every pattern group.

The Laplace SDWP path also still called `_sdwp_for_draw` per posterior draw.

## Decision

- `_support_maps` encodes `(entity, frame)` pairs as integers, uses
  `np.unique` and `searchsorted` to build the per-agent and per-deck frame
  sets, and never loops over seats or patterns in Python.
- `_gate` computes the zero-support mask across all replicate rows at once
  (`weights[:, support] != 0`) and counts once.
- `_replicate_targets` uses `np.bincount` on flat `pattern * 4 + winner`
  indices. The worker state precomputes the flat index once, so a refit only
  scatters the sampled cluster values.
- `build_patterns` encodes `(agents, decks, swaps)` as integer columns,
  groups with `np.unique(axis=0, return_inverse=True)`, builds winner counts
  with `np.bincount`, and builds cluster counts with one `np.unique` over
  `(pattern, winner, cluster)` triples.
- `Posterior.sdwp(method="laplace")` uses `_sdwp_matrix` for compact
  posteriors; the dense fixture path keeps `_sdwp_for_draw`.

## Measurements

Local dev machine unless stated; `benchmarks/p2_spike.py` for the synthetic
`perf_large` shape.

| Measurement | Before | After |
|---|---:|---:|
| `_support_maps`, standard 600 clusters | 5.9 ms | 2.3 ms |
| `_gate` | 0.54 ms | 0.10 ms |
| Bootstrap SDWP, standard 20 agents | 0.43 s | 0.10 s |
| Laplace SDWP, 100 draws × 20 agents × 400 contexts | per-draw path | 0.11 s |
| `build_patterns`, 40k rows / 40k patterns | 0.345 s | 0.089 s |
| `build_patterns`, 400k rows / 400k patterns | 6.14 s | 1.35 s |
| Cold fit (design + compression + Newton + Cholesky), 400k rows | 9.12 s | 4.33 s |

The 400k cold fit is still dominated by Newton/Cholesky (2.98 s) and the
remaining pattern-index construction (part of the 1.35 s compression), not by
Python grouping. The full 1M synthetic cold fit is expected to improve from
about 23.6 s to roughly 10 s; the reference run is still pending.

## Equivalence

- The vectorised `build_patterns` arrays are byte-identical to the previous
  implementation on a 40k-row / 40k-pattern synthetic frame
  (`agent_indices`, `deck_indices`, `targets`, cluster arrays, reduction and
  ordering).
- `test_batched_sdwp_matches_per_draw` checks the batched SDWP against the
  per-draw reference at `rtol=1e-12`.
- The 224 fast tests and the 3 statistical tests pass; the SC1/SC2 `ci_smoke`
  coverages are unchanged to the displayed precision.

## Remaining hotspots (not changed here)

- `GameTable.from_records` and I0 ingestion: about 6.9 s and 12.4 s for 400k
  rows. These are outside the §12 cold-fit definition; improving them needs a
  columnar `GameTable`/normalisation change, not a local vectorisation.
- Newton/likelihood Hessian: 2.98 s at 400k rows (6.6 s at 1M, provisional).
  The kernels are already batched; the six seat-pair `bincount` accumulation is
  the next possible micro-optimisation, but it is within the current target.
- `predict_proba`: 0.53 s for 10k contexts, still a per-context loop building a
  dense `4 × P` design. The compact batched path is the obvious next step.
- Bootstrap refits: about 2.1 s per 100 refits on `standard`; these are full
  Newton solves and are expected to dominate once the report path is batched.

No model, prior, tolerance or statistical contract changed.
