# Changelog

All notable changes to `podrank` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses
semantic versioning.

## [Unreleased]

### Added

- P1 data layer: immutable content-hashed `AgentRegistry` and `DeckRegistry`
  snapshots, an order-preserving raw `GameTable`, `validate`, `Inclusion`, and
  the shared I0–I5 pipeline with canonical `data_hash`, I3 counts, I1 row
  filters, I2 complete-only manifests, I4 decisive-game selection, and I5
  support/saturation diagnostics.
- P1 fixtures and tests for F7, F22, canonical hashing, boundary validation,
  manifest ownership, null singleton clusters, and counts reconciliation.
- Coverage enforcement at 90% on core in CI.
- P2 rating core: `ModelSpec`, compressed Luce likelihood and objective,
  damped Newton solver, `fit`, posterior contrasts/components/eligibility with
  D1–D5 statuses and Laplace intervals, canonical fields with materialisation
  and persistence, SDWP with FD6 labels, prediction, and posterior save/load.
- P2 fixtures/tests for F1, F5, F8, F8b, F8c, F10, F11, F11b, F14, F15, F17,
  F20, F20b and the P2 F22/F23 boundary cases.
- P2 performance refactor: compact typed pattern arrays, reduced-coordinate
  likelihood derivatives mapped back to the full parameter space, fused
  objective/gradient/Hessian/delta evaluation, null-space FD6 projection, and a
  reproducible `benchmarks/p2_spike.py`.
- P2 performance spike recorded in ADR 0002: 1M patterns, P=199, local cold fit
  23.59 s, SDWP 10k contexts 0.302 s, pattern arrays 77.5 MiB; the official
  16-core reference run is pending.
- Posterior row retention removed: outcome counts are aggregated by agent and
  suite, cluster table/strata are retained, and a loaded posterior reproduces
  reports and seeded bootstrap frame state without the original `GameTable`.
- Schema-2 posterior archive: small JSON manifest plus typed `.npy` members,
  with strict member/dtype/shape/size/cross-index validation and atomic replace.
  Local 1M baseline: 84.2 MiB, 0.60 s save, 0.52 s load; 24 members, 84.23 MiB
  uncompressed, within the documented loader limits.
- Crossing stratum labels are stored as canonical frame indices per column and
  resolve back to tagged cluster keys; crossing in one column does not affect
  other strata or unstratified state. Schema-1 inventory found no released
  artifacts, so unsupported schema-1 files fail without a migration path.
- I0 dedup now encodes canonical row bytes only for repeated `game_id`s and
  streams the canonical JSON array into SHA-256. Local 1M I0: 43.38 s → 33.75 s.

### Added

- P3 cluster-bootstrap core: compact-frame multinomial resampling, stratified
  resampling, frozen-index warm-start refits, B4/B5 suppression gates, B6
  percentiles, contrast/log-worth/seat/deck/swap/SDWP bootstrap reports and
  diagnostics under `podrank.uncertainty`.
- Bootstrap smoke tests for frame state, stratification, crossing rejection,
  failed replicates, entity gates and unavailable deck baselines.
- `podrank.sim.reference_world("doc_tiny_balanced")` with the frozen 60-cluster
  world, 24-context `sim_field`, and the end-to-end records → fit → eligibility
  → SDWP → 100-replicate bootstrap vertical slice.
- Ordinary `doc_tiny` world with the standard composition stream, shared-luck
  Gumbel noise, 5% draws and 1% caps.
- Frozen `ci_smoke`, `standard` and `perf_large` world definitions,
  deterministic SC5 disconnected-component, SC6 all/draw-only and SC8
  unsupported-anchor checks.
- Pinned-deck scenario generator with exclusive agent/deck pairs, reserved
  decisive clusters, anchor connectivity and the `v'Gv <= 1e-12 ||G||` D4
  null-space assertion; SC7 classifies both pinned agents `aliased` in every
  pass and omits them from headline SDWP.
- SC3b crossover augmentation: exactly two crossover clusters make the first
  pinned agent's anchor contrast pass D4 while the second stays aliased.
- SC3 sparse-support generator: 3/6/12 reserved clusters for the three sparse
  agents, an exclusive pinned deck, exact likelihood-bearing counts, pinned
  aliasing and `insufficient_clusters` bootstrap suppression for the 3- and
  6-cluster agents only.
- `difference_gram()` exposes the D4 Gram used by the scenario generator
  assertions without materialising difference rows.
- `podrank.sim.studies` SC1/SC2 harness: per-replication seeds, δ redrawn from
  N(0, 0.5²), `ρ_div` override, the deduplicated §11.3 scoring list, an
  independent SDWP truth calculation, and coverage/emission/bias reporting with
  explicit target-miss diagnostics. The 50-replication `ci_smoke` and
  1,000-replication `standard` SC1/SC2 release runs are recorded in ADRs 0006
  and 0010.
- `sim.standard_rows` exposes the ordinary world generator with run-seed and δ
  overrides and stays byte-identical to `world.game_rows()` at default settings.
- SC4 misspecified generator with ordered-pair agent × deck and agent ×
  opponent effects and a deterministic kingmaker penalty (ADR 0005); the
  report check shows SDWP ordering is field-invariant while field values differ.
- A 50-replication SC1/SC2 `ci_smoke` study now runs in the CI `stat` job.
  SC1 pooled coverage passes; SC2 pooled coverage is 0.9095 and misses the
  [0.93, 0.97] target, recorded with its bias and per-quantity results in ADR
  0006.
- Standard SC1/SC2 release-shape studies pass: 1000/1000 replications each,
  41,000/41,000 intervals, emission 1.0, pooled coverage 0.9433658536585365
  (SC1 rerun, identical to ADR 0006) and 0.9422195121951219 (SC2,
  `rho_div = 0`), no target misses and no per-quantity coverage below 90%
  (ADR 0010).
- R1 documentation: quickstart, ratings/fields, identification and uncertainty
  concepts, assumptions, limitations, glossary, the fit-from-logs and
  define-a-field how-tos, and a mkdocstrings API reference. Concept examples use
  the frozen `doc_tiny` world and show pinned outputs; `tests/test_docs_examples.py`
  pins log-worth, SDWP and bootstrap intervals to 1e-6.
- MkDocs math support (`pymdownx.arithmatex` plus MathJax) for the model
  formulas.
- R2 comparison data surfaces: canonical `PodDistribution` with a payload-only
  hash, atomic persistence and seeded `sample`; `BlockDescriptor` validation
  and `from_draw`; focal and duplicate/paired block scoring with E4 directional
  bounds; `summarise_blocks`; and the fixed-mixture `Wealth` with E1 thresholds
  (C1–C3, E1, F2–F4, F21 summary case).
- `podrank.compare` is exposed lazily as `pr.compare`, so importing or using R1
  never loads comparison code.
- R2 `Comparison` core: atomic descriptor declaration, C2 row matching, event
  classification (reject/transport duplicate/id conflict/integrity), stochastic
  and deterministic resolution, frozen block consumption, budget extensions,
  the failure rule, replay invalidation, status/report surfaces and revisioned
  rating export. Fixtures F9, F12, F16, F19, F21 and E5 are covered.
- R2 confidence sequences (E6/F13), full report contents, the §10.4 pairing
  report (duplicate variance ratio and paired matched draw/cap denominator),
  and schema-1 `Comparison.save/load` round-trips.
- Reduced S5/S11 runner (`benchmarks/r2_studies.py`): 72 S5 boundary cells at
  10,000 replications with no limit violations, and 16 S11 cells at 5,000
  replications with two recorded flags; the full predefined-world reference run
  remains open (ADR 0012).
- R2 public-name scope guard and comparison how-to/reference docs.
- P4 R1 diagnostics: `predictive_quality` (log-loss/Brier, baselines,
  cluster-level means, reliability), `posterior_predictive` (agent/deck/seat/tag
  z flags), `forward_split` (burn-in, absent-entity exclusion), `tune_sigma_beta`
  (Laplace evidence) and `tune_sigma_beta_cv` (whole-cluster CV).
- `ModelSpec.sigma_beta` is persisted in the posterior archive.
- P4 reports: markdown renderers, optional matplotlib reliability/PPC plots, and
  the `podrank calibrate --world …` CLI entry point.
- Whole-cluster K-fold CV evaluation (`cluster_cv`) added; S3 now reports both
  CV and forward split.
- S2a NumPyro/SBC reference and S3/S4 studies recorded in ADRs 0015–0016. S4's Laplace evidence now includes the prior normalisation, selects the true
  sigma_beta=0.5 on the reference dataset and five replications, and cluster CV
  remains within the 2x range.
- Field context/recipe validation tests cover the malformed-context and
  malformed-recipe branches.
- Batched bootstrap SDWP: `Posterior._sdwp_matrix` evaluates all agents and
  replicate modes together. On `standard`, 100-refit bootstrap SDWP drops from
  9.22 s to 0.41 s per replication; the 10-replication SC1/SC2 smoke drops from
  about 128 s to 33 s with identical coverage (ADR 0007).
- Vectorised bootstrap setup and pattern compression (ADR 0008):
  `_support_maps` 5.9 → 2.3 ms, `_gate` 0.54 → 0.10 ms, `build_patterns`
  6.14 → 1.35 s at 400k rows, and 400k-row cold fit 9.12 → 4.33 s. Laplace
  SDWP uses the batched matrix. Arrays and statistical results are unchanged.
- Compact `predict_proba` 0.534 → 0.144 s for 10k contexts; ingestion fast
  paths cut `GameTable.from_records` 6.9 → 3.3 s and I0 12.4 → 10.3 s at 400k
  rows (ADR 0009).
- `run_study(..., workers=...)` spreads replications over a process pool:
  10 standard SC1 reps 26.4 → 10.2 s on 4 workers with identical coverage.
- Numba was tested and rejected: the warm Hessian kernel is ~1.7× faster but
  the ~1.7 s JIT compile makes the one-shot 400k fit slower. Core stays
  NumPy/SciPy; process parallelism is the effective lever (ADR 0009).
- Fit-level reuse: the final Newton likelihood Hessian is used for D2 instead
  of recomputing it; cluster strata aggregate in one pass; the component
  union-find is array-indexed with identical persisted roots (0.735 → 0.478 s
  at 400k rows).
- Canonical row serialization now uses the C ASCII string encoder and fixed key
  order instead of `json.dumps(sort_keys=True)`, with cached quoting for
  repeated strings; row validation has exact-type fast paths. Row
  normalisation 1.21 → 0.91 s and canonical encoding 0.78 → 0.55 s per 100k;
  400k I0 10.3 → 8.55 s. 1M cold fit 9.87 s; full 1M fit including I0
  38.7–42.4 s; 1M SDWP 10k 0.27–0.33 s.
- A NumPy `unique`/`bincount` entity-count implementation was tested and
  reverted (0.343 vs 0.203 s per 100k); the combined Python pass is faster at
  these entity counts.
- Standard SC1 release-shape study passes: 1000/1000 replications, 41,000/41,000
  intervals, emission 1.0, pooled coverage 0.9433658536585365, no target misses.
  Standard SC1 was rerun with the same result, and standard SC2 (`rho_div = 0`)
  passes with 1000/1000 replications, 41,000/41,000 intervals, emission 1.0,
  pooled coverage 0.9422195121951219 and no target misses (ADR 0010).
- Bootstrap `spawn`/serial parity is covered by the Linux test suite.
- Controlled F6-style frozen-index/prior-only and F18-style gate precedence
  tests, including real deck-support B4 and all-failed B5 extension cases.
- Collision-safe tuple diagnostics keys for contrast, log-worth and
  field-hash-keyed SDWP.
- Standard-library process-pool bootstrap execution with picklable worker
  state, replicate-index RNG streams, ordered results and serial/Linux
  `fork`/Linux `spawn` parity tests.
- `workers=1` remains the no-pool default; worker-count changes are tested not
  to alter statistical results. ADR 0004 records the API and the benchmark-only
  BLAS one-thread policy.

### Changed

- CI now runs the test suite under coverage on both Linux Python versions.
- Added `scipy-stubs` to the development group for strict typing.

### Removed

- Windows support and the `windows-smoke` CI job. The process start method now
  follows the platform default instead of branching on `win32`; Windows is not
  a supported or tested platform (ADR 0011).

## [0.0.1] - 2026-11-26

- Initial placeholder version reserved for the package name.
- P0 scaffold: Hatchling packaging, locked development dependencies, typed
  `podrank` package with `py.typed`, smoke tests, ruff/mypy/pytest configuration,
  documentation scaffold with an ADR template, and Linux CI.
