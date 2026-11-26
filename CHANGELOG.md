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
  explicit target-miss diagnostics. The full 50/1,000-replication release runs
  remain pending.
- `sim.standard_rows` exposes the ordinary world generator with run-seed and δ
  overrides and stays byte-identical to `world.game_rows()` at default settings.
- SC4 misspecified generator with ordered-pair agent × deck and agent ×
  opponent effects and a deterministic kingmaker penalty (ADR 0005); the
  report check shows SDWP ordering is field-invariant while field values differ.
- A 50-replication SC1/SC2 `ci_smoke` study now runs in the CI `stat` job.
  SC1 pooled coverage passes; SC2 pooled coverage is 0.9095 and misses the
  [0.93, 0.97] target, recorded with its bias and per-quantity results in ADR
  0006. The 1,000-replication `standard` release run remains pending.
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
- Windows CI now runs the bootstrap `spawn`/serial parity subset.
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

## [0.0.1] - 2026-11-26

- Initial placeholder version reserved for the package name.
- P0 scaffold: Hatchling packaging, locked development dependencies, typed
  `podrank` package with `py.typed`, smoke tests, ruff/mypy/pytest configuration,
  documentation scaffold with an ADR template, and Linux/Windows CI.
