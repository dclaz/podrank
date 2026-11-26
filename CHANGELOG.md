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

### Changed

- CI now runs the test suite under coverage on both Linux Python versions.
- Added `scipy-stubs` to the development group for strict typing.

## [0.0.1] - 2026-11-26

- Initial placeholder version reserved for the package name.
- P0 scaffold: Hatchling packaging, locked development dependencies, typed
  `podrank` package with `py.typed`, smoke tests, ruff/mypy/pytest configuration,
  documentation scaffold with an ADR template, and Linux/Windows CI.
