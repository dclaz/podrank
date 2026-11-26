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

### Changed

- CI now runs the test suite under coverage on both Linux Python versions.

## [0.0.1] - 2026-11-26

- Initial placeholder version reserved for the package name.
- P0 scaffold: Hatchling packaging, locked development dependencies, typed
  `podrank` package with `py.typed`, smoke tests, ruff/mypy/pytest configuration,
  documentation scaffold with an ADR template, and Linux/Windows CI.
