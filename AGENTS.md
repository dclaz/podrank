# AGENTS.md

## Mission and scope

Build `podrank`, a Python library for rating fixed, versioned agents in four-player free-for-all games. R1 fits ratings from supplied results and reports ranks, contrasts, Standardised Decisive Win Probability (SDWP), and uncertainty. R2 adds pairwise comparison evidence from declared blocks. R1 must import and run without `podrank.compare`.

The library consumes records; it does not play games, train agents, schedule matches, manage checkpoints, or decide which agents to promote or release. Checkpoint-to-training-step joins and learning-curve plots belong to the caller. Deck and seat terms adjust ratings; this is not a deck-strength or deck-selection model. Do not add embeddings, engine/RL dependencies, or unrequested features.

## Read before editing

1. Inspect `git status --short`, existing code, and test/CI configuration. Preserve unrelated work and don't overwrite existing files just to match a proposed layout.
2. Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §0, §1, and §15. Then read only the contracts, §14 tests, Appendix A fixtures, and Appendix C interfaces relevant to the task. Use §13 for packaging. Read `PROJECT_DECISIONS.md` if present and relevant; its project-specific fields and tags are not generic library defaults.
3. Cite the applicable rule and fixture IDs in the task summary. The plan is the specification; this file is the workflow. If a material contract, fixture, or public signature conflicts with another or is underspecified, document the exact conflict and propose a short ADR in `docs/decisions/`. Stop only the affected work. Don't silently choose a side, relax a test, or adopt an unapproved behavioural change. An approved change updates the plan, fixture, §14 test list, and §15 gate together.

## Build order and first task

| Phase | Implement | Acceptance focus |
|---|---|---|
| P0 | Repository, `pyproject.toml` (Hatchling), uv lockfile, minimal typed package, smoke test, lint/type/test CI, docs scaffold and ADR template | Package imports; smoke test, lint and strict typing pass. Linux Python 3.12/3.13 and Windows smoke in CI. |
| P1 | Immutable registry snapshots, `GameTable`, validation and I0–I5 inclusion | F7 and boundary tests, including malformed rows through `fit` once wired, duplicate game IDs, manifest ownership, null singleton keys, and reconciled counts. |
| P2 | Luce rating core, compressed likelihood, solver, identification/eligibility, fields, prediction and posterior persistence | Relevant Appendix A fixtures, derivatives and invariance, field canonicalisation and round trips; record the §12 performance spike. |
| P3 | Weak-direction diagnostics, cluster bootstrap, scenario checks | F6 and F18; `doc_tiny_balanced` runs records → fit → eligibility → SDWP → bootstrap; §11/§15 gates satisfied. |

**If the repo is empty, do P0 only on the first pass.** Scaffold, lock dependencies, and run checks. Do not write a placeholder `fit` that appears to work or implement R2 scaffolding that exposes a public API. If P0 already exists, work on the earliest unmet gate unless the user directs otherwise. R1 ships after P3. After R1, P4 rating extras and P5 comparison are independent; P5 gates R2.

## Implementation rules

- Use Python 3.12+, `numpy>=2.0` and `scipy>=1.13` as the only core runtime dependencies. Keep optional backends behind their extras. Use typed boundary records, `py.typed`, explicit public exports, and the package boundaries in §13. No parallel frameworks or abstractions without a concrete testable need.
- Preserve the contract order for identity validation, deduplication, inclusion, and likelihood eligibility. Retain the source outcome/count information; a draw is not a winner and a failure is not a draw. Keep the four-seat ordering and canonical parameter and cluster keys stable.
- Use the plan's plain-text formulas, float64 numerics, solver stopping rule, seeded RNG streams and canonical serialisation. Keep the expanded likelihood as a small test oracle for compression, not a second production path. No explicit matrix inverse where a Cholesky solve suffices.
- Keep status, numeric value, headline eligibility and interval availability separate. The supported anchor has exact zero self-contrast but a numeric SDWP. Bootstrap eligibility is frozen from the original fit; replicate failures cannot suppress exact zero intervals or override an original-fit B4 suppression.
- Once implementing R2, treat stochastic different-ID reruns as diagnostics, not contradictions. Keep same-ID integrity checks, deterministic replay, event order and revisioned exports as specified. Compare complete-data means at a fixed horizon for S11, not means selected by stopping.

## Verification

Use the repo's configured entry points when present. For the P0 scaffold, configure the `dev` group and these commands. Run from the repository root:

```bash
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --strict -p podrank
uv run --locked pytest -q -m 'not stat'
```

Generate or regenerate `uv.lock` with `uv lock` only when bootstrapping or intentionally changing dependencies. Start with the narrow fixture/test for the change, then run the checks above. At the relevant phases, run `uv run --locked pytest -q -m stat` and the named §11 study, coverage, benchmark and release checks configured by CI. The fast test selection is not a substitute for the statistical gates.

- Prefer deterministic tests for contracts, ordering, counts, hashes, labels and errors. Use the specified tolerances for numeric assertions and explicit seeds for stochastic fixtures. Test save/load and the R1 import boundary where affected.
- Don't recalculate golden outputs from the function under test. Use an independent calculation or an existing fixture. Do not change seeds, tolerances or expected outputs just to get green. A failing statistical target is reported under §11/§15, not silently turned into a pass.
- Do not claim CI ran locally or claim a phase passed if only its narrow tests ran. If the environment blocks a check, report the exact command and blocker.

## Task handoff

Keep the diff scoped to one phase or a clearly bounded contract. At the end, report: (1) phase and rules/fixtures implemented, (2) checks run and outcomes, (3) blockers or unresolved ADRs, and (4) the next unmet acceptance gate. Do not commit, push, publish or change external systems unless explicitly requested.
