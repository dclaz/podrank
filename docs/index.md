# podrank

Documentation scaffold for `podrank`. The implementation is currently in the
**P2 rating core** phase: immutable registry snapshots, the order-preserving
`GameTable`, `validate`, the I0–I5 `Inclusion` pipeline, `fit`, posterior
contrasts and eligibility, fields, SDWP, prediction and persistence are
available. The §12 cold-fit and SDWP targets pass on the local synthetic
`perf_large` proxy; the official 16-core reference run is still required, and
end-to-end time is dominated by I0. The P3 cluster-bootstrap core is available
under `podrank.uncertainty`; ordinary `doc_tiny` and the `doc_tiny_balanced`
vertical slice pass, and serial/Linux `fork`/Linux `spawn` bootstrap parity is
tested. A Windows `spawn` parity step is wired into CI but has not yet run
green. `ci_smoke`, `standard` and `perf_large` world definitions,
deterministic SC3/SC4/SC5/SC6/SC7/SC8 checks and the SC3b crossover report check
exist; a 50-replication SC1/SC2 `ci_smoke` study runs in CI. SC1 passes pooled
coverage; SC2 misses it at `ci_smoke` size, recorded in ADR 0006. The full
`standard` SC1 study passes (1000 replications, pooled coverage 0.9434); the
`standard` SC2 run is scheduled on the main computer. Green Windows CI and the
16-core bootstrap benchmark remain pending. The comparison API is not exposed
yet. Details are recorded in
`docs/decisions/0002-p2-performance-spike.md`,
`docs/decisions/0004-bootstrap-workers.md`,
`docs/decisions/0005-sc4-generator.md` and
`docs/decisions/0006-sc1-sc2-ci-smoke.md`.

## Where the specification lives

- `IMPLEMENTATION_PLAN.md` is the specification: contracts, fixtures, tests and
  phased delivery.
- `AGENTS.md` is the implementation workflow.

Both live at the repository root and are the normative reference; these pages
will grow alongside each phase.

## The two questions

1. **How do these agents rank against a stated field, and by how much?**
   `podrank.rating` (R1) fits a batch Bayesian Luce winner model and reports
   ranks, contrasts, Standardised Decisive Win Probability (SDWP), and
   uncertainty.
2. **Is agent A better (or not worse) than agent B, and is the evidence
   sufficient yet?** `podrank.compare` (R2) uses declared blocks and an
   anytime-valid betting test.

## Working on the code

```bash
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --strict -p podrank
uv run --locked pytest -q -m 'not stat'
```
