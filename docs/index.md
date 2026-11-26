# podrank

Documentation scaffold for `podrank`. The implementation is currently at the
**P1 data** phase: immutable registry snapshots, the order-preserving
`GameTable`, `validate`, and the shared I0–I5 `Inclusion` pipeline are available.
The rating and comparison APIs are not exposed yet.

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
