# podrank

`podrank` rates and compares **fixed policies (agents)** in **four-player
free-for-all games with a single winner or a draw**. It consumes result records;
it does not play games, train agents, schedule matches, or decide what to do with
the ratings.

| Question | Module | Kind of evidence |
|---|---|---|
| How do these agents rank against a stated field, and by how much? | `podrank.rating` (core, R1) | Model-based: batch Bayesian Luce winner model |
| Is agent A better (or not worse) than agent B, and is the evidence sufficient yet? | `podrank.compare` (optional, R2) | Design-based: block scores + anytime-valid betting test |

R1 never depends on `podrank.compare`.

## Status

**P2 rating core implemented.** The package provides immutable `AgentRegistry`
and `DeckRegistry` snapshots, an order-preserving `GameTable`, `validate`, the
shared I0–I5 inclusion pipeline, and the R1 rating core: `fit`, posterior
contrasts/components/eligibility, reference fields, SDWP, prediction and
persistence. The §12 cold-fit and SDWP targets pass on the local synthetic
`perf_large` proxy; the official 16-core reference run is still required, and
end-to-end time is currently dominated by I0. See
[`docs/decisions/0002-p2-performance-spike.md`](docs/decisions/0002-p2-performance-spike.md).
The P3 cluster-bootstrap core (`podrank.uncertainty`), tuple-key diagnostics,
serial/Linux `fork`/Linux `spawn` replicate execution, ordinary `doc_tiny`,
the `doc_tiny_balanced` vertical slice and a Windows `spawn` parity CI step are
available. `ci_smoke`, `standard` and `perf_large` world definitions plus
deterministic SC5/SC6/SC8 smoke checks exist. Windows CI has not yet run green;
SC1/SC2/SC3/SC3b/SC4/SC7 full gates and the 16-core bootstrap benchmark remain
pending. The comparison API is not implemented. Implementation phases follow
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §15.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --strict -p podrank
uv run --locked pytest -q -m 'not stat'
```

Documentation is built with `uv run --locked mkdocs build`.

## Documentation

Concept pages, assumptions, limitations and decision records live in `docs/`.
Architectural decisions use the template in `docs/decisions/`.
