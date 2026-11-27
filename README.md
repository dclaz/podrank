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

**R1 rating core implemented.** The package provides immutable `AgentRegistry`
and `DeckRegistry` snapshots, an order-preserving `GameTable`, `validate`, the
shared I0–I5 inclusion pipeline, and the R1 rating core: `fit`, posterior
contrasts/components/eligibility, fields, SDWP, prediction and persistence.
The P3 cluster-bootstrap core (`podrank.uncertainty`) provides tuple-key
diagnostics, serial and process-parallel replicate execution using the
platform's default start method, and the seeded, frozen-frame resampling rules. Ordinary `doc_tiny`, the
`doc_tiny_balanced` vertical slice, the R1 concept/assumption/how-to/reference
pages and a pinned-output documentation test are in place.

`ci_smoke`, `standard` and `perf_large` world definitions plus deterministic
SC3/SC4/SC5/SC6/SC7/SC8 checks and the SC3b crossover report check exist. The
50-replication SC1/SC2 `ci_smoke` study runs in CI: SC1 passes pooled coverage,
while SC2 misses it (ADR 0006). Both full `standard` studies pass at 1000
replications: SC1 pooled coverage 0.9434 and SC2 (`rho_div = 0`) 0.9422, with
emission 1.0 and no bias misses (ADR 0010). The §12 cold-fit and SDWP targets
pass on the local synthetic `perf_large` proxy; the official 16-core reference
run is still required, and end-to-end time is currently dominated by I0 (ADR
0002). The 16-core reference benchmark remains pending. The R2 comparison core is implemented: canonical pod distributions, block
descriptors, block scoring, fixed-mixture betting, the `Comparison` state
machine (validation, resolution, consumption, invalidation and revisioned
export), confidence sequences, full reports and schema-1 persistence. The full predefined-world S5/S11 run is recorded in ADR 0013; real-log
validation remains an open R2 release item.
Implementation phases follow [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §15.

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
Start with the [quickstart](docs/index.md), then
[ratings and fields](docs/concepts/ratings.md),
[identification](docs/concepts/identification.md),
[uncertainty](docs/concepts/uncertainty.md),
[assumptions](docs/assumptions.md) and
[limitations](docs/limitations.md). The
[API reference](docs/reference/index.md) is generated from the package.
Architectural decisions use the template in `docs/decisions/`.
