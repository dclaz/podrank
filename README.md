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

**P1 data layer.** The package provides immutable `AgentRegistry` and
`DeckRegistry` snapshots, an order-preserving `GameTable`, `validate`, and the
shared I0–I5 inclusion pipeline (`Inclusion`). The rating (`fit`) and comparison
APIs are not implemented yet. Implementation phases follow
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
