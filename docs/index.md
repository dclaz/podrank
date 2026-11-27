# podrank

`podrank` rates fixed, versioned agents in four-player free-for-all games. It
consumes game records; it does not play games, train agents, schedule matches or
decide which agents to promote.

The R1 rating core is available now: immutable registry snapshots, an
order-preserving game table, the I0–I5 inclusion pipeline, a Bayesian additive
Luce fit, identification and headline eligibility, fields and SDWP, prediction,
posterior persistence and the cluster bootstrap. The R2 comparison core is available: pod distributions, descriptors, block
scoring, betting, the `Comparison` state machine, confidence sequences, reports
and persistence. The full predefined-world S5/S11 run is recorded in ADR 0013; real-log
validation remains an open R2 release item. See the
[limitations](limitations.md) for what R1 deliberately does not do.

## Install

Requires Python 3.12+; NumPy and SciPy are the only core dependencies. From a
checkout:

```bash
uv sync --locked --group dev
```

## Quickstart

This example uses the frozen `doc_tiny` reference world (6 agents, 3 brackets,
12 decks, 60 clusters of 4 games, world seed 101). It is the same world used by
the concept pages, so the numbers are reproducible.

```python
import podrank as pr

world = pr.sim.reference_world("doc_tiny")
rows = world.game_rows()
agents = world.agent_registry()
decks = world.deck_registry()

games = pr.GameTable.from_records(rows)

# Optional preview of I0 steps 1-3. fit always runs the full pipeline.
report = pr.validate(games, agents, decks)
assert report.ok

post = pr.fit(games, agents, decks)
counts = post.counts()
print(counts.eligible, counts.likelihood_used)
# 240 223
```

`eligible` counts games that pass inclusion; `likelihood_used` is smaller
because draws and caps are not wins. The model describes who wins, given that
the game is decisive.

```python
for row in post.log_worth():
    print(row.agent_id, round(row.value, 6))
# agent_00  0.0
# agent_01 -0.599385
# agent_02 -0.190913
# agent_03  0.326982
# agent_04  0.373443
# agent_05  0.614616
```

Log-worth is relative to the anchor (`agent_00`), whose value is exactly zero.

```python
field = world.field(24)
for row in post.sdwp(field).rows:
    print(row.agent_id, round(row.sdwp, 6))
# agent_05 0.454114
# agent_04 0.400392
# agent_03 0.390259
# agent_00 0.321843
# agent_02 0.284809
# agent_01 0.214389
```

SDWP is the focal agent's conditional decisive win probability averaged over the
field's fixed contexts. It ignores draws and caps; read it beside
`post.outcome_rates()`.

```python
boot = pr.uncertainty.cluster_bootstrap(post, n=100, seed=101)
result = boot.contrast("agent_04", "agent_00")
print(round(result.value, 6), round(result.interval.lo, 6), round(result.interval.hi, 6))
# 0.373443 0.10017 0.694306
```

The point value is the original-fit MAP contrast; the interval is the
cluster-bootstrap percentile range.

To rate your own logs, see [fit from logs](how-to/fit-from-logs.md). To choose an
evaluation distribution, see [define a field](how-to/define-a-field.md).

## What R1 reports

| Output | Meaning |
|---|---|
| `post.log_worth()` | agent log-worth relative to the anchor, with Laplace intervals |
| `post.contrast(a, b)` | any agent, seat, deck or vector contrast |
| `post.seats()`, `post.decks()` | mean-centred seat effects and deck nuisance values |
| `post.headline_eligibility()` | D1–D5 status and headline inclusion per agent |
| `post.components()`, `post.identification_report()` | connectivity, weak directions, gauge count |
| `post.sdwp(field)` | MAP plug-in SDWP per headline-eligible agent |
| `post.outcome_rates()` | empirical draw, cap and failed rates |
| `post.predict_proba(contexts)` | outcome-free four-seat win probabilities |
| `pr.uncertainty.cluster_bootstrap(post, ...)` | block-aware intervals |

## Learn more

- [Ratings and fields](concepts/ratings.md): the model, log-worth and SDWP.
- [Identification](concepts/identification.md): components, weak directions and
  prior-dependent outputs.
- [Uncertainty](concepts/uncertainty.md): Laplace versus cluster bootstrap.
- [Assumptions](assumptions.md): what the data and the caller must provide.
- [Limitations](limitations.md): what the model does not claim.
- [API reference](reference/index.md)
- [Decisions](decisions/index.md): the ADRs behind the implementation.

## Where the specification lives

- `IMPLEMENTATION_PLAN.md` is the normative specification: contracts, fixtures,
  tests and phased delivery.
- `AGENTS.md` is the implementation workflow.

## Working on the code

```bash
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --strict -p podrank
uv run --locked pytest -q -m 'not stat'
```
