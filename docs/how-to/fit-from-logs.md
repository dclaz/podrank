# Fit from logs

**Summary.** Turn recorded games into an immutable table, bind them to agent and
deck registry snapshots, and call `pr.fit`. The same pipeline runs whether or
not you call `validate` first; `validate` is a preview.

## 1. Prepare three inputs

`fit` needs:

1. a `GameTable` of game records;
2. an `AgentRegistry` with exactly one anchor;
3. a `DeckRegistry` with a bracket and split for every deck.

Registry rows are plain mappings:

```python
agent_rows = [
    {"agent_id": "heur", "kind": "heuristic", "policy_version": "h1", "is_anchor": True},
    {"agent_id": "ckpt_10", "kind": "checkpoint", "policy_version": "c10"},
]
deck_rows = [
    {"deck_id": "deck_a", "bracket": 0, "split": "train", "deck_version": "a1"},
    {"deck_id": "deck_b", "bracket": 0, "split": "train", "deck_version": "b1"},
]
```

`kind` is one of `random`, `heuristic`, `checkpoint`, `exploiter`, `other`.
`split` is one of `train`, `validation`, `holdout`. Each registry is sorted and
content-hashed when constructed; the snapshot IDs appear in the posterior's
provenance.

## 2. Load game records

Required fields per game:

| Field | Type | Notes |
|---|---|---|
| `game_id` | string | unique; exact duplicates are removed, conflicting ones raise |
| `played_at` | RFC 3339 string | explicit offset; normalised to UTC |
| `agents` | four strings | seat order |
| `decks` | four strings | seat order |
| `policy_versions` | four strings | must match the agent snapshot |
| `outcome` | `win` / `draw` / `cap` / `failed` | |

Optional fields: `deck_versions`, `swap_counts` (default all zero), `suite`,
`cluster_id`, `block_id`, `rotation`, `role`, `deal_id`, `spec_hash`,
`engine_version`, `audit`. For a `win`, `winner_seat` is required and must be
`0..3`; for every other outcome it must be absent or `None`.

A CSV loader using only the standard library:

```python
import csv
import json

REQUIRED = ("game_id", "played_at", "agents", "decks", "policy_versions", "outcome")


def load_games(path):
    records = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = {key: row[key] for key in REQUIRED if row[key] != ""}
            for key in ("agents", "decks", "policy_versions"):
                record[key] = json.loads(row[key])          # ["a", "b", "c", "d"]
            if row.get("winner_seat"):
                record["winner_seat"] = int(row["winner_seat"])
            for key in ("suite", "cluster_id", "engine_version"):
                if row.get(key):
                    record[key] = row[key]
            records.append(record)
    return records
```

The library is count-agnostic; it does not impose a file format. Convert
whatever your harness writes into mappings with these keys before constructing
the table.

## 3. Build the table and preview

```python
import podrank as pr

games = pr.GameTable.from_records(records)
agents = pr.AgentRegistry.from_records(agent_rows)
decks = pr.DeckRegistry.from_records(deck_rows)

report = pr.validate(games, agents, decks)
print(report.ok, report.received, report.deduplicated, report.data_hash)
```

`validate` runs I0 steps 1–3 (normalise, deduplicate, identity and mirror
checks) without the row or completeness filters. It raises on errors by
default; pass `strict=False` to collect `ValidationIssue` rows and inspect them.
`fit` runs the full I0–I5 pipeline, so an invalid row raises there too.

## 4. Fit

```python
post = pr.fit(games, agents, decks)
counts = post.counts()
print(
    counts.received,
    counts.exact_duplicates,
    counts.eligible,
    counts.likelihood_used,
    counts.likelihood_bearing_clusters,
)
# 240 0 240 223 60
```

The default model has seats, brackets, deck deviations and no swap effect. Use
`ModelSpec(swap_effect=True)` when swaps are recorded and should be modelled;
with the default spec, nonzero swaps require
`allow_pooled_variants=True`, which pools swapped variants instead.

Useful fit keyword arguments:

| Argument | Purpose |
|---|---|
| `spec=` | model settings (`ModelSpec`) |
| `inclusion=` | suites, engine versions, splits, agents, time range, complete clusters, manifest |
| `split_policy=` | which deck splits each agent or suite may use |
| `cluster_strata=` | custom bootstrap strata keyed by cluster |
| `tag_schema=` | required/allowed deck tags |
| `warm_start=` | values matched by parameter name for a refit |

`fit` always runs the same pipeline; there is no separate "fast path" that
skips validation.

## 5. Read the result

```python
post.counts()                 # received, eligible, likelihood-used, per-entity counts
post.log_worth()              # theta[a] - theta[anchor]
post.contrast("ckpt_10", "heur")
post.seats()                  # mean-centred seat effects
post.decks()                  # deck nuisance relative to train decks
post.outcome_rates()          # empirical draw/cap/failed rates
post.components()             # anchor component and any disconnected agents
post.headline_eligibility()   # D5 status and headline inclusion
post.identification_report()  # gauge count, weak directions
post.unrankable()             # agents/decks without likelihood-used games
```

With the frozen `doc_tiny` world (240 games, 60 clusters, 6 agents), the default
fit reports:

```text
received=240  exact_duplicates=0  eligible=240  likelihood_used=223
likelihood_bearing_clusters=60
```

`eligible` counts games that survive inclusion. `likelihood_used` is smaller
because draws and caps are not wins. Both counts reconcile:
`received - exact_duplicates = eligible + sum(excluded)`.

Note that `post.counts()` is a typed `CountsReport`, not a formatted string;
read its fields directly as in the example above.

## 6. Evaluate a field

```python
field = pr.Field.load("fields/sim_field_v1.json")   # saved with Field.save
report = post.sdwp(field)
for row in report.rows:
    print(row.agent_id, round(row.sdwp, 4), row.field_label)
```

The field must reference decks and opponents with likelihood-used games in the
anchor component; otherwise `sdwp` raises `UnsupportedFieldError`. With the swap
effect off, a field context with nonzero swaps raises `SwapsNotModelledError`.
See [define a field](define-a-field.md).

## 7. Save and reload

```python
post.save("posterior.podrank")
loaded = pr.Posterior.load("posterior.podrank")
assert loaded.sdwp(field).rows == report.rows
```

The archive contains the fitted state, the registries and the retained cluster
table, so reports, strata and seeded bootstrap frames reproduce without the
original `GameTable`. It is validated strictly on load and written atomically.

## 8. Add intervals

```python
boot = pr.uncertainty.cluster_bootstrap(post, n=400, seed=7)
print(boot.contrast("ckpt_10", "heur").interval)
print(boot.sdwp(field).rows[0].interval)
```

Use the cluster bootstrap when games share deals; use the Laplace intervals on
the posterior for a quick model-based check. See
[uncertainty](../concepts/uncertainty.md).

## Common errors

| Error | Cause |
|---|---|
| `ValidationError` | missing/unknown field, bad length, bad enum, naive timestamp |
| `DuplicateGameError` | same `game_id`, different canonical bytes |
| `UnknownEntityError` | agent or deck ID absent from the snapshot |
| `IdentityMismatchError` | zero or multiple anchors, or a policy-version mismatch |
| `ManifestMismatchError` | manifest ownership or complete-cluster mismatch |
| `NoRankableEvidence` | no likelihood-used games at all |
| `SwapsNotModelledError` | nonzero swaps with the swap effect off |
| `UnsupportedFieldError` | field references an unsupported deck or opponent |
| `ConvergenceError` | the Newton solver did not converge |
