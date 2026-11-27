# Run a comparison

**Summary.** A comparison is a declared, fixed-horizon or anytime-valid
statement about one pair of agents under a frozen pod distribution. Declare the
blocks before play, ingest the reported games, then read the status, report and
revisioned rating export.

## 1. Bind the snapshots and pod distribution

```python
import podrank as pr

agents = pr.AgentRegistry.from_records(
    [
        {"agent_id": "a", "kind": "checkpoint", "policy_version": "a1", "is_anchor": True},
        {"agent_id": "b", "kind": "checkpoint", "policy_version": "b1", "is_anchor": False},
    ]
)
decks = pr.DeckRegistry.from_records(
    [
        {"deck_id": f"d{index}", "bracket": 0, "split": "train", "deck_version": f"v{index}"}
        for index in range(1, 5)
    ]
)
pods = pr.compare.PodDistribution.from_support(
    "pods",
    1,
    "duplicate",
    [{"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 1.0}],
)
```

`PodDistribution` is canonical and hashed. `pods.sample(n, seed, start_index=0)`
draws pod records; the caller binds them into descriptors with the deck and
policy versions from the snapshots.

## 2. Declare blocks

```python
comparison = pr.compare.Comparison(
    a="a",
    b="b",
    design="duplicate",
    agents=agents,
    decks=decks,
    engine_version="engine",
    pod_distribution=pods,
    hypothesis=pr.compare.Superiority(0.0),
    alpha=0.05,
    max_blocks=4000,
    replay="deterministic",
)
comparison.declare_blocks(descriptors)
```

Declaration validates every descriptor atomically: contiguous indices, unique
IDs, design/engine/pod binding, deck versions, zero swaps, support membership
and policy versions. A bad batch raises `ValueError` and declares nothing.

## 3. Ingest reports

`ingest` accepts a `GameTable` or an iterable of row mappings. Each row must
match its declared block, deal, rotation, role and seat assignment. The returned
`IngestReport` counts accepted rows, rejects, transport duplicates and ID
conflicts. A same-ID, same-position row with a different outcome is an integrity
violation; it quarantines the position and invalidates the evidence.

Explicit failures use `mark_failed(block_id, rotation, role)` or the atomic
`mark_block_failed(block_id)`. `extend_budget(new_max)` continues a
`BUDGET_EXHAUSTED` comparison without changing the accumulated wealth.

## 4. Read the result

```python
print(comparison.status())          # CONTINUE, ABOVE, BELOW, BUDGET_EXHAUSTED, ...
report = comparison.report()
print(report.decision_at_stop, report.evidence_valid)
print(report.mean_lift_bounds, report.confidence_sequence)

table, export = comparison.rating_rows()
print(len(table), export.export_revision)
```

- `status()` is the **effective** status; `INCONCLUSIVE_INVALID` means a replay
  or integrity violation invalidated the evidence.
- `decision_at_stop` is historical and never rewritten.
- `rating_rows()` is a revisioned snapshot of consumed, non-quarantined
  successful rows plus failed game rows. Replace earlier exports; never append
  or feed the raw event log to a fit.
- A confidence sequence is `None` when nothing has been consumed or the
  evidence is invalid.

## 5. Persist and resume

```python
comparison.save("comparison.json")
resumed = pr.compare.Comparison.load("comparison.json")
assert resumed.status() == comparison.status()
assert resumed.rating_rows() == comparison.rating_rows()
```

The schema-1 JSON envelope contains the bound configuration, snapshots, pod
distribution, descriptors, event log, resolution/quarantine state, frozen
scores, wealth, budget and export revision. Loading does not replay games.

## What a comparison does not do

A comparison makes one claim under assumptions A1–A5. It does not aggregate
several comparisons, control family-wise error, choose a pod distribution or
decide a promotion/release policy. Those are caller responsibilities; each
comparison reports its evidence independently.

See [assumptions](../assumptions.md) and §7–§8 for the full rules.
