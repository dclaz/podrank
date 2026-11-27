# Define a field

**Summary.** A field is a finite, weighted list of evaluation contexts: four
decks, three opponents, a focal seat and optional swaps. `podrank` canonicalises
and hashes the list, then evaluates SDWP on it; choosing the field is a
modelling decision the caller owns.

## Contexts and weights

Each context fixes:

| Key | Value |
|---|---|
| `decks` | four deck IDs in seat order |
| `swap_counts` | four non-negative integers (default all zero) |
| `opponents` | three agent IDs, one for each non-focal seat |
| `focal_seat` | `0..3` |
| `weight` | positive finite number |

The focal agent is not named in the context; SDWP evaluates every eligible agent
in that seat in turn. A context's weight is relative: `from_contexts` normalises
all weights to sum to one.

## Build directly from contexts

```python
import podrank as pr

field = pr.Field.from_contexts(
    "tiny_field",
    1,
    [
        {
            "decks": ("deck_b0_00", "deck_b0_01", "deck_b0_02", "deck_b0_03"),
            "swap_counts": (0, 0, 0, 0),
            "opponents": ("agent_00", "agent_01", "agent_02"),
            "focal_seat": 0,
            "weight": 1.0,
        },
        {
            "decks": ("deck_b1_00", "deck_b1_01", "deck_b1_02", "deck_b1_03"),
            "swap_counts": (0, 0, 0, 0),
            "opponents": ("agent_00", "agent_01", "agent_02"),
            "focal_seat": 1,
            "weight": 1.0,
        },
    ],
)
print(len(field.contexts), [c.weight for c in field.contexts])
print(field.field_hash[:16])
```

```text
2 [0.5, 0.5]
7d711700acc15464
```

Canonicalisation guarantees:

- duplicate contexts (same decks, swaps, opponents and focal seat) are merged
  by summing weights;
- contexts are sorted by canonical key: focal seat, decks, swaps, opponents;
- weights are normalised to one and checked to `1e-12`;
- the SHA-256 hash is computed over `field_id`, `version` and the canonical
  contexts. Reordering or splitting duplicate inputs gives the same hash.

Two fields with different `field_id` or `version` have different hashes even
when their contexts match. The hash is the identity used in report provenance
and bootstrap diagnostic keys.

## Build from a recipe

`Field.materialise(recipe, decks=...)` draws a reproducible field from a
declared recipe:

```python
recipe = {
    "field_id": "my_field",
    "version": 1,
    "recipe": {
        "deck_split": "train",
        "pod": {
            "bracket_policy": "same_bracket",
            "bracket_weights": "equal",
            "min_decks_per_bracket": 4,
            "distinct_decks": True,
            "swap_counts": 0,
        },
        "opponents": {
            "pool": ["agent_00", "agent_01", "agent_02"],
            "draw": "independent_uniform_per_seat",
        },
        "focal_seat": "uniform",
        "n_contexts": 24,
        "seed": 101,
    },
}
world = pr.sim.reference_world("doc_tiny")
field = pr.Field.materialise(recipe, decks=world.deck_registry())
print(len(field.contexts), field.field_hash[:16])
```

Recipe rules:

| Key | Allowed values |
|---|---|
| `deck_split` | `train`, `validation`, `holdout` |
| `pod.bracket_policy` | `same_bracket` or `any` |
| `pod.bracket_weights` | `equal` |
| `pod.min_decks_per_bracket` | integer; at least 4 are required in practice |
| `pod.distinct_decks` | `true` |
| `pod.swap_counts` | non-negative integer, applied to all four seats |
| `opponents.pool` | non-empty, duplicate-free list of agent IDs |
| `opponents.draw` | `independent_uniform_per_seat` |
| `focal_seat` | `uniform` |
| `n_contexts` | positive integer |
| `seed` | non-negative integer |

With `same_bracket`, brackets that cannot supply four decks in the chosen split
are excluded with a warning; if none qualify, `NoEligiblePodsError` is raised.
The recipe is stored in the field's provenance together with the deck snapshot
ID and the library version, but only the contexts participate in the hash.

For `doc_tiny` with 24 contexts, the recipe above produces a field whose
contexts start:

```text
focal_seat=0 decks=('deck_b0_01', 'deck_b0_03', 'deck_b0_02', 'deck_b0_00')
            opponents=('agent_00', 'agent_02', 'agent_01') weight=1/24
```

## Persist and reload

```python
field.save("fields/my_field_v1.json")
loaded = pr.Field.load("fields/my_field_v1.json")
assert loaded.field_hash == field.field_hash
assert loaded.contexts == field.contexts
```

The JSON envelope has three keys: `payload`, `field_hash` and `provenance`.
The hash is computed over the payload only; provenance is preserved but not
rehashed. The loader rejects duplicate JSON keys, non-canonical order, weight
violations and hash mismatches with `FieldIntegrityError`.

## Evaluate

```python
post = pr.fit(
    pr.GameTable.from_records(world.game_rows()),
    world.agent_registry(),
    world.deck_registry(),
)
report = post.sdwp(field)
for row in report.rows:
    print(row.agent_id, row.sdwp, row.field_label, row.failing_contexts)
```

`sdwp` performs support checks before evaluation:

- every deck must have likelihood-used games;
- every opponent must be supported and in the anchor component;
- with the swap effect off, every context's `swap_counts` must be zero.

Failures raise `UnsupportedFieldError` or `SwapsNotModelledError`; they are not
silently replaced by a nearby field. A field that references supported entities
but contains seat-utility differences the data never determined is still
evaluated, and each affected agent's row is labelled `prior_dependent_field`
with the failing context indices. That label is reporting only: the value is
kept and the model is unchanged (FD6).

## Pinned example

The frozen `doc_tiny` world ships `sim_field` at 24 contexts. Its hash and SDWP
ordering are:

```text
field_id='sim_field' version=1 contexts=24
field_hash prefix=203392dd51855ad9
agent_05 0.454114
agent_04 0.400392
agent_03 0.390259
agent_00 0.321843
agent_02 0.284809
agent_01 0.214389
```

Every row is `field_identified` because `doc_tiny` is fully connected and the
field contexts stay within supported differences.

## Rules and fixtures

- §4.6: field semantics, canonicalisation, hashing and persistence.
- FD1–FD6: duplicate keys, context-version fingerprint, recipes, round-trip,
  weight preservation and field identification.
- Fixtures: F8 (SDWP), F8b (canonicalisation and hash), F8c (round-trip with
  raw weights), F15 (field identification), F20 (swaps and prediction),
  F23 (persistence).
