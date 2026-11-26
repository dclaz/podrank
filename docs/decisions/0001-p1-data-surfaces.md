# 0001. P1 data-layer surfaces not pinned by the plan

- **Status:** Proposed
- **Date:** 2026-11-26
- **Phase:** P1
- **Contract/section:** §3.4, §3.5, Appendix C.1

## Context

P1 must expose `pr.validate(...) -> ValidationReport` and accept an optional
`tag_schema` (§3.4), but the plan does not pin the report's fields, the exact
mapping shape for `tag_schema`, or the population used for per-entity counts.
`PROJECT_DECISIONS.md` is not present, so no project-specific tag schema exists.
The affected P1 work otherwise proceeds; this record proposes the surfaces used
by the implementation and tests.

## Decision

- `ValidationReport` carries `received`, `exact_duplicates`, the deduplicated
  and identity-validated `valid_rows`, the canonical `data_hash`, collected
  `errors`, duplicate `warnings`, and both registry `snapshot_id`s. It exposes
  `ok` and `deduplicated` convenience properties.
- `tag_schema` is a mapping whose ordinary keys map a deck-tag key to either
  `"wubrg"` (characters are restricted to `wubrg`, case-insensitively) or an
  iterable of allowed string values. The reserved key `"required"` maps to the
  required tag keys. A required key must be present; allowed values are checked
  only when the key is present.
- `CountsReport` per-entity `games` and `seat_appearances` describe the eligible
  population; overall counts and exclusion reasons remain the I3 counts.
- Registry `snapshot_id`s hash canonical FD3 JSON of rows sorted by ID. Rows are
  normalised defensively (sorted unique agent tags, sorted deck-tag maps).

## Consequences

- P2 can consume the report and counts without redefining their fields.
- The plan should either accept these surfaces or pin different ones before a
  release; the P1 tests currently pin the choices above.
- A future `PROJECT_DECISIONS.md` tag schema can be expressed in this encoding
  without adding project-specific defaults to the library.

## Alternatives considered

- Separate `TagSchema` dataclass — more explicit, but adds a public type the plan
  did not name; the mapping form can be wrapped later without breaking callers.
- Per-entity counts over all received rows — less useful for rating diagnostics
  and does not match the F7 hand counts for included rows.
- A free-form JSON object in `ValidationReport` — weaker typing and harder to
  test; typed fields are cheap to extend additively.
