# API reference

The public surface is grouped by package. Every name here is exported from the
package shown and is covered by the R1 test suite.

| Package | Use |
|---|---|
| [`podrank.data`](data.md) | registries, `GameTable`, inclusion and `validate` |
| [`podrank.rating`](rating.md) | `fit`, `Posterior`, fields, identification reports, prediction |
| [`podrank.uncertainty`](uncertainty.md) | cluster bootstrap and its diagnostics |
| [`podrank.sim`](sim.md) | frozen reference worlds and scenario generators |
| [`podrank.errors`](errors.md) | exception and warning types |
| [`podrank.compare`](compare.md) | pod distributions, descriptors, block scoring, betting and the comparison state machine |
| [`podrank.diagnostics`](diagnostics.md) | decisive-game log-loss, Brier, baselines and reliability bins |

The top-level `podrank` package re-exports the data, rating and error names that
R1 users need, plus the `sim` and `uncertainty` subpackages. Importing
`podrank` does not import `podrank.compare`.
