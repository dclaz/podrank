# 0015. S3 forward split and S4 sigma-beta selection

- **Status:** Accepted
- **Date:** 2026-11-27
- **Phase:** P4
- **Contract/section:** §10.1, §11.4 S3/S4

## S3 forward split

`forward_split(..., n_windows=5, n_burn=20)` on the `standard` world fits on
earlier clusters and scores each later window once. Absent entities are excluded
from scoring and counted; the first up-to-20 new-entity clusters are burned into
the training fold.

| Window | Train clusters | Scored | Excluded | Full log-loss | Best baseline |
|---:|---:|---:|---:|---:|---|
| 1 | 120 | 448 | 0 | 1.229204 | θ+deck 1.247299 |
| 2 | 240 | 453 | 0 | 1.227409 | θ+deck 1.234675 |
| 3 | 360 | 447 | 0 | 1.210383 | θ+deck 1.232714 |
| 4 | 480 | 460 | 0 | 1.245080 | θ+deck 1.234593 |

Pooled over held-out clusters: full `1.229599`, best baseline `theta_deck`,
difference `-0.009173`, cluster SE `0.004940`, `not_worse=True`. The full model
is not worse than the best baseline by more than two SE.

### Whole-cluster CV

`cluster_cv(..., folds=5)` on the same standard data (whole clusters assigned to
folds before fitting, 2,263 held-out decisive games):

| Fold | Train clusters | Scored | Full log-loss | Best baseline |
|---:|---:|---:|---:|---|
| 0 | 480 | 451 | 1.208926 | θ+deck 1.221539 |
| 1 | 480 | 449 | 1.183161 | θ+deck 1.194774 |
| 2 | 480 | 446 | 1.234047 | θ+deck 1.246063 |
| 3 | 480 | 460 | 1.237182 | θ+deck 1.251290 |
| 4 | 480 | 457 | 1.236828 | θ+deck 1.233854 |

Pooled: full `1.222534`, best baseline `theta_deck`, difference `-0.009864`,
cluster SE `0.004165`, `not_worse=True`. Both S3 evaluation modes pass.

## S4 sigma-beta

Truth for the reference world is `σ_β = 0.5`; the acceptance range is
`[0.25, 1.0]`.

The first implementation omitted the prior normalisation
`-0.5·Σ log(2π/λ_i)` from the Laplace evidence. That made the evidence favour
large `σ_β`. With the term restored, the same standard dataset gives:

```text
0.25  -2812.9325
0.50  -2795.1238   <- selected
1.00  -2807.8372
2.00  -2833.2656
```

Laplace evidence now selects `0.5`, exactly the generating value. Five
independent outcome/composition replications (`standard_rows` with run seeds
7000–7004, same true deck effects) both selectors within the two-times range:

| Replication | Evidence | Cluster CV |
|---:|---:|---:|
| 0 | 0.5 | 0.5 |
| 1 | 0.5 | 0.5 |
| 2 | 0.5 | 0.5 |
| 3 | 0.5 | 0.25 |
| 4 | 0.5 | 0.25 |

The earlier evidence miss was an implementation defect in the evidence
constant, not a property of the model. No prior, world or tolerance was changed
to force agreement.

## Consequences

- S3 passes the "not worse than best baseline" report criterion in both forward
  and whole-cluster CV modes.
- S4 passes: evidence selects 0.5 on the reference dataset and on all five
  replications; cluster CV selects 0.25–0.5, also inside `[0.25, 1.0]`.
- `ModelSpec.sigma_beta` is persisted in the posterior archive (additive
  optional field, default 0.5 for older payloads).

- S3 passes the "not worse than best baseline" report criterion.
- S4 passes via cluster CV within 2× of truth; evidence selection is a reported
  miss.
- `ModelSpec.sigma_beta` is persisted in the posterior archive (additive
  optional field, default 0.5 for older payloads).
