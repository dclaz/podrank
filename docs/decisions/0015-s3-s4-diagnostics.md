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

## S4 sigma-beta

Truth for the reference world is `σ_β = 0.5`; the acceptance range is
`[0.25, 1.0]`.

Laplace evidence over `[0.25, 0.5, 1.0, 2.0]` selects `2.0`:

```text
0.25  -2790.4356
0.50  -2739.3558
1.00  -2718.7982
2.00  -2710.9555
```

That is outside the two-times range, so it is reported as an evidence miss rather
than hidden. Five-fold whole-cluster CV on the same grid selects `0.25`
(log-loss `1.221791` vs `1.222534` at 0.5, `1.226316` at 1.0 and `1.227910` at
2.0), which is inside `[0.25, 1.0]`.

S4 acceptance is recorded on the cluster-CV selector; the Laplace-evidence
result is recorded as a discrepancy for review. No prior, world or tolerance was
changed to force agreement.

## Consequences

- S3 passes the "not worse than best baseline" report criterion.
- S4 passes via cluster CV within 2× of truth; evidence selection is a reported
  miss.
- `ModelSpec.sigma_beta` is persisted in the posterior archive (additive
  optional field, default 0.5 for older payloads).
