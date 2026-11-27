# 0010. Standard SC2 result and SC1 rerun

- **Status:** Accepted
- **Date:** 2026-11-27
- **Phase:** P3
- **Contract/section:** §11.2, §11.3, §15

## Context

ADR 0006 recorded the `ci_smoke` SC1/SC2 operating characteristics and the
completed `standard` SC1 release-shape run, but left the `standard` SC2 run
open: the local attempt was interrupted and the follow-up scheduled the
1,000-replication study on the main computer. Until that result was recorded,
the P3 statistical gate for SC2 remained open (ADR 0006, "Consequences").

This record closes that follow-up and also re-runs `standard` SC1 to confirm the
recorded result is reproducible after the P2/P3 performance changes.

## What was run

Both studies used the frozen `standard` world (§11.2: A = 20, D = 48, 600
clusters × 4 games) and the §11.3 release settings, from the repository root:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  run_study("SC1", world_name="standard", replications=1000,
            bootstrap_replicates=400, seed=303, workers=28)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  run_study("SC2", world_name="standard", replications=1000,
            bootstrap_replicates=400, seed=303, rho_div=0.0, workers=28)
```

Environment: AMD Ryzen 9 9950X3D (16 cores / 32 threads), 91 GiB RAM,
Linux 7.0.0-44, CPython 3.13.17, NumPy 2.5.5, SciPy 1.18.1, one BLAS thread per
worker. This is not the 16-core reference machine; as with ADR 0006, the study
size, seed and scoring list are the specified ones, so the local result is the
recorded release-shape result.

Both runs used the §11.3 scoring list: 21 deduplicated log-worth contrasts
(three order contrasts plus every agent vs the anchor) and the SDWP of all 20
agents, 41,000 intervals per study. SC1/SC2 redraw true δ from `N(0, 0.5²)` per
replication; SC2 sets `ρ_div = 0`.

## Results

| Metric | SC1 rerun | SC2 standard |
|---|---:|---:|
| Replications | 1000 / 1000 | 1000 / 1000 |
| Failed replications | 0 | 0 |
| Quantities scored / emitted | 41,000 / 41,000 | 41,000 / 41,000 |
| Emission rate | 1.0 | 1.0 |
| Pooled coverage | **0.9433658536585365** | **0.9422195121951219** |
| Target range | [0.93, 0.97] | [0.93, 0.97] |
| Target misses | none | none |
| Per-quantity coverage below 90% | none | none |
| Elapsed (28 workers) | 247.4 s | 245.6 s |

SC1 reproduced the ADR 0006 value exactly, as expected from the frozen seeds.
The Monte Carlo standard error of pooled coverage at 41,000 intervals and
`p ≈ 0.94` is `sqrt(0.94·0.06/41000) ≈ 0.0012`; both results are inside the
target with no bias or emission failures.

Per-quantity detail:

| Metric | SC1 rerun | SC2 standard |
|---|---:|---:|
| Log-worth coverage range | 0.925 – 0.957 | 0.931 – 0.956 |
| SDWP coverage range | 0.936 – 0.956 | 0.935 – 0.956 |
| Max `\|mean bias\|`, log-worth | 0.00532 (`agent_02` vs anchor) | 0.00997 (`agent_03` vs anchor) |
| Max `\|mean bias\|`, SDWP | 0.00120 (`agent_13`) | 0.00172 (`agent_17`) |
| Lowest log-worth quantity | `agent_04` vs anchor, 0.925 | `agent_06` vs anchor, 0.931 |
| Lowest SDWP quantity | `agent_12`, 0.936 | `agent_19`, 0.935 |

The practical bias tolerances (§11.3: 0.05 log-worth, 0.01 SDWP) are met by a
wide margin for every quantity.

## Review

The `ci_smoke` SC2 pooled miss (0.9095 in ADR 0006) does **not** reproduce at
the release size. At 200 clusters, `ρ_div = 0` makes the four games of a cluster
share one Gumbel draw; the cluster bootstrap treats clusters as independent, so
the small CI study has both strong within-cluster dependence and only 50
replications. At 600 clusters and 1,000 replications the intervals attain
nominal pooled coverage. No implementation defect was found, and no model,
prior, scenario or tolerance was changed to obtain this result.

## Consequences

- SC1 and SC2 both meet the §11.3 statistical targets on the `standard` world.
  The P3 statistical gate is closed.
- The `ci_smoke` 50-replication result remains an operating characteristic of
  the small CI study, not a release sign-off; its SC2 miss is now explained by
  small-sample behaviour rather than carried as an open gate.
- The CI `stat` job keeps running the 50-replication SC1/SC2 study and keeps
  recording coverage/bias misses as warnings.
- No fixture, test, public signature or world value changes.

## Alternatives considered

- Re-running SC2 at 50 replications on `standard` — rejected: §11.3 requires
  1,000 replications for the release target.
- Raising `ci_smoke` to `standard` size in CI — rejected: the plan defines
  `ci_smoke` as the small CI study and the statistical target as a release
  review, not a merge gate.
- Adjusting the bootstrap or tolerance to make the `ci_smoke` SC2 miss pass —
  rejected by §11.3; the release run shows no defect to fix.
