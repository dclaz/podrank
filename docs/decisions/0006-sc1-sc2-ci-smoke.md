# 0006. SC1/SC2 statistical study results

- **Status:** Accepted (SC1 standard passed; SC2 standard pending)
- **Date:** 2026-11-26
- **Phase:** P3
- **Contract/section:** §11.3, §14

## What was run

The `podrank.sim.studies` harness was run on the frozen `ci_smoke` world with the
CI-smoke settings from §11.2:

```text
run_study("SC1", world_name="ci_smoke", replications=50,
          bootstrap_replicates=100, seed=202, rho_div=None)
run_study("SC2", world_name="ci_smoke", replications=50,
          bootstrap_replicates=100, seed=202, rho_div=0.0)
```

SC1/SC2 redraw δ from `N(0, 0.5²)` per replication; the field is the world's
400-context `sim_field`; the scoring list is the deduplicated §11.3 list (11
log-worth contrasts plus the SDWP of all 10 agents). Both runs had 50/50
successful replications and emitted 1050/1050 intervals (emission rate 1.0).

## Results

| Scenario | Pooled coverage | Target range | Emission | Bias misses | Per-quantity < 90% |
|---|---|---|---|---|---|
| SC1 | 0.9343 | [0.93, 0.97] | 1.0 | none | `agent_03` vs anchor (0.88), `agent_04` vs anchor (0.86) |
| SC2 | 0.9095 | [0.93, 0.97] | 1.0 | `agent_00` vs `agent_01` (0.0546) and its reverse | `agent_00`/`agent_01`, `agent_05` vs anchor, `agent_09` vs `agent_08`, SDWP `agent_05`, SDWP `agent_09` |

Monte Carlo standard error for pooled coverage at 50 replications and
`p ≈ 0.91` is about `sqrt(0.91·0.09/1050) ≈ 0.0088`, so SC2's pooled result is
roughly 2.3 standard errors below the lower target.

## Standard SC1 result (completed)

The release-shape SC1 study was run locally on the 8-core dev machine:

```text
run_study("SC1", world_name="standard", replications=1000,
          bootstrap_replicates=400, seed=303, workers=8)
```

with `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` in the parent
environment. It took 2,397 s.

| Metric | Result |
|---|---:|
| Replications | 1000 / 1000 (0 failed) |
| Quantities scored / emitted | 41,000 / 41,000 |
| Emission rate | 1.0 |
| Pooled coverage | **0.9433658536585365** |
| Target misses | none |
| Per-quantity coverage below 90% | none |

`SC1` therefore meets the §11.3 statistical target on the release world. The
local run is not the 16-core reference environment, but the study size and
seed are the specified ones.

## Standard SC2 status

The local SC2 run was started and interrupted before completion. It produced
no partial result (`run_study` returns only at completion). SC2 will be run on
the main computer; until that result is recorded, the SC2 statistical gate
remains open.

## Review

This is a **statistical miss** under §11.3. It is not converted into a test
tolerance change, a model change, a prior change or a scenario change.

Likely contributors, in order:

1. The 50-replication `ci_smoke` run is the CI smoke, not the 1000-replication
   `standard` release study. `ci_smoke` has 10 agents and 200 clusters; the
   release target uses 20 agents and 600 clusters.
2. SC2 sets `ρ_div = 0`, so the four games in a cluster share one Gumbel draw.
   The cluster bootstrap treats clusters as independent, but the strong shared
   deals reduce effective within-cluster information and can make the intervals
   optimistic.
3. The extreme anchor-vs-lowest-agent contrast has visible finite-sample
   shrinkage bias (0.0546 in log-worth units), consistent with fixed-prior MAP
   attenuation rather than a bug.
4. Individual quantities have only 50 intervals each, so a per-quantity
   coverage below 90% is noisy and is reported, not treated as a second
   independent target.

No implementation defect was identified in the harness: the generating θ/δ,
the field contexts, the MAP values and the bootstrap dependencies are the
documented ones, and SC1's pooled coverage is inside the target range.

## Consequences

- The CI `stat` job runs the 50-replication SC1/SC2 study. It blocks on
  structural failure (failed fits, no intervals, invalid coverage) but records
  coverage/bias misses as warnings, as §11.3 requires.
- SC1 meets the statistical target on `standard` (1000 replications, bootstrap
  400). SC2's `standard` run remains to be completed on the main computer.
- P3 is **not** claimed complete for the statistical gate until SC2 `standard`
  is recorded. A miss there requires the same review; it must not be fixed by
  relaxing the model, priors or tolerances.
- The `ci_smoke` result is an operating characteristic of the current
  implementation at small size, not a release sign-off.

## Follow-up

- Run the `standard` 1000-replication SC2 study on the main computer and record
  the per-quantity coverage, emission and bias here.
- If the standard run also misses, investigate whether the cluster bootstrap's
  independence assumption under strong shared deals needs a documented
  limitation or a revised uncertainty method; do not silently change the
  tolerance.
