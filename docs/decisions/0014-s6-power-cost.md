# 0014. S6 power and cost report

- **Status:** Accepted
- **Date:** 2026-11-27
- **Phase:** R2 release
- **Contract/section:** §11.4 S6

## Context

S6 reports R2 power and cost over the frozen `cmp_effect` worlds: effect size ×
`ρ_div` × duplicate vs paired. It is a report, not a release threshold.

## Run

- 4 target lifts × 2 designs × 4 `ρ_div` = 32 cells.
- 1,000 replications per cell, `max_blocks = 4,000`, `alpha = 0.05`,
  `Superiority(0)`.
- Each replication uses the frozen `cmp_effect` world and calibrated `theta_A`;
  the stopping time is the first upper crossing `K_up >= 1/alpha`.
- Cost is the mean/median consumed blocks; censored replications use all 4,000
  blocks.

## Power

| μ | design | ρ=0 | ρ=0.3 | ρ=0.6 | ρ=1 |
|---:|---|---:|---:|---:|---:|
| 0.005 | duplicate | 0.189 | 0.090 | 0.085 | 0.056 |
| 0.005 | paired | 0.133 | 0.103 | 0.074 | 0.075 |
| 0.010 | duplicate | 0.610 | 0.344 | 0.214 | 0.149 |
| 0.010 | paired | 0.424 | 0.243 | 0.179 | 0.128 |
| 0.020 | duplicate | 1.000 | 0.956 | 0.798 | 0.639 |
| 0.020 | paired | 0.953 | 0.732 | 0.558 | 0.408 |
| 0.050 | duplicate | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.050 | paired | 1.000 | 1.000 | 1.000 | 0.998 |

## Mean consumed blocks

| μ | design | ρ=0 | ρ=0.3 | ρ=0.6 | ρ=1 |
|---:|---|---:|---:|---:|---:|
| 0.005 | duplicate | 3499 | 3728 | 3735 | 3836 |
| 0.005 | paired | 3664 | 3710 | 3796 | 3771 |
| 0.010 | duplicate | 2631 | 3214 | 3468 | 3584 |
| 0.010 | paired | 3030 | 3383 | 3536 | 3637 |
| 0.020 | duplicate | 810 | 1494 | 2144 | 2680 |
| 0.020 | paired | 1431 | 2257 | 2741 | 3086 |
| 0.050 | duplicate | 123 | 223 | 322 | 471 |
| 0.050 | paired | 279 | 425 | 578 | 824 |

## Observations

- Power falls sharply as `ρ_div` rises: shared deals reduce effective
  information, and the duplicate design is more affected in absolute blocks.
- At 2 percentage points the duplicate design reaches full power at `ρ=0` and
  still has substantial power at `ρ=1`; paired is weaker at high `ρ`.
- At 0.5 percentage points power is low even at `ρ=0`, as expected from the
  sizing approximation in §7.9.
- Costs are right-censored at 4,000 blocks for the low-effect cells; the mean is
  therefore a lower bound on the cost needed for those effects.

## Consequences

- S6 is recorded; no model, threshold or world was changed.
- Real-log validation remains the open R2 release item.
