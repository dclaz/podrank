# 0012. Reduced S5/S11 run and the open reference grid

- **Status:** Superseded in part by [0013](0013-full-s5-s11.md)
- **Historical status:** Accepted
- **Date:** 2026-11-27
- **Phase:** P5
- **Contract/section:** §11.4, §14, §15

## Context

P5 requires the S5 boundary check and the S11 fixed-horizon lift-recovery
report. The predefined `cmp_null` and `cmp_effect` worlds and the full
predefined-world simulator are not yet part of the repository. Running the
literal S5 cross-product (10,000 replications per condition, including replay,
retry, late-recovery and arrival-permutation variants) is a multi-day reference
run, not a unit-test-sized check.

## Decision

- Implement and record a **reduced, dimension-complete** run over simulated
  block scores:
  - S5: 3 boundary conditions × 2 designs × 4 `ρ_div` × 3 failure
    configurations = 72 cells, 10,000 replications, 4,000 blocks per replication,
    `alpha = 0.05`, `min_blocks_for_failure_rule = 50`,
    `max_failed_fraction = 0.02`.
  - S11: 4 target lifts × 2 designs × 2 `ρ_div` = 16 cells, 5,000
    replications, 4,000 blocks per replication, complete-data means only.
- The run is produced by `benchmarks/r2_studies.py` using
  `podrank.sim.compare_studies`.
- Treat the result as **supporting evidence**, not the full release gate. The
  full predefined-world S5/S11 run remains an explicit open R2 release item.
- Do not change any model, threshold, world or tolerance to remove a flag. S11
  flags are reported as observed.

## Results (reduced run)

S5: 72 cells at 10,000 replications, **no cell exceeded**
`alpha + 2·MC SE = 0.0544` in either direction. S11: 16 cells at 5,000
replications (above the 1,000 required), two cells
flagged at the two-SE rule:

| Cell | Mean | Truth | Bias | MC SE |
|---|---:|---:|---:|---:|
| `duplicate ρ=0 μ=0.05` | 0.0501523 | 0.05 | +0.0001523 | 0.0000500 |
| `paired ρ=0.3 μ=0.05` | 0.0497793 | 0.05 | −0.0002207 | 0.0001061 |

The duplicate cell is about 3 MC SE and the paired cell about 2.1 MC SE. They
are recorded, not corrected; the full grid and independent target calculation
remain the release decision.

## Consequences

- P5 has an executable S5/S11 runner and a recorded reduced result.
- The full predefined-world S5/S11 run, including replay/retry/arrival/resume
  variants, is still required before an R2 release claim.
- No fixture, model, threshold or public signature changes.

## Alternatives considered

- **Skip the statistical run until the full worlds exist.** Rejected: it would
  leave no evidence at all for the betting and boundary implementation.
- **Run the literal full grid now.** Rejected: the compute cost is
  disproportionate to the current implementation-complete scope, and the
  predefined worlds are not yet frozen in-repo.
- **Relax the S11 flag threshold.** Rejected: flags are observations under the
  specified rule.
