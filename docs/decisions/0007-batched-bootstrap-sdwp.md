# 0007. Batched bootstrap SDWP

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P3
- **Contract/section:** §5.3 B3, §12

## Context

The SC1/SC2 `standard` study was dominated by one calculation. A single
replication cost about 12.8 s:

| Step | Time |
|---|---|
| Generate 2,400 rows | 0.12 s |
| `GameTable.from_records` | 0.02 s |
| Fit (design + Newton + D4) | 0.10 s |
| 100 bootstrap refits | 2.15 s |
| **Bootstrap SDWP (20 agents)** | **9.22 s** |
| Truth SDWP (20 agents) | 0.11 s |

`BootstrapResult.sdwp` called `Posterior._sdwp_for_draw` for every
`(agent, replicate mode)` pair. With 20 agents × 100 modes × 400 field contexts
that is 800,000 softmax evaluations through a Python context loop. The
computation is correct but needlessly serial.

## Decision

- Add `Posterior._sdwp_matrix(field, agent_ids, modes)`, a compact batched
  equivalent of `_sdwp_for_draw` returning shape `(len(modes), len(agent_ids))`.
- `BootstrapResult.sdwp` uses the batched matrix once per field and slices each
  agent's replicate values. Its support-gate dependency sets are unchanged.
- Keep `_sdwp_for_draw` as the readable reference path and test the two against
  each other. Do not keep a second production path: the batched method is the
  only one used by the bootstrap; `_sdwp_for_draw` is retained for the public
  posterior MAP path and equivalence tests.

## Measured result

On the local 4-core dev machine, one `standard` replication with 100 bootstrap
refits and a 400-context field:

- Bootstrap SDWP: **9.22 s → 0.41 s**.
- Ten-replication SC1 standard smoke: **127.7 s → 33.3 s**.
- Ten-replication SC2 standard smoke: **129.6 s → 34.8 s**.

The pooled coverages are unchanged to the displayed precision:
SC1 `0.9341463414634147`, SC2 `0.9317073170731708`; the target-miss counts are
unchanged.

## Consequences

- The bootstrap refits (about 2.1 s per replication) are now the main cost of a
  study replication, which is the expected shape.
- A 1,000-replication `standard` study is still release-machine work: roughly
  55 minutes per scenario serially on the dev box, less with replication-level
  parallelism or more cores. It is not a CI job.
- No statistical result, interval method, gate or tolerance changes.

## Verification

- `test_batched_sdwp_matches_per_draw` compares the batched matrix against
  `_sdwp_for_draw` for multiple modes and agents at `rtol=1e-12`.
- `test_bootstrap_sdwp_uses_batched_mode_values` recomputes the percentile
  intervals from the batched mode values and compares them to the report.
- Existing bootstrap, `doc_tiny_balanced` and parity tests pass unchanged.
