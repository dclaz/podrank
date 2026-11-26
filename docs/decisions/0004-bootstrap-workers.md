# 0004. Bootstrap worker API and BLAS thread policy

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P3
- **Contract/section:** Appendix C.1, §5.3, §12

## Context

The P3 bootstrap needs a process-pool execution option without changing the
statistical result, and the §12 reference benchmark requires one BLAS thread per
worker. Setting thread environment variables inside a worker initializer does
not prove that an already-loaded BLAS honoured them, and temporarily changing
the caller's environment inside `cluster_bootstrap()` is an unwanted global side
effect.

## Decision

- `cluster_bootstrap(..., workers=1)` exposes a keyword-only, positive integer
  execution option. Booleans and non-positive values are rejected.
- `workers=1` is the default and creates no process pool.
- With the same inputs and seed, worker count must not change replicate order,
  diagnostics, statuses or intervals beyond the specified numeric tolerance.
- Process-pool workers receive picklable compact read-only state once, derive
  their RNG stream from the replicate index, and results are collected in
  replicate order. Linux uses `fork`; Windows uses `spawn`.
- `cluster_bootstrap()` does not mutate process-wide thread environment
  variables. One BLAS thread per worker is a **verified reference-benchmark
  requirement**, not a runtime guarantee. Set the relevant thread variables
  before Python starts; use `threadpoolctl` only as a dev/benchmark inspection
  tool.
- A run without verified one-thread workers is labelled exploratory and cannot
  be used to claim the reference §12 target.

## Consequences

- The public worker option is pinned in Appendix C.1.
- Linux `fork` and Linux `spawn` parity are tested; a Windows `spawn` parity
  step is wired into CI, but a green Windows run and the 16-core 400-refit
  benchmark remain pending.
- No `threadpoolctl` runtime dependency is added to the NumPy/SciPy core.
- Reference benchmark reports must state the process start method, verified
  worker BLAS limits, elapsed time and peak memory across parent and workers,
  including how shared memory is counted.

## Alternatives considered

- Environment-variable switching inside the library — rejected; global-side
  effect and unverifiable after BLAS initialisation.
- Threads instead of processes — rejected; Python-level solver work does not
  provide the required scaling and BLAS threading is harder to bound.
- Adding `threadpoolctl` to core — rejected; it is a benchmark/development
  tool, not a NumPy/SciPy runtime requirement.
