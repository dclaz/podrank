# 0011. Drop Windows support and the Windows CI job

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P0, P3
- **Contract/section:** §12, §13, §14; `AGENTS.md` P0 acceptance

## Context

The P0 scaffold required a Windows smoke job, and the bootstrap and study
harnesses special-cased the process start method:

```python
multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
```

The rating model, likelihood, identification, fields, persistence and study
code are ordinary Python, NumPy and SciPy and have no Windows-specific logic.
The only platform fork was the multiprocessing start method plus the CI job that
exercised it. Windows was never a product requirement, and the job added a
second runner solely to validate a branch that the Linux suite can already
exercise with the `spawn` start method.

## Decision

- **Windows is not a supported or tested platform.** The `windows-smoke` CI job
  is removed. CI is Linux Python 3.12 and 3.13 plus the statistical smoke job.
- The process start method is no longer selected by OS name. Both
  `podrank.uncertainty.bootstrap` and `podrank.sim.studies` use
  `multiprocessing.get_context()`, the interpreter's default for the platform.
  This keeps `spawn`-based POSIX environments working without a Windows
  commitment.
- The `spawn`/serial parity test is retained and renamed from
  `test_linux_spawn_bootstrap_parity` to
  `test_spawn_start_method_bootstrap_parity`. It validates the start method, not
  Windows.
- §12 now asks benchmarks to report the process start method rather than
  requiring separate Linux and Windows reports. §13, §14, `AGENTS.md`, the
  README and the changelog no longer claim Windows coverage.

## Consequences

- No Windows compatibility is claimed or tested. Windows users may run the pure
  Python path at their own risk, but results are outside the release evidence.
- The library remains OS-agnostic in the ordinary sense: no OS-name branches,
  no platform-specific imports, and no platform-specific packaging.
- The `spawn`/serial parity test still runs on Linux, so a `spawn` regression is
  caught without a Windows runner.
- No numerical, statistical, fixture or public-signature change.
- `uv.lock` still contains wheel entries for every platform; that is lockfile
  metadata, not a support claim.

## Alternatives considered

- **Keep the Windows smoke job.** Rejected: it adds a platform commitment and a
  second runner for no product requirement; the Linux suite already covers the
  start method.
- **Hard-code `fork` and delete the `spawn` test.** Rejected: it would make the
  package less portable than the Python code requires and would break
  `spawn`-based POSIX environments for no benefit.
- **Keep the `win32` branch without CI.** Rejected: an untested platform branch
  is worse than no platform claim.
