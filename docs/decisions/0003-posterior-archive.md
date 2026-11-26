# 0003. Posterior archive format: JSON manifest plus typed `.npy` members

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P2
- **Contract/section:** Appendix C.2.4, §14 Posterior archive, §5.3

## Context

The row-free JSON posterior was a useful baseline: at 1M games it was about
76.7 MiB, saved in 4.96 s and loaded in 5.64 s. That is workable, but bulk arrays
still travel as JSON lists, which inflates file size and parse/serialise cost.
C.2 previously specified JSON envelopes for both `Posterior` and `Comparison`.
No schema-1 posterior artifact has been released; the only files are local spike
outputs, so a migration layer would add dead complexity.

## Decision

`Posterior` uses a versioned single-file archive. `Field` and `Comparison`
formats are unchanged.

- **Container:** ZIP with `manifest.json` and `arrays/<name>.npy` members.
- **Manifest:** exactly `kind` (`podrank.posterior`), `schema_version` (integer
  `2`), `package_version`, and `payload`. The payload contains JSON metadata,
  ordered records, tagged keys, outcome-rate numerators/denominators, cluster
  table/strata, provenance and an `arrays` descriptor map
  (`member`, `dtype`, `shape`) for each bulk array.
- **Bulk arrays:** typed NumPy `.npy` members, uncompressed inside the archive.
  No JSON lists for bulk arrays, no base64, no pickle, no raw game rows.
- **Atomic write:** same-directory temporary file followed by `os.replace`.
- **Load validation:** reject duplicate, missing or extra ZIP members; invalid
  or unknown member names; unsupported schema version; bad descriptors;
  unsupported dtypes; dtype/shape mismatches; oversized members; and
  cross-array index/shape errors. Use `numpy.load(..., allow_pickle=False)`.
- **No migration:** schema-1 posterior JSON files are rejected as unsupported.
  Re-fit is the supported path for local spike artifacts.
- **Scope:** only compact fitted posterior state is archived. A manually
  constructed dense reference posterior (not a `fit` product) raises
  `PersistenceError` before any file is created, rather than falling back to
  JSON bulk arrays. Persistence remains opt-in; `fit → report → bootstrap` is
  entirely in-memory.
- **Crossing keys:** for each built-in stratum column (`suite`,
  `engine_version`), a cluster that carries conflicting labels records its
  canonical frame indices in `stratum_crossings`. `crossing_keys(column)`
  resolves those indices back to tagged cluster keys. A crossing in `suite`
  affects only `suite`; it does not split the cluster or choose a label, and it
  does not affect unstratified or other-column state. Custom strata are
  single-label by construction.

## Schema-1 inventory

Checked on 2026-11-26:

- no git tags and no release history beyond the two local commits;
- `gh release list` unavailable in this environment;
- PyPI `podrank` JSON endpoint returns HTTP 404;
- no schema-1 Posterior artifacts or shared `.json` posterior files in the
  repository; the only posterior archives are local spike files.

Conclusion: schema 1 was not released. Unsupported schema-1 posterior files now
fail with a clear `PersistenceError`; no compatibility reader or migration is
added.

## Loader limits

These are archive-loader limits, separate from `max_dense_bytes = 2 GiB`:

| Limit | Value |
|---|---:|
| Maximum ZIP members | 256 |
| Maximum manifest bytes | 16 MiB |
| Maximum per-member bytes | 8 GiB |
| Maximum total uncompressed bytes | 16 GiB |
| Maximum archive file bytes | 16 GiB |

The loader checks member count, total uncompressed size, descriptor dtype and
shape, expected member size (`npy` payload plus header allowance) and cross-array
indices before returning state. The measured 1M artifact uses 24 members and
84.23 MiB uncompressed, well inside the limits.

## Measured baseline (provisional, local 4-core-class machine)

Synthetic 1M-game / 1M-pattern / P=199 workload, fresh process:

| Measurement | Row-free JSON | Schema-2 archive |
|---|---:|---:|
| Artifact size | 76.7 MiB | **84.2 MiB** |
| Save | 4.96 s | **0.60 s** |
| Load | 5.64 s | **0.52 s** |
| Fit from existing `GameTable` | — | 61.39 s |
| Peak RSS, fit + save/load | — | 2,812 MiB |
| RSS after input construction | — | 2,223 MiB |
| Outcome/cluster/strata/crossing parity | yes | yes |

The archive is slightly larger because it carries typed cluster/table string
arrays, but save/load are about 8–11× faster. No file-size gate is set; this is
a first uncompressed typed-array baseline, not a compression result. The local
machine is not the §12 16-core reference.

## Consequences

- C.2.4 distinguishes the schema-2 Posterior archive from the unchanged
  schema-1 Comparison JSON envelope and Field FD3b envelope.
- A loaded archive must reproduce reports and the seeded bootstrap frame without
  the original `GameTable`; this is covered by parity tests.
- File-size and save/load gates remain unset until measured. The local 4-core
  numbers are provisional; canonical `perf_large` acceptance stays on the
  16-core reference machine.
- The archive is deliberately uncompressed first; compression is a separate
  measurement.

## Alternatives considered

- Base64 blobs inside the schema-1 JSON envelope — rejected; hidden expansion,
  larger files and no typed-member validation.
- A single `np.savez` without a manifest — rejected; less explicit schema,
  versioning and metadata.
- Keep JSON lists for bulk arrays — rejected; the measured 76.7 MiB row-free
  baseline already shows the avoidable overhead, and it will grow with arrays.
- A schema-1 compatibility reader — rejected; no released schema-1 artifacts.
