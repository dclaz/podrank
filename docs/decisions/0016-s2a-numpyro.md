# 0016. S2a Laplace vs NUTS and SBC reference

- **Status:** Accepted
- **Date:** 2026-11-27
- **Phase:** P4
- **Contract/section:** §11.4 S2a

## Setup

`podrank.reference.fit_numpyro` fits the same additive Luce model with NumPyro
NUTS, using the R1 priors and `doc_tiny`. The NumPyro extra (`jax`, `numpyro`)
was installed only for this reference study. The study is
`benchmarks/s2a_reference.py`.

## Laplace vs NUTS

One `doc_tiny` dataset, NUTS with 500 warm-up and 1,000 samples. For every
non-anchor agent contrast, the NUTS posterior sd is compared with the Laplace sd
from the 95% podrank interval:

```text
ratio min  0.9892
ratio max  1.0446
ratio mean 1.0119
```

All ratios are inside `[0.9, 1.1]`.

## SBC

100 prior draws of the six agent log-worths (anchored at 0), four seats, three
bracket effects and twelve deck deviations; independent decisive games on the
`doc_tiny` design; NUTS with 100 warm-up and 200 samples; ranks for the five
non-anchor contrasts.

```text
rank bins (5 equal bins, 500 ranks): 108, 90, 99, 92, 111
chi-square vs uniform: 3.50 (df 4, p ≈ 0.48)
max bin deviation: 0.022
```

The rank histogram is consistent with uniformity; S2a passes its sd-ratio and
SBC checks on this reference run.

## Consequences

- The optional reference extra is exercised and the R1 Laplace approximation is
  validated against NUTS on `doc_tiny`.
- This is a small reference study, not a claim about every world or dataset.
- No core dependency was added; the extra remains optional.
