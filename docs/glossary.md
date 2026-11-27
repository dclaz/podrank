# Glossary

| Term | Meaning |
|---|---|
| **agent** | a fixed, versioned policy. `podrank` never trains or promotes it. |
| **anchor** | the one agent with `is_anchor=True`; log-worth is reported relative to it. |
| **bracket** | a group of decks sharing a bracket effect \(\mu_b\). |
| **cluster** | a group of games that share dependence (for example one deal). The bootstrap resamples whole clusters. |
| **contrast** | a linear combination of parameters, usually \(\theta_a - \theta_b\) or a mean-centred seat or deck difference. |
| **data share** | the fraction of a contrast's prior variance removed by the data; below 0.5 labels the contrast `weak`. |
| **deck nuisance** | \(\beta_d - \operatorname{mean}_{\text{ref}}(\beta)\) over rankable `train` decks. Not a deck ranking. |
| **D1–D5** | identification and contrast rules: components, unsupported anchor, information, data share, null-space check, statuses. |
| **field** | a finite, weighted, hashed list of evaluation contexts. |
| **gauge direction** | an exact symmetry of the likelihood, such as adding a constant to all \(\theta\). |
| **likelihood-bearing cluster** | a cluster with at least one likelihood-used game; the bootstrap frame. |
| **log-worth** | \(\theta_a - \theta_{\text{anchor}}\), the additive skill scale. |
| **MAP** | maximum a posteriori parameter estimate; all headline values are MAP plug-ins. |
| **outcome** | `win`, `draw`, `cap` or `failed`. Only `win` enters the likelihood. |
| **pattern** | a group of likelihood-used games with the same design row; the compressed likelihood stores winner counts per pattern. |
| **posterior** | the fitted result object: mode, curvature, registries, counts, reports and persistence. |
| **prior-dependent** | a value the data does not determine; reported only as a labelled diagnostic, never as a headline result. |
| **SDWP** | standardised decisive win probability: the focal agent's conditional decisive win probability averaged over a field. |
| **seat effect** | \(s_i\), reported mean-centred over the four seats. |
| **stratum** | a group used for stratified bootstrap resampling (`suite`, `engine_version` or a custom column). |
| **swap** | a per-seat count of deck swaps; modelled only when `swap_effect=True`. |
| **unrankable** | an agent or deck with no likelihood-used games; it is not ranked from the prior. |
