# Uncertainty

**Summary.** `podrank` reports two uncertainty methods: fast, model-based
Laplace intervals from the penalised curvature, and a cluster bootstrap over
likelihood-bearing clusters. Both are estimates of sampling variability for the
regularised MAP statistic; neither establishes identification, which is decided
once on the original fit.

## What an interval is attached to

Intervals are attached to **MAP plug-in** quantities: log-worth contrasts, seat
contrasts, deck nuisance contrasts, the swap effect and SDWP. Posterior-averaged
SDWP is available through `method="laplace"` but is not bootstrapped in R1.

Eligibility is frozen from the original fit:

- withheld quantities (`unrankable`, `disconnected`, `aliased`) have no interval;
- a supported zero contrast gets the exact interval `[0.0, 0.0]`;
- a nonzero contrast gets an interval only when it is `weak` or `identified`,
  subject to the rules below.

Every interval carries its method (`exact`, `laplace` or `cluster_bootstrap`)
and any warning labels.

## Laplace intervals

For a linear contrast vector \(c\) and penalised Hessian \(H\) at the MAP, solve

\[
H z = c, \qquad \operatorname{sd}(c) = \sqrt{c' z},
\]

and report \(c'\hat\psi \pm z_{0.975}\operatorname{sd}(c)\), using the normal
0.025/0.975 quantiles. Plain text: invert the curvature along the contrast
direction and use the normal approximation.

Laplace intervals are **model-based**: they assume games are independent given
the parameters and that the posterior is approximately Gaussian near the mode.
Shared deals can move the true uncertainty in either direction. They are cheap
and are the default for `post.log_worth()`, `post.contrast()`, `post.seats()`
and `post.decks()`.

For SDWP, `post.sdwp(field, method="laplace", n_draws=..., seed=...)` draws from
the Laplace approximation, adds `sdwp_posterior_mean`, and returns a linear
interval from the draw-level SDWP values. It requires an explicit seed. The
headline `sdwp` value remains the MAP plug-in value and its ordering is not
replaced by the posterior mean.

## Cluster bootstrap

```python
boot = pr.uncertainty.cluster_bootstrap(post, n=400, seed=7)
boot.contrast("agent_05", "agent_01")
boot.sdwp(field)
```

The resampling unit is the **likelihood-bearing cluster** of the original fit:
a cluster with at least one likelihood-used game, with a null `cluster_id`
treated as a singleton. Clusters with no likelihood-used games contribute
nothing to the objective and are not in the frame. Clusters are never split.

Each replicate:

1. draws multinomial cluster weights (unstratified, or within strata when
   `strata=` names a column of the retained cluster table);
2. rebuilds the winner counts from the posterior's outcome-and-cluster map;
3. refits with the **original** parameter index, priors, inclusion, spec and
   field contexts, warm-started from the original mode.

The reported interval is the original MAP value plus the 2.5% and 97.5%
percentiles of the converged replicate values (linear method).

### Gates and suppression

For each quantity, the bootstrap first applies an **original-fit gate** (B4):

- at least `min_clusters` likelihood-bearing clusters overall (default 50);
- at least `min_entity_clusters` for every dependency (default 10: the focal
  agent, opponents, decks, and every nonzero contrast coefficient).

A failure yields `insufficient_clusters` and no resampling for that quantity.

For quantities that pass B4, a replicate gate (B5) applies:

- if `failed_fraction > 0.05`, suppress as `unstable_bootstrap`;
- otherwise count converged replicates where a dependency has no resampled
  frame clusters; if `n_zero_support / n_converged > 0.05`, suppress as
  `insufficient_clusters`;
- remaining intervals use all converged replicates and add an
  `unstable_bootstrap` warning label when `failed_fraction > 0.01`.

The comparisons are strict; equality does not trigger a threshold.
`boot.diagnostics()` reports `n_converged`, `n_zero_support` (`None` when not
computed) and `n_used` per quantity.

Crucially, rankability, anchor, component and headline-eligibility checks are
applied **once, to the original fit**. A replicate with no data for an entity
simply has a prior-only direction for it. That is why a replicate failure can
never suppress an exact zero interval or override an original-fit suppression.

### Strata

`strata=None` resamples all frame clusters. Pass `strata="suite"` or
`strata="engine_version"` to resample within that column, or supply a named
custom column through `cluster_strata=` when fitting. Stratification is
allowed only when every frame cluster lies wholly within one stratum;
otherwise the call raises and names the crossing clusters. Strata survive
`Posterior.save/load`.

### What coverage means

Bootstrap coverage is a property of the sampling frame and the scenario, not a
promise for any particular dataset. It is checked only in the predefined
simulation scenarios (§11.3), never inferred from real logs. Real logs are used
for stability and runtime.

## Which method to use

| Question | Method |
|---|---|
| Quick uncertainty for a contrast | Laplace |
| Uncertainty that respects shared deals | cluster bootstrap |
| SDWP posterior mean under the model | Laplace with an explicit seed |
| SDWP interval that respects cluster dependence | cluster bootstrap |
| Is the quantity identified at all? | neither; use D4/FD6 |

Both methods describe the variability of the regularised estimator. Neither
turns a prior-dependent quantity into a data-identified one.

## Pinned example

With the `doc_tiny` fit and `sim_field` (24 contexts), a 100-replicate
cluster bootstrap with seed 101 gives:

```python
boot = pr.uncertainty.cluster_bootstrap(post, n=100, seed=101)
for left, right in [("agent_04", "agent_00"), ("agent_05", "agent_01")]:
    interval = boot.contrast(left, right).interval
    print(left, right, round(interval.lo, 6), round(interval.hi, 6), interval.method)

sdwp_row = next(row for row in boot.sdwp(field).rows if row.agent_id == "agent_05")
print("agent_05 SDWP", round(sdwp_row.interval.lo, 6), round(sdwp_row.interval.hi, 6))
```

```text
agent_04 agent_00 0.10017 0.694306 cluster_bootstrap
agent_05 agent_01 0.786326 1.718352 cluster_bootstrap
agent_05 SDWP 0.381084 0.529913
```

The point values are the original-fit MAP contrasts and SDWP; the interval is
the percentile range over converged replicates. Worker count changes runtime,
not the seeded statistical result.

## Rules and fixtures

- §5.2: Laplace intervals.
- §5.3 B1–B6: frame, frozen replicate, statistic, gates, percentiles.
- §11.3: coverage is evaluated only in the predefined scenarios, and a miss
  requires a recorded release review.
- Fixtures: F6 (frozen bootstrap index), F18 (suppression precedence), F23
  (bootstrap frame, streams and strata).
- ADRs 0004, 0007, 0008 and 0010 record the worker model, batched SDWP, setup
  vectorisation and the SC1/SC2 release studies.
