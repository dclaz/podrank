# Limitations

**Summary.** `podrank` rates the agents in the records it is given and evaluates
a declared field. It does not establish causality, choose a field, model
matchups, or replace a designed experiment.

## Model limitations

- **Additive Luce only.** Utilities are additive in agent, seat, bracket, deck
  and swap terms. If the true process is non-additive, the fit is the best
  additive approximation in the KL sense, not the truth.
- **Fixed coefficients.** There is no time-varying skill, no agent-by-opponent
  interaction and no agent-by-deck interaction in R1. Effects of those kinds
  appear as unexplained residual variation.
- **Fixed priors.** The priors shrink estimates toward zero. This produces
  genuine finite-sample bias, especially for extreme agents and small samples.
  The SC1/SC2 studies measure it; it is not tuned away.
- **Decisive-only frame.** Draws, caps and failures are reported as outcome
  rates but do not inform the win model. SDWP is not a win rate among decisive
  games, and it is not the same as a score that gives credit for draws.
- **One anchor.** Anchor-relative reports need an anchor with likelihood-used
  games. If it has none, the anchor-dependent reports return
  `AnchorUnsupported`; there is no automatic replacement.
- **Deck split is prior-dependent.** The split of a deck effect into bracket
  and deviation is never data-identified, and deck nuisance values are not a
  deck-strength ranking.

## Identification limitations

- Connectivity, support counts and data share are necessary but not sufficient.
  A contrast that lies outside the likelihood's row space is withheld even when
  both agents have many games.
- Identification is computed once on the original fit. It does not adapt to
  which clusters a particular bootstrap replicate happened to draw.
- Small worlds can have weakly supported nuisance directions even when the
  agents are rankable. The identification report says so rather than hiding it.

## Uncertainty limitations

- **Laplace** is a model-based normal approximation. It assumes games are
  independent given the parameters; shared deals can make it optimistic or
  conservative.
- **Cluster bootstrap** assumes likelihood-bearing clusters are independent and
  exchangeable (within strata). Misstated clusters invalidate the interval.
- Coverage is an operating characteristic measured in predefined simulations.
  It is not guaranteed for an arbitrary dataset or field.
- Bootstrap suppression (`insufficient_clusters`, `unstable_bootstrap`) is not a
  pass: it means the requested interval could not be supported.

## Scope limitations

- Four-player free-for-all games with a single winner or a non-decisive outcome.
  Other table sizes and ranked outcomes are out of scope.
- The library consumes records. It does not play games, schedule matches,
  manage checkpoints, join training steps, or decide promotions and releases.
- Deck and seat terms adjust ratings. This is not a deck-strength or
  deck-selection model.
- Unseen-deck prediction, agent embeddings, engine/RL dependencies and
  evaluation-deck leakage detection are out of scope.
- R2 comparison is a separate API and is not exposed in R1. The R1 import
  boundary is deliberate: rating a table never loads comparison code.

## Operational limitations

- Records must name registered, versioned agents and decks. Unknown IDs raise.
- Timestamps must carry an explicit offset.
- Duplicate `game_id`s with different bytes are errors, not last-write-wins.
- `cluster_id` is the caller's statement of dependence. The library cannot
  recover the true deal grouping from flat rows.
- Field support is checked at evaluation. A field that references an unseen
  deck or an opponent outside the anchor component raises rather than silently
  substituting a supported entity.

See [assumptions](assumptions.md) for the full list of caller obligations and
[identification](concepts/identification.md) for the status model.
