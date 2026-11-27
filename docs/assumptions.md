# Assumptions

**Summary.** `podrank` is a rating library for records, not an experiment
designer. This page lists what the records, the statistical model and the
caller are assumed to provide, and what remains the caller's responsibility.

## Data assumptions

A record is one four-player game with a single winner or a non-decisive
outcome. `GameTable.from_records` preserves the supplied rows; normalisation and
the inclusion pipeline run inside `fit`, so calling `validate` first is optional
but recommended.

- **Four seats, four agents, four decks.** Arrays are in seat order and have
  exactly four entries. Duplicate entities inside one game are invalid.
- **Outcomes.** `win` has an integer `winner_seat` in `0..3`. `draw`, `cap` and
  `failed` have no winner. A draw is not a winner and a failure is not a draw.
  Only `win` rows enter the likelihood.
- **Identity.** Agent and deck IDs must exist in the supplied immutable
  registry snapshots. The row's `policy_versions` must match the snapshot, and
  deck versions must be present when the deck registry supplies them. There is
  exactly one anchor in the agent snapshot.
- **Timestamps.** `played_at` is an RFC 3339 timestamp with an explicit offset
  and is normalised to UTC. Naive timestamps are rejected.
- **Duplicates.** Rows with the same `game_id` and identical canonical bytes are
  exact duplicates and are removed with a warning. The same `game_id` with
  different bytes is a `DuplicateGameError`.
- **Clusters.** `cluster_id` groups games that share a deal or other dependence.
  A null `cluster_id` is a singleton cluster. Clusters are never split, and the
  bootstrap resamples whole clusters.
- **Suites and engine versions** are ordinary row fields. They can be filtered
  by inclusion settings and used as bootstrap strata.

## Statistical assumptions

The R1 model is

\[
u_i = \theta_{a_i} + \mu_{b(d_i)} + \delta_{d_i} + s_i + \eta k_i,
\qquad
P(\text{seat } i \text{ wins} \mid \text{decisive})
= \operatorname{softmax}_i(u).
\]

The fit assumes:

- **Additive Luce utilities.** The winner is the argmax of independent Gumbel
  utilities, so the conditional win probability is the softmax. If the true
  process is not additive, the fit is the best additive approximation in the KL
  sense for the included data.
- **Fixed coefficients.** Agent, seat, bracket, deck and swap effects are fixed
  for the fitted data. There is no time-varying skill, no agent-by-opponent
  interaction, and no agent-by-deck interaction in R1.
- **Exchangeable clusters.** The cluster bootstrap assumes likelihood-bearing
  clusters are independent and exchangeable, within strata if used. Shared
  deals within a cluster are respected by resampling the cluster, not the game.
- **Conditional independence of games** given the parameters, for the Laplace
  approximation. Shared deals can move the true uncertainty in either
  direction, which is why the block-aware method is the cluster bootstrap.
- **Fixed priors.** The R1 priors are part of the estimator. They shrink
  contrasts toward zero, and that finite-sample bias is not a bug.
- **Decisive-only frame.** Draws, caps and failures are reported as outcome
  rates but do not inform the win model. The model describes who wins given
  that the game is decisive.

## Assumptions the caller makes

`podrank` cannot verify these from arrival order or from the records alone:

- **Inclusion is a deliberate choice.** Filters for suites, engine versions,
  splits, agents, time ranges, complete clusters and manifests are supplied by
  the caller. The default inclusion uses the available data.
- **The field is the evaluation distribution.** A field is a declared, finite,
  weighted set of contexts. Choosing it is a modelling decision; the library
  evaluates it and hashes it, but does not choose it.
- **The anchor is meaningful.** The anchor must have likelihood-used games for
  anchor-relative reports. Changing the anchor means a new registry snapshot
  and a refit.
- **Clusters reflect real dependence.** Games that share a deal, an opponent
  draw or any other common randomness should share a `cluster_id`. Misstating
  the cluster structure invalidates bootstrap intervals.
- **Failures are reported, not hidden.** If a caller retries a failed game, the
  new game gets a new `game_id` and the failed attempt is reported. R2's
  comparison evidence additionally requires that only one attempt is in flight
  per position; that is a caller obligation documented in §8.
- **No leakage between splits.** Deck splits are honoured when a split policy
  or inclusion filter is supplied. Training on evaluation decks is the caller's
  choice, not something the library can detect.

## Out of scope

The library does not play games, train agents, schedule matches, manage
checkpoints, or decide which agents to promote or release. Checkpoint-to-step
joins and learning-curve plots belong to the caller: outputs carry `agent_id`
and provenance for joining. Deck and seat terms adjust ratings; this is not a
deck-strength or deck-selection model. Unseen-deck prediction, agent embeddings
and engine/RL dependencies are not part of R1.

## Where the rules live

- §3: records, identity, inclusion and outcome semantics.
- §4: model, priors, identification and reporting.
- §5: posterior and uncertainty.
- §8: comparison assumptions (R2).
- §11.2: the frozen reference worlds used by examples, tests and studies.
