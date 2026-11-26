# 0005. SC4 misspecified-world generator

- **Status:** Accepted
- **Date:** 2026-11-26
- **Phase:** P3
- **Contract/section:** §11.2 SC4, §11.3

## Context

§11.2 specifies SC4 as “agent × deck effects `N(0, 0.3²)`, agent ×
opponent effects `N(0, 0.3²)`, kingmaker penalty `0.2` on the leading player's
utility”, but does not pin the pairing, aggregation, draw stream or tie rule.
SC4 is a report-only misspecification scenario; it must not change the fitted
model or be treated as a hard identification result.

## Decision

- **Agent × deck:** independent `E[a, d] ~ N(0, 0.3²)` for every ordered
  `(agent, deck)` pair. `E[a, d]` is added to `a`'s utility whenever `a` plays
  `d`.
- **Agent × opponent:** independent `O[a, o] ~ N(0, 0.3²)` for every ordered
  `a != o`. `a`'s utility receives `sum_o O[a, o]` over the other seats in the
  pod, including independent self-play copies.
- **Kingmaker:** after the additive terms, brackets, decks and seats are summed,
  the seat with the largest utility loses `0.2` before the Gumbel draw. Ties
  choose the lowest seat index deterministically.
- **Stream and order:** all interaction draws come from
  `SeedSequence(study_seed, spawn_key=(4, replication))`. Agent × deck pairs are
  drawn in sorted agent order then sorted deck order; agent × opponent pairs are
  drawn in sorted agent order then sorted opponent order.
- The severity of the misspecification is fixed at `0.3` for both effects and
  `0.2` for the kingmaker penalty. Changing them requires a new scenario name.
- SC4 remains report-only. The fit ignores the interactions; the report shows
  that SDWP ordering is field-invariant while field-level SDWP values differ
  from the misspecified observed world.

## Consequences

- The generator is deterministic for a `(seed, replication)` pair.
- `sim.sc4_interactions()` exposes the drawn effects for reports and tests.
- A misspecified-world coverage or bias result is not a model failure; it
  documents the limits of the additive, non-matchup-aware R1 model.

## Alternatives considered

- Antisymmetric agent × opponent effects — rejected; the plan describes a
  misspecified world, and an ordered-pair draw is simpler and reproducible.
- Drawing interactions per cluster — rejected; effects are properties of the
  agent/deck/opponent pair, not the deal.
- Applying the kingmaker penalty to the true θ leader — rejected; the plan says
  the leading player's utility, which is the current additive utility.
