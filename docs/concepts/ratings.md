# Ratings and fields

**Summary.** `podrank` fits a Bayesian additive Luce model to decisive
four-player games and reports each agent's log-worth relative to the anchor,
with intervals and identification diagnostics. A field turns the fitted
utilities into a standardised decisive win probability (SDWP) for a declared
evaluation distribution.

## The model

For a decisive game, put the four agents in seats \(i = 0, 1, 2, 3\), let
\(d_i\) be the deck in seat \(i\), \(b(d_i)\) its bracket, and \(k_i\) its swap
count. The additive utility and the conditional win probability are

\[
u_i = \theta_{a_i} + \mu_{b(d_i)} + \delta_{d_i} + s_i + \eta k_i,
\qquad
p_i = \frac{\exp(u_i)}{\sum_{j=0}^{3} \exp(u_j)}.
\]

In plain text: each agent contributes a skill term, each deck contributes its
bracket effect plus a deck-specific deviation, each seat contributes a seat
effect, and an optional swap term applies when swaps are modelled. The winner
is the seat with the largest utility, so the probability that seat \(i\) wins
given that the game is decisive is the softmax above.

Only `win` rows enter the likelihood (rule I4). `draw`, `cap` and `failed`
rows are retained, counted and reported as outcome rates, but they are not
treated as wins or as each other. The model describes **who wins, given that
the game is decisive**.

If reality is not additive, the fit is the best additive approximation in the
KL sense for the included data; it is not a claim about "true skill".

## Parameters and priors

| Block | Meaning | Size | Prior |
|---|---|---|---|
| \(\theta\) | agent log-worth | \(A\) | \(N(0, 3^2)\) |
| \(s\) | seat effect | 4 | \(N(0, 0.5^2)\) |
| \(\mu\) | bracket effect | \(B\) | \(N(0, 1^2)\) |
| \(\delta\) | deck deviation within bracket | \(D\) | \(N(0, 0.5^2)\) |
| \(\eta\) | swap effect (off by default) | 1 | \(N(0, 0.1^2)\) |

The priors are fixed in R1. They regularise the fit and make it stable, and
they introduce genuine finite-sample shrinkage; a contrast estimate is not an
unbiased estimate of a generating parameter.

## Log-worth

Log-worth is defined relative to the anchor,

\[
\ell_a = \theta_a - \theta_{\text{anchor}},
\]

so the anchor's own value is exactly zero. For display, the Elo conversion is
\(\text{Elo} = 400/\ln 10 \cdot \ell\). Adding a constant to every
\(\theta\) leaves every probability unchanged, so only differences are
reported.

The same idea applies to seats, brackets and decks:

- a **seat contrast** is \(s_i - \operatorname{mean}(s)\);
- a **deck nuisance** is \(\beta_d - \operatorname{mean}_{\text{ref}}(\beta)\)
  over rankable `train` decks, where \(\beta_d = \mu_{b(d)} + \delta_d\);
- an **agent contrast** is \(\theta_a - \theta_b\), including between agents
  that are not the anchor.

Deck nuisance values are not a deck ranking, and the split of \(\beta_d\) into
\(\mu_b + \delta_d\) is always prior-dependent.

## SDWP

For a canonical field \(F\) with contexts \(c\) and weights \(w_c\), SDWP for
agent \(a\) is

\[
\operatorname{SDWP}(a; F) = \sum_c w_c \, p_{\text{focal}}(a, c; \hat\psi),
\]

where \(p_{\text{focal}}\) is the model's conditional decisive win probability
for \(a\) in the focal seat of context \(c\), evaluated at the MAP parameters
\(\hat\psi\). Plain text: average the agent's predicted win probability over the
field's fixed contexts.

Three consequences matter:

1. **SDWP is not a win rate.** The field weights are fixed; contexts are not
   reweighted by how often they produce decisive games. A field with one
   context that is almost always drawn and one that is almost always decisive
   gives a value between the two conditional win probabilities, not the observed
   share of decisive wins.
2. **SDWP ignores draws and caps.** Read it next to `post.outcome_rates()`.
3. **The field does not make the model matchup-aware.** For fixed parameters,
   higher log-worth means a higher win probability in every context. Changing
   the field changes the size of the gaps, not their order.

SDWP is reported only for headline-eligible agents (see
[identification](identification.md)) and is sorted by descending value, with
agent ID breaking ties. A numeric SDWP can still be labelled
`prior_dependent_field` when the field contains seat-utility differences the
data never determined; the label is reported, the number is retained, and the
model is unchanged. See [fields](ratings.md#fields) below and rule FD6.

## Fields

A field is a finite, weighted list of evaluation contexts. Each context fixes
four decks in seat order, the three opponents, the focal seat, optional swap
counts and a positive weight. Fields are canonicalised when constructed:

- duplicate contexts (same decks, swaps, opponents and focal seat) are merged
  by summing weights;
- contexts are sorted by a canonical key;
- weights are normalised to sum to one;
- the result is hashed with SHA-256.

The hash identifies the exact evaluation distribution. Two fields with the
same contexts and weights have the same hash even if the input order differed.
See [define a field](../how-to/define-a-field.md) for construction, recipes,
persistence and the support checks that SDWP performs.

## Prediction

`post.predict_proba(contexts)` evaluates arbitrary four-seat, outcome-free
inputs at the MAP:

\[
\bigl(p_0, p_1, p_2, p_3\bigr), \qquad \sum_i p_i = 1.
\]

Prediction does not require the agent to be headline-eligible. It does require
every agent and deck to have likelihood-used games, and it applies the FD6
label per input. Prediction of unseen decks is not part of R1.

## Pinned example

The frozen `doc_tiny` world (6 agents, 3 brackets, 12 decks, 60 clusters,
world seed 101) gives the following log-worth and SDWP values. The field is
`sim_field` with 24 contexts.

```python
import podrank as pr

world = pr.sim.reference_world("doc_tiny")
post = pr.fit(
    pr.GameTable.from_records(world.game_rows()),
    world.agent_registry(),
    world.deck_registry(),
)

for row in post.log_worth():
    print(row.agent_id, round(row.value, 6))

field = world.field(24)
for row in post.sdwp(field).rows:
    print(row.agent_id, round(row.sdwp, 6))
```

```text
agent_00  0.0
agent_01 -0.599385
agent_02 -0.190913
agent_03  0.326982
agent_04  0.373443
agent_05  0.614616

agent_05  0.454114
agent_04  0.400392
agent_03  0.390259
agent_00  0.321843
agent_02  0.284809
agent_01  0.214389
```

The `doc_tiny` input contains draws and caps, so `post.counts().eligible` is
240 while `post.counts().likelihood_used` is 223. Every value above is a MAP
plug-in number; intervals and their labels are covered in
[uncertainty](uncertainty.md).

## Rules and fixtures

- §4.1–4.4: model, priors, gauge directions, compressed likelihood.
- §4.6–4.7 and FD6: field materialisation, hashing, SDWP and its labels.
- §10: reporting surfaces, outcome rates and deck nuisance.
- Fixtures: F1 (Luce likelihood), F5 (compressed likelihood, gradient and
  Hessian), F8/F8b/F8c (SDWP, field canonicalisation, round-trip), F10 (gauge
  versus objective), F15 (field identification), F17 (D5 output statuses),
  F20 (prediction and swaps).
