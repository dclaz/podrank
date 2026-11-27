# Identification

**Summary.** A fitted coefficient is only reportable when the data, not the
prior, determines the contrast that uses it. `podrank` separates graph
connectivity, information, and exact null-space aliasing, and reports a
contrast status for every quantity instead of silently printing a prior-driven
number.

## Why identification is separate from fitting

The likelihood is flat along some directions, so the prior alone sets the
answer there. There are two distinct cases:

- **Gauge directions** are exact symmetries of every probability, such as
  adding a constant to all agent log-worths. They are harmless once reports use
  differences, but they must not be confused with zero-information contrasts.
- **Aliasing directions** appear when a quantity cannot be separated from a
  nuisance effect. If agent A only ever plays deck X, and no other agent ever
  plays X, the split between A's log-worth and X's deck effect is decided by the
  prior even with a million games.

`podrank` runs the checks once, on the original fit. Bootstrap replicates and
subsequent reports never reclassify an entity, so a replicate that happens to
omit a rare crossover cluster cannot turn an identified agent into an
unidentified one, or vice versa.

## Components (D1)

Build the agent graph whose edges join agents that appeared together in at
least one likelihood-used game. The component containing the anchor is the
**anchor component**.

- Agents outside the anchor component are fitted but marked `disconnected`.
  Their anchor-relative log-worth, cross-component contrasts and SDWP are
  withheld from default outputs, and they are omitted from the headline
  ordering. `post.components()` lists them separately.
- Contrasts between two agents in the **same non-anchor component** are
  evaluated like any other contrast; `disconnected` describes the contrast to
  the anchor, not the agents' ability to be compared with each other.

If the anchor has no likelihood-used games, the anchor-dependent reports return
a single `AnchorUnsupported(anchor_id=...)` result (D1a). The fit still includes
every supported entity. There is no silent replacement of the anchor: change the
registry snapshot and refit.

## Information diagnostic (D2)

Let \(F\) be the positive likelihood curvature at the mode (the Hessian of the
negative log-likelihood, without the prior), and \(\Lambda\) the prior
precision. In prior-whitened coordinates,

\[
M = \Lambda^{-1/2} F \Lambda^{-1/2}.
\]

After removing the gauge directions, eigen-decompose the remaining block.
Directions with eigenvalue below `weak_ratio` (default 1) are **weakly
supported**: the data carries less information than the prior. The report lists
each weak direction's eigenvalue and top-loading parameter names, so a caller
can see which parameters are driving it. D2 is diagnostic only; it does not
withhold a contrast.

## Data share (D3)

For a nonzero contrast vector \(c\),

\[
\text{data\_share}(c)
= 1 - \frac{c' H^{-1} c}{c' \Lambda^{-1} c},
\]

where \(H\) is the penalised Hessian at the mode. Plain text: the fraction of
the contrast's prior variance that the data has removed. A data share below
0.5 labels the contrast `weak`; the numeric value is still reported and the
agent remains in the headline with a flag. A zero contrast (a self-contrast or
the anchor's own log-worth) has no data share and is `zero` instead.

## Null-space aliasing (D4)

Connectivity and cluster counts are not enough. D4 checks whether the contrast
vector lies in the row space of the **within-game difference design**:

1. For every likelihood-used pattern and seats \(i = 1,2,3\), take the
   difference row \(X_{j,i} - X_{j,0}\). Pattern weights do not matter.
2. Accumulate the \(P \times P\) Gram matrix \(G\) of those rows and keep the
   eigenvectors with eigenvalue above `tol_eig * max(eigenvalue)`. Their span
   is the row space \(V_r\).
3. For a contrast \(c\), compute the aliasing residual
   \[
   r(c) = \frac{\lVert c - V_r V_r' c \rVert}{\lVert c \rVert}.
   \]
4. If \(r(c) > \texttt{tol\_alias}\) (default \(10^{-6}\)), the contrast is
   **aliased**. Otherwise it is identified by the data, subject to D3's
   `weak` label.

The residual is reported for every nonzero public contrast: agent contrasts,
seat effects, deck nuisance contrasts, the swap effect and
`post.contrast_vector`. Working with the Gram matrix rather than a thin SVD is
deliberate: when there are fewer difference rows than parameters, a thin SVD
can return no null vectors at all (F14).

## Contrast statuses (D5)

The first matching row applies.

| Status | Meaning | Default value |
|---|---|---|
| `unrankable` | entity has no likelihood-used games | withheld |
| `zero` | supported contrast vector is exactly zero | `0.0`, exact `[0, 0]` interval |
| `disconnected` | explicit contrast spans graph components | withheld |
| `aliased` | nonzero contrast fails D4 | withheld |
| `weak` | passes D4, data share < 0.5 | numeric, flagged |
| `identified` | passes D4, data share >= 0.5 | numeric |

A missing ID is not a status: it raises `UnknownEntityError`.

`headline_eligibility()` returns each agent's anchor-relative status, an
independent `eligible` boolean and reasons. An agent is eligible exactly when it
is the supported anchor (`zero`) or its status is `weak` or `identified`.
SDWP emits one numeric row per eligible agent, including the anchor, and carries
the log-worth status as provenance.

## Prior-dependent values

Withheld is not the same as unavailable. `log_worth(include_withheld=True)` and
`contrast(..., include_withheld=True)` put the prior-dependent MAP number for a
`disconnected` or `aliased` contrast in `prior_dependent_value`; `value` stays
empty. These diagnostics are never values in a headline table. `unrankable`
entities have no diagnostic number at all.

The split of \(\beta_d\) into \(\mu_b\) and \(\delta_d\), and every deck nuisance
contrast, is always labelled prior-dependent.

## Field identification (FD6)

An identified agent ordering does not guarantee identified SDWP **levels**. A
field can contain seat-utility differences the data never determined, for
example two deck groups that each have many games but never meet (F15).

For every positive-weight context and candidate focal agent, FD6 computes the
three focal-versus-opponent design differences and their aliasing residuals
against the stored \(V_r\). If any residual exceeds `tol_alias`, that agent's
SDWP is labelled `prior_dependent_field` with the failing context indices. The
number is still reported and the agent stays in the headline if D4 allows.

## Pinned example

`doc_tiny` is fully connected: all six agents are in the anchor component, with
no other components. The identification report has six gauge directions. Two
bracket-level directions are weakly supported in this small world, which is an
honest statement about 60 clusters, not a refusal to rank the agents.

```python
post = pr.fit(
    pr.GameTable.from_records(world.game_rows()),
    world.agent_registry(),
    world.deck_registry(),
)
components = post.components()
print(components.anchor_id, components.anchor_component, components.other_components)
print(post.identification_report().n_gauge)
```

```text
agent_00 ('agent_00', 'agent_01', 'agent_02', 'agent_03', 'agent_04', 'agent_05') ()
6
```

## Rules and fixtures

- §4.3: D1, D1a, D2, D3, D4, D5.
- §4.7 and FD6: labels for SDWP and predictions.
- §5.3 B4: identification is frozen from the original fit.
- Fixtures: F10 (gauge), F11 (connected but confounded), F11b (unsupported
  anchor), F14 (wide-design aliasing), F15 (field identification), F17 (D5
  statuses), F18 (suppression precedence).
