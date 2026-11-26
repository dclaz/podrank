# IMPLEMENTATION_PLAN.md — `podrank`

**Target:** Python ≥ 3.12 · **Audience:** a coding agent implementing the module, and the humans reviewing it. Changes to any contract below require an ADR. Project-specific configuration lives in `PROJECT_DECISIONS.md`, not here.

> Working name: **`podrank`** (§17). Confirm the name is free on PyPI before the first release.

---

## 0. How to use this document (for the implementing agent)

1. Implement P0 → P1 → P2 → P3 for R1; after R1, P4 and P5 are independent, and P5 gates R2 (§15). A phase is done only when all of its acceptance criteria pass in CI.
2. Rules marked **MUST** are implementation rules. Each has a fixture in Appendix A; the fixture is the ground truth. If a rule and a fixture disagree, stop and raise an ADR (a short decision record in `docs/decisions/`).
3. These sections are **contracts**: §2.2 (public API), §3 (data), §4 (rating model), §5 (uncertainty), §6–7 (comparison). Don't change them silently.
4. **Plain-text formulas are normative.** Rendered formulas in user docs are explanatory.
5. **Guarantees come from proofs under written assumptions, not from simulations.** Simulations check the implementation and measure behaviour in predefined scenarios (§11).
6. The core package depends **only on NumPy and SciPy**. Everything else is an optional extra (§13).
7. Correctness first, then clarity, then speed.
8. `N_SEATS = 4` is a module constant. v1 supports exactly 4-player games.
9. **Two releases.** R1 (rating) ships standalone, with no dependency on `podrank.compare`. R2 (pairwise comparison evidence) follows (§15).
10. **Scope test for any new feature:** does it help answer *which agents are better than others, by how much, and when is the evidence sufficient*? If it's about what to *do* with the answer (promote, release, retire, retry, schedule), it belongs to the caller's RL system (§1.4).

---

## 1. Purpose, scope and ownership boundary

`podrank` rates and compares **fixed policies (agents)** in **4-player free-for-all games with a single winner or a draw**. The motivating case is an RL bot for Magic: The Gathering Commander that should play many decks well, but `podrank` has no concept of training, checkpoints or releases: an agent is just a fixed, versioned policy.

| Question | Module | Kind of evidence |
|---|---|---|
| **How do these agents rank against a stated field, and by how much?** | `podrank.rating` (core, R1) | Model-based: a batch Bayesian Luce winner model |
| **Is agent A better (or not worse) than agent B, and is the evidence sufficient yet?** | `podrank.compare` (optional, R2) | Design-based: block scores + anytime-valid betting test |

The rating module never depends on the comparison module.

### 1.1 Ownership boundary

| `podrank` owns | The caller owns (out of scope, not deferred) |
|---|---|
| Result schema and validation | The game engine, its RNG design and replay behaviour |
| Rating fits, rankings, uncertainty, diagnostics | Executing games, workers, distribution, retries |
| Block **score semantics** and block validation (R2) | Building and scheduling evaluation games |
| Pairwise sequential **evidence** (A vs B) | Deciding what to do with the evidence (promote, release, retire) |
| An optional pure pod-sampling helper | Checkpoints, lineage, queues, preemption, attempt tracking |
| Log-based reports (e.g. how much luck rotations share) | Combining comparisons into a policy; error control across comparisons |

Correct seat order, policy identity, replay behaviour and the sampling of comparison blocks are **input assumptions** (§9). `podrank` validates what it can see in the data; it can't verify how games were executed.

### 1.2 Not in scope

Game-theoretic population analysis (α-Rank, CCE/deviation ratings), engine changes, harness orchestration, RL training changes. Nothing depends on OpenSpiel.

### 1.3 Deck effects are nuisance controls

> **Comparing or ranking deck strength is not a goal of `podrank`.** Deck parameters exist only so that agent ratings aren't biased by which decks an agent happened to play. They depend on the agents, suites and seat mix in the fit, so they are not a deck power ranking. **Downstream RL agents will be much better at judging deck strength**: they see every card and the full game state and can evaluate decks via their value networks or controlled play-outs. Don't extend `podrank` to do this, and don't add agent × deck interactions just to give the headline rating a stronger interpretation.

### 1.4 Belongs to the caller's RL system

Out of scope permanently: promotion gates and gate configs; candidate/incumbent roles and promotion margins; attempt logs and α spending across attempts; spec issuing, run seeds and spec ledgers; draw-guardrail decisions; anything keyed on checkpoints, generations or training steps. `docs/examples/rl-pipeline.md` shows (without shipping) how a caller might build these from `podrank` outputs.

---

### 1.5 Key terms (read first)

| Term | Meaning | Where it matters |
|---|---|---|
| **Agent** | A fixed policy: weights + inference settings (`policy_version`). Different settings = different agent | Everywhere |
| **Pod** | Which 4 decks sit in seats 0–3 (seat 0 moves first) | Fields, comparisons |
| **Deal** | A pod plus everything random about the game's setup (shuffles, seed), identified by `deal_id` | Comparisons |
| **Cluster** | A set of games that may be correlated because they share a deal. Stored as `cluster_id` on each game; a game with no `cluster_id` is a cluster of one. R1 uses clusters **only** to keep correlated games together when estimating uncertainty (§5.3) | R1 |
| **Block** | A cluster with a fixed structure, used by R2: one deal played several times with the agent under test (the *focal* agent) rotated through the seats. A **duplicate block** is 4 games (focal agent A in seats 0, 1, 2, 3 in turn; three copies of B elsewhere). A **paired block** is 8 games (each rotation played once with A focal and once with B focal, against the same opponents). A block's `block_id` is its `cluster_id` | R2 |
| **Rotation** | Which seat the focal agent occupies within a block (0–3) | R2 |
| **Role** | In a paired block, whether A or B is the focal agent in that game | R2 |
| **Context** | One fully specified situation for prediction: decks and swaps per seat, the opponents, and the focal seat | Fields |
| **Field** | A fixed, weighted list of contexts that every agent is scored against (§4.6) | R1 headline (SDWP) |

**Example duplicate block.** One deal: decks Atraxa, Krenko, Kinnan, Ghave in seats 0–3, fixed seed. It's played four times:

| Rotation | Seat 0 (Atraxa) | Seat 1 (Krenko) | Seat 2 (Kinnan) | Seat 3 (Ghave) |
|---|---|---|---|---|
| 0 | **A** | B | B | B |
| 1 | B | **A** | B | B |
| 2 | B | B | **A** | B |
| 3 | B | B | B | **A** |

A plays every seat and deck once with the same luck, so much of the luck cancels (like duplicate bridge). The block score is A's average points minus 0.25 (§6.3). All four games share one `cluster_id` (= `block_id`), and uncertainty methods resample them together, never individually.

## 2. Design principles and public API

### 2.1 Principles

1. **Comparison evidence is design-based; ratings are model-based.** Never mix them silently; label every number with its kind. `podrank` reports evidence; it never makes operational decisions.
2. **Name what is estimated.** R1 estimates *additive-model log-worth* and the *field-standardised decisive win probability* (SDWP). R2 estimates the *mean block score lift of A over B*. These are different quantities (§4.7, §6.6).
3. **Batch, order-independent ratings** from one fit over all included games.
4. **No silent data loss and no silent prior-only results.** Failed games are never dropped quietly; quantities without data support are labelled or withheld (§3.5, §4.3).
5. **Reproducible.** Seeds via `numpy.random.SeedSequence`; every output records the package version, data hash, registry snapshot ids, field hash and seeds.
6. **Closed-form linear algebra, dense Cholesky** at v1 scale (≈ 200 parameters).
7. **Simple API with good defaults.** Typed, documented, with runnable examples.

### 2.2 Public API at a glance (contract)

```python
import podrank as pr

# ---------- Rating (R1, standalone) ----------
games  = pr.GameTable.from_records(records)            # in-memory; no extras needed
# or: games = pr.io.read_parquet("logs/")              # requires podrank[io]
agents = pr.AgentRegistry.from_records(agent_rows)     # immutable, content-hashed snapshot
decks  = pr.DeckRegistry.from_records(deck_rows)

pr.validate(games, agents, decks)                      # optional preview; raises on errors by default
post = pr.fit(games, agents, decks)                    # always runs the same pipeline (I0); ModelSpec(), Inclusion.default()

post.counts()                                          # received / eligible / likelihood-used
post.components()                                      # anchor component vs other components
post.headline_eligibility()                            # per-agent contrast_status + eligible, or AnchorUnsupported
field = pr.Field.load("fields/eval_field_v1.json")     # finite, hashed list of contexts
post.sdwp(field)                                       # headline: MAP plug-in SDWP
post.log_worth(); post.contrast("ckpt_0050", "ckpt_0040")
post.outcome_rates(); post.unrankable(); post.identification_report()

boot = pr.uncertainty.cluster_bootstrap(post, n=400, seed=7)
boot.sdwp(field)                                       # MAP plug-in SDWP + cluster-bootstrap interval
boot.contrast("ckpt_0050", "ckpt_0040")

# ---------- Comparison (R2, optional) ----------
pods = pr.PodDistribution.load("pods/train_v1.json")   # frozen, versioned
draws = pods.sample(n_blocks=64, seed=1234, start_index=0)  # optional pure helper; returns PodDraws
# ... caller binds draws into descriptors, declares them, then builds and runs games ...
cmp = pr.compare.Comparison(a="ckpt_0050", b="ckpt_0040", design="duplicate",
                            agents=agents, decks=decks, engine_version="engine_hash",
                            pod_distribution=pods,
                            hypothesis=pr.compare.Superiority(margin=0.0),
                            alpha=0.05, max_blocks=4000, replay="deterministic")
                            # optional: alpha_up=, alpha_low=, cs_alpha=
cmp.declare_blocks(descriptors)    # BlockDescriptor records (C1), declared before play
cmp.ingest(games)                  # ordinary GameTable rows
cmp.mark_failed(block_id, rotation, role=None)   # or cmp.mark_block_failed(block_id)
cmp.extend_budget(new_max)         # continues with the same wealth (E0)
cmp.status()                       # effective status (E7): ABOVE | BELOW | CONTINUE | BUDGET_EXHAUSTED
                                   #   | INCONCLUSIVE | INCONCLUSIVE_INVALID
cmp.decision_at_stop; cmp.evidence_valid
cmp.report()                       # evidence, mean-lift bounds, confidence sequence, rates
cmp.rating_rows(); cmp.failure_records()   # revisioned export snapshot (E8)
cmp.save(path); pr.compare.Comparison.load(path)
```

`podrank` never builds, schedules or runs games. Running validate → fit → bootstrap → SDWP never imports `podrank.compare` (tested, §14).

---

## 3. Data contracts

### 3.1 Registries

Registries are **immutable snapshots** with a `snapshot_id` (content hash of all rows). Edits create a new snapshot. Every output records the snapshot ids it used.

**AgentRegistry**

| Field | Type | Notes |
|---|---|---|
| `agent_id` | str | Unique, stable |
| `kind` | enum | `random`, `heuristic`, `checkpoint`, `exploiter`, `other`. Descriptive only; no behaviour attached |
| `policy_version` | str | Weights hash + inference config. **Same weights, different config = different agent.** Repeated on every game row |
| `is_anchor` | bool | Exactly one anchor. Use a heuristic bot, not the random bot |
| `tags` | list[str] | Caller-defined, descriptive |
| `metadata` | dict | Free-form; never interpreted (except as an explicitly named grouping key) |

**DeckRegistry**

| Field | Type | Notes |
|---|---|---|
| `deck_id` | str | Stable id |
| `bracket` | int | 1–5 or own scheme |
| `split` | enum | `train`, `validation`, `holdout`. A label only; any number of decks per label, including none (§20) |
| `deck_version` | str | Hash of the canonical card list |
| `tags` | dict[str, str] | Reporting only; **never enter the model** |
| `metadata` | dict | Free-form |

### 3.2 Game table (one row per game; shared by R1 and R2)

| Field | Type | Notes |
|---|---|---|
| `game_id` | str | Unique per execution attempt, including retries; reuse only for retransmitting the identical report |
| `played_at` | timestamp (UTC) | **Required** |
| `agents[4]`, `decks[4]` | str | Per seat; **seat 0 takes the first turn** |
| `policy_versions[4]` | str | **Required**; must match the snapshot |
| `deck_versions[4]` | str \| null | If supplied, must match |
| `swap_counts[4]` | int | 0 = canonical list |
| `outcome` | enum | `win`, `draw`, `cap` (turn/time cap: a game outcome), `failed` (execution failure: **no outcome**) |
| `winner_seat` | int \| null | 0..3 for `win`; null otherwise |
| `suite` | str | Caller-defined label |
| `cluster_id` | str \| null | Shared-deal cluster label (R1 meaning: games that may be dependent). Null = singleton cluster. For R2 games it equals the `block_id` |
| `block_id`, `rotation`, `role` | str, int, enum \| null | R2 only (§6); ignored by R1 apart from `cluster_id` |
| `deal_id` | str \| null | R2 only: opaque deal identity (e.g. deal seed); must match the block descriptor |
| `spec_hash` | str \| null | Optional caller-supplied specification fingerprint, retained for audit only; not a replay or acceptance key |
| `engine_version` | str | |
| `audit` | dict \| null | Optional provenance; if present, a JSON object with string keys and finite numbers, stored without affecting inference |

`draw` and `cap` are outcomes; `failed` is missing data. They are never conflated.

### 3.3 Pre-game covariates only

Only agent, seat, deck and swap count may enter the model (`ModelSpec` whitelist). Variables partly caused by the agents' play (mulligans, game length…) would bias skill estimates.

### 3.4 R1 validation (`pr.validate`)

R1 treats `cluster_id` as a label only. **R1 has no knowledge of 4- or 8-game block layouts**; rotation, role and position checks are R2's (§6.5).

R1 checks, at the boundary:
- **Shape and types:** `agents`, `decks`, `policy_versions`, `deck_versions`, `swap_counts` have length exactly 4; `outcome` and `role` are known enum values; `played_at` must be a timezone-aware datetime or RFC 3339 string with an explicit offset; reject naive values and normalise to UTC; `swap_counts` are non-negative integers; `winner_seat` is 0..3 when `outcome == "win"` and null otherwise.
- **Uniqueness:** exact duplicate rows (identical in every field) are deduplicated with a warning; rows sharing a `game_id` with different content raise `DuplicateGameError`.
- **Identity:** every agent and deck id exists in the bound snapshots; `policy_versions` match the snapshot; `deck_versions`, if non-null, match the snapshot (null means "not supplied" and is accepted); exactly one anchor.
- **Mirror seats:** an agent may appear in several seats of one game (default allowed, since comparisons use copies of B); `allow_mirror=False` rejects this.
- **Split governance (opt-in):** `podrank` has no built-in rule about which suites may use which deck splits. A caller may pass `split_policy=` (a mapping `split -> allowed suites`); rows that violate it are errors. Without a policy, nothing is checked (§20).

Optional `tag_schema=` (a dict of tag key → allowed values or `"wubrg"`, plus a list of required keys) checks deck tags; `PROJECT_DECISIONS.md` gives this project's schema. Tags still never enter the model.

Validation can't prove execution matched the declaration (§9). Returns a `ValidationReport`; raises on errors unless `strict=False`. Both entry points apply the same `allow_mirror`, `split_policy` and `tag_schema` checks when given equivalent inputs. `strict=False` is available only on `pr.validate` (as a preview); `pr.fit` always raises on validation errors.

### 3.5 Inclusion and counts

**MUST (I0). One pipeline.** `pr.fit` runs these steps in this order on every call, whether or not `pr.validate` was called first; `pr.validate` runs steps 1–3 only. Both delegate to internal `data.inclusion.run_pipeline(games, agents, decks, *, inclusion, split_policy=None, tag_schema=None, validate_only=False, strict=True)`: `validate` supplies an inclusion with only its mirror policy overridden, sets `validate_only=True`, and returns a `ValidationReport`; `fit` consumes the full pipeline result and always uses `strict=True`. In non-strict preview, collect errors without passing malformed rows to downstream stages. The full pipeline runs:
1. schema and shape checks (§3.4), including UTC timestamp normalisation and canonical typed optional defaults;
2. exact-duplicate removal by equality of canonical row bytes (C.2); conflicting `game_id`s raise;
3. identity checks against the snapshots, including optional split and tag policies; on the full path, validate complete-only manifest ownership **here**, before I1;
4. row filters (I1);
5. completeness filter (I2);
6. outcome selection for the likelihood (I4) and the support rule (I5).

**MUST (I1). Row filters select individual games.** Apply filters in I3 order. Suite and engine version test exact membership; `played_at` uses inclusive UTC bounds. `None` for a set filter disables it; an empty set selects no rows. Filters on entities have fixed meanings for mixed pods:
- **split filter:** a game passes only if **all four** of its decks carry a selected split;
- **agent filter:** a game passes if **at least one** seat holds a selected agent.

There are no other entity-filter semantics in v1. A cluster keeps whichever of its rows survive. The report lists clusters that were partially filtered and how many rows each lost.

**MUST (I2). Cluster filter.** Default `clusters="use_available"`: all rows that survive I1 are used. `clusters="complete_only"` requires an explicit **expected-games manifest** (`cluster_id -> set of game_ids`):
- Without a manifest, `complete_only` raises `InclusionError`.
- **Exact membership.** Build a reverse index `game_id -> cluster_id` from the manifest; reject a game ID listed under multiple clusters. On deduplicated, identity-validated rows **before I1**, require every non-null `cluster_id` to match the reverse index for that row, and require every null `cluster_id` game to be absent from the index. Violations raise `ManifestMismatchError`, even when I1 would filter out the row.
- **Completeness.** A manifested cluster is kept only if every listed game is present, passed I1 and is not `failed`. Otherwise all its rows are dropped, with the reason (`missing`, `filtered` or `failed`) per game.
- **Null `cluster_id`:** the row is a singleton cluster. It needs no manifest entry and is kept if it passes I1 and is not `failed`.
- Missing games are **never inferred** from a bare `cluster_id`.

**MUST (I3). Counts** (`post.counts()`), overall and per agent and deck:

| Count | Meaning |
|---|---|
| received | rows supplied, **before** deduplication |
| exact_duplicates | rows removed in I0 step 2 |
| eligible | rows that pass I0 steps 1–5 |
| excluded, by reason | rows removed by each filter (`split`, `agent`, `suite`, `engine_version`, `played_at`, `cluster_missing`, `cluster_filtered`, `cluster_failed`); a row is counted under the first filter that removes it, in the order listed |
| likelihood-used | eligible and `outcome == "win"` (I4) |
| likelihood-bearing clusters | clusters with ≥ 1 likelihood-used game |

Overall counts are **games**. Per-entity counts give both **games** (distinct games containing the entity) and **seat appearances** (a mirror-seat game counts once per seat). Overall: `received - exact_duplicates = eligible + sum(excluded)`.

Outcome rates (§10.5) are computed on **eligible** rows. A separate, clearly labelled **pre-inclusion** panel reports draw/cap/failed rates on the deduplicated input before steps 4–5, so failures removed by `complete_only` stay visible.

**MUST (I4). Draws.** R1 has **one fitting mode**: only `win` games enter the likelihood. `draw`, `cap` and `failed` are counted and reported (§10.5) but never fitted.

**MUST (I5). Evidence.** No likelihood-used games → `NoRankableEvidence`. An agent or deck with no likelihood-used games is removed from the parameter vector and listed by `post.unrankable()`; it is never ranked from the prior. Agents whose decisive games are all wins or all losses are flagged `saturated`.

---

## 4. Rating model

### 4.1 Model

```
u[g,i]  = theta[a] + mu[b(d)] + delta[d] + seat[i] (+ eta * k[g,i])
p[g,i]  = exp(u[g,i]) / sum_j exp(u[g,j])
```
where a = agent, d = deck, b(d) = its bracket, k = swap count. Only `win` games are fitted (I4). The model describes **who wins, given that the game is decisive**.

If reality isn't additive, the fit is the best additive approximation (in the KL sense) for the included data, not "true skill".

### 4.2 Parameters, priors, encoding

| Block | Size | Prior | Default |
|---|---|---|---|
| θ (agents) | A | N(0, 3²) | on |
| s (seats) | 4 | N(0, 0.5²) | on |
| μ (brackets) | B | N(0, 1²) | on |
| δ (deck deviations) | D | N(0, σ_β²), **σ_β = 0.5 fixed in R1** | on |
| η (swap effect) | 1 | N(0, 0.1²) | off |

One-hot columns for agent, seat, bracket, deck. ≈ 200 parameters at A = 50, D = 140, B = 5.

**Gauge directions** (exact likelihood symmetries; fixed only by the prior): all-ones on θ; all-ones on seats; all-ones on μ; and for each bracket b, (+1 on μ_b, −1 on every δ_d with b(d) = b). Moving along a gauge direction leaves every probability and `likelihood_nll` unchanged, but **changes the objective** by the change in the prior penalty (§4.4, F10). Only gauge-invariant quantities are reported (§4.7).

Gauge directions are one part of the likelihood's **null space**. The design can have further null directions (e.g. an agent that always plays the same deck; §4.3 D4).

### 4.3 Identification and components

**MUST (D1). Components.** Build the agent graph whose edges join agents that appeared together in at least one likelihood-used game. The **anchor component** is the component containing the anchor. For agents outside it:
- the fit still includes them;
- they have status `disconnected` (D5): their anchor-relative log-worth, their contrasts with agents in other components and their SDWP are **withheld** from default outputs;
- they are **omitted from the headline ordering** and listed separately by `post.components()`;
- contrasts between two agents **in the same non-anchor component** are evaluated by D4 like any other contrast.

**MUST (D1a). Unsupported anchor.** Still fit every supported entity. `components()`, `headline_eligibility()`, `log_worth()` and `sdwp()` (including their bootstrap equivalents) return one `AnchorUnsupported(anchor_id=...)` result, not a row per agent; `identification_report()` retains its diagnostics and uses that result for the anchor-dependent component section. For these methods, short-circuit before field evaluation or row construction, after validating method arguments unrelated to field support. Explicit agent contrasts use D5 normally, including `unrankable` for a contrast involving the unsupported anchor. No automatic replacement: use a new registry snapshot and refit to change the anchor.

**MUST (D2). Information diagnostic.** Let F be the **positive** likelihood curvature at the mode (the Hessian of the negative log-likelihood, without the prior; positive semidefinite). In prior-whitened coordinates, `M = Lambda^(-1/2) F Lambda^(-1/2)`. Remove the gauge directions in those coordinates: each gauge vector g maps to `Lambda^(1/2) g`. Build an orthonormal basis `Q` (P × (P − n_gauge)) of the orthogonal complement of their span (e.g. from a full QR of the mapped gauge vectors), and eigen-decompose `Q' M Q`. Working in this basis guarantees gauge directions never reappear as zero eigenvalues. Directions with eigenvalue < `weak_ratio` (default 1: the data carries less information than the prior) are **weakly supported**; report their top-loading parameters.

**MUST (D3). Data share.** For each reported nonzero contrast vector c: `data_share = 1 - (c' H^-1 c) / (c' Lambda^-1 c)`. Below 0.5 → `weak` (D5). For c = 0 (a self-contrast) no data share is computed; it is reported as not applicable.

**MUST (D4). Data-identified headline contrasts (null-space check).** Connectivity (D1) and support counts aren't enough. If agent A always plays deck X **and no other agent ever plays X**, A's log-worth can't be separated from X's deck effect, however many games it has; the prior alone would decide the split (F11). (If other agents also play X, their crossover games may identify the split; D4 decides, not a support count.)

Procedure, at the fitted design (likelihood-used games only):
1. Build the **within-game difference rows**: for every pattern j and seats i = 1..3, the row `X[j,i] - X[j,0]` (the softmax depends only on these differences). Pattern weights don't matter for the row space.
2. Accumulate the **P × P Gram matrix** `G = sum over difference rows of row' row` (P = number of parameters, ≈ 200; built per pattern, never materialising all rows). Compute `w, V = numpy.linalg.eigh(G)`. Keep eigenvectors with `w > tol_eig * max(w)` (default `tol_eig = 1e-10`). Let `V_r` be the retained columns: an orthonormal basis of the **row space** (the directions the data determines). Working with G avoids the thin-SVD pitfall: when there are fewer difference rows than parameters, a thin SVD returns no null vectors at all (F14).
3. Define the **aliasing residual** of any contrast vector c as the part of c outside the row space:
   `r(c) = || c - V_r (V_r' c) || / || c ||`.
   For each agent a in the anchor component, use `c = e_theta[a] - e_theta[anchor]`. **For c = 0** (the anchor itself, or any self-contrast) no residual is computed; its status comes from D5.
4. Classify:
   - `r(c) > tol_alias` (default 1e-6): **aliased**. The contrast isn't determined by the data at all. The agent is **withheld from the headline ordering and the headline SDWP table**, with reason `aliased`, and the top-loading null-space parameters are reported (e.g. `theta[A]` with `delta[X]`). The fit and its diagnostics remain available; the agent's row in `post.log_worth()` shows status `aliased` with no value (D5).
   - otherwise **identified**. If D3's data share is < 0.5, also label `weak` (shown in the headline, flagged).
5. `post.headline_eligibility()` returns each agent's anchor-relative `contrast_status`, separate `eligible` boolean, and reasons (D5), or the method-level result in D1a.

The same check applies to **every public contrast**: agent contrasts, seat effects, deck nuisance contrasts, the swap effect and `post.contrast_vector(c)`. Field contexts use it through FD6. The row-space basis `V_r` is computed once per fit and stored in the posterior. The bootstrap (§5.3) never changes any classification; it is computed once on the original fit.

**MUST (D5). Contrast status and headline eligibility.** For contrast results, apply the first matching row. Method-level anchor handling (D1a) happens before this table. Reports without a contrast (e.g. counts or provenance) have no contrast status.

| Status | Condition | Default contrast value |
|---|---|---|
| `unknown` | A supplied ID is absent from the bound snapshots | Raise `UnknownEntityError` |
| `unrankable` | A named entity has no likelihood-used games | Withheld |
| `zero` | The supported contrast vector is exactly zero | `0.0`, data share absent, exact interval `[0.0, 0.0]` |
| `disconnected` | An explicit agent contrast spans graph components (D1) | Withheld |
| `aliased` | Nonzero contrast fails D4 | Withheld |
| `weak` | Passes D4 and `data_share < 0.5` | Numeric, flagged |
| `identified` | Passes D4 and `data_share >= 0.5` | Numeric |

- **Diagnostic values.** `log_worth(include_withheld=True)` and `contrast(..., include_withheld=True)` put the prior-dependent MAP number for `disconnected` or `aliased` in `prior_dependent_value`, never in `value`. No diagnostic number exists for an unrankable entity.
- **Headline.** `eligible` is true exactly when the agent is the supported anchor (its anchor-relative contrast status is `zero`) or its anchor-relative status is `weak` or `identified`. SDWP emits one numeric row per eligible agent, including the anchor. Carry `contrast_status` as provenance; it describes log-worth, not the SDWP value or interval. Apply FD6 separately, retaining the number with a field-dependence label if needed. Sort by descending MAP SDWP, then by agent ID for ties, including when displaying posterior means or bootstrap intervals.
- **Intervals.** Supported zero *contrasts* have exact `[0, 0]` intervals and bypass uncertainty checks. Nonzero contrasts get intervals only when weak or identified, subject to method-specific rules. Numeric SDWP for the anchor uses the same uncertainty rules as for any other eligible agent; it never gets a zero interval merely because its contrast is zero.

**Also reported:** agents with < 3 distinct decks or a seat distribution far from uniform (possible confounding), and suggested crossover games. Generating them is the caller's choice.

**Always labelled prior-dependent:** the split of β_d into μ_b and δ_d.

### 4.4 Compressed likelihood, gradient and Hessian

**MUST (L1). Summed target mass throughout.** Group eligible decisive games by identical design row (agents, decks, seats, swap counts) into patterns j. For pattern j store:

```
T[j,i] = number of fitted games in pattern j won by seat i      (summed target mass)
c[j]   = sum_i T[j,i]                                              (pattern weight)
```

and use, with p[j] the softmax of the pattern's utilities:

```
likelihood_nll(psi) = - sum_j sum_i T[j,i] * log p[j,i]
prior_penalty(psi)  = 0.5 * psi' Lambda psi
objective(psi)      = likelihood_nll(psi) + prior_penalty(psi)       # what the solver minimises

grad_u (pattern j)  = c[j] * p[j] - T[j]
Hess_u (pattern j)  = c[j] * ( diag(p[j]) - p[j] p[j]' )
grad_psi            = X' (stacked grad_u)  +  Lambda psi          # gradient of objective
H_psi               = sum_j X_j' Hess_u(j) X_j  +  Lambda         # Hessian of objective
```

The implementation exposes `likelihood_nll`, `prior_penalty` and `objective` as separate functions with these exact names. "NLL" alone is never used for the penalised objective.

**Never** multiply a summed target by c again (i.e. never `c * (p - T)`). Normalised targets aren't used anywhere. Appendix A F5 checks all three quantities against the expanded rows.

**Reduced parameterisation.** Deck effects enter only through β = M (μ, δ) with M = [bracket one-hot | I]. Compute in the reduced vector (θ, β, s, η) — 4 non-zeros per row — then `H_psi = J' H_red J + Lambda`.

**Solver:** dense SPD Cholesky (`scipy.linalg.cho_factor`) only. Before allocating, estimate the dense memory needed (Hessian, Cholesky factor, Gram matrix and its eigenvectors, bootstrap worker copies). Raise `ModelTooLargeError` if the parameter count exceeds `max_params` (default 10,000) **or** the estimate exceeds `max_dense_bytes` (default 2 GiB). Budget at least `4 * 8 * P**2` bytes per simultaneously active dense workspace (one in serial fit; include the parent and worker workspaces if bootstrap is parallel). Include additional concurrent dense allocations in the preflight estimate. The 2 GiB limit normally binds first: four matrices alone reach it at P = 8,192, and additional allocations may lower that ceiling. The parameter limit is a separate cap.

### 4.5 Fitting

`pr.fit` takes the inputs and keyword arguments listed in Appendix C.

1. Run the I0 pipeline (validation, I1–I2, counts I3, I4–I5).
2. Build the design and parameter index using Appendix C ordering.
3. **Compress (MUST, L2):** build the outcome-and-cluster map `(pattern j, winner seat i, cluster_id) -> count`. The fit uses `T[j,i] = sum over clusters`; the bootstrap uses the map directly.
4. Solve by damped Newton using the initialisation, stopping criterion and Armijo constants in Appendix C.
5. Max-subtracted log-softmax; float64; floating-point warnings are errors in tests.
6. Warm starts are keyword-only and matched by parameter name (Appendix C).
7. Return `Posterior` (mode, Cholesky factor, parameter index, spec, inclusion, counts, components, D5 statuses, row-space basis `V_r`, priors, outcome-and-cluster map, retained cluster table for strata, diagnostics, provenance).

### 4.6 Reference fields

**MUST (FD1). A field is a finite weighted list of contexts.** Each context declares:

```
decks[4], swap_counts[4], opponents (ordered, for the three non-focal seats), focal_seat, weight
```

Input weights must be positive and finite. They need not sum to 1; normalisation happens during canonicalisation (FD3).

**MUST (FD2). Procedural recipes are materialised once.** A recipe (e.g. "uniform over 4 distinct training decks, 3 × B, uniform focal seat") is expanded into the explicit list by `pr.Field.materialise(recipe, decks=…)`. The **only** seed is the recipe's `seed` key; there is no `seed=` keyword. The materialised list is what gets saved, hashed and used.

**Recipe schema (v1).** `pr.Field.materialise(recipe, decks=<DeckRegistry snapshot>)` accepts exactly this form; the deck registry snapshot is a **required** input (it supplies deck ids, brackets and splits), and its `snapshot_id` is recorded in the materialised field; unknown keys raise. Values shown are illustrative; every value is a caller parameter (`min_decks_per_bracket` defaults to 1). Allowed values: `bracket_policy` ∈ {`same_bracket`, `any`}; `bracket_weights` ∈ {`equal`}; `opponents.draw` ∈ {`independent_uniform_per_seat`}; `focal_seat` ∈ {`uniform`}; `distinct_decks` must be `true`; `swap_counts` a non-negative integer; `n_contexts` ≥ 1; `pool` non-empty; `seed` a non-negative integer. The example's `min_decks_per_bracket: 8` is illustrative; the `doc_tiny` world (4 decks per bracket) pins its own value of 4 in its world file.

```json
{
  "field_id": "example_field",
  "version": 1,
  "recipe": {
    "deck_split": "train",
    "pod": {"bracket_policy": "same_bracket", "bracket_weights": "equal",
            "min_decks_per_bracket": 8, "distinct_decks": true, "swap_counts": 0},
    "opponents": {"pool": ["heur_a", "heur_b"], "draw": "independent_uniform_per_seat"},
    "focal_seat": "uniform",
    "n_contexts": 2000,
    "seed": 12345
  }
}
```

**Effective minimum:** a bracket is eligible only if it has at least `max(4, min_decks_per_bracket)` decks in the chosen split (4 distinct decks are needed per pod; for `bracket_policy: "any"`, the split as a whole needs ≥ 4). If no bracket is eligible, raise `NoEligiblePodsError` naming the split and the per-bracket counts.

Use `rng = Generator(PCG64(SeedSequence(seed, spawn_key=(k,))))` for context k. Sort opponent IDs and reject duplicates (`ValueError`); sorting does not remove entries. Sort brackets numerically and decks by ID. Call the RNG in this order:

1. For `same_bracket`, choose `eligible_brackets[rng.integers(0, len(eligible_brackets))]`, then use that bracket's sorted decks. Warn for excluded undersized brackets. For `any`, use the entire sorted split and skip the bracket RNG call.
2. Use `rng.choice(len(deck_pool), size=4, replace=False)` in draw order for seats 0–3.
3. Set `focal_seat = int(rng.integers(0, 4))`.
4. Use `rng.integers(0, len(opponent_pool), size=3)` to assign opponents to non-focal seats in ascending seat order.

Assign each draw weight 1 and canonicalise with FD3. The materialised list defines the field, so it carries no sampling error.

**MUST (FD3). One canonical form, used for both hashing and evaluation.**
1. **Coalesce:** contexts with identical `(focal_seat, decks, swap_counts, opponents)` are merged; their weights are summed with `math.fsum` (correctly rounded, so independent of input order).
2. **Order:** sort the merged contexts by the key `(focal_seat, decks, swap_counts, opponents)`, comparing lists element-wise as strings/integers.
3. **Normalise:** divide each weight by `math.fsum` of all merged weights (float64). Check `abs(math.fsum(weights) - 1) <= 1e-12` afterwards; no further adjustment.
4. **Serialise:** `{"contexts": [...], "field_id": ..., "version": ...}`, each context with keys `decks, focal_seat, opponents, swap_counts, weight`; `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`, floats via Python's shortest round-trip `repr`.
5. `field_hash = sha256(utf-8 bytes).hexdigest()`.

**Evaluation uses exactly this canonical list** (same order, same weights). Guarantees: **reordering** the input contexts (including duplicates) never changes the hash or any result. **Splitting** a context's weight into duplicates preserves the hash only if the merged canonical weights are bit-identical. Float64 can't promise that in general (e.g. 0.1 + 0.2 ≠ 0.3), so callers who need stable hashes should supply already-merged weights (F8b). The serialised payload includes `field_id` and `version`, so both are covered by `field_hash`.

**MUST (FD3c). Field outputs and compatibility.** Every field output carries `field_id`, `version`, `field_hash` and a **context-version fingerprint**: sha256 hex digest of the FD3 canonical UTF-8 JSON object `{"agents": [[id, version], ...], "decks": [[id, version], ...]}`. Each list contains one pair per distinct referenced ID, sorted by ID, using versions from the bound snapshots. Include only the field’s context opponents and decks; focal policy versions are carried separately per result row and do not change the field fingerprint. This keeps the fingerprint comparable when the same frozen field is used across different focal cohorts. The payload hash identifies the field; the fingerprint identifies what its ids meant when it was evaluated. Two outputs are comparable only if both match. `podrank` does not compare outputs itself; it exposes both values so the caller can check.

**MUST (FD3a). Construct vs load.** Canonicalisation (coalesce, order, normalise) happens **only** when a field is built from raw contexts or a recipe. `pr.Field.load(path)` reads a saved canonical field and **validates without renormalising**: it parses the JSON rejecting duplicate keys at any level; validates the payload schema; checks positive finite weights, `abs(math.fsum(weights) - 1) <= 1e-12`, canonical order and unique contexts; re-serialises the **parsed** payload with the FD3 rules; recomputes the hash and compares it with the stored `field_hash`. No raw-byte comparison of the original file is made or claimed. Renormalising on load is forbidden: normalised float64 weights need not sum to exactly 1 (e.g. raw `[1, 6, 15]` normalise to weights summing to 0.9999999999999999), and dividing again changes them and the hash (F8c).

**MUST (FD3b). Saved file format.** A saved field is a JSON **envelope** with exactly three keys:

```
{
  "payload":    <the FD3 canonical object: contexts, field_id, version>,
  "field_hash": sha256 of the FD3 canonical serialisation of "payload" only,
  "provenance": {"recipe": <recipe or null>, "deck_snapshot_id": <str or null>,
                 "materialised_with": <podrank version>}
}
```

The hash covers **only the payload**. `field_hash` and `provenance` sit outside it, so the definition isn't circular, and provenance can be recorded without changing the field's identity: two fields with identical contexts and weights have the same hash however they were produced. `Field.load` rejects an envelope whose payload doesn't hash to `field_hash` (`FieldIntegrityError`), and preserves `provenance` unchanged (F8c).

**MUST (FD4). Supported contexts only.** Every deck in a field must have likelihood-used games; every opponent must be rankable and in the anchor component. Otherwise `Field` evaluation raises `UnsupportedFieldError`, naming the offending entries. There is no unseen-deck prediction in R1 (§19).

**MUST (FD4a). Swap counts.** Field construction, recipe materialisation and loading validate non-negative integer swaps but do not consult a model. After I2, `fit` checks every eligible row: if η is off, nonzero swaps raise `SwapsNotModelledError` unless `ModelSpec(allow_pooled_variants=True)` was set. In that case fitting ignores the swaps and all posterior-derived outputs carry `pooled_variants`. `predict_proba` likewise accepts and ignores nonzero swaps only with that pooling opt-in when η is off. Field evaluation with η off always rejects nonzero swaps, including on pooled posteriors. With η on, fit, prediction and field evaluation use the supplied counts.

**MUST (FD5). Seeded Monte Carlo.** Any posterior-sampling computation takes an explicit `seed`. With the same seed, inputs, package version and numerical environment (NumPy/SciPy/BLAS build, platform), results are bit-identical. Across environments, results agree with `rtol=1e-10, atol=1e-12` for portable numeric fixture checks (§15); RNG stream goldens are exact within the pinned reference environment.

**MUST (FD6). Field identification label.** An identified agent ordering (D4) doesn't guarantee identified SDWP *levels*: a field context can contain seat-utility differences the data never determined (e.g. two deck groups that each have many games but never meet; F15). For every positive-weight context and every candidate focal agent, compute the three focal-versus-opponent design differences `X[context, opp_seat] - X[context, focal_seat]` and their aliasing residuals with the stored `V_r` (D4 step 3). If any residual exceeds `tol_alias`, that agent's SDWP for this field carries the label **`prior_dependent_field`**, listing the failing contexts and top-loading parameters. This is a **reporting label only**: the number is still reported, the agent stays in the headline ordering if D4 allows, and the model is unchanged. A field where every context passes is labelled `field_identified`.

### 4.7 Estimands and reporting

| Output | Definition (plain text) | Kind |
|---|---|---|
| **SDWP (headline)** | `SDWP(a; F) = sum_c w_F(c) * p_focal(a, c; psi_hat)`: MAP plug-in, the focal agent's decisive-game win probability per context, averaged with the field's fixed (canonical) weights. Reported only for headline-eligible agents (D4) | model-based |
| SDWP, posterior mean (optional) | same, averaged over Laplace posterior draws; labelled `sdwp_posterior_mean` | model-based |
| Log-worth | `theta[a] - theta[anchor]` (Elo display = × 400/ln 10) | model-based |
| Log-worth contrast | `theta[a] - theta[b]` | model-based |
| Seat effect | `seat[i] - mean(seat)` | model-based |
| Deck nuisance | `beta[d] - mean_ref(beta)`, where the reference set is the rankable decks with split `train`; if that set is empty, the baseline is `unavailable` and no deck nuisance values are returned (never silently another split). Not a deck ranking | model-based, prior-dependent split |
| Outcome rates | empirical `draw`, `cap`, `failed` rates per agent, with counts | empirical |

**What SDWP is and isn't.** The field's weights are fixed; contexts are not reweighted by how often they produce decisive games. So SDWP is **not** the win rate among the decisive games a field would produce. Example: two equally weighted contexts with conditional win probabilities 40% and 10% and decisive rates 1% and 99% give SDWP = 25%, while the win rate among decisive games is about 10.3%. Read SDWP beside draw and cap rates.

**The field doesn't make the model matchup-aware.** For fixed parameters, an agent with higher log-worth has the higher predicted win probability in every context. Changing the field changes the size of the gaps, not their order. SDWP translates additive skill into win-probability units; matchup effects show up in field-specific observed results (§10.2).

**Log-worth contrasts** are contrasts in the additive model, not averages of deck-specific differences. Claims that A beats B use R2's design-based lift, reported separately.

**Draws.** SDWP ignores draws. R2's evaluation score `P(win) + 0.25 × P(draw or cap)` is a different quantity; under it, turning a loss into a draw improves the score.

**Progress plots** (e.g. SDWP against training step) are the caller's: outputs carry `agent_id` and provenance for joining. Recommended pattern: **one joint fit** over all checkpoints being plotted, evaluated against **one frozen field**. Don't splice SDWPs from separate refits as if only the policy changed; the priors, other agents and deck effects differ between fits.

---

## 5. Posterior API and uncertainty

### 5.1 Methods

```python
post.sdwp(field)                      # agent, sdwp (MAP plug-in), component, labels, field_hash
post.sdwp(field, method="laplace", n_draws=1000, seed=…)   # sdwp_posterior_mean + interval
post.log_worth(); post.contrast(a, b); post.contrast_vector(c)   # Laplace intervals, labelled
post.seats(); post.decks(); post.swap_effect()
post.counts(); post.components(); post.unrankable(); post.outcome_rates()
post.identification_report()
post.sample(n, seed)                  # psi = mode + solve(L', z), H = L L', z ~ N(0, I)
post.predict_proba(contexts)          # outcome-free PredictionInput records, Appendix C
post.provenance(); post.save(path); pr.Posterior.load(path)
```

Contrast and headline outputs use D5; predictions use FD6 labels. Intervals carry their method (`exact`, `laplace` or `cluster_bootstrap`) and applicable labels. `insufficient_clusters` and `unstable_bootstrap` may suppress bootstrap intervals as specified by B4–B5 without suppressing their point estimates.

### 5.2 Laplace intervals

`sd = sqrt(c' z)` with `H z = c`. **Model-based**: they assume games are independent given the parameters. Shared deals can move true uncertainty in either direction.

### 5.3 Cluster bootstrap (the one block-aware method)

```python
boot = pr.uncertainty.cluster_bootstrap(post, n=400, seed=…, strata=None)
boot.sdwp(field); boot.log_worth(); boot.contrast(a, b); boot.contrast_vector(c)
boot.seats(); boot.decks(); boot.swap_effect(); boot.diagnostics()       # per quantity: n_converged, n_zero_support, n_used
```

`n` must be a positive integer. `strata` must be `None` or the name of a column in the posterior's **retained cluster table** (`suite`, `engine_version`, or a named custom column supplied through `cluster_strata` to `pr.fit`); `Posterior.save/load` preserves that table. Anything else is rejected before resampling.

`cluster_strata` is `Mapping[str, Mapping[ClusterKey, str]] | None`: column name to nonempty stratum label per canonical C.2 key, including `("game", game_id)` for a singleton. Reserve built-in names `suite` and `engine_version`. Each custom column must cover every frame key; missing keys or keys absent from the deduplicated input raise `ValueError`. Ignore supplied labels for non-frame keys known to that input. Persist a column's entries in canonical key order with keys encoded as `["cluster", id]` or `["game", id]`. For a built-in column, require one value per frame cluster over its likelihood-used rows; report crossing keys when the column is selected.

**What it estimates.** The sampling variability of the **regularised MAP statistic**, assuming likelihood-bearing clusters are independent and exchangeable (within strata, if used), conditional on the decisive-only frame (I4). It does **not** establish identification (that is D4/FD6, computed once on the original fit, and their labels carry over to every interval), and its coverage is checked only in the predefined scenarios (§11.3), not guaranteed. A replicate can omit the rare crossover clusters that identify a contrast; that instability is measured by the SC3b scenario (§11.3), not recomputed per replicate.

**MUST (B1). Sampling frame.** The resampling units are the **likelihood-bearing clusters** of the original fit (clusters with ≥ 1 likelihood-used game; null `cluster_id` → singleton). Clusters with no likelihood-used games contribute nothing to the objective and are excluded from the frame. Clusters are never split.
- Unstratified: each replicate draws `C` clusters with replacement from the `C` frame clusters (multinomial weights summing to C).
- Stratified (`strata=` a game-level column): allowed only if every frame cluster lies within one stratum; otherwise raise naming the crossing clusters. Within each stratum h, draw exactly `C_h` clusters with replacement, where `C_h` is that stratum's original count of frame clusters.

Use `Generator(PCG64(SeedSequence(seed, spawn_key=(k,))))` for replicate k. In canonical frame order, `w = np.bincount(rng.integers(0, C, size=C), minlength=C)`. For stratification use the same generator once per stratum in sorted label order, replace C with the stratum size, and scatter counts to the canonical indices for that stratum.

**MUST (B2). Frozen replicate.** Each replicate draws multinomial cluster weights w_c, rebuilds `T_rep[j,i] = sum_c w_c * count(j, i, c)` from the outcome-and-cluster map, and refits with:
- the **original parameter index** (no parameters added or removed, even if a replicate contains no data for some agent or deck);
- the **original priors** (all scales unchanged);
- the original inclusion, spec and (for SDWP) the original materialised field contexts;
- a warm start from the original mode.

Rankability, anchor, component and headline-eligibility (D4) checks are applied **once, to the original fit**, never per replicate. A replicate with no data for an entity simply has a prior-only direction for it (Appendix A F6).

**MUST (B3). Statistic.** The bootstrapped statistic is the **MAP plug-in** value (SDWP, log-worth, contrast, seat effect). Output: the original fit's MAP plug-in value plus the 2.5%/97.5% percentiles of the replicate values. Posterior-averaged statistics are never bootstrapped in v1.

**MUST (B4). Original-fit gate.** First honor D1a/D5 and FD4: withheld quantities have no interval; return the exact interval for a supported zero contrast. For each remaining quantity require at least `min_clusters` likelihood-bearing clusters (default 50) overall and at least `min_entity_clusters` (default 10) for every dependency. A failure at this gate yields `insufficient_clusters`; do not run the B5 per-quantity checks. Dependencies are fixed from the original fit. An entity’s support is the set of B1 frame clusters in which it occupies at least one seat in at least one likelihood-used game. Replicate support is zero exactly when all weights for that set are zero:

- Linear contrasts: collect every agent and deck with a nonzero coefficient in the final, combined contrast vector. In particular a deck nuisance contrast includes the target and every non-cancelled deck in the reference mean. Bracket, seat and swap coefficients introduce no entity dependencies.
- SDWP: the focal agent and every distinct field opponent and field deck, including when the focal agent is the anchor.

**MUST (B5). Replicate gate.** Apply only to quantities that passed B4 and need a stochastic interval. A replicate fails if the solver raises `ConvergenceError`; unexpected errors propagate. `failed_fraction = n_failed / n`.

1. If `failed_fraction > 0.05`, suppress each such interval as `unstable_bootstrap`; do not compute zero-support counts.
2. Otherwise compute `n_converged = n - n_failed`. Count converged replicates where any B4 dependency has no resampled frame clusters. If `n_zero_support / n_converged > 0.05`, suppress as `insufficient_clusters`.
3. For each remaining interval use all converged replicates, including zero-support replicates, and add an `unstable_bootstrap` warning label if `failed_fraction > 0.01`.

The comparisons are strict; equality does not trigger a threshold. Diagnostics give `n_converged`, `n_zero_support` (`None` when not computed) and `n_used` (0 for exact, withheld or suppressed intervals). A zero contrast does not require successful resampling; `n_converged` is a run-level count, not a dependency of the exact result.

**MUST (B6). Percentiles.** Over the converged replicates, the interval endpoints are `numpy.quantile(values, [0.025, 0.975], method="linear")`. The replicate streams are defined in B1 and do not depend on which worker runs a replicate.

**Validation:** coverage is evaluated only in predefined simulation scenarios (§11). Real logs are used for stability and runtime only.

The cluster sandwich estimator is deferred (§19).

---

## 6. Comparison blocks (R2)

### 6.1 Definitions

- A **pod**: 4 distinct decks for seats 0–3, plus swap counts (all 0 in R2 v1; C2).
- A **deal**: a pod plus an opaque `deal_id` (e.g. the deal seed). The caller's engine turns it into a game given the agents per seat.
- A **block** plays one deal with a focal agent rotated through seats 0–3; in rotation f the focal agent sits in seat f.
- Comparisons are between two named agents, **A** (focal) and **B** (reference).

### 6.2 Block descriptors (caller input)

**MUST (C1).** Before ingesting a block's games, the caller declares a descriptor:

```
block_id, block_index, design ("duplicate" | "paired"), deal_id,
decks[4], deck_versions[4], swap_counts[4], opponents[3] (paired only),
policy_versions, engine_version, pod_distribution_id, pod_distribution_version, pod_distribution_hash
```

`policy_versions` is a mapping from every distinct agent ID used by the block (A, B and any opponents) to its exact version. `opponents` is null for duplicate and a clockwise triple for paired. `block_index` values are 0, 1, 2, … in draw order, assigned before play. The expected positions are duplicate: `(rotation, None)` for rotations 0–3; paired: `(rotation, role)` for those rotations and roles `a`, `b`. Declare contiguous indices beginning at 0. `block_id` and `deal_id` must each be unique in the comparison.

**MUST (C2). Identity binding.** A comparison is bound at construction to the agent and deck snapshots (hence exact `policy_version`s and `deck_version`s), one `engine_version`, and one `PodDistribution` (id, version, hash). Every ingested row must match these: `policy_versions`, non-null `deck_versions` (null is rejected in R2), `engine_version`, `cluster_id == block_id`, and the declared position. **R2 v1 accepts only `swap_counts == 0`**; swapped variants are deferred (§19). `deal_id` is a logical draw identity, unique within the comparison; the same ordered pod drawn twice is valid with distinct `deal_id`s. Descriptors are declared before play **as a caller attestation**: `podrank` rejects detectable conflicts but cannot verify timing or i.i.d. sampling.

**MUST (C3). PodDistribution.** A canonical JSON payload (hashed like FD3): one design, ID, version and a finite list of support points with positive weights. Coalesce on `(decks, opponents)` and order by that key before normalising. A support point contains an ordered list of four distinct deck IDs and, for paired, an ordered opponent triple; duplicate uses null opponents. Swaps are implicitly zero. Reject a descriptor outside this joint support. `pods.sample(n_blocks, seed, start_index=0)` returns immutable `PodDraw` records containing only `block_index`, `deal_id`, `decks`, `opponents` and the distribution identity (ID, version, hash). For absolute draw index k use `Generator(PCG64(SeedSequence(seed, spawn_key=(k,))))` and `rng.choice(n_support, size=None, p=weights_normalised)`. For sampling only, divide the canonical float64 weights by their float64 sum; leave the stored payload and hash unchanged. The helper’s `deal_id` is `f"{pod_distribution_hash[:12]}:{seed}:{k}"`. It is not game randomness. For each draw the caller builds a `BlockDescriptor`: supply `block_id` and the bound design/engine version, copy the draw fields, set all four swaps to zero and look up the deck and policy versions from the snapshots. `declare_blocks` validates all bindings before any games are ingested.

**PodDistribution persistence.** The three-key envelope is `payload`, `pod_distribution_hash`, `provenance`. The payload has exactly `design`, `pod_distribution_id`, `version`, `support`; each support item has exactly `decks`, `opponents`, `weight`. Use FD3's UTF-8 encoding and payload-only hash. Use FD3b's provenance shape (`recipe` and `deck_snapshot_id` are null unless supplied by a producer). Load checks schema, duplicate keys, canonical order/uniqueness, the FD3 weight check and hash without renormalisation; reject invalid files with `PersistenceError`. Normalisation for a NumPy draw does not change the stored payload.

**Pod sampling.** Anytime validity requires whole blocks drawn i.i.d. under A1, including the caller’s deal randomness. The sampler draws the pod and opponent support point only; it cannot attest to the caller’s randomness or pre-play declaration. Quota and low-discrepancy schedules are descriptive-only (§7.8).

### 6.3 Duplicate block (default; 4 games)

Opponents: 3 copies of B. `X_dup = mean_f(r_A[f]) - 0.25`, with r = 1 win, 0.25 draw or cap, 0 loss; range [−0.25, 0.75].

**Why 0.25 is exact:** every rotation's counterfactual is the same all-B game, whose four seat scores sum to exactly 1. So `X_dup` equals the 8-game paired score deal by deal under deterministic replay **provided the all-B game is the same game in every rotation** (the engine's randomness doesn't depend on which seat is focal or on the role), and in expectation under stochastic execution provided each rotation's all-B game has the same outcome distribution. Determinism of each position on its own is not enough. This holds **only** when all opponents are copies of B. Variance reduction is measured (§10.4), not guaranteed.

### 6.4 Paired block (8 games; mixed opponent fields)

Ordered opponent triple (o1, o2, o3) per block, placed clockwise from the focal seat: `seat (f+1) mod 4 = o1, (f+2) mod 4 = o2, (f+3) mod 4 = o3`. Each rotation is played with A focal (`role="a"`) and B focal (`role="b"`), identical otherwise. `X_pair = 0.25 * sum_f (r_A[f] - r_B[f])`, range [−1, 1].

### 6.5 R2 block validation

On ingest, each game must match a declared descriptor: same `block_id`, `deal_id`, decks and swap counts per seat; the focal agent is in seat f for rotation f; opponents are placed per the design; the position `(rotation, role)` is expected. Mismatches are rejected with reasons. **Accepting a report is separate from applying it:** a report for an already-resolved position is accepted and stored in the event log (for diagnostics and the replay check in E3/E7), but it never changes the resolved position or a consumed score (E4), and it is never exported for ratings (E8).

### 6.6 Design-based estimands

| Quantity | Definition |
|---|---|
| **Block score lift** μ | `E[X]` under the declared `PodDistribution` (and opponent field) |
| Evaluation score | `P(win) + 0.25 × P(draw or cap)` of the focal agent; turning losses into draws raises it |

Never mixed with rating outputs.

---

## 7. Anytime-valid sequential evidence (R2)

### 7.1 Assumptions

A1. Whole blocks are i.i.d. (§6.2), independently of other blocks and past outcomes (caller's declaration).
A2. A, B, opponents, pod distribution, design and config are fixed for the comparison.
A3. Blocks are consumed in `block_index` order; each bet uses only earlier blocks.
A4. Missing results are handled per §7.5, the declared replay mode is truthful, and at most one attempt per position is in flight. Report a failure before starting a retry with a fresh ID; never pick a result from parallel attempts based on arrival time or outcome.
A5. Outcomes depend only on the game's own specification and randomness, not run history.

Under A1–A5 the complete-data scores X_1, X_2, … are i.i.d. with mean μ.

### 7.2 Hypotheses and α

**MUST (E1).**
- `Superiority(margin=m)`: upper threshold `m_up = m`, lower threshold `m_low = 0`. ABOVE = evidence μ > m; BELOW = evidence μ < 0.
- `NonInferiority(margin=m)`, m > 0: `m_up = m_low = -m`. ABOVE = evidence μ > −m; BELOW = evidence μ < −m.
- `alpha=` sets `alpha_up = alpha_low = alpha`: each directional claim holds at its level. `alpha_up` and `alpha_low` can be set separately (overriding `alpha`).
- `cs_alpha` is the level of the two-sided confidence sequence (E6). Default: `min(alpha_up, alpha_low)`. It is a reporting level only and never affects decisions.
- `alpha_up`, `alpha_low` and `cs_alpha` must each lie strictly in (0, 1); otherwise `ValueError`. The report lists all three separately.
- Valid thresholds: `x_min < m_low ≤ m_up < x_max` strictly, for the design's range (duplicate [−0.25, 0.75], paired [−1, 1]). Anything else raises `ValueError`.

### 7.3 Fixed-mixture betting wealth

```
upper:  lam_bound = 0.5 / (m_up - x_min)
        K_up[t]   = mean_k prod_{s<=t} (1 + lam_k * (Xup_s - m_up))
lower:  lam_bound = 0.5 / (x_max - m_low)
        K_low[t]  = mean_k prod_{s<=t} (1 + lam_k * (m_low - Xlow_s))
grid:   lam_k = lam_bound * k / G,   k = 1..G,   G = 16,   weights 1/G
```

`Xup_s` and `Xlow_s` are the frozen directional scores (§7.5); with no failures they're equal. Each factor is ≥ 0.5. Log-wealth per component, updated with `log1p`.

**Why the guarantee holds with failures (proof sketch, documented in `docs/concepts/evidence.md`).** Let `X_s` be the **complete-data** score of block s (what it would have scored had every position succeeded). Under A1–A5 the `X_s` are i.i.d. with mean μ. Define the complete-data wealth `K*_up[t]` exactly as `K_up[t]` but with `X_s` in place of `Xup_s`. Under the upper null (μ ≤ m_up) each component of `K*_up` is a nonnegative supermartingale, so their fixed mixture is too, and Ville's inequality gives `P(sup_t K*_up[t] >= 1/alpha_up) <= alpha_up`. By construction (E4) `Xup_s <= X_s <= Xlow_s` for every block, whichever positions failed and why. Since `lam_k > 0` and every factor is ≥ 0.5 > 0, each upper factor computed from `Xup_s` is ≤ the factor computed from `X_s`, so `K_up[t] <= K*_up[t]` for every t and every k. An observed upper crossing therefore implies a complete-data crossing, and `P(ABOVE under the null) <= alpha_up`. The lower direction is symmetric with `Xlow_s >= X_s`. The argument holds for **any** failure mechanism, including outcome-dependent ones. **Not claimed:** that observed wealth is itself a supermartingale, or that the guarantee holds conditional on later passing a replay check.

### 7.4 Statuses

Evaluated after each consumed block, in this order:

```
1. up  = K_up  >= 1/alpha_up;   low = K_low >= 1/alpha_low     (>= : reaching the threshold counts)
   up and low      -> INCONCLUSIVE (conflict)     terminal
   up only         -> ABOVE                       terminal
   low only        -> BELOW                       terminal
2. failure rule (§7.5, F-rule) triggered -> INCONCLUSIVE (failures)   terminal
3. n == max_blocks -> BUDGET_EXHAUSTED            non-terminal; cmp.extend_budget(new_max) continues with the same wealth
4. otherwise       -> CONTINUE
```

Evidence thresholds take precedence over the failure rule: conservative scoring keeps ABOVE/BELOW valid even with failures. Starting over with fresh wealth for the same A and B is a new comparison whose claim stands alone.

**MUST (E7). Decision at stop vs evidence validity.** The comparison stores two separate fields:
- `decision_at_stop`: the status when the comparison first became terminal (ABOVE, BELOW, INCONCLUSIVE). It is **never rewritten**, and consumed scores are never rewritten.
- `evidence_valid` (bool, default true) with `invalidation_reasons`.

A verified `replay_violation` or `game_id_integrity_violation` sets `evidence_valid = false` whenever it arrives. If not terminal, set `decision_at_stop = INCONCLUSIVE` with that reason; otherwise preserve the historical decision. Append the violation kind and position to `invalidation_reasons`.

**Effective status.** `cmp.status()` returns the **effective** status: if `evidence_valid` is false it returns `INCONCLUSIVE_INVALID` (with `invalidation_reasons`), never ABOVE or BELOW. Otherwise it returns the live status (§7.4). The historical decision is read from `cmp.decision_at_stop`. `save`/`load` preserve both, the event log, the event cursor, the quarantine set and the export revision exactly.

### 7.5 Reports, retries, consumption and frozen evidence

**MUST (E0). Per-event transition.** Each event is one game row, one `mark_failed` call or one `mark_block_failed` call. `ingest(table)` is exactly equivalent to ingesting its rows one at a time in table order (F19). Every execution attempt, including a retry, must have a fresh `game_id`; a retransmission of that exact row retains its ID. A caller may use `"{spec_hash}:{attempt}"` only if that pair is unique across its game records (otherwise namespace by position or use another unique ID). For each event, in this order:
1. **Validate and classify** (C1–C3, §6.5): `reject` (malformed, unknown position, identity mismatch), `transport_duplicate` (same `game_id`, identical content: dropped silently and counted), `id_conflict` (same `game_id`, different content: rejected and audited; see step 4), or `accepted`.
2. **Record** accepted events in the event log with the next event number (E2).
3. **Resolve** an accepted position per E3, or leave it unresolved/provisional. Later reports never replace the resolved report.
4. **Check integrity before draining.** For a deterministic different-ID replay violation (E3), quarantine the position and invalidate evidence (E7). In either replay mode, two individually valid rows with the same `game_id`, same declared position and different `(outcome, winner_seat)` are a `game_id_integrity_violation`: reject and retain the conflicting row in the audit log (not the accepted event log), quarantine the position and invalidate evidence. All other ID conflicts are rejected/audited without changing resolution, quarantine or validity; transport duplicates do not change state beyond their counter.
5. **Drain:** while the comparison is non-terminal, fewer than `max_blocks` blocks are consumed, and the next block in `block_index` order has all positions resolved, consume it (E4) and evaluate §7.4.

Rules:
- `mark_block_failed(block_id)` is **one atomic event**: every unresolved or provisional position in the block resolves as failed; already-resolved positions are unchanged. It records the block's descriptor so failure records can be built (E8).
- **At budget** (`BUDGET_EXHAUSTED`), later events are still validated, recorded and resolved, but no further block is consumed and wealth doesn't change. `extend_budget(new_max)` requires `new_max > max_blocks`, is allowed only in non-terminal state, keeps the same wealth and immediately runs step 5. After draining, if still non-terminal, always recompute §7.4 rules 3–4 against the new maximum, even if no block was consumed. Do not change the export revision just because a budget changes.
- **After a terminal decision**, no further blocks are consumed. Events are still validated and recorded, and steps 3–4 still run, so late contradictions are detected (E7, F12).

**MUST (E2). Event sequence.** Every accepted event (a game row, a `mark_failed` call or a `mark_block_failed` call) gets a monotonically increasing event number when ingested. Within one `ingest` call, rows are numbered in table order. The event log is persisted.

**MUST (E3). Position resolution.** A position is `(block_id, rotation, role)`.
- `replay="stochastic"` (default): the first report or explicit failure mark resolves the position. Later valid reports with different IDs are diagnostic only, even when outcomes disagree. They never quarantine the position, change the export, or invalidate evidence. The same-ID rule in E0 still applies.
- `replay="deterministic"`: a `failed` row is **provisional**. The position resolves at the first successful report, or at an explicit `mark_failed` / `mark_block_failed`. A later success with a **different** outcome (`outcome`, `winner_seat`) from an earlier success for the same position is a `replay_violation`, handled by E7. It may arrive before or after the block is consumed, and before or after a terminal decision.

**MUST (E4). Consumption.** Block k is consumed when blocks 0..k−1 are consumed and all of block k's positions are resolved. At consumption, compute and **persist** its two directional scores:

| Unresolved-as-failed game | `Xup` (pessimistic for "better") | `Xlow` (pessimistic for "worse") |
|---|---|---|
| A's game | r_A = 0 | r_A = 1 |
| B's game (paired) | r_B = 1 | r_B = 0 |

After consumption these scores are frozen. Later reports for that block (e.g. a late recovery) are stored in the event log for diagnostics only, and are never exported for ratings (E8); the one exception is that a late report can reveal an integrity or replay violation, which changes evidence validity and can remove a row from the export (E7–E8), never the frozen scores.

**MUST (E5). Order invariance.** With fixed descriptors, the same per-position report/mark history (count an atomic block mark in the history of each affected position), globally consistent game IDs, no integrity or replay violation and the same budget extensions at the same consumed-prefix boundaries, inter-block arrival order does not change frozen scores, the terminal decision or final effective status. Event logs retain arrival order. For an E0/E3 contradiction with the same valid reports, evidence is invalid in either order; the terminal decision can differ if detection precedes the original stopping point (F12).

**MUST (E8). Rating export (a revisioned snapshot).** `cmp.rating_rows()` returns `(rows, export_report)` for **consumed blocks only**:
- **At most one row per position:** for each non-quarantined position resolved by a success, the successful report used in its consumed score. Quarantined positions are omitted.
- **Failed positions** export a `failed` row only if the failure event was a game row (which carries every `GameTable` field). Positions failed by `mark_failed` / `mark_block_failed` export no row; they appear in `cmp.failure_records()` (block_id, rotation, role, descriptor, event number). `podrank` never invents `played_at`, versions or outcomes.
- Late repeats, recovered provisional failures and every report of a quarantined position stay in the event log only.
- `export_report` contains `comparison_id`, `export_revision`, every omitted consumed position with its reason and `evidence_valid`. Start revision at 0. After an event or budget extension, advance it once **per newly consumed block** and once **per newly quarantined consumed position**, plus once for an `evidence_valid` change only if neither of those increments occurred. Do not increment for an ordinary repeat; quarantining a position that was never consumed does not by itself change the export. For a quarantined consumed position show the violation kind as the omission reason, regardless of how it was originally resolved.

The export is a **snapshot, not an append-only feed**: a later contradiction can remove a previously exported row. Callers must **replace** a comparison's earlier rows with the latest snapshot (keyed by `comparison_id`, `export_revision`), or export only after they have stopped sending reports. Never feed the raw event log to a rating fit (F16).

**F-rule (failures; a usefulness rule, not a validity requirement).** After at least `min_blocks_for_failure_rule` (default 50) consumed blocks, if the fraction of consumed blocks with any failed position exceeds `max_failed_fraction` (default 0.02), the comparison ends INCONCLUSIVE (failures).

Duplicate identical rows and conflicting rows for the same `game_id` are handled by E0 step 1 and step 4.

### 7.6 Mean-lift bounds

`cmp.report()` gives `[mean(Xup), mean(Xlow)]` over consumed blocks: bounds on the observed complete-data mean lift. With no failures they coincide and are reported as a single mean. An unqualified complete-case mean is never shown alongside the confidence sequence.

### 7.7 Confidence sequence (on demand)

**MUST (E6).** Grid points are built from integers: `m_k = x_min + k * 0.0025` for `k = 1 … K-1`, where `K = round((x_max - x_min) / 0.0025)` (400 for duplicate, 800 for paired); the endpoints are never included. For each m, replay the frozen scores:
- m is excluded **from below** if the upper test `H0: mu <= m` built from **Xup** reached `2/cs_alpha` at any time;
- m is excluded **from above** if the lower test `H0: mu >= m` built from **Xlow** reached `2/cs_alpha` at any time.

No confidence sequence is returned (`cs = None` with a reason) when zero blocks are consumed or `evidence_valid` is false. The descriptive API (§7.8) has no confidence-sequence field.

The reported interval is the **outward bracket**: `lo` = the largest grid point excluded from below (or x_min if none), `hi` = the smallest grid point excluded from above (or x_max if none).
- If `lo < hi`, report `[lo, hi]`, even when no grid point lies strictly between them (e.g. lo = 0.1000, hi = 0.1025; the continuous interval may still be nonempty).
- Only if `lo >= hi` (the bounds actually cross) report the bracket with the flag `cs_conflict = true`.
- `cs_conflict` is a **reporting flag only**; it never changes `decision_at_stop` or `evidence_valid`.
Never computed inside `ingest()`.

### 7.8 Descriptive summaries

Use `pr.compare.summarise_blocks(blocks, *, design) -> DescriptiveSummary` for logs without a declared i.i.d. schedule. The input is a finite sequence of already-resolved `ScoredBlockInput` records: a unique `block_id` and a mapping from each expected `(rotation, role)` to focal points `0`, `0.25`, `1`, or `None` for an explicitly failed position. Duplicate accepts role `None` only. Require exactly the expected keys and no duplicate block IDs; reject bad shapes or values with `ValueError`. The caller selects the resolved row per position before invoking this API; do not infer missing entries or resolve retries here. Reuse E4 for directional scores, then return `n_blocks`, `n_failed_blocks`, `failed_block_fraction` and `mean_lift_bounds`; the last two are `None` for empty input. No `Comparison` instance, wealth, F-rule, confidence sequence, bootstrap or status is created.

### 7.9 Sizing (rough; S6 measures it)

`E[N_blocks] ≈ 2 * var(X) * ln(1/alpha) / mu^2 × (1 to 2)`. With independent rotations, duplicate var(X) ≈ 0.047, so at α = 0.05: about 700–1,400 blocks for +2 pp, 110–230 for +5 pp.

### 7.10 Report contents (`cmp.report()`)

Effective status and `decision_at_stop`, with their meaning in words; `evidence_valid` and reasons; hypothesis, thresholds, `alpha_up`, `alpha_low` and `cs_alpha`, design, pod distribution id/version, replay mode; blocks consumed, waiting and failed; mean-lift bounds; confidence sequence; A's draw and cap rates (descriptive). For paired designs, the A − B draw+cap rate on matched rotations as a **descriptive** difference. In duplicate designs, B's rate on the same deals is unobserved, so no difference is reported.

---

## 8. Combining comparisons (caller's responsibility)

A single comparison makes one claim: if its null hypothesis is true and assumptions A1–A5 hold, the probability of reaching ABOVE is at most `alpha_up` (and BELOW at most `alpha_low`). Nothing more.

Running several comparisons (against different opponent fields or deck suites), repeating them for successive agents, or combining their statuses into a policy is the caller's job, including any error control across comparisons (§1.4). `podrank` reports each comparison's evidence independently and never aggregates statuses. Suite-difference reports, tag breakdowns and batch mode are not part of R2 (§19).

---

## 9. Input assumptions about execution (caller's responsibility)

1. **Declared replay mode:** `deterministic` (same game position, deal and per-seat seeds always give the same game; the current engine is believed to satisfy this) or `stochastic`. It affects only how retries resolve (E3).
2. **Seat order as given:** seat 0 moves first; the harness doesn't reshuffle seats within comparison blocks.
3. **Policy identity:** each seat's agent is exactly the declared `policy_version`.
4. **No history dependence** (A5).
5. **Failures reported, not hidden (A4):** only one attempt may be in flight per position; a retry starts only after the previous attempt was reported as failed, and gets a new `game_id`. Do not launch speculative parallel retries or silently re-seed. This is a caller obligation, not enforceable from arrival order alone. Document it in `docs/assumptions.md`.
6. **Blocks sampled as declared** (A1), with `block_index` assigned before play.

Documented in `docs/assumptions.md`. Optional `audit` metadata is stored and summarised.

---

## 10. Diagnostics

### 10.1 Predictive quality (post-R1)
Decisive-game log-loss on **whole held-out clusters** (baseline ln 4); Brier; reliability diagrams; baselines: uniform, seat-only, θ-only, θ + deck.

**Forward split (post-R1):** windows by `played_at` (or a caller-named agent metadata key). Rule for absent entities: games in window k involving agents or decks with no likelihood-used games in the training fold (earlier windows plus the burn-in) are **excluded from scoring and counted**. Burn-in: the first `n_burn` clusters (default 20) in window k that involve a new entity are added to the training fold.

### 10.2 Posterior predictive checks
Observed decisive win rates per agent, deck, seat, tag and opponent field vs MAP predictions; flag |z| > 3 using cluster-bootstrap SEs where eligible. Field-specific gaps are where the additive ordering breaks down.

### 10.3 Identification
`identification_report()`: components (D1), weak directions (D2), data shares (D3), confounding flags, saturation.

### 10.4 Pairing report (R2, from logs only)
For duplicate blocks: observed var(X) against the variance expected if rotations were independent with the same per-rotation outcome rates. With `r_f` the focal score in rotation f (1, 0.25 or 0), the independent-rotation baseline is `V_ind = sum_f Var(r_f) / 16`, where `Var(r_f) = E[r_f^2] - E[r_f]^2`, `E[r_f] = P(win) + 0.25·P(draw or cap)` and `E[r_f^2] = P(win) + P(draw or cap)/16`, all estimated per rotation from consumed blocks. Report `ratio = var(X) / V_ind` (below 1 = pairing helped). If `V_ind == 0` the ratio is reported as undefined. For paired blocks the matched draw/cap diagnostic uses only rotations where both the A and B positions succeeded; its denominator is that count.

### 10.5 Outcome rates
Per agent and suite: draw, cap and failed rates with counts.

---

## 11. Simulations and validation

### 11.1 Synthetic world (`pr.sim`)

True θ, seats, μ, δ, optional η; a draw/cap process; optional agent × deck and agent × opponent effects; drift; random and outcome-dependent failures; deterministic or stochastic execution. Winner = `argmax_i(u_i + Gumbel_i)` (exact Luce). Shared deals: Gumbel noise keyed by (cluster, seat), redrawn per game with probability ρ_div. `SimExecutor` (test infrastructure only) turns descriptors into game rows.

### 11.2 Reference worlds (the module's own concrete numbers)

The library itself is count-agnostic (§20), but its **simulations, examples, docs, tests and benchmarks** use a small set of named, frozen reference worlds, so every number in the repo comes from one place. They live in `podrank/sim/worlds/*.json` and load with `pr.sim.reference_world(name)`. **No example, statistical test, doc or benchmark may use ad-hoc sizes**; it names a world (optionally with a documented variant). Appendix A fixtures and small property-based tests are exempt.

**Shared conventions (all worlds unless stated):**

| Item | Value |
|---|---|
| Agent ids | `agent_00` … `agent_{A-1}`; `agent_00` is the anchor (`kind="heuristic"`) |
| True θ | `agent_00` = 0; the rest are `0.05 + ` the quantiles of N(0, 0.6²) at probabilities `(k - 0.5)/(A - 1)`, k = 1…A−1, assigned in increasing order. The +0.05 offset stops the middle agent tying the anchor. Deterministic, no seed. E.g. `doc_tiny`: −0.719, −0.265, 0.050, 0.365, 0.819 |
| Brackets | B brackets with IDs 0 through B−1, with equal numbers of decks; true μ evenly spaced on [−0.3, 0.3] |
| Deck ids | `deck_b{b}_{nn}`, e.g. `deck_b2_07` |
| True δ | within each bracket, quantiles of N(0, 0.5²) (matching σ_β = 0.5), assigned in deck order at probabilities `(k - 0.5)/n_b`, k = 1…n_b, for n_b decks in the bracket |
| True seats | `[0.15, 0.05, -0.05, -0.15]` (first-player advantage) |
| η | off |
| Outcomes | 5% draws, 1% caps, independent of skill; 0% failures |
| Cluster structure | one deal per cluster: 4 distinct decks from **one** bracket (bracket uniform), 4 distinct agents, sampled in draw order without replacement from sorted eligible IDs for that cluster; **4 games** with the agents rotated cyclically (game r: agent k in seat (k + r) mod 4), so each agent plays every seat and deck once |
| Shared luck | ρ_div = 0.3 |
| Reference field (`sim_field`) | 400 contexts from the FD2 recipe (a single frozen field shared by all agents; self-play against independent copies of the focal policy is intentional): same-bracket pods from the world's decks, opponent pool `agent_00`–`agent_02`, uniform focal seat |
| Bootstrap | n = 400 (n = 100 in CI smoke) |

**Worlds:**

| Name | Used by | A | B × decks/bracket (D) | Clusters × 4 games | Seed | Notes |
|---|---|---|---|---|---|---|
| `doc_tiny` | quickstart, concept pages, doctests | 6 | 3 × 4 (12) | 60 (240 games) | 101 | Quickstart-sized; use `doc_tiny_balanced` for the deterministic interval-emission fixture; `sim_field` with 24 contexts |
| `ci_smoke` | CI statistical smoke tests (SC1–SC8 at small size) | 10 | 3 × 8 (24) | 200 (800 games) | 202 | 50 replications per scenario in CI |
| `standard` | SC1–SC8 release gates, S2a–S4 | 20 | 4 × 12 (48) | 600 (2,400 games) | 303 | ≈ 120 clusters per agent, ≈ 50 per deck; 1,000 replications |
| `doc_progress` | `how-to/plot-progress.md` | 14 | 3 × 8 (24) | 480 (1,920 games) | 505 | Agents: `heur_a` (anchor, θ = 0), `heur_b` (θ = 0.2), `ckpt_00`…`ckpt_11` with θ = −0.8 + 0.15k (−0.8 to 0.85). Each checkpoint has 40 clusters against `heur_a`/`heur_b` (opponents drawn per seat) |
| `perf_large` | §12 benchmarks | 50 | 5 × 28 (140) | 250,000 (1,000,000 games) | 404 | ≈ 200 parameters; `perf_bootstrap` = its first 50,000 clusters (200,000 games) |

**Scenario variants of `standard` (§11.3):**

| Scenario | Change from `standard` |
|---|---|
| SC1 | none; true δ additionally redrawn from N(0, 0.5²) per replication |
| SC2 | ρ_div = 0 |
| SC3 | Reserve 21 of the world’s clusters for sparse agents (3, 6, 12); exclude those agents from ordinary draws. Each reserved cluster contains exactly one sparse agent and three non-sparse, non-pinned agents. Force rotation 0 to be decisive and sample its winner from the ordinary Luce model; sample the other rotations normally. Assert the exact likelihood-bearing counts. Generate `agent_16` on exclusive `deck_b1_00` using the pinned-deck generator. |
| SC3b | Generate and validate the SC7 base first. Append two likelihood-bearing clusters with the first pinned agent in a pod of four non-pinned decks from its pinned deck’s bracket, and three ordinary agents (include the anchor; exclude the second pinned agent). Rotate everyone through all seats and force rotation 0 decisive as in SC3. Keep the two pinned decks exclusive. Assert exactly two crossover clusters, that the first pinned agent’s anchor contrast passes D4 on the augmented design, and that the second remains aliased. The base-only null-space assertion does not apply to the first pinned agent after crossover. Report omission rate and interval widths, without reclassifying replicates. |
| SC4 | agent × deck effects N(0, 0.3²), agent × opponent effects N(0, 0.3²), kingmaker penalty 0.2 on the leading player's utility |
| SC5 | agents split into `agent_00`–`agent_09` and `agent_10`–`agent_19`; clusters never mix the groups |
| SC6 | (a) every game a draw; (b) `standard` with every `agent_19` game a draw |
| SC7 | `agent_18` plays only `deck_b1_00` and `agent_19` only `deck_b2_00`, each deck exclusive to its agent (pinned-deck generator, below) |
| SC8 | every game involving `agent_00` is a draw |

**Pinned-deck generator (SC3, SC7).** These scenarios need an agent whose skill is genuinely inseparable from a deck. The standard rotation generator would move every agent through all four decks of a cluster, adding exactly the crossover games that remove the aliasing. So for each pinned agent a with pinned deck d:
- **a plays only d, and d is played only by a** (the deck is exclusive to its agent). Pods that contain a always include d, drawn from d's bracket; pods without a never include d. At most one pinned agent per cluster.
- In each rotation, a sits in whichever seat holds d. Order the other agents and remaining seats initially, then cyclically shift those agents by `r % 3` in rotation r = 0, 1, 2, 3. In `ci_smoke` SC3, ordinary clusters use the six non-sparse, non-pinned agents; include the anchor in the first reserved cluster for each sparse or pinned agent. In all worlds check connectivity and the stated D4 condition before scoring simulation results.

**Before fitting, each replication asserts:** (a) every game of a used d, and every game with d had a in that seat; (b) each pinned agent appears in ≥ 30 likelihood-bearing clusters (reserve 30 distinct clusters per pinned agent, forcing rotation 0 decisive as in SC3; draw remaining ordinary clusters without pinned agents); (c) the agent graph is connected and contains the anchor; (d) the alias vector `v = e_theta[a] - e_delta[d]` lies in the likelihood's null space: `v' G v <= 1e-12 * ||G||` with G the D4 Gram matrix. A failed assertion is a **generator bug**, reported as such, never as a D4 result. Only then is D4 tested.

**Agent mapping for smaller worlds.** Variants are defined by **role**, mapped to agent ids per world:

| Role | `standard` (A = 20) | `ci_smoke` (A = 10) |
|---|---|---|
| SC3 sparse agents (3, 6, 12 clusters) | `agent_17`, `agent_18`, `agent_19` | `agent_07`, `agent_08`, `agent_09` |
| SC3 pinned agent / deck | `agent_16` / `deck_b1_00` | `agent_06` / `deck_b1_00` |
| SC5 groups | `agent_00`–`agent_09`, `agent_10`–`agent_19` | `agent_00`–`agent_04`, `agent_05`–`agent_09` |
| SC6(b) draw-only agent | `agent_19` | `agent_09` |
| SC7 pinned agents / decks | `agent_18` / `deck_b1_00`, `agent_19` / `deck_b2_00` | `agent_08` / `deck_b1_00`, `agent_09` / `deck_b2_00` |
| SC8 anchor | `agent_00` | `agent_00` |
| SC3b crossover agent | `agent_18` | `agent_08` |

**R2 worlds** (duplicate design unless stated; decks and brackets as `standard`; seed 707):

| Name | Used by | Setup |
|---|---|---|
| `doc_compare` | `how-to/run-a-comparison.md` | A vs B with true lift μ = 0.05; up to 400 blocks; α = 0.05 |
| `cmp_null` | S5 | Each directional test at its own null boundary: (i) Superiority(m = 0), μ = 0 (both directions); (ii) Superiority(m = 0.02), μ = 0.02 (upper direction); (iii) NonInferiority(m = 0.02), μ = −0.02 (both directions). θ_A calibrated numerically so the world's true lift equals the stated μ (stored in the world file). Duplicate and paired designs; `max_blocks` = 4,000; α = 0.05; 10,000 replications per condition; ρ_div ∈ {0, 0.3, 0.6, 1}; failures 0% and 2% (random and outcome-dependent) |
| `cmp_effect` | S6, S11 | target μ ∈ {0.005, 0.01, 0.02, 0.05}; θ_A calibrated numerically so the world's true lift equals the target (stored in the world file); 1,000 replications per cell; duplicate and paired (paired opponents drawn from `agent_00`–`agent_02`) |

**Pinned outputs.** Docs show real outputs from these worlds. They're generated at first release, pinned in the docs, and checked in CI to 1e-6 within the reference environment (FD5). Changing a world's numbers requires a new world name or version, never an in-place edit.

**Ordinary cluster composition.** Give cluster index `k` the composition stream `Generator(PCG64(SeedSequence(run_seed, spawn_key=(1,k))))`, where `run_seed` is the frozen world seed for a standalone world and the replication seed for a study. In that stream draw the eligible bracket index with `integers`, the four distinct deck indices with `choice(..., size=4, replace=False)`, then four distinct agent indices with the same `choice` form. Use sorted eligible pools and retain draw order. A scenario that prescribes a composition overrides this sequence for the affected clusters. Keep outcome randomness on a separate stream (not consumed from the composition generator).

### 11.3 Predefined R1 scenarios

Each scenario is frozen in `sim/scenarios/` with its seed, size, field and a **predeclared scoring list**. 1,000 replications each.

**Two kinds of check, with different consequences:**
- **Hard gates (CI blocks merge and release):** all Appendix A fixtures, derivative and invariance tests, and the **deterministic behaviour checks** in the table below: flags, labels, exceptions and classifications (SC3 flags, SC5 labels, SC6 exceptions, SC7 aliasing, SC8 `AnchorUnsupported`), plus the generator assertions.
- **Statistical targets (release review):** SC1/SC2 coverage, emission rate and bias. Run in full on `standard` before release. A miss **doesn't** fail CI; it requires an **explicit release review**, recorded as an ADR, that examines the cause (implementation bug, scenario too small, or a genuine limitation of the method). A miss must never be "fixed" by changing the model, priors or tolerances just to pass.

**Targets.** Log-worth contrasts at the generating θ, and **SDWP evaluated at the generating parameters** (`SDWP(a; F, psi_true)`) for the scenario's field. The MAP estimate is an estimator of these; with fixed priors it has real finite-sample shrinkage bias, which is not a bug.

**Predeclared scoring list per scenario:** the contrasts scored (SC1/SC2: the top-vs-2nd, top-vs-median and median-vs-bottom agents by true θ, plus every agent vs the anchor) and the SDWP of every agent in the field.

**Coverage is conditional on an interval being emitted.** Each scenario also reports the **emission rate** (intervals emitted / quantities scored); suppressed intervals don't count as misses or hits.

**Bias** is reported **per estimand** (never pooled across estimands, so signs can't cancel), with a practical tolerance rather than a Monte Carlo test: `|mean bias|` ≤ 0.05 in log-worth units and ≤ 1.0 percentage point in SDWP for each scored quantity. Coverage is reported pooled **and** per quantity; the pooled range is the target, and any single quantity below 90% is called out in the release review.

| Scenario | Setup | Check |
|---|---|---|
| **SC1 well-supported** | `standard` world (§11.2): A = 20, D = 48, 600 clusters of 4 games, ρ_div = 0.3, correctly specified, δ redrawn from the prior per replication | **Statistical targets:** pooled cluster-bootstrap coverage over the scoring list in [93%, 97%]; emission rate ≥ 99%; bias within the practical tolerance; per-quantity coverage reported |
| **SC2 strong shared deals** | as SC1, ρ_div = 0 | Same statistical targets |
| **SC3 sparse support** | `standard` variant SC3 (§11.2) | `insufficient_clusters` and confounding flags fire as specified; coverage **reported only** |
| **SC3b rare crossover** | `standard` variant SC3b (§11.2) | Report only: replicate omission rate of the crossover clusters, interval widths |
| **SC4 misspecified** | `standard` variant SC4 (§11.2) | Report only; documents failure modes; SDWP ordering field-invariant while observed field results differ (§4.7) |
| **SC5 disconnected** | `standard` variant SC5 (§11.2) | D1 labels; disconnected agents absent from headline ordering |
| **SC6 all-draw / draw-only agent** | `standard` variant SC6 (§11.2) | `NoRankableEvidence`; draw-only agent listed unrankable |
| **SC7 connected but confounded** | `standard` variant SC7 (§11.2) | D4 classifies both `aliased` in **every** replication that passes the generator assertions; no headline SDWP for them; other agents' gates as SC1 |
| **SC8 unsupported anchor** | `standard` variant SC8 (§11.2) | `AnchorUnsupported` returned; no silent replacement |

### 11.4 Post-R1 and R2 studies

| ID | Study | Acceptance |
|---|---|---|
| S2a | Laplace vs NUTS (extra `[reference]`), SBC on independent games | sd ratio in [0.9, 1.1]; SBC uniform |
| S3 | Whole-cluster CV and forward split (§10.1) | Report; full model not worse than best baseline by > 2 SE in correctly specified worlds |
| S4 | σ_β selection (evidence, cluster CV) | Within 2× of truth |
| S5 | **R2 implementation check** (`cmp_null`): every boundary cell, both designs; outcome-dependent failures; stochastic and deterministic retries; late recoveries after consumption; block arrival order permutations; resume | False claims reported **by direction** against that direction's α: false ABOVE ≤ `alpha_up` + 2·MC SE and false BELOW ≤ `alpha_low` + 2·MC SE (10k reps), in every cell; E5 holds exactly |
| S6 | R2 power and cost (`cmp_effect`) vs effect size × ρ_div × duplicate vs paired | Report |
| S10 | Scalability (`perf_large`, `perf_bootstrap`) | Meets §12 |
| S11 | Fixed-horizon lift recovery under SC4-style agent × deck effects, using the `cmp_effect` design and frozen world cells. | Compare the mean of independent complete-data replication means with the independently calculated truth; flag absolute bias greater than twice the Monte Carlo SE across replication means. Report stopped means and missing-result bounds separately, not as unbiased estimates. |

---

**S11 protocol.** Use 1,000 replications of 4,000 complete blocks per frozen cell, with no stopping, retry or failure in the lift-recovery statistic. Freeze one interaction matrix per cell. Compute its target independently: enumerate the finite pod/opponent distribution and rotations; integrate each role’s marginal expected points using the true Luce probabilities and the specified draw/cap process. For duplicate subtract 0.25 after averaging A’s four rotations; for paired subtract B’s matching average. Use the resulting interaction-world target, not the unperturbed `cmp_effect` target. Dependence within a block affects variance but not this marginal target. Run threshold stopping and failure variants separately and report their consumed-prefix means and bounds as operating characteristics.

## 12. Performance targets (16-core desktop CPU; data from `perf_large` / `perf_bootstrap`, §11.2)

| Operation | Size | Target |
|---|---|---|
| Fit, cold | 1M games, ~200 params | < 30 s |
| Fit, warm | +50k games | < 5 s |
| 1,000 contrasts | ~200 params | < 20 ms |
| Cluster bootstrap | 400 refits, 200k games | < 5 min on 16 processes |
| SDWP | 10k contexts | < 0.5 s |
| Sequential update | G = 16 | < 20 µs |
| `ingest()` overhead | 32 blocks | < 5 ms (excluding disk) |

Reduced parameterisation; CSR design reused across iterations; pattern compression (benchmark the gain); no Python loops over games; `pytest-benchmark` with a 25% regression gate.

**What each timing includes.** "Fit, cold" = design build + compression + Newton + Cholesky, timed separately from parsing/validation and from identification (D2/D4). "Fit, warm" = a full refit on the enlarged data warm-started by parameter name, **including** recompression (there is no incremental ingestion). "SDWP" is timed cold (including FD6 aliasing checks, done with batched projections onto the cached `V_r`, never one projection per context) and cached separately. Also benchmark a low-compression variant of `perf_large` (few repeated patterns): compression is an optimisation, not an assumption. Bootstrap workers share read-only arrays and set BLAS threads to 1 per worker; report Linux (fork) and Windows (spawn) separately. Every benchmark report records CPU, RAM, OS, NumPy/SciPy/BLAS versions, process count, thread settings and peak RSS.

**Status of the targets.** They are **release-review targets**: a miss requires a recorded review, like §11.3's statistical targets. A **performance spike in P2** measures cold fit, identification, SDWP and one bootstrap replicate on `perf_large` before the R1 API freezes.

---

## 13. Packaging

- `pyproject.toml` (hatchling), `requires-python = ">=3.12"`, `uv` lockfile.
- **Core:** `numpy>=2.0`, `scipy>=1.13`.
- **Extras:** `io` (pyarrow), `polars`, `plot` (matplotlib), `reference` (numpyro, jax), `speed` (numba), `cli` (typer, rich), `all`.
- **Dev:** pytest, pytest-xdist, pytest-benchmark, hypothesis, ruff, mypy `--strict`, coverage ≥ 90% on core, mkdocs-material, mkdocstrings, pre-commit.
- CI: Python 3.12, 3.13 on Linux; Windows smoke job. `py.typed`; explicit `__all__`; semantic versioning; `CHANGELOG.md`.

```
podrank/
  __init__.py  _version.py  constants.py  types.py  errors.py
  data/        registry.py  table.py  validate.py  inclusion.py
  io/          parquet.py                                  # [io]
  rating/      spec.py  design.py  compress.py  likelihood.py  solve.py  fit.py
               posterior.py  field.py  components.py  identification.py  predict.py
  uncertainty/ bootstrap.py
  compare/     pods.py  descriptors.py  blocks.py  events.py  betting.py
               comparison.py  cs.py  reports.py  pairing.py  summary.py
  diagnostics/ calibration.py  ppc.py  outcomes.py  forward.py
  sim/         world.py  executor.py  scenarios/  studies/  report.py
  reference/   numpyro_model.py  compare.py                # [reference]
  report/      tables.py  plots.py  markdown.py
  cli.py
tests/  benchmarks/  docs/  examples/
```

---

## 14. Testing

**R1 (release-blocking)**
- **Appendix A fixtures:** F1, F20, F5 (compressed likelihood, gradient, Hessian vs expanded rows; the double-weighting form must fail), F6 (frozen bootstrap index), F7 (I0–I3: mixed pods, manifests, singletons, counts), F8 (SDWP), F8b (field canonicalisation and hash), F10 (gauge vs objective), F11 (connected-but-confounded), F11b (unsupported anchor), F14 (wide-design aliasing), F15 (field identification), F17 (D5 output statuses), F18 (bootstrap suppression precedence).
- **Derivatives:** analytic gradient and Hessian of `objective` (the function the solver minimises) vs finite differences of `objective`, incl. the reduced-parameter chain rule; relative error < 1e-6.
- **Invariances:** gauge shifts leave all probabilities and `likelihood_nll` unchanged, and change `objective` by exactly the change in `prior_penalty` (F10); consistent seat relabelling; agent/deck relabelling permutes the solution; compressed = uncompressed fit (1e-10); every per-cluster score from the map equals the uncompressed per-cluster score.
- **Data:** F7/F22, C.1–C.2 configuration and canonical data, and I0 pipeline enforced by `pr.fit` without a prior `pr.validate` (invalid rows raise in `fit`), boundary checks (lengths, enums, negative swaps, `winner_seat`), opt-in `split_policy`, I1–I5 behaviours, counts reconcile (`received - exact_duplicates = eligible + sum(excluded)`) and match hand counts, swap policy FD4a, `policy_versions` mismatch errors, snapshot ids in provenance, `cluster_id` null → singleton, no block-layout checks in R1.
- **Solver:** F20b, including rounded objective, actual compressed Luce change and the step-50 convergence boundary.
- **Identification:** D1 component labels and headline omission; D1a `AnchorUnsupported`; D2 gauge directions absent from weak-direction output (complement-basis construction); D3 data share, not computed for c = 0; D4 classification (F11: aliased before crossover games, identified after; F14: wide design, where a thin-SVD implementation must fail the test), tolerances documented; D5 contrasts and separate headline eligibility (F17), including diagnostic `prior_dependent_value` only with `include_withheld=True`; deck baseline `unavailable` with no rankable train decks; FD6 labels (F15).
- **Fields:** F23 seeded and byte-format checks; FD1–FD6, FD3a–FD3c (duplicate JSON keys rejected; context-version fingerprint changes when a referenced version changes), recipe enum and size validation, FD3a and FD3b (payload-only hash, provenance preserved, tampered payload rejected), including F8c (save/load round-trip with raw weights `[1, 6, 15]` preserves canonical bytes, hash and SDWP; a loader that renormalises must fail), `NoEligiblePodsError` and the `max(4, min_decks_per_bracket)` rule, F8b (reordered duplicates give the same hash and SDWP; a split whose merged weight isn't bit-identical is documented, not asserted), `UnsupportedFieldError`, seeded Monte Carlo bit-identical within one environment.
- **Bootstrap:** F23 frame, stream and custom strata checks; B1–B6, including F18 precedence and strict-threshold boundaries, the all-failed case, `n <= 0` and unsupported strata rejected, strata surviving save/load, the likelihood-bearing frame, per-stratum counts, stratification refusal, unchanged parameter index and priors, eligibility suppression, failure and zero-support denominators, percentile method, and identical results regardless of how replicates are distributed over processes.
- **Posterior archive:** schema-2 round-trip; metadata/report parity; seeded bootstrap frame parity without the original `GameTable`; crossing stratum keys stored as canonical frame indices and resolved back to tagged keys; artifact size/timing recorded; reject duplicate/extra/missing ZIP members, wrong schema, bad descriptors, dtype/shape mismatches, member-count/total-size/oversized-member violations, and cross-array index errors.
- **Import isolation:** validate → fit → bootstrap → SDWP never imports `podrank.compare`.
- **End to end (`doc_tiny_balanced`, a frozen `doc_tiny` fixture variant):** records → fit → eligibility → SDWP → bootstrap (100 replicates, seed 101), without importing comparison. Keep the six agents, three brackets, four decks per bracket and 60 clusters. For cluster k = 0…59, use bracket `k % 3` (zero-based sorted brackets); circularly shift its four sorted decks by `(k // 3) % 4`, and use agents `(k + j) % 6` for j = 0…3. Rotate agents as in `doc_tiny`; force all 240 games decisive and sample winners with the world's seeded Luce generator. Use 24 FD2 contexts from these decks with zero swaps, `min_decks_per_bracket=4`, and world seed 101. Assert 240 likelihood rows, 60 likelihood-bearing clusters, all six eligible, anchor `contrast_status=zero`, six numeric `field_identified` SDWP rows in the prescribed order, and six emitted non-exact bootstrap intervals. Check MAP SDWP against explicit context-by-context softmax, and exact round-trip ordering and labels. Do not require a fixed `weak` versus `identified` split for nonanchors. Ordinary `doc_tiny` examples keep the default world generator.
- **Statistical:** SC1–SC8 on `ci_smoke` in CI (`@pytest.mark.stat`, 50 replications); on `standard` with 1,000 replications before release. Hard gates and statistical targets as defined in §11.3: deterministic checks block merge and release; SC1/SC2 statistical misses require a recorded release review.

**R2 (release-blocking for R2)**
- Fixtures F2, F3, F4, F9, F12, F13, F16, F19, F21.
- F9/F16/F19/F21/F23, including retry identity, per-row events, an empty-buffer budget extension, and PodDistribution persistence.
- C2 identity binding (changed policy version, null or changed deck version, mixed engine versions, `cluster_id != block_id`, nonzero swaps rejected); C3 support membership, repeated pods with distinct `deal_id`s accepted, `start_index` continuation reproduces one stream; C1 descriptor validation and detectable-conflict rejection; E1 threshold validation; E2–E5 including the order-invariance property (hypothesis-based tests permuting inter-block arrival order with fixed per-position histories, **on streams without replay violations**); a separate test that a replay violation sets `evidence_valid = false` in every arrival order; F-rule precedence; E0 transitions (F19: batch vs single-row equivalence, atomic block failure, buffered suffix at budget then `extend_budget`, events after a terminal decision); E6 confidence-sequence outward bracket, `cs_conflict` only when bounds cross, `2/cs_alpha` with asymmetric alphas, integer grid without endpoints, no CS for zero blocks or invalidated evidence; descriptive summary has no CS field (F13), checked against an independent reference implementation; E7 decision/validity separation and effective status `INCONCLUSIVE_INVALID`; E8 revisioned export (at most one row per non-quarantined position; failure records without rows; revision increments on a late contradiction; F16); terminal states frozen; resume reproduces state and event log exactly; descriptive-only summaries for undeclared blocks.
- **Scope guard:** split each public snake_case name on `_`; reject whole tokens `promote`, `promotion`, `incumbent`, `candidate`, `gate`, `attempt` and `checkpoint` (the last is allowed only as a `kind` value, not as a public name). Do not reject `aggregate`, `propagate` or `delegate`.

---

## 15. Phased delivery

| Phase | Deliverables | Acceptance |
|---|---|---|
| **P0 Scaffold** | repo, CI, lint/type/test/docs skeleton, ADR template | CI green; `mypy --strict` clean |
| **P1 Data** | snapshots, `GameTable`, R1 validation, I0 pipeline, inclusion (I1–I5) | Data tests; F7/F22 pipeline cases |
| **P2 Rating core** | design, compression (L1–L2), `likelihood_nll` / `prior_penalty` / `objective`, Newton, Cholesky, posterior, components (D1, D1a), data share (D3), headline eligibility (D4–D5), fields (FD1–FD6), SDWP, prediction, Appendix C numerical and save/load contracts | F1, F5, F8, F8b, F8c, F10, F11, F11b, F14, F15, F17, F20, F20b; F7/F22 fit cases; F23 field cases; derivative, invariance, compression tests; performance spike (§12) recorded |
| **P3 Identification + bootstrap** | D2, cluster bootstrap (B1–B6), simulator, scenarios SC1–SC8 | F6, F18, F23 bootstrap cases; B-rule tests; `doc_tiny` vertical-slice assertions; SC3–SC8 deterministic checks pass; SC3b reported; SC1/SC2 statistical targets met or a release review recorded |
| **R1 release** | P0–P3 | All R1 tests pass; import isolation; rating docs complete |
| **P4 Post-R1 rating extras** | calibration, PPC, forward split, σ_β tuning, NumPyro reference, reports/plots, CLI | S2a, S3, S4 reports |
| **P5 Comparison core** | descriptors and identity binding (C1–C3), block validation and scoring, state machine (E0), event log (E2–E5), revisioned export (E8), confidence sequence (E6), fixed-mixture evidence, statuses, reports, resume, `pods.sample`, descriptive summary, pairing report | F2, F3, F4, F9, F12, F13, F16, F19, F21, F23 distribution cases; C2–C3, E0, E6, E7 and E8 tests (§14); S5 (all boundary cells), S11 |
| **R2 release** | P5 | All R2 tests incl. scope guard; S6 report; validated on supplied real logs |
| **P6 Hardening** | performance targets (S10), docs complete | Targets met; docs build cleanly |

R1 does not wait for anything after P3. After R1, P4 is independent of P5 and does not gate R2. **Start P0/P1 now**; they don't depend on any open modelling question.

**Process during implementation.** Issues are found as failing fixtures or tests and resolved by ADRs, not by editing contracts silently. Any change to a contract updates, in the same change, its fixture in Appendix A, the §14 test list and the §15 acceptance column.

**Test tolerances.** Compare counts, statuses, labels, hashes and canonical payloads exactly. Use unrounded formula-based expectations with `rtol=1e-10, atol=1e-12` for portable numeric fixtures unless a test specifies otherwise. Printed rounded values in Appendix A are explanatory, not high-precision assertions. Bit-identical numeric expectations apply only within a pinned environment (FD5).

---

## 16. Documentation

```
docs/
  index.md               # the two questions; quickstart (in-memory + [io])
  concepts/
    ratings.md           # Luce model, log-worth, SDWP, what fields can and can't do, draws
    identification.md    # components, weak directions, prior-dependent outputs
    uncertainty.md       # Laplace vs cluster bootstrap; B-rules; eligibility
    blocks.md            # descriptors, duplicate and paired blocks, the 0.25 baseline
    evidence.md          # betting, A1–A5, retries and frozen evidence, statuses, CS
  assumptions.md
  how-to/ fit-from-logs.md  define-a-field.md  run-a-comparison.md  plot-progress.md
  examples/rl-pipeline.md # illustrative only; not shipped code
  reference/  simulations/  glossary.md  limitations.md  decisions/
```

Every concept page: a two-sentence summary, the math in LaTeX and plain text, a runnable example, and links to the MUST rules and fixtures behind each claim.

**Examples use reference worlds only (§11.2):** quickstart and concept pages use `doc_tiny`; `plot-progress.md` uses `doc_progress`; `run-a-comparison.md` uses `doc_compare`; `fit-from-logs.md` and `define-a-field.md` use `doc_tiny` exported to records/Parquet. Every example states the world name and shows its pinned output.

---

## 17. Module name

**`podrank`**: "pod" is the Commander word for a table. A web search (26 Nov 2026) found no PyPI package of that name. Confirm at `https://pypi.org/project/podrank/` and reserve with a `0.0.1` placeholder. Alternatives: `tablerank`, `lucepod`, `seatwise`.

---

## 18. Risks and open questions

| Risk | Mitigation |
|---|---|
| Compressed targets double-weighted | L1 + F5 |
| Bootstrap replicates change the parameter space | B2 + F6 |
| Filters silently create partial blocks | I1–I2 + F7 |
| Field results not reproducible | FD1–FD5 + F8, F8b |
| Prior-only rankings presented as results | I5, D1, D1a, D3, D4 (row-space aliasing check), B4 |
| Prior-dependent SDWP levels presented as data-determined | FD6 + F15 |
| Shrinkage bias mistaken for a bug | Targets at generating parameters; practical bias tolerance; fixtures as hard gates (§11.2) |
| SDWP read as the win rate among decisive games, or as matchup-aware | §4.7; SC4 |
| Intervals miscalibrated under sparse support or misspecification | Predefined scenarios; eligibility rules; labels |
| Two implementations consume different evidence or intervals | E2–E5 + F9; B1, B5, B6 |
| A late contradiction rewrites history | E7 + F12 |
| An invalidated decision read as valid | Effective status (E7) |
| Two implementations fit different rows | I0–I3 + F7 |
| Withheld or prior-only numbers shown as results | D5 + F17 |
| Rating export double-counts or keeps contradicted rows | E8 revisioned snapshot + F16 |
| Memory blow-up at large P | `max_dense_bytes` (§4.4); P2 performance spike |
| Retries select favourable outcomes | E3 |
| Many comparisons inflate false findings | Each claim stands alone; combining is the caller's |
| Scope creep | §0 scope test; §1.4; CI scope guard |

Open questions for the project owner (R2 only; R1 has none):
1. The first pod distribution(s) for comparisons.
2. Typical comparison settings (margins, `max_blocks`), so S6 reports the relevant table.

Project-specific choices (reference field, deck tags, deck splits) are recorded in `PROJECT_DECISIONS.md`, outside the library spec.

---

## 19. Deferred

| Item | Condition to revisit |
|---|---|
| `draws="split"` | Only as an explicitly non-headline sensitivity analysis, with a defined decisive-support requirement |
| Unseen-deck prediction (prior-predictive path for decks without games) | Specified and labelled as prior-predictive |
| σ_β tuning, NumPyro reference, CLI/plots, forward evaluation | P4 (post-R1) |
| Cluster sandwich variance | After its centring under shrinkage and stratification is specified and validated |
| Swapped deck variants in R2 comparisons | An exact swapped-list identity (hash) contract is specified |
| Batch mode, suite differences, tag breakdowns (R2) | After R2 ships, with batch identifiers, completion rules and budget units defined |
| Inference for unknown schedules | Not planned; descriptive summaries only |
| Generative draw/cap model | If the win rate among decisive games is needed |
| Deck embeddings in the deck prior (Appendix B) | Real-data gain shown; PCA fitted within folds |
| Cyclic term, active pod selection, online monitor, adaptive betting | Concrete need shown |
| Agent × deck models; deck strength ranking | Not planned (§1.3) |

---

## 20. Generality: no project-specific numbers in the library

`podrank` makes **no assumptions** about how many agents, decks, brackets, deck splits or held-out decks a project uses, or how they were chosen. Any of these may be zero, one or thousands, subject only to the support and identification rules (I5, D1–D4, FD4, FD6) and the performance limit `max_params`.

- `split` is a label on each deck (`train`, `validation`, `holdout`). Any number of decks may carry each label, including none. `podrank` doesn't create, size, balance or seal splits; it only records the label, filters on it (I1) and reports by it.
- Fields, pod recipes and tag schemas are **caller data**. `podrank` provides the mechanisms (recipe materialisation, canonicalisation, optional tag validation) with documented parameter defaults, not project values.
- Choosing reference opponents, deck sizes per split, bracket quotas, leakage rules between splits and when to build a holdout are project decisions. They live in the caller's repository.

This project's choices are recorded separately in **`PROJECT_DECISIONS.md`** (not part of the library specification). Changing them never requires a change to `podrank`.

## Appendix A — Hand-computable fixtures

**F1. One decisive game.** `u = [0.5, 0, 0, -0.5]`, seat 0 wins.
```
p = [0.387456, 0.235004, 0.235004, 0.142537];  log p[0] = -0.948154
grad_u = p - y = [-0.612544, 0.235004, 0.235004, 0.142537]
Hess_u row 0 = [0.237334, -0.091054, -0.091054, -0.055227]
```

**F2 (R2). Duplicate equals paired.** `r_A = [1, 0, 0.25, 0]`; all-B game won by seat 2 → `r_B = [0, 0, 1, 0]`.
```
X_dup = mean(r_A) - 0.25 = 0.0625;   X_pair = 0.25 * (sum(r_A) - sum(r_B)) = 0.0625
```

**F3 (R2). Fixed-mixture update** (illustrative G = 4). Duplicate, upper test, m_up = 0 → `lam_bound = 2.0`, `lam = [0.5, 1, 1.5, 2]`.
```
X = 0.5   -> [1.25, 1.5, 1.75, 2.0]              mixture 1.625
X = -0.25 -> [1.09375, 1.125, 1.09375, 1.0]      mixture 1.078125
X = 0.25  -> [1.230469, 1.40625, 1.503906, 1.5]  mixture 1.410156
```

**F4 (R2). Directional scores.** Duplicate block, rotation 1 resolved as failed, others `[1, ·, 0.25, 0]`.
```
Xup  (r_A[1] = 0) = 1.25/4 - 0.25 = 0.0625
Xlow (r_A[1] = 1) = 2.25/4 - 0.25 = 0.3125
mean-lift bounds for this block: [0.0625, 0.3125]
```

**F5. Compressed likelihood (L1).** One pattern seen twice; at ψ = 0, `p = [0.25, 0.25, 0.25, 0.25]`. Seat 0 wins in cluster b1, seat 1 in cluster b2.
```
T = [1, 1, 0, 0],  c = 2
likelihood_nll (pattern) = -(log 0.25 + log 0.25)  = 2.772589
grad_u         = c*p - T = [-0.5, -0.5, 0.5, 0.5]   (equals the sum over the two expanded rows)
Hess_u         = 2*(diag(p) - p p')  -> diagonal 0.375, off-diagonal -0.125
WRONG (must fail): c*(p - T) = [-1.5, -1.5, 0.5, 0.5]
per-cluster grad_u: b1 = [-0.75, 0.25, 0.25, 0.25],  b2 = [0.25, -0.75, 0.25, 0.25]
bootstrap weights (b1, b2) = (2, 0)  ->  T_rep = [2, 0, 0, 0]
```

**F6. Frozen bootstrap index (B2).** Agent X appears only in cluster c3. A replicate with weights (c1, c2, c3) = (2, 1, 0):
```
parameter index unchanged (theta[X] still present); no error raised
theta[X] has no likelihood terms -> its MAP is exactly the prior mean: theta_rep[X] = 0
the replicate's log-worth of X is therefore -theta_rep[anchor] (prior-only direction)
this replicate counts toward X's zero-cluster fraction (B5)
```

**F7. Cluster filtering (I1–I3).** Cluster b1 has games g1–g4 with `played_at` t1 < t2 < t3 < t4; g2 is a draw. Filter `played_at <= t3`:
```
cluster b1 kept with rows {g1, g2, g3}; reported as partially filtered (lost 1 row)
received 4, eligible 3, likelihood-used 2 (g1, g3), likelihood-bearing clusters 1
clusters="complete_only" without a manifest -> InclusionError
with manifest {b1: {g1, g2, g3, g4}} -> b1 dropped (g4 filtered out; reason "filtered")
```
Extensions (decks d1–d4 split `train`, d5 `holdout`; agents H anchor, A, B, C):
```
g5: decks [d1, d2, d3, d5], agents [A, B, B, C], win
    split filter {train}  -> g5 excluded (reason "split"; not all four decks are train)
    agent filter {A}      -> g5 kept (A is in at least one seat)
    per-entity counts for B: games 1, seat appearances 2
g6: agents [B, B, C, C], win; agent filter {A} -> g6 excluded (reason "agent")
complete_only, manifest {b1: {g1, g2, g3, g4}}:
    extra row g9 with cluster_id b1          -> ManifestMismatchError (g9 not listed)
    a game_id listed in two clusters          -> ManifestMismatchError
  a listed game with null/wrong cluster_id  -> ManifestMismatchError before I1, even if filtered
  row g10 with cluster_id b7 (not in manifest) -> ManifestMismatchError
    row g8 with cluster_id null, outcome win -> kept as a singleton cluster
    manifest lists g4 but g4 is absent        -> b1 dropped (reason "missing")
    g3 replaced by a failed row               -> b1 dropped (reason "failed")
duplicates (base case g1–g4, played_at filter): g1 supplied twice, identical -> received 5, exact_duplicates 1, eligible 3
P1: run_pipeline on a row with three agents -> raises at I0 step 1.
P2: pr.fit on the same row, without a prior pr.validate -> same error.
reconciliation: received - exact_duplicates == eligible + sum(excluded)
```

**F8. SDWP (FD1).** θ_A = 0.5, θ_B = 0, all deck effects 0, seats `[0.2, 0, 0, -0.2]`. Two contexts, weight 0.5 each: A in seat 0 vs 3 × B; A in seat 3 vs 3 × B.
```
p_focal(seat 0) = 0.416712;  p_focal(seat 3) = 0.295292
SDWP = 0.5*0.416712 + 0.5*0.295292 = 0.356002
```

**F8b. Canonical field (FD3).** Contexts: `ctxA` = focal seat 0, decks `[d1, d2, d3, d4]`, swaps `[0,0,0,0]`, opponents `[B, B, B]`; `ctxB` = the same with focal seat 3. Input list (weights): `[(ctxA, 1.0), (ctxB, 1.0), (ctxA, 2.0)]`, `field_id = "f"`, `version = 1`.
```
canonical contexts: ctxA weight 0.75, then ctxB weight 0.25
canonical JSON:
{"contexts":[{"decks":["d1","d2","d3","d4"],"focal_seat":0,"opponents":["B","B","B"],"swap_counts":[0,0,0,0],"weight":0.75},{"decks":["d1","d2","d3","d4"],"focal_seat":3,"opponents":["B","B","B"],"swap_counts":[0,0,0,0],"weight":0.25}],"field_id":"f","version":1}
field_hash = 74c61ecb69d988879d4c89221a3d76414cff99807425113575aedb111850c610
input [(ctxB, 1.0), (ctxA, 2.0), (ctxA, 1.0)] -> identical JSON, hash and SDWP
```

**F8c. Field save/load round-trip (FD3a).** Three distinct contexts with raw weights `[1, 6, 15]`.
```
canonical weights = [0.045454545454545456, 0.2727272727272727, 0.6818181818181818]
math.fsum(canonical weights) = 0.9999999999999999        (within 1e-12: valid)
save -> load: payload bytes, field_hash, provenance and SDWP identical
edit any payload value in the saved file (e.g. a weight) -> load raises FieldIntegrityError
edit only provenance -> loads; field_hash unchanged
renormalising on load would divide by 0.9999999999999999 and change all three weights -> forbidden
```

**F9 (R2). Retries and frozen evidence (E3–E4).** Duplicate block 0, rotation 1. Events: (1) rotation 1 `failed`; (2–4) one report each for rotations 0, 2, 3 with focal points 1, 0.25, 0 respectively; (5) rotation 1 `win`. Each execution uses a distinct ID.
```
stochastic:    event 1 resolves rotation 1 as failed; block consumed at event 4
               -> Xup = 0.0625, Xlow = 0.3125 (frozen); event 5 kept in the event log only; not exported for ratings (E8)
deterministic: event 1 is provisional; block consumed at event 5 with r_A[1] = 1
               -> Xup = Xlow = 2.25/4 - 0.25 = 0.3125
deterministic + a further report for rotation 1 under a new `game_id` with outcome="win", winner_seat=2 (A, in seat 1, lost)
               -> replay_violation (different outcome from the earlier success) -> handled by E7 (see F12)
```
(There is no `loss` outcome value: A losing is encoded as `outcome="win"` with a `winner_seat` other than A's seat.)

**F10. Gauge shift vs objective.** Two agents, θ = [0.5, 0.0], prior N(0, 3²), everything else fixed. Shift both by +1:
```
probabilities and likelihood_nll: unchanged
prior_penalty(theta) before = (0.25 + 0)/18 = 0.0138889
prior_penalty(theta) after  = (2.25 + 1)/18 = 0.1805556
objective changes by exactly +0.1666667
```

**F11. Connected but confounded (D4).** One bracket. Agent A always plays deck X, B always plays Y, anchor H plays Z and W. Every game seats {A on X, B on Y, H on Z, H on W}, over all 24 seat permutations (each agent appears in plenty of clusters; the graph is connected).
```
parameter order: theta[A,B,H], seat[0..3], mu[b1], delta[W,X,Y,Z]  (12 params)
rank(Ddiff) = 6, nullity = 6
aliasing residual r(theta[A] - theta[H]) = 0.65192  -> aliased
aliasing residual r(theta[B] - theta[H]) = 0.65192  -> aliased
aliasing residual r(theta[A] - theta[B]) = 0.70711  -> aliased
(theta[A] + delta[X]) - (theta[B] + delta[Y]) is identified (residual ~3e-16), but it's an agent+deck
combination, not an agent contrast; it is never presented as a headline rating
Ddiff @ (e_theta[A] - e_delta[X]) = 0 exactly: the aliasing direction
```
Add a second set of 24 games seating {A on Z, B on W, H on X, H on Y}. Then `r(theta[A] - theta[H]) ≈ 1e-16` → identified. Headline eligibility changes from `aliased` to `identified` (or `weak`, depending on D3).

**F11b. Unsupported anchor (D1a).** Agents A, B, C and anchor H; decks rotate freely. Games seating {A, B, C, C} are decisive; every game that includes H is a `draw`. H has no likelihood-used games and is removed by I5.
```
post.log_worth(), anchor-relative SDWP and anchor-component labels -> AnchorUnsupported(anchor="H")
post.contrast("A", "B") -> reported (identified by D4)
no other agent becomes the anchor; refitting with anchor="A" is the caller's choice
```

**F14. Wide-design aliasing (D4 step 2).** One decisive game: H (anchor) on X in seat 0, A on Y in seat 1, B on Z in seat 2, C on W in seat 3; one bracket. 13 parameters (θ[A,B,C,H], seat[0..3], μ, δ[W,X,Y,Z]), 3 difference rows.
```
thin SVD of the 3 x 13 difference matrix: singular values [3.464102, 1.732051, 1.732051], none zero
    -> a thin-SVD null-space rule finds NO null vectors and would wrongly call every contrast identified
Gram / row-space method: rank 3; r(theta[A] - theta[H]) = 0.816497 -> aliased (correct)
anchor self-contrast: status zero (D5), value 0.0; no residual computed
```

**F15. Field identification (FD6).** One bracket, decks P, Q, R, S (group 1) and T, U, V, W (group 2). Agents H (anchor), A, B. Games: every distinct seating of {H, A, B, B} × every permutation of a group's four decks, for each group separately. The two deck groups never appear in the same game.
```
agent contrasts: r(theta[A] - theta[H]) ≈ 8e-16, r(theta[B] - theta[H]) ≈ 8e-16 -> identified (D4 passes)
field context 1: A focal in seat 0 on P; H on Q, R, S in seats 1–3
    residuals of the three focal-vs-opponent differences: [0, 0, 0]          -> passes
field context 2: A focal in seat 0 on P; H on Q, T, U in seats 1–3
    residuals: [0, 0.288675, 0.288675]                                      -> fails
a field containing context 2 with positive weight: A's SDWP labelled prior_dependent_field
(A stays in the headline ordering; only the SDWP level is labelled)
```

**F12 (R2). Late contradiction after a terminal decision (E7).** Deterministic replay. The comparison reaches ABOVE at block 40. Later, a second report arrives for block 12, rotation 2, with a different `winner_seat` from the result already consumed.
```
decision_at_stop = ABOVE                 (unchanged)
evidence_valid   = false; invalidation_reasons = ["replay_violation: block 12, rotation 2"]
frozen scores for blocks 0..40           (unchanged)
cmp.status()     = INCONCLUSIVE_INVALID
the same contradiction arriving before block 40 is consumed ->
    decision_at_stop = INCONCLUSIVE (replay_violation), evidence_valid = false
```

**F13 (R2). Confidence-sequence bracket (E6).**
```
largest point excluded from below = 0.1000, smallest excluded from above = 0.1025
    -> report [0.1000, 0.1025], cs_conflict = false   (no grid point survives, but bounds don't cross)
largest excluded from below = 0.1050, smallest excluded from above = 0.1025
    -> report the bracket with cs_conflict = true; decision_at_stop and evidence_valid unchanged
alpha_up = 0.01, alpha_low = 0.10, cs_alpha unset -> cs_alpha = 0.01
    decisions: ABOVE at K_up >= 100, BELOW at K_low >= 10; CS exclusion at wealth >= 2/0.01 = 200
alpha_up = 0.01, alpha_low = 0.10, cs_alpha = 0.05 -> CS exclusion at 40; decisions unchanged
duplicate grid: K = 400; points -0.2475, -0.2450, ..., 0.7475 (399 points; -0.25 and 0.75 never included)
```

---

**F16 (R2). Rating export (E8).** Duplicate block 0, deterministic replay. Each execution has a fresh ID. Events: 1 = rotation 1 failed; 2, 3, 4 = first reports for rotations 0, 2, 3, respectively (all wins, including rotation 2); 5 = rotation 1 win, winner seat 1 (A); 6 = rotation 1 repeat with the same outcome as event 5; 7 = rotation 2 repeat with a different winner seat from event 3.
```
after event 5: block 0 consumed; export_revision = 1; rating_rows(): 4 rows
after event 6: no change (export_revision = 1); event 6 in the event log only
after event 7: rotation 2 quarantined; export_revision = 2; rating_rows(): 3 rows
    rotation 0 <- event 2;  rotation 1 <- event 5;  rotation 3 <- event 4
    export_report omits (block 0, rotation 2, "quarantined: replay_violation")
event log only: event 1 (provisional failure, recovered), event 6 (repeat), event 7 (contradiction)
evidence_valid = false (event 7, E7); cmp.status() = INCONCLUSIVE_INVALID
event 5's row sent again with the same game_id and identical content -> transport_duplicate: counted, not recorded
```

**F17. Output statuses (D5).** One fit containing: anchor H; A and B as in F11 (aliased); C and D, which only ever play each other (a second component); E, whose games are all draws; every deck has split `holdout` (no train decks).
```
post.contrast("H", "H")  -> status zero, value 0.0, data_share n/a, interval [0, 0]
post.contrast("A", "H")  -> status aliased, no value
post.contrast("A", "H", include_withheld=True) -> prior_dependent_value present; value column empty
post.contrast("C", "H")  -> status disconnected, no value
post.contrast("C", "D")  -> evaluated by D4 (same component)
post.contrast("E", "E")  -> status unrankable (precedes zero)
post.contrast("Z", "H")  -> UnknownEntityError
post.decks()             -> baseline unavailable; no deck nuisance values
post.headline_eligibility() -> H (zero, true); A, B (aliased, false); C, D (disconnected, false); E (unrankable, false)
post.sdwp(valid_field_with_H_as_each_opponent) -> one numeric row, for H, with a non-exact interval when requested
```
With the anchor's games all draws (F11b data), `post.log_worth()`, `post.headline_eligibility()`, `post.components()` and `post.sdwp(field)` each return one `AnchorUnsupported(anchor_id="H")` result, while `post.contrast("A", "B")` is still evaluated by D4.

**F18. Bootstrap precedence (B5).** n = 100.
```
3 failed (3%); Q1 zero-support in 6 of 97 converged (6.19%); Q2 in 4 of 97 (4.12%)
    -> Q1 suppressed (insufficient_clusters); Q2 emitted, labelled unstable_bootstrap,
       percentiles over all 97 converged values; diagnostics Q2: n_converged 97, n_zero_support 4, n_used 97
6 failed (6%)       -> every stochastic interval that passed B4 suppressed (unstable_bootstrap); zero-support not computed
100 failed          -> every stochastic interval that passed B4 suppressed (unstable_bootstrap); no division performed
5 failed (exactly 5%); Q zero-support 0 -> Q emitted, labelled unstable_bootstrap (5% is not > 5%)
1 failed (exactly 1%)                   -> intervals emitted without the warning label
0 failed; Q zero-support 5 of 100 (exactly 5%) -> Q emitted
```

**F18 extension.** With 60 total clusters and an identified train-deck nuisance contrast, give target X 30 clusters and baseline deck Y 9; both have nonzero coefficients in the final contrast. X's point estimate remains numeric, but its interval is `insufficient_clusters` from B4 even when all replicates fail. In that run, H vs H has an exact `[0, 0]` interval and zero replicates used; an eligible stochastic contrast with all dependencies supported is `unstable_bootstrap`. With only one rankable train deck X, the centred contrast for X is exactly zero and has no bootstrap dependencies.

**F19 (R2). State transitions (E0).** Duplicate design, `Superiority(margin=0)`, `alpha=0.01`, `max_blocks=2`, default F-rule. With at most four blocks and these thresholds, neither direction can cross even at the best score (maximum upper factor 2.5, so wealth is at most 39.0625 < 100; the lower maximum is smaller). Use F12 or a separate deliberately terminal setup for the post-ABOVE case.
```
batch vs single: ingest([block 1 rows, block 0 rows]) and the same rows one at a time in that order
    -> identical event logs; block 0 consumed at its last row's event, block 1 drained in the same step
budget: blocks 0, 1, 2 fully reported -> blocks 0, 1 consumed; BUDGET_EXHAUSTED; block 2 buffered; wealth unchanged
    extend_budget(2) -> ValueError;  extend_budget(3) -> block 2 consumed during the call
separate empty-buffer case: consume blocks 0 and 1, then extend_budget(3) with no complete block 2
    -> CONTINUE; same wealth, consumed count and export revision
atomic failure (after extend_budget(4) in the buffered case): block 3 with rotation 0 already resolved (win), then mark_block_failed("block 3")
    -> one event; rotations 1-3 resolved as failed; rotation 0 unchanged; block 3 consumed
    -> failure_records(): 3 entries; rating_rows() for block 3: the rotation 0 row only
after ABOVE: rows for a new block are validated and recorded but never consumed
```

**F20. Prediction, swaps, and solver (Appendix C and FD4a).** Add one registered but unused deck to F14’s registry; keep the F14 likelihood unchanged. The observed four-seat input predicts four probabilities summing to one and is `field_identified`. Move only the decks in seats 0 and 1 while leaving agents fixed: return four probabilities with `prior_dependent_field`. Replace a deck with an unknown ID: `UnknownEntityError`; a registered deck excluded from the likelihood: `UnsupportedPredictionError`. A field containing a nonzero swap constructs and round-trips independently of the model, but its evaluation rejects the swap with η off, even on a pooled posterior. On a pooled posterior `predict_proba` ignores nonzero swap counts and labels the result. Solver verification is specified separately in F20b.

**F20b. Solver boundary (C.2).** Use one-dimensional callbacks. For `f(x)=x²/2`, start at x=1 with gradient x and Hessian 1; take one full step, return x=0, decrement²=0 and lower factor 1. Repeat with `f(x)=1_250_000 + x²/2`, x0=1e-5, and the exact callback `delta(x,h)=h*(x+h/2)`. The unshifted and shifted aggregate objectives at the endpoints round to the same float in the second test, but the callback must accept the full step and converge. For the actual Luce change, use F14’s four fixed seat/agent/deck tuples and F5’s winner counts `[1_000_000, 1_000_000, 0, 0]` in the compressed objective (`η` off, specified priors). Test the solver directly on compressed counts rather than constructing two million game objects. Start at zero with `decrement_scale=2_000_000` and require `decrement_sq <= 2e-14`; at `psi=0`, use utility increments `[1e-8, -2e-8, 3e-8, -4e-8]` via the seat coefficients and compare the objective delta including the ridge to a 50-digit Decimal reference with `rtol=1e-10, atol=1e-12`. For the limit, use `f(x)=exp(x+100)` with objective, gradient and Hessian equal to the same positive value and x0=0. The mathematical Newton step is −1, Armijo permits each step, and decrement² remains well above threshold. Assert `ConvergenceError` after checking the iterate near −50, with no objective trial near −51.

**F21 (R2). Replay/integrity matrix and descriptive summary.** In each replay mode send two individually valid reports for one position with two different winners. Vary same vs distinct IDs and delivery before vs after block consumption:

| IDs | Replay mode | Expected result |
|---|---|---|
| Distinct | stochastic | First result resolves; second recorded only; validity, export and revision unchanged |
| Distinct | deterministic | Second recorded; `replay_violation`; quarantine and invalidate |
| Same | either | Second rejected and audited; `game_id_integrity_violation`; quarantine and invalidate |

Also test failed → success in both modes. With a fresh game ID, deterministic replay resolves on success and stochastic replay retains its first resolution; neither invalidates. Under the same ID, the change violates E0 in either mode and invalidates. Identical failure retransmissions remain transport duplicates.

Before consumption, invalidation stops draining; no block or row is exported. After consumption, the scores stay frozen, the position's row is removed, and revision advances once. Also test a same-ID metadata-only conflict and a cross-position ID conflict: reject/audit both without invalidation. Pass F4's four positions (including the explicit `None`) to `summarise_blocks`; expect one block, one failed block, failure fraction 1 and bounds `[0.0625, 0.3125]`. Empty input has zero counts and no bounds/rate; a missing position is rejected.

**F22. Shared pipeline and canonical data (I0–I2, C.1–C.2).** Extend F7 with an aware timestamp and an equivalent timestamp in another offset: assert one canonical row, not an ID conflict, for the pair; reject a naive timestamp. Reorder F7, add an exact duplicate and vary inclusion without changing `data_hash`; change an audited value on an excluded row and require a different hash. Compare to an independent canonical-json/SHA-256 oracle. With `allow_mirror=False`, verify a repeated policy row fails both preview and full pipeline. With a required declared tag absent, require the same error on both paths; P2 adds the actual `fit(..., tag_schema=...)` boundary test. Check both endpoints of the inclusive time range and `None` vs empty filters. Cover a multiply-owned manifest ID and a listed game with null/wrong cluster key before I1.

**F23. Seeded streams and canonical bytes (FD2–FD3, B1, C3).** With seed 12345, two eligible brackets of four sorted decks each and two sorted opponents, the first three raw FD2 draws have `(bracket index, deck indices, focal seat, opponent indices)` equal to `(0,[1,2,0,3],1,[0,1,0])`, `(1,[0,1,2,3],0,[0,0,0])`, `(0,[1,2,3,0],0,[0,0,1])`. Check them before coalescing; input registry and pool permutation leave canonical bytes unchanged. Reject a duplicate pool ID. With five canonical cluster keys, replicate 0 at the same seed has weights `[0,1,1,1,2]`. Compare `SeedSequence(seed, spawn_key=(k,))` with `.spawn(n)[k]` in the pinned NumPy environment. Verify tagged singleton and named-cluster keys with the same string do not collide, custom strata survive save/load, and draw/cap-only occurrences of an entity do not increase its support. Verify a non-ASCII ID is escaped in FD3 bytes, a referenced version change affects the fingerprint, and an unrelated entity or focal-only version change does not. R2: save/load PodDistribution without changing bytes or hash; reject bad schema, duplicate keys and tampering; a split sample using `start_index` matches the corresponding full stream.

## Appendix B — Producing pooled deck embeddings (deferred feature; recommendations)

Kept for when §19's condition is met. Aim: capture what differentiates a deck (plan, speed, interaction density, synergy), not what every Commander deck shares.

1. Card embeddings from the RL project's card pipeline (text + reminder text, type line, cost, P/T, keywords).
2. Drop basic lands; optionally down-weight other lands.
3. Weight by rarity across a large external decklist corpus (IDF `log(N/(1+n_c))` or SIF `a/(a+f_c)`, a ≈ 1e-3).
4. Weight the commander 5–10× a normal card.
5. Weighted mean, then remove the top common principal direction of a reference corpus; L2-normalise.
6. Upgrades: co-occurrence card embeddings trained on decklists (biggest gain); concatenated text + co-occurrence vectors; spread statistics; RL-agent embeddings (versioned).
7. Checks: nearest neighbours share commander/archetype; probes predict archetype, not just colours; PC1 doesn't dominate; small swaps move a deck only slightly.
8. Version everything; fit PCA within each validation fold.

## Appendix C — Public boundaries and deterministic implementation rules

This appendix pins interface details; the numbered rules own statistical and state semantics. Named return types are immutable value records (or tuples of those records); absent numeric values are `None`, not NaN sentinels. IDs sort as Python strings; bracket IDs sort numerically. Public record collections do not depend on registry insertion order. `GameTable` preserves input row order and does not deduplicate at construction: R1 applies I0, R2 applies E0. R1 imports do not import R2.

### C.1 R1 signatures and result shapes

`ModelSpec` and `Inclusion` are frozen dataclasses. Normalise and defensively copy collections at construction. `Inclusion.default()` is equivalent to `Inclusion()`.

| Record | Field and type | Default |
|---|---|---|
| `ModelSpec` | `swap_effect: bool`; `allow_pooled_variants: bool` | `False` for each |
| | `weak_ratio: float`; `tol_eig: float`; `tol_alias: float` | `1.0`; `1e-10`; `1e-6` |
| | `max_params: int`; `max_dense_bytes: int` | `10_000`; `2 * 2**30` |
| `Inclusion` | `suites`, `engine_versions`, `splits`, `agents`: each `frozenset[str] \| None` | `None` for each |
| | `played_at: tuple[datetime \| None, datetime \| None]` | `(None, None)` |
| | `clusters: Literal["use_available", "complete_only"]` | `"use_available"` |
| | `manifest: Mapping[str, frozenset[str]] \| None` | `None` |
| | `allow_mirror: bool` | `True` |

Require finite `weak_ratio > 0`, `0 < tol_eig < 1`, `0 <= tol_alias < 1`, and positive integer limits. Timestamp bounds obey §3.4 and must be ordered when both are given. `manifest` is required for `complete_only` and must be absent for `use_available` (otherwise `InclusionError`); its keys and game-ID sets must be nonempty. `fit` applies `inclusion.allow_mirror`; `validate` passes its `allow_mirror` into the same shared checks. `tag_schema` and `split_policy` have the same meaning on both entry points.


```text
GameTable.from_records(records) -> GameTable
AgentRegistry.from_records(records) -> AgentRegistry
DeckRegistry.from_records(records) -> DeckRegistry
validate(games, agents, decks, *, strict=True, allow_mirror=True,
         split_policy=None, tag_schema=None) -> ValidationReport
fit(games, agents, decks, spec=ModelSpec(), inclusion=Inclusion.default(),
    split_policy=None, cluster_strata=None, *, warm_start=None, tag_schema=None) -> Posterior
Field.from_contexts(field_id, version, contexts) -> Field
Field.materialise(recipe, *, decks) -> Field
Field.load(path) -> Field; Field.save(path) -> None
Posterior.load(path) -> Posterior; post.save(path) -> None
post.counts() -> CountsReport; post.components() -> ComponentsReport | AnchorUnsupported
post.unrankable() -> UnrankableReport; post.outcome_rates() -> OutcomeRatesReport
post.identification_report() -> IdentificationReport; post.provenance() -> Provenance
post.headline_eligibility() -> tuple[EligibilityRow, ...] | AnchorUnsupported
post.log_worth(*, include_withheld=False) -> tuple[AgentContrastRow, ...] | AnchorUnsupported
post.contrast(a, b, *, include_withheld=False) -> ContrastResult
post.contrast_vector(c) -> ContrastResult
post.seats() -> tuple[SeatContrastRow, ...]
post.decks() -> DeckNuisanceReport; post.swap_effect() -> ContrastResult | EffectNotModelled
post.sdwp(field, *, method="map", n_draws=1000, seed=None) -> SdwpReport | AnchorUnsupported
post.predict_proba(contexts) -> tuple[PredictionRow, ...]
post.sample(n, seed) -> float64 array of shape (n, P)
cluster_bootstrap(post, n=400, *, seed, strata=None, min_clusters=50,
                  min_entity_clusters=10) -> BootstrapResult
```

- `EligibilityRow`: `agent_id`, `contrast_status`, `eligible`, `reasons`. `ContrastResult`: `status`, `value`, `data_share`, `aliasing_residual`, `prior_dependent_value`, `interval`, `labels`, `reasons`. Typed agent/seat/deck rows include the relevant ID or seat. `Interval` has `lo`, `hi`, `method`, `labels`. `DeckNuisanceReport` has baseline status, sorted reference deck IDs and rows; missing baseline produces no rows. `swap_effect()` returns `EffectNotModelled` when η is off. Linear Laplace intervals use `value ± scipy.stats.norm.ppf(0.975)*sd`.
- `SdwpReport` carries field identity, context-version fingerprint, fit provenance, and D5-ordered rows. Each row has `agent_id`, `sdwp` (always MAP), `contrast_status`, `field_label`, `failing_contexts`, `labels` and optional `interval`. `method="laplace"` requires an explicit seed, adds `sdwp_posterior_mean`, and uses the linear 0.025/0.975 quantiles of draw-level SDWP. It does not replace the MAP estimate or its ordering. `method="map"` has no interval. Bootstrap result contrast/SDWP methods return the same shapes with B4–B6 intervals and no posterior mean. It exposes `diagnostics()` keyed by the statistic and its identifying inputs.
- `PredictionInput`: exactly `agents`, `decks`, `swap_counts`, all length four in seat order; input is a sequence of typed records. IDs refer to the posterior's snapshots; no winner, outcome, suite or timestamp is accepted. Validate shape and swaps first (`ValueError` for malformed inputs), then IDs (`UnknownEntityError` for missing registry IDs), support (`UnsupportedPredictionError` for entities without likelihood-used games), then FD4a. Predict at MAP. For each input use FD6's three seat differences relative to seat 0; a zero difference, if present, passes. Return `PredictionRow(probabilities, field_label, failing_seats, labels)` in input order. Prediction does not require the anchor or reuse the leaderboard eligibility filter: cross-component predictions can be returned with the prior-dependence label. Field evaluation retains its stricter FD4 rules.
- Field opponents occupy the three non-focal seats in ascending seat order (R2's clockwise placement is separate). For raw fields, reject empty contexts or invalid shape/weights with `ValueError`; reject invalid saved field schema or hash with `FieldIntegrityError`. FD4 violations at evaluation raise `UnsupportedFieldError`; FD4a violations raise `SwapsNotModelledError`.
- Order log-worth, eligibility, entity counts and unrankable rows by ID; seat rows by index. For components, sort IDs within each component; list the anchor component first, then other components by smallest ID. `contrast_vector` takes a finite float64-compatible vector of length P in parameter-index order, not an ID mapping. Numeric method arguments outside their allowed domain raise `ValueError`. Use `DuplicateGameError`, `ManifestMismatchError`, `NoRankableEvidence`, `ModelTooLargeError` and other named domain errors at their specified boundaries. `AnchorUnsupported` is a result, not an exception.

### C.2 Numerical and persistence rules

1. **Canonical keys.** Order parameters as θ by supported agent ID, seats 0–3, μ by supported bracket ID, δ by supported deck ID, then η if enabled. A tagged tuple distinguishes a supplied cluster `("cluster", cluster_id)` from a singleton `("game", game_id)` even if their strings match. Sort those tuples lexicographically for the frame. Sort pattern keys lexicographically by `(agents, decks, effective_swap_counts)`; seat is positional and swaps are all zero in the key when η is off. Aggregation uses integer counts before converting to float64. Scan patterns and bootstrap-frame entries in canonical order.
2. **Solver.** Internal `rating.solve.newton(objective, grad, hess, x0, *, objective_delta=None, decrement_scale=1) -> SolveResult(mode, lower_factor, accepted_steps, decrement_sq)`. Use it for fit and bootstrap, passing the stable delta callback below and `decrement_scale=max(1, N_used)` where `N_used` is the weighted sum of likelihood-used game counts in the objective being solved. The helper is not public. Start at zero unless a finite `warm_start` mapping is supplied: copy matching entries, initialise missing entries to zero and ignore names not in the index. B2 starts at the original mode. At each iterate solve `H d = -g` by Cholesky and set `decrement_sq = -g' d`. Treat a negative value of magnitude at most `1e-20` as zero; reject materially negative or nonfinite decrements. Converge when `decrement_sq / decrement_scale <= 1e-20`, after the negative-roundoff handling. The scale counts likelihood observations, not the absolute objective, so adding a constant cannot loosen the stopping rule. Otherwise try `t = 1, 1/2, …, 2^-40` and take the first trial with finite candidate coordinates, finite candidate objective, and finite `delta` satisfying `delta <= 1e-4*t*g'd`, where `delta = objective_delta(psi, t*d)` if provided, otherwise `objective(psi + t*d) - objective(psi)`. Check convergence at the initial point and after every accepted step, including step 50. If still not converged after that check, raise `ConvergenceError` without taking step 51. Also raise on failed factorisation or exhausted backtracking. Return the lower factor evaluated at the returned mode.

   **Stable Armijo change for the Luce objective.** For each compressed pattern, let `u` be the original utilities, `h = X @ step` the utility increment, `p = softmax(u)`, and `q = h - h[0]`. For each winner i the change in negative log likelihood is `logZ_ratio - q[i]`. Compute `logZ_ratio = log1p(fsum(p[j]*expm1(q[j]) for j in seats))` if `max(abs(q)) <= 0.5`; otherwise use `logsumexp(u + q) - logsumexp(u)`. Add the ridge change `sum_j precision[j] * step[j] * (psi[j] + step[j]/2)`. Use float64 and `math.fsum` in canonical pattern/seat/parameter order to sum the count-weighted changes and the ridge change. Use the same likelihood weights as the objective, also in bootstrap. This retains Armijo's test without comparing two nearly equal aggregate objectives or accepting a step solely because its decrement is small.
**Canonical row bytes and `data_hash`.** Use FD3's UTF-8 JSON encoding for the complete normalised §3.2 row object, all keys present, absent nullable fields as null, arrays in seat order, and UTC timestamps formatted `YYYY-MM-DDTHH:MM:SS.ffffffZ` (always six digits; reject unsupported sub-microsecond precision rather than truncate nonzero digits). Preserve audit payloads as JSON. Hash the FD3 JSON array of those rows **after I0 deduplication and before row/cluster/outcome filters**, sorted by `game_id`. `data_hash` is the lowercase SHA-256 hex of that array; filtering is represented separately by inclusion/provenance. Reordering input or adding an identical row cannot change this hash; changing even a filtered-out row can.

3. **Field persistence.** The FD3b envelope is schema 1; the payload's `version` is the caller's field version, not the persistence schema. Do not add a fourth envelope key. Preserve canonical payload and provenance as specified by FD3a–b.
4. **Posterior and Comparison persistence.** **Posterior** (R1) uses a versioned single-file archive: a ZIP container with one small UTF-8 `manifest.json` and typed `arrays/<name>.npy` members for bulk numeric arrays. The manifest has exactly `kind` (`podrank.posterior`), `schema_version` (integer `2`), `package_version`, and `payload`. The payload carries named metadata, ordered records and tagged keys as JSON, plus an `arrays` map whose entries name the member, dtype and shape. Bulk parameter vectors, matrices, compact pattern arrays and cluster-count arrays must not be JSON lists, base64, pickle or raw game rows. Load rejects duplicate ZIP members, unexpected or missing members, unknown member names, unsupported dtypes/schema versions, malformed descriptors, dtype/shape mismatches, size-bound violations and cross-array index errors with `PersistenceError`; use `numpy.load(..., allow_pickle=False)`. **Comparison** (R2) keeps the UTF-8 JSON envelope with exactly `kind` (`podrank.comparison`), `schema_version` (integer `1`), `package_version`, and `payload`. **Field** keeps its FD3b envelope. Reject duplicate JSON keys, malformed payloads and unsupported schema versions with `PersistenceError`. No pickle or arbitrary code on load. Save all inputs needed to reproduce returned state without refitting or replaying on load: the Posterior archive contains snapshots, spec, inclusion, parameter index, mode, lower factor, row-space and null-space bases, D1–D5 reports and classifications, integer outcome counts and rates, compact pattern table, outcome-and-cluster map, retained cluster table/strata including crossing indices, and provenance, and does not retain raw game rows; Comparison payload contains bound configuration/snapshots/distribution, descriptors, accepted and audit events, cursor, seen-ID index/counters, resolution/quarantine state, consumed scores, log wealth, budget, terminal/effective state and export revision. Save via same-directory temporary file and atomic replace. Schema changes require an ADR and a round-trip test. Do not claim compatibility with other schemas unless an explicit migration is implemented.

### C.3 R2 signatures and ordering

```text
PodDistribution.load(path) -> PodDistribution; pods.save(path) -> None
pods.sample(n_blocks, seed, start_index=0) -> tuple[PodDraw, ...]
summarise_blocks(blocks, *, design) -> DescriptiveSummary
Comparison(*, a, b, design, agents, decks, engine_version, pod_distribution,
           hypothesis, alpha=0.05, alpha_up=None, alpha_low=None, cs_alpha=None,
           max_blocks, replay="stochastic", min_blocks_for_failure_rule=50,
           max_failed_fraction=0.02) -> Comparison
cmp.declare_blocks(descriptors) -> None
cmp.ingest(games) -> IngestReport
cmp.mark_failed(block_id, rotation, role=None) -> None
cmp.mark_block_failed(block_id) -> None; cmp.extend_budget(new_max) -> None
cmp.status() -> Status; cmp.report() -> ComparisonReport
cmp.rating_rows() -> tuple[GameTable, ExportReport]
cmp.failure_records() -> tuple[FailureRecord, ...]
cmp.save(path) -> None; Comparison.load(path) -> Comparison
```

Reject invalid configuration, descriptors or explicit failure marks with `ValueError`; a descriptor batch validates atomically. `ingest` handles per-row failures as E0 classifications and reports them in input order. Distinct from a caller's game/block IDs, generate `comparison_id` once at construction with `uuid.uuid4().hex` and preserve it on load. Compare independent instances' state for E0/E5 tests excluding this identifier and the event-log ordering where E5 permits it. Export rows and failures in consumed `block_index`, rotation, role order (`a` before `b`). `status()` returns the effective enum; detailed reasons and the historical stop are in `report()`.

