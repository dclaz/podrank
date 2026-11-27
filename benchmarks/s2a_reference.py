"""S2a Laplace-vs-NUTS sd ratio and small SBC study on doc_tiny."""

from __future__ import annotations

import json
import time

import numpy as np

from podrank import GameTable, fit, sim
from podrank.reference import fit_numpyro, numpyro_available

REPLICATIONS = 100
WARMUP = 100
SAMPLES = 200


def _prior_draw(rng: np.random.Generator, world, spec) -> tuple[np.ndarray, ...]:
    ids = list(world.agent_ids)
    theta = rng.normal(0.0, 3.0, size=len(ids))
    theta[ids.index("agent_00")] = 0.0
    seat = rng.normal(0.0, 0.5, size=4)
    mu = rng.normal(0.0, 1.0, size=world.brackets)
    delta = rng.normal(0.0, spec, size=len(world.deck_ids))
    return theta, seat, mu, delta


def _simulated_rows(world, theta, seat, mu, delta, rng):
    ids = {name: index for index, name in enumerate(world.agent_ids)}
    decks = {name: index for index, name in enumerate(world.deck_ids)}
    brackets = {bracket: bracket for bracket in range(world.brackets)}
    rows = []
    for row in world.game_rows():
        utilities = np.asarray(
            [
                theta[ids[row["agents"][s]]]
                + mu[brackets[world.deck_registry().by_id[row["decks"][s]].bracket]]
                + delta[decks[row["decks"][s]]]
                + seat[s]
                for s in range(4)
            ]
        )
        probabilities = np.exp(utilities - utilities.max())
        probabilities /= probabilities.sum()
        updated = dict(row)
        updated["outcome"] = "win"
        updated["winner_seat"] = int(rng.choice(4, p=probabilities))
        rows.append(updated)
    return rows


def main() -> None:
    if not numpyro_available():
        print(json.dumps({"status": "blocked", "reason": "numpyro extra not installed"}))
        return
    started = time.perf_counter()
    world = sim.reference_world("doc_tiny")
    table = GameTable.from_records(world.game_rows())
    agents = world.agent_registry()
    decks = world.deck_registry()
    post = fit(table, agents, decks)
    samples = fit_numpyro(table, agents, decks, seed=11, num_warmup=500, num_samples=1000)
    ids = list(samples["agent_ids"])
    anchor = ids.index("agent_00")
    ratios = []
    for index, agent in enumerate(ids):
        if index == anchor:
            continue
        nuts_sd = float(np.std(samples["theta"][:, index] - samples["theta"][:, anchor]))
        result = post.contrast(agent, "agent_00")
        assert result.interval is not None
        laplace_sd = (result.interval.hi - result.interval.lo) / (2 * 1.959963984540054)
        ratios.append(nuts_sd / laplace_sd)

    ranks = []
    rng = np.random.default_rng(404)
    for replication in range(REPLICATIONS):
        theta, seat, mu, delta = _prior_draw(rng, world, 0.5)
        rows = _simulated_rows(world, theta, seat, mu, delta, rng)
        fitted = fit_numpyro(
            GameTable.from_records(rows),
            agents,
            decks,
            seed=replication,
            num_warmup=WARMUP,
            num_samples=SAMPLES,
        )
        sample_ids = list(fitted["agent_ids"])
        for index, agent in enumerate(sample_ids):
            if agent == "agent_00":
                continue
            true_value = theta[sample_ids.index(agent)] - theta[sample_ids.index("agent_00")]
            draws = fitted["theta"][:, index] - fitted["theta"][:, sample_ids.index("agent_00")]
            ranks.append(float(np.mean(draws < true_value)))

    bins = [0] * 5
    for rank in ranks:
        bins[min(4, int(rank * 5))] += 1
    total = len(ranks)
    print(
        json.dumps(
            {
                "status": "ok",
                "sd_ratio_min": min(ratios),
                "sd_ratio_max": max(ratios),
                "sd_ratio_mean": float(np.mean(ratios)),
                "sbc_replications": REPLICATIONS,
                "sbc_ranks": total,
                "sbc_bins": bins,
                "sbc_max_bin_deviation": max(abs(count / total - 0.2) for count in bins),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
