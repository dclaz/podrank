"""NumPyro reference model for R1 posterior validation (extra [reference])."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import GameTable, normalize_game_row
from podrank.rating.spec import ModelSpec


def numpyro_available() -> bool:
    try:
        import numpyro  # noqa: F401
    except ImportError:
        return False
    return True


def _arrays(
    games: GameTable | Iterable[Mapping[str, object]],
    agents: AgentRegistry,
    decks: DeckRegistry,
) -> dict[str, np.ndarray]:
    agent_ids = sorted(agents.by_id)
    deck_ids = sorted(decks.by_id)
    agent_index = {name: index for index, name in enumerate(agent_ids)}
    deck_index = {name: index for index, name in enumerate(deck_ids)}
    bracket_ids = sorted({record.bracket for record in decks.by_id.values()})
    bracket_index = {bracket: index for index, bracket in enumerate(bracket_ids)}
    rows = [
        row
        for row in (normalize_game_row(record) for record in games)
        if row.outcome == "win" and row.winner_seat is not None
    ]
    n = len(rows)
    agent_array = np.empty((n, 4), dtype=np.int32)
    deck_array = np.empty((n, 4), dtype=np.int32)
    bracket_array = np.empty((n, 4), dtype=np.int32)
    winner = np.empty(n, dtype=np.int32)
    for index, row in enumerate(rows):
        winner[index] = row.winner_seat if row.winner_seat is not None else 0
        for seat, agent in enumerate(row.agents):
            agent_array[index, seat] = agent_index[agent]
            deck_array[index, seat] = deck_index[row.decks[seat]]
            bracket_array[index, seat] = bracket_index[decks.by_id[row.decks[seat]].bracket]
    return {
        "agent": agent_array,
        "deck": deck_array,
        "bracket": bracket_array,
        "winner": winner,
        "agent_ids": np.asarray(agent_ids),
        "deck_ids": np.asarray(deck_ids),
        "n_agents": np.asarray(len(agent_ids)),
        "n_decks": np.asarray(len(deck_ids)),
        "n_brackets": np.asarray(len(bracket_ids)),
    }


def fit_numpyro(
    games: GameTable | Iterable[Mapping[str, object]],
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    spec: ModelSpec | None = None,
    seed: int = 0,
    num_warmup: int = 300,
    num_samples: int = 300,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(1)
    spec = ModelSpec() if spec is None else spec
    arrays = _arrays(games, agents, decks)
    agent = jnp.asarray(arrays["agent"])
    deck = jnp.asarray(arrays["deck"])
    bracket = jnp.asarray(arrays["bracket"])
    winner = jnp.asarray(arrays["winner"])
    n_agents = int(arrays["n_agents"])
    n_decks = int(arrays["n_decks"])
    n_brackets = int(arrays["n_brackets"])

    def model() -> None:
        theta = numpyro.sample("theta", dist.Normal(0.0, 3.0).expand([n_agents]).to_event(1))
        seat = numpyro.sample("seat", dist.Normal(0.0, 0.5).expand([4]).to_event(1))
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([n_brackets]).to_event(1))
        delta = numpyro.sample(
            "delta",
            dist.Normal(0.0, spec.sigma_beta).expand([n_decks]).to_event(1),
        )
        utilities = theta[agent] + mu[bracket] + delta[deck] + seat
        numpyro.sample("obs", dist.Categorical(logits=utilities), obs=winner)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    samples = mcmc.get_samples()
    return {
        "theta": np.asarray(samples["theta"]),
        "seat": np.asarray(samples["seat"]),
        "mu": np.asarray(samples["mu"]),
        "delta": np.asarray(samples["delta"]),
        "agent_ids": tuple(str(name) for name in arrays["agent_ids"]),
        "deck_ids": tuple(str(name) for name in arrays["deck_ids"]),
        "samples": int(num_samples),
    }


__all__ = ["fit_numpyro", "numpyro_available"]
