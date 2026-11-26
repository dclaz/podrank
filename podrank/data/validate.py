"""Public ``pr.validate`` preview wrapper."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from podrank.data.inclusion import Inclusion, run_pipeline
from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import GameTable
from podrank.types import ValidationReport


def validate(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    strict: bool = True,
    allow_mirror: bool = True,
    split_policy: Mapping[str, Collection[str]] | None = None,
    tag_schema: Mapping[str, object] | None = None,
) -> ValidationReport:
    """Preview I0 steps 1–3 without running row or completeness filters."""

    return run_pipeline(
        games,
        agents,
        decks,
        inclusion=Inclusion(allow_mirror=allow_mirror),
        split_policy=split_policy,
        tag_schema=tag_schema,
        validate_only=True,
        strict=strict,
    )


__all__ = ["validate"]
