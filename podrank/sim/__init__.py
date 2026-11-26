"""Synthetic reference worlds and deterministic simulators."""

from podrank.sim.scenarios import (
    alias_vector,
    append_crossover_clusters,
    assert_alias_in_null_space,
    assert_pinned_design,
    crossover_cluster_ids,
    pinned_deck_rows,
    pinned_roles,
    sc3_roles,
    sc3_rows,
    sc7_rows,
    standard_rows,
)
from podrank.sim.world import SimWorld, reference_world

__all__ = [
    "SimWorld",
    "alias_vector",
    "append_crossover_clusters",
    "assert_alias_in_null_space",
    "assert_pinned_design",
    "crossover_cluster_ids",
    "pinned_deck_rows",
    "pinned_roles",
    "reference_world",
    "sc3_roles",
    "sc3_rows",
    "sc7_rows",
    "standard_rows",
]
