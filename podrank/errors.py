"""Domain errors and warnings for the data boundary."""

from __future__ import annotations


class PodrankError(Exception):
    """Base class for all podrank domain errors."""


class DataError(PodrankError):
    """Base class for errors caused by supplied data or inclusion settings."""


class ValidationError(DataError, ValueError):
    """A boundary record violates the data contract in IMPLEMENTATION_PLAN.md §3."""


class DuplicateGameError(DataError, ValueError):
    """Two different rows share one ``game_id``."""


class UnknownEntityError(DataError, ValueError):
    """A row references an agent or deck absent from the bound snapshots."""


class IdentityMismatchError(ValidationError):
    """A row's policy or deck version does not match the snapshot."""


class InclusionError(DataError, ValueError):
    """An inclusion configuration is invalid or inconsistent."""


class ManifestMismatchError(InclusionError):
    """Rows do not match the complete-only expected-games manifest."""


class NoRankableEvidence(InclusionError):
    """No likelihood-used (decisive) games remain after inclusion."""


class DuplicateRowWarning(UserWarning):
    """Exact duplicate rows were removed during I0 deduplication."""


__all__ = [
    "DataError",
    "DuplicateGameError",
    "DuplicateRowWarning",
    "IdentityMismatchError",
    "InclusionError",
    "ManifestMismatchError",
    "NoRankableEvidence",
    "PodrankError",
    "UnknownEntityError",
    "ValidationError",
]
