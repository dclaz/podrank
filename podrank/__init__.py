"""podrank: rating fixed, versioned agents in four-player free-for-all games.

The P1 data layer is available: immutable registry snapshots, an order-preserving
game table, boundary validation, and the shared I0–I5 inclusion pipeline. The
rating (``fit``) and comparison entry points arrive in later phases.
"""

from podrank._version import __version__
from podrank.constants import N_SEATS
from podrank.data import AgentRegistry, DeckRegistry, GameTable, Inclusion, validate
from podrank.errors import (
    DataError,
    DuplicateGameError,
    DuplicateRowWarning,
    IdentityMismatchError,
    InclusionError,
    ManifestMismatchError,
    NoRankableEvidence,
    PodrankError,
    UnknownEntityError,
    ValidationError,
)
from podrank.types import (
    AgentRecord,
    CountsReport,
    DeckRecord,
    EntityCounts,
    GameRow,
    OutcomeCounts,
    OutcomeRates,
    PipelineResult,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "AgentRecord",
    "AgentRegistry",
    "CountsReport",
    "DataError",
    "DeckRecord",
    "DeckRegistry",
    "DuplicateGameError",
    "DuplicateRowWarning",
    "EntityCounts",
    "GameRow",
    "GameTable",
    "IdentityMismatchError",
    "Inclusion",
    "InclusionError",
    "ManifestMismatchError",
    "N_SEATS",
    "NoRankableEvidence",
    "OutcomeCounts",
    "OutcomeRates",
    "PipelineResult",
    "PodrankError",
    "UnknownEntityError",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "validate",
]
