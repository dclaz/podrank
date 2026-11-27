"""podrank: rating fixed, versioned agents in four-player free-for-all games.

The R1 rating core is available: immutable registry snapshots, an
order-preserving game table, the shared I0–I5 inclusion pipeline, ``fit``,
posterior reports, fields, prediction, archive persistence and cluster
bootstrap. The comparison entry point arrives in a later phase.
"""

from __future__ import annotations

from types import ModuleType

from podrank import diagnostics, sim, uncertainty
from podrank._version import __version__
from podrank.constants import N_SEATS
from podrank.data import AgentRegistry, DeckRegistry, GameTable, Inclusion, validate
from podrank.errors import (
    ConvergenceError,
    DataError,
    DuplicateGameError,
    DuplicateRowWarning,
    FieldIntegrityError,
    IdentityMismatchError,
    InclusionError,
    ManifestMismatchError,
    ModelTooLargeError,
    NoEligiblePodsError,
    NoRankableEvidence,
    PersistenceError,
    PodrankError,
    SwapsNotModelledError,
    UnknownEntityError,
    UnsupportedFieldError,
    UnsupportedPredictionError,
    ValidationError,
)
from podrank.rating import (
    AgentContrastRow,
    AnchorUnsupported,
    ComponentsReport,
    ContrastResult,
    DeckContrastRow,
    DeckNuisanceReport,
    EffectNotModelled,
    EligibilityRow,
    Field,
    FieldContext,
    IdentificationReport,
    Interval,
    ModelSpec,
    OutcomeRateRow,
    OutcomeRatesReport,
    Posterior,
    PredictionInput,
    PredictionRow,
    Provenance,
    SdwpReport,
    SdwpRow,
    SeatContrastRow,
    UnrankableReport,
    fit,
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
    "AgentContrastRow",
    "AgentRecord",
    "AgentRegistry",
    "AnchorUnsupported",
    "ComponentsReport",
    "ContrastResult",
    "ConvergenceError",
    "CountsReport",
    "DataError",
    "DeckContrastRow",
    "DeckNuisanceReport",
    "DeckRecord",
    "DeckRegistry",
    "DuplicateGameError",
    "DuplicateRowWarning",
    "EffectNotModelled",
    "EligibilityRow",
    "EntityCounts",
    "Field",
    "FieldContext",
    "FieldIntegrityError",
    "GameRow",
    "GameTable",
    "IdentificationReport",
    "IdentityMismatchError",
    "Inclusion",
    "InclusionError",
    "Interval",
    "ManifestMismatchError",
    "ModelSpec",
    "ModelTooLargeError",
    "N_SEATS",
    "NoEligiblePodsError",
    "NoRankableEvidence",
    "OutcomeCounts",
    "OutcomeRateRow",
    "OutcomeRates",
    "OutcomeRatesReport",
    "PersistenceError",
    "PipelineResult",
    "PodrankError",
    "Posterior",
    "PredictionInput",
    "PredictionRow",
    "Provenance",
    "SdwpReport",
    "SdwpRow",
    "SeatContrastRow",
    "SwapsNotModelledError",
    "UnknownEntityError",
    "UnrankableReport",
    "UnsupportedFieldError",
    "UnsupportedPredictionError",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "compare",
    "diagnostics",
    "fit",
    "sim",
    "uncertainty",
    "validate",
]


def __getattr__(name: str) -> ModuleType:
    """Expose the R2 package lazily so R1 imports never load it."""

    if name == "compare":
        from importlib import import_module

        return import_module("podrank.compare")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
