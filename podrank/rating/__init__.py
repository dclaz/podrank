"""R1 rating core: fit, fields, identification and prediction."""

from podrank.rating.field import Field, FieldContext
from podrank.rating.fit import fit
from podrank.rating.identification import (
    AgentContrastRow,
    AnchorUnsupported,
    ComponentsReport,
    ContrastResult,
    DeckContrastRow,
    DeckNuisanceReport,
    EffectNotModelled,
    EligibilityRow,
    IdentificationReport,
    Interval,
    SeatContrastRow,
    UnrankableReport,
)
from podrank.rating.posterior import (
    OutcomeRateRow,
    OutcomeRatesReport,
    Posterior,
    Provenance,
    SdwpReport,
    SdwpRow,
)
from podrank.rating.predict import PredictionInput, PredictionRow
from podrank.rating.spec import ModelSpec

__all__ = [
    "AgentContrastRow",
    "AnchorUnsupported",
    "ComponentsReport",
    "ContrastResult",
    "DeckContrastRow",
    "DeckNuisanceReport",
    "EffectNotModelled",
    "EligibilityRow",
    "Field",
    "FieldContext",
    "IdentificationReport",
    "Interval",
    "ModelSpec",
    "OutcomeRateRow",
    "OutcomeRatesReport",
    "Posterior",
    "PredictionInput",
    "PredictionRow",
    "Provenance",
    "SdwpReport",
    "SdwpRow",
    "SeatContrastRow",
    "UnrankableReport",
    "fit",
]
