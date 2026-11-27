"""R2 comparison surfaces: pods, descriptors, block scoring and evidence.

The comparison state machine is added incrementally. Importing this package
does not import or change any R1 rating code.
"""

from __future__ import annotations

from podrank.compare.betting import (
    Hypothesis,
    NonInferiority,
    Superiority,
    Thresholds,
    Wealth,
    thresholds,
)
from podrank.compare.blocks import (
    BlockScores,
    DescriptiveSummary,
    ScoredBlockInput,
    focal_points,
    score_block,
    summarise_blocks,
)
from podrank.compare.comparison import (
    Comparison,
    ComparisonReport,
    EventRecord,
    ExportReport,
    FailureRecord,
    IngestIssue,
    IngestReport,
    PairingReport,
    Status,
)
from podrank.compare.cs import ConfidenceSequence, confidence_sequence, grid_points
from podrank.compare.descriptors import BlockDescriptor, Position, expected_positions
from podrank.compare.pods import Design, PodDistribution, PodDraw, PodSupport

__all__ = [
    "BlockDescriptor",
    "BlockScores",
    "Comparison",
    "ConfidenceSequence",
    "ComparisonReport",
    "DescriptiveSummary",
    "EventRecord",
    "ExportReport",
    "FailureRecord",
    "Design",
    "Hypothesis",
    "IngestIssue",
    "IngestReport",
    "NonInferiority",
    "PairingReport",
    "PodDistribution",
    "PodDraw",
    "PodSupport",
    "Position",
    "ScoredBlockInput",
    "Status",
    "Superiority",
    "Thresholds",
    "Wealth",
    "confidence_sequence",
    "expected_positions",
    "grid_points",
    "focal_points",
    "score_block",
    "summarise_blocks",
    "thresholds",
]
