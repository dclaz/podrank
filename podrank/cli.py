"""Small R1 CLI: version and a diagnostics report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from podrank._version import __version__
from podrank.data import GameTable
from podrank.rating import fit
from podrank.report import calibration_markdown
from podrank.sim import reference_world


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="podrank")
    parser.add_argument("--version", action="version", version=f"podrank {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    calibrate = subparsers.add_parser("calibrate", help="report decisive-game calibration")
    calibrate.add_argument("--world", default="doc_tiny")
    arguments = parser.parse_args(argv)
    if arguments.command == "calibrate":
        world = reference_world(arguments.world)
        post = fit(
            GameTable.from_records(world.game_rows()),
            world.agent_registry(),
            world.deck_registry(),
        )
        from podrank.diagnostics import predictive_quality

        print(calibration_markdown(predictive_quality(post, world.game_rows())))
        return 0
    parser.print_help()
    return 0


__all__ = ["main"]
