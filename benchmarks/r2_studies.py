"""Reduced S5/S11 boundary and lift-recovery run for the R2 core.

Run from the repository root:

    uv run --locked python benchmarks/r2_studies.py

Uses the frozen cmp_null and cmp_effect worlds; the prescribed S5/S11
replications and results are recorded in ADR 0013.
"""

from __future__ import annotations

import json
import time

from podrank.sim.compare_studies import run_s5_cell, run_s11_cell, s5_cells

REPLICATIONS = 10000
BLOCKS = 4000


def main() -> None:
    started = time.perf_counter()
    s5 = []
    violations = []
    for index, cell in enumerate(s5_cells()):
        result = run_s5_cell(cell, replications=REPLICATIONS, blocks=BLOCKS, seed=1000 + index)
        limit = result.alpha + 2 * result.mc_se
        upper_ok = result.false_above / result.replications <= limit
        lower_ok = result.false_below / result.replications <= limit
        if not (upper_ok and lower_ok):
            violations.append(
                {
                    "cell": cell.name,
                    "false_above": result.false_above,
                    "false_below": result.false_below,
                    "limit": limit,
                }
            )
        s5.append(
            {
                "cell": cell.name,
                "replications": result.replications,
                "false_above": result.false_above,
                "false_below": result.false_below,
                "alpha": result.alpha,
                "mc_se": result.mc_se,
                "limit": limit,
            }
        )

    s11 = []
    for mean in (0.005, 0.01, 0.02, 0.05):
        for design in ("duplicate", "paired"):
            for rho in (0.0, 0.3):
                result = run_s11_cell(
                    design=design,  # type: ignore[arg-type]
                    mean=mean,
                    rho=rho,
                    replications=5000,
                    blocks=BLOCKS,
                    seed=2000 + len(s11),
                    cell_index=len(s11),
                )
                s11.append(
                    {
                        "cell": result.cell.name,
                        "replications": result.replications,
                        "blocks": result.blocks,
                        "mean": result.mean,
                        "truth": result.truth,
                        "bias": result.bias,
                        "mc_se": result.mc_se,
                        "flagged": result.flagged,
                    }
                )

    print(
        json.dumps(
            {
                "s5_cells": len(s5),
                "s5_replications": REPLICATIONS,
                "blocks": BLOCKS,
                "s5_limit_violations": violations,
                "s11_cells": len(s11),
                "s11_flagged": [item for item in s11 if item["flagged"]],
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
