"""S6 power and cost report over the frozen cmp_effect worlds."""

from __future__ import annotations

import json
import time

from podrank.sim.compare_studies import run_s6_cell

REPLICATIONS = 1000
BLOCKS = 4000


def main() -> None:
    started = time.perf_counter()
    results = []
    for mean in (0.005, 0.01, 0.02, 0.05):
        for design in ("duplicate", "paired"):
            for rho in (0.0, 0.3, 0.6, 1.0):
                result = run_s6_cell(
                    design=design,  # type: ignore[arg-type]
                    mean=mean,
                    rho=rho,
                    replications=REPLICATIONS,
                    blocks=BLOCKS,
                    seed=4000 + len(results),
                )
                results.append(
                    {
                        "cell": result.cell.name,
                        "power": result.power,
                        "mean_blocks": result.mean_blocks,
                        "median_blocks": result.median_blocks,
                        "censored_fraction": result.censored_fraction,
                    }
                )
                print(
                    f"{result.cell.name} power={result.power:.3f} "
                    f"mean_blocks={result.mean_blocks:.1f}",
                    flush=True,
                )
    print(
        json.dumps(
            {
                "cells": len(results),
                "replications": REPLICATIONS,
                "elapsed": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
