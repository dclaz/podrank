"""Model configuration for R1."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Frozen rating-model settings from Appendix C.1."""

    swap_effect: bool = False
    allow_pooled_variants: bool = False
    weak_ratio: float = 1.0
    tol_eig: float = 1e-10
    tol_alias: float = 1e-6
    max_params: int = 10_000
    max_dense_bytes: int = 2 * 2**30

    def __post_init__(self) -> None:
        if not isinstance(self.swap_effect, bool):
            raise ValueError("swap_effect must be a boolean")
        if not isinstance(self.allow_pooled_variants, bool):
            raise ValueError("allow_pooled_variants must be a boolean")
        if not math.isfinite(self.weak_ratio) or self.weak_ratio <= 0:
            raise ValueError("weak_ratio must be finite and > 0")
        if not math.isfinite(self.tol_eig) or not 0 < self.tol_eig < 1:
            raise ValueError("tol_eig must satisfy 0 < tol_eig < 1")
        if not math.isfinite(self.tol_alias) or not 0 <= self.tol_alias < 1:
            raise ValueError("tol_alias must satisfy 0 <= tol_alias < 1")
        if isinstance(self.max_params, bool) or not isinstance(self.max_params, int):
            raise ValueError("max_params must be a positive integer")
        if self.max_params <= 0:
            raise ValueError("max_params must be a positive integer")
        if isinstance(self.max_dense_bytes, bool) or not isinstance(self.max_dense_bytes, int):
            raise ValueError("max_dense_bytes must be a positive integer")
        if self.max_dense_bytes <= 0:
            raise ValueError("max_dense_bytes must be a positive integer")


__all__ = ["ModelSpec"]
