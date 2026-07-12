"""Literal guarded-minima interpolation reference method."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from .models import EnvelopeConfig
from .support import guarded_block_minima


def minima_spline(
    y: NDArray[np.float64],
    sampling_rate: float,
    config: EnvelopeConfig,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
    """Fit a shape-preserving curve through robust block minima.

    This mode most literally implements "approximate at the lowest points". It
    is intentionally offered as a transparent reference rather than the default:
    its answer changes with block duration and can follow physiological ECG Q/S
    troughs or sustained artifacts.
    """

    indices, values, diagnostics = guarded_block_minima(
        y,
        sampling_rate,
        config.minima_window_seconds,
        config.minima_overlap,
        config.guard_seconds,
        config.outlier_sigma,
        valid_mask=valid_mask,
    )
    if indices.size == 0:
        approximation = np.full_like(y, np.min(y))
    elif indices.size == 1:
        approximation = np.full_like(y, values[0])
    else:
        interpolation = PchipInterpolator(indices.astype(float), values, extrapolate=False)
        grid = np.arange(y.size, dtype=float)
        approximation = np.asarray(interpolation(grid), dtype=np.float64)
        approximation[: indices[0]] = values[0]
        approximation[indices[-1] + 1 :] = values[-1]

    diagnostics.update(
        {
            "algorithm": "guarded_block_minima_pchip",
            "iterations": 1,
            "converged": True,
            "interpolator": "PCHIP",
            "endpoint_policy": "constant extension from first/last support",
        }
    )
    return approximation, indices, values, diagnostics
