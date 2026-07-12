"""Metrics that make lower-envelope behavior inspectable."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def pinball_loss(
    signal: NDArray[np.float64], approximation: NDArray[np.float64], quantile: float
) -> float:
    residual = signal - approximation
    loss = np.where(residual >= 0.0, quantile * residual, (quantile - 1.0) * residual)
    return float(np.mean(loss))


def envelope_metrics(
    signal: NDArray[np.float64],
    approximation: NDArray[np.float64],
    sampling_rate: float,
    quantile: float,
    side: str,
    valid_mask: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Calculate coverage, fidelity, and smoothness diagnostics."""

    mask = np.ones(signal.size, dtype=bool) if valid_mask is None else valid_mask
    observed = signal[mask]
    fitted = approximation[mask]
    if side == "lower":
        tail_fraction = float(np.mean(observed < fitted))
        oriented_signal = observed
        oriented_fit = fitted
    else:
        tail_fraction = float(np.mean(observed > fitted))
        oriented_signal = -observed
        oriented_fit = -fitted

    second_derivative = np.diff(approximation, n=2) * sampling_rate**2
    roughness_rms = (
        float(np.sqrt(np.mean(second_derivative * second_derivative)))
        if second_derivative.size
        else 0.0
    )
    return {
        "sample_count": int(signal.size),
        "valid_sample_count": int(np.count_nonzero(mask)),
        "interpolated_sample_count": int(mask.size - np.count_nonzero(mask)),
        "empirical_tail_fraction": tail_fraction,
        "target_tail_fraction": float(quantile),
        "coverage_error": abs(tail_fraction - quantile),
        "pinball_loss": pinball_loss(oriented_signal, oriented_fit, quantile),
        "roughness_rms_per_second2": roughness_rms,
        "residual_median": float(np.median(signal - approximation)),
        "residual_mad": float(
            np.median(np.abs((signal - approximation) - np.median(signal - approximation)))
        ),
    }


def truth_metrics(
    approximation: NDArray[np.float64], truth: NDArray[np.float64]
) -> dict[str, float]:
    error = approximation - truth
    return {
        "truth_rmse": float(np.sqrt(np.mean(error * error))),
        "truth_mae": float(np.mean(np.abs(error))),
        "truth_bias": float(np.mean(error)),
        "truth_max_abs_error": float(np.max(np.abs(error))),
    }
