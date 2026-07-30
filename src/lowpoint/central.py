"""Central-trend comparisons for conditioned biosignals."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .kalman import KalmanStepTrace, asymmetric_kalman
from .models import EnvelopeConfig
from .preprocessing import odd_sample_count, prepare_signal


def _prepare_with_validity(
    signal: ArrayLike,
    valid_mask: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    prepared, finite_mask = prepare_signal(signal)
    if valid_mask is None:
        effective_mask = finite_mask
    else:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.ndim != 1 or supplied_mask.size != prepared.size:
            raise ValueError("valid_mask must be one-dimensional and match the signal length")
        effective_mask = finite_mask & supplied_mask
    if not np.any(effective_mask):
        raise ValueError("at least one sample must be valid for a central approximation")
    return prepared, effective_mask


def symmetric_kalman_trend(
    signal: ArrayLike,
    sampling_rate: float,
    config: EnvelopeConfig,
    *,
    valid_mask: ArrayLike | None = None,
    trace_out: dict[str, KalmanStepTrace] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return a central local-linear Kalman trend with equal deviation weights.

    The process, measurement, clipping, initialization, and known-gap handling
    settings are shared with the asymmetric Kalman estimator. Only the
    measurement asymmetry is changed: positive and negative innovations both
    receive weight 0.5.
    """

    config.validate(sampling_rate)
    prepared, effective_mask = _prepare_with_validity(signal, valid_mask)
    symmetric_config = replace(config, method="kalman", side="lower", quantile=0.5)
    approximation, _indices, _values, diagnostics = asymmetric_kalman(
        prepared,
        sampling_rate,
        symmetric_config,
        valid_mask=effective_mask,
        trace_out=trace_out,
    )
    diagnostics.update(
        {
            "algorithm": "symmetric_local_linear_kalman",
            "target_type": "central trend with equal positive/negative innovation weights",
            "innovation_weight_below": 0.5,
            "innovation_weight_above": 0.5,
            "predictive": bool(diagnostics["causal"]),
            "clinical_baseline": False,
        }
    )
    return np.asarray(approximation, dtype=np.float64), diagnostics


def centered_rolling_median(
    signal: ArrayLike,
    sampling_rate: float,
    window_seconds: float,
    *,
    valid_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return an offline centered rolling median that ignores invalid samples."""

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive and finite")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    prepared, effective_mask = _prepare_with_validity(signal, valid_mask)
    window_samples = min(
        prepared.size,
        odd_sample_count(window_seconds, sampling_rate, minimum=1),
    )
    if window_samples > 1 and window_samples % 2 == 0:
        window_samples -= 1
    values = prepared.copy()
    values[~effective_mask] = np.nan
    approximation = (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )
    finite_approximation = np.isfinite(approximation)
    filled_samples = int(np.count_nonzero(~finite_approximation))
    if filled_samples:
        index = np.arange(approximation.size)
        approximation = np.interp(
            index,
            index[finite_approximation],
            approximation[finite_approximation],
        )
    diagnostics: dict[str, Any] = {
        "algorithm": "centered_rolling_median",
        "window_seconds": float(window_seconds),
        "window_samples": int(window_samples),
        "valid_sample_count": int(np.count_nonzero(effective_mask)),
        "excluded_sample_count": int(effective_mask.size - np.count_nonzero(effective_mask)),
        "filled_trend_samples": filled_samples,
        "centered": True,
        "causal": False,
        "predictive": False,
        "clinical_baseline": False,
    }
    return np.asarray(approximation, dtype=np.float64), diagnostics
