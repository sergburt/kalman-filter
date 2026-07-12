"""Explicit, conservative signal preparation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def prepare_signal(signal: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return a finite 1-D float signal and the original finite-sample mask.

    Interior gaps are linearly interpolated and leading/trailing gaps use the
    nearest finite value. The returned mask makes this transformation visible to
    callers; no missing sample is silently presented as originally observed.
    """

    y = np.asarray(signal, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if y.size < 5:
        raise ValueError("signal must contain at least 5 samples")
    valid = np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        raise ValueError("signal must contain at least 3 finite samples")
    if not np.all(valid):
        index = np.arange(y.size)
        y = np.interp(index, index[valid], y[valid]).astype(np.float64, copy=False)
    return y, valid


def robust_location_scale(y: NDArray[np.float64]) -> tuple[float, float]:
    """Median and a robust, strictly positive scale estimate."""

    location = float(np.median(y))
    mad = float(np.median(np.abs(y - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        q25, q75 = np.percentile(y, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = float(np.std(y))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = 1.0
    return location, scale


def odd_sample_count(seconds: float, sampling_rate: float, minimum: int = 1) -> int:
    count = max(minimum, int(round(seconds * sampling_rate)))
    return count if count % 2 == 1 else count + 1
