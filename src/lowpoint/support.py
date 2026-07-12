"""Robust low-point extraction used by the literal spline and diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from .preprocessing import odd_sample_count


def guarded_block_minima(
    y: NDArray[np.float64],
    sampling_rate: float,
    window_seconds: float,
    overlap: float,
    guard_seconds: float,
    outlier_sigma: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], dict[str, float | int]]:
    """Extract one robust minimum from each overlapping block.

    A short median guard removes isolated one/few-sample negative impulses before
    minima are selected. A rolling MAD rule then rejects amplitude-inconsistent
    anchors. This deliberately estimates *observed troughs*, not an ECG
    isoelectric baseline.
    """

    n = y.size
    guard = odd_sample_count(guard_seconds, sampling_rate, minimum=1)
    if guard > 1:
        guarded = median_filter(y, size=guard, mode="reflect")
    else:
        guarded = y.copy()

    window = max(3, int(round(window_seconds * sampling_rate)))
    window = min(window, n)
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, max(1, n - window + 1), stride))
    last_start = max(0, n - window)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    candidates: dict[int, float] = {}
    for start in starts:
        stop = min(n, start + window)
        local = int(np.argmin(guarded[start:stop])) + start
        candidates[local] = float(guarded[local])

    indices = np.asarray(sorted(candidates), dtype=np.int64)
    values = np.asarray([candidates[int(i)] for i in indices], dtype=np.float64)
    raw_count = int(indices.size)

    # Reject isolated anchor-amplitude excursions while retaining genuine drift.
    if indices.size >= 5:
        trend_width = min(7, indices.size if indices.size % 2 else indices.size - 1)
        trend = median_filter(values, size=trend_width, mode="nearest")
        deviation = values - trend
        local_mad = median_filter(np.abs(deviation), size=trend_width, mode="nearest")
        global_mad = float(np.median(np.abs(deviation - np.median(deviation))))
        floor = max(1.4826 * global_mad, np.finfo(np.float64).eps)
        scale = np.maximum(1.4826 * local_mad, floor)
        keep = np.abs(deviation) <= outlier_sigma * scale
        if np.count_nonzero(keep) >= 3:
            indices = indices[keep]
            values = values[keep]

    diagnostics: dict[str, float | int] = {
        "guard_samples": int(guard),
        "window_samples": int(window),
        "stride_samples": int(stride),
        "raw_support_count": raw_count,
        "rejected_support_count": raw_count - int(indices.size),
    }
    return indices, values, diagnostics
