"""CSV loading and sampling-rate inference with explicit validation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd


def read_table(source: str | Path | BinaryIO | TextIO | bytes) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = BytesIO(source)
    try:
        frame = pd.read_csv(source, sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError):
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("input table is empty")
    return frame


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]


def infer_sampling_rate(time_values: np.ndarray) -> tuple[float, float]:
    """Return sampling rate and relative interval jitter (MAD/median)."""

    time_values = np.asarray(time_values, dtype=float)
    if time_values.ndim != 1 or time_values.size < 3 or not np.all(np.isfinite(time_values)):
        raise ValueError("time column must contain at least 3 finite values")
    intervals = np.diff(time_values)
    if np.any(intervals <= 0):
        raise ValueError("time values must be strictly increasing")
    median_interval = float(np.median(intervals))
    jitter = float(np.median(np.abs(intervals - median_interval)) / median_interval)
    return 1.0 / median_interval, jitter


def export_frame(
    time: np.ndarray,
    signal: np.ndarray,
    approximation: np.ndarray,
    residual: np.ndarray,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": time,
            "signal": signal,
            "envelope": approximation,
            "signal_minus_envelope": residual,
            "original_sample_valid": valid_mask,
        }
    )
