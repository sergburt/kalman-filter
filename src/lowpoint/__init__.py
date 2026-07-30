"""Conditioning and lower-envelope estimation for uniformly sampled biosignals."""

from .central import centered_rolling_median, symmetric_kalman_trend
from .filtering import preprocess_signal
from .models import (
    SOFTWARE_VERSION,
    EnvelopeConfig,
    EnvelopeResult,
    SignalFilterConfig,
    SignalFilterResult,
)
from .pipeline import estimate_envelope

__all__ = [
    "EnvelopeConfig",
    "EnvelopeResult",
    "SignalFilterConfig",
    "SignalFilterResult",
    "centered_rolling_median",
    "estimate_envelope",
    "preprocess_signal",
    "symmetric_kalman_trend",
]
__version__ = SOFTWARE_VERSION
