"""Conditioning and lower-envelope estimation for uniformly sampled biosignals."""

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
    "estimate_envelope",
    "preprocess_signal",
]
__version__ = SOFTWARE_VERSION
