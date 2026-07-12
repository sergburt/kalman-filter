"""Lower-envelope estimation for ECG and other uniformly sampled biosignals."""

from .models import SOFTWARE_VERSION, EnvelopeConfig, EnvelopeResult
from .pipeline import estimate_envelope

__all__ = ["EnvelopeConfig", "EnvelopeResult", "estimate_envelope"]
__version__ = SOFTWARE_VERSION
