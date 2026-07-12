"""Public data models and validation for lower-envelope estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Method = Literal["quantile", "minima", "kalman"]
Side = Literal["lower", "upper"]
SOFTWARE_VERSION = "0.1.0"


@dataclass(frozen=True)
class EnvelopeConfig:
    """Configuration shared by the three estimators.

    Parameters use physical units where possible. ``quantile`` is interpreted in
    the transformed lower-tail domain; requesting ``side='upper'`` mirrors the
    signal internally and mirrors the result back.
    """

    method: Method = "quantile"
    side: Side = "lower"
    quantile: float = 0.05

    # Offline quantile smoother.
    smoothness_hz: float = 0.35
    max_iterations: int = 60
    tolerance: float = 3e-4
    smoothing_epsilon: float = 0.002
    edge_padding_seconds: float = 0.0

    # Literal guarded-minima spline and support-point diagnostics.
    minima_window_seconds: float = 0.80
    minima_overlap: float = 0.50
    guard_seconds: float = 0.012
    outlier_sigma: float = 4.0

    # Causal asymmetric local-linear-trend Kalman filter.
    kalman_process_variance: float = 2e-4
    kalman_measurement_variance: float = 0.08
    kalman_innovation_clip: float = 4.0
    kalman_warmup_seconds: float = 2.0

    def validate(self, sampling_rate: float | None = None) -> None:
        if self.method not in {"quantile", "minima", "kalman"}:
            raise ValueError(f"Unsupported method: {self.method!r}")
        if self.side not in {"lower", "upper"}:
            raise ValueError("side must be 'lower' or 'upper'")
        if not 0.001 <= self.quantile < 0.5:
            raise ValueError("quantile must be in [0.001, 0.5)")
        if not np.isfinite(self.smoothness_hz) or self.smoothness_hz <= 0:
            raise ValueError("smoothness_hz must be positive and finite")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0 < self.tolerance < 1:
            raise ValueError("tolerance must be between 0 and 1")
        if not 0 < self.smoothing_epsilon <= 0.5:
            raise ValueError("smoothing_epsilon must be in (0, 0.5]")
        if not np.isfinite(self.edge_padding_seconds) or self.edge_padding_seconds < 0:
            raise ValueError("edge_padding_seconds must be nonnegative and finite")
        if not np.isfinite(self.minima_window_seconds) or self.minima_window_seconds <= 0:
            raise ValueError("minima_window_seconds must be positive")
        if not 0 <= self.minima_overlap < 0.95:
            raise ValueError("minima_overlap must be in [0, 0.95)")
        if not np.isfinite(self.guard_seconds) or self.guard_seconds < 0:
            raise ValueError("guard_seconds must be nonnegative and finite")
        if not np.isfinite(self.outlier_sigma) or self.outlier_sigma <= 0:
            raise ValueError("outlier_sigma must be positive")
        if not np.isfinite(self.kalman_process_variance) or self.kalman_process_variance <= 0:
            raise ValueError("kalman_process_variance must be positive")
        if (
            not np.isfinite(self.kalman_measurement_variance)
            or self.kalman_measurement_variance <= 0
        ):
            raise ValueError("kalman_measurement_variance must be positive")
        if not np.isfinite(self.kalman_innovation_clip) or self.kalman_innovation_clip <= 0:
            raise ValueError("kalman_innovation_clip must be positive")
        if not np.isfinite(self.kalman_warmup_seconds) or self.kalman_warmup_seconds < 0:
            raise ValueError("kalman_warmup_seconds must be nonnegative and finite")
        if sampling_rate is not None:
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                raise ValueError("sampling_rate must be positive and finite")
            if self.smoothness_hz >= sampling_rate / 2:
                raise ValueError("smoothness_hz must be below the Nyquist frequency")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvelopeResult:
    """Estimator output with enough provenance for reproducible analysis."""

    approximation: NDArray[np.float64]
    residual: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    support_indices: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    support_values: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config: EnvelopeConfig = field(default_factory=EnvelopeConfig)

    def __post_init__(self) -> None:
        n = len(self.approximation)
        if len(self.residual) != n or len(self.valid_mask) != n:
            raise ValueError("approximation, residual, and valid_mask lengths must match")
        if len(self.support_indices) != len(self.support_values):
            raise ValueError("support_indices and support_values lengths must match")
