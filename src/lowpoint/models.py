"""Public data models and validation for lower-envelope estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Method = Literal["quantile", "minima", "kalman"]
Side = Literal["lower", "upper"]
FilterPhase = Literal["zero_phase", "causal"]
BaselineMethod = Literal["none", "median", "highpass"]
SOFTWARE_VERSION = "0.1.0"


@dataclass(frozen=True)
class SignalFilterConfig:
    """Optional conditioning applied before lower-envelope estimation.

    All stages are disabled by default. Filter order is the per-pass design
    order; zero-phase forward/backward processing doubles the effective order.
    """

    phase_mode: FilterPhase = "zero_phase"

    mains_enabled: bool = False
    mains_frequency_hz: float = 50.0
    mains_quality_factor: float = 30.0
    mains_harmonics: int = 1

    highpass_enabled: bool = False
    highpass_cutoff_hz: float = 0.5
    highpass_order: int = 2

    lowpass_enabled: bool = False
    lowpass_cutoff_hz: float = 40.0
    lowpass_order: int = 4

    baseline_method: BaselineMethod = "none"
    baseline_window_seconds: float = 0.8
    baseline_cutoff_hz: float = 0.5
    baseline_order: int = 2

    @property
    def enabled(self) -> bool:
        return bool(
            self.mains_enabled
            or self.highpass_enabled
            or self.lowpass_enabled
            or self.baseline_method != "none"
        )

    def validate(self, sampling_rate: float) -> None:
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        if self.phase_mode not in {"zero_phase", "causal"}:
            raise ValueError("phase_mode must be 'zero_phase' or 'causal'")
        if self.baseline_method not in {"none", "median", "highpass"}:
            raise ValueError("baseline_method must be 'none', 'median', or 'highpass'")

        nyquist = sampling_rate / 2.0
        if self.mains_enabled:
            _validate_frequency(self.mains_frequency_hz, nyquist, "mains_frequency_hz")
            if not np.isfinite(self.mains_quality_factor) or self.mains_quality_factor <= 0:
                raise ValueError("mains_quality_factor must be positive and finite")
            _validate_integer(self.mains_harmonics, "mains_harmonics", 1, 20)

        if self.highpass_enabled:
            _validate_frequency(self.highpass_cutoff_hz, nyquist, "highpass_cutoff_hz")
            _validate_integer(self.highpass_order, "highpass_order", 1, 12)
        if self.lowpass_enabled:
            _validate_frequency(self.lowpass_cutoff_hz, nyquist, "lowpass_cutoff_hz")
            _validate_integer(self.lowpass_order, "lowpass_order", 1, 12)
        if (
            self.highpass_enabled
            and self.lowpass_enabled
            and self.highpass_cutoff_hz >= self.lowpass_cutoff_hz
        ):
            raise ValueError("highpass_cutoff_hz must be below lowpass_cutoff_hz")

        if self.baseline_method == "median":
            if not np.isfinite(self.baseline_window_seconds) or self.baseline_window_seconds <= 0:
                raise ValueError("baseline_window_seconds must be positive and finite")
        elif self.baseline_method == "highpass":
            _validate_frequency(self.baseline_cutoff_hz, nyquist, "baseline_cutoff_hz")
            _validate_integer(self.baseline_order, "baseline_order", 1, 12)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_frequency(value: float, nyquist: float, name: str) -> None:
    if not np.isfinite(value) or not 0 < value < nyquist:
        raise ValueError(f"{name} must be finite and strictly between 0 and Nyquist")


def _validate_integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")


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


@dataclass
class SignalFilterResult:
    """Conditioning output with raw/prepared/processed provenance."""

    raw_signal: NDArray[np.float64]
    prepared_signal: NDArray[np.float64]
    processed_signal: NDArray[np.float64]
    baseline_estimate: NDArray[np.float64]
    removed_component: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    config: SignalFilterConfig = field(default_factory=SignalFilterConfig)

    def __post_init__(self) -> None:
        n = len(self.raw_signal)
        arrays = (
            self.prepared_signal,
            self.processed_signal,
            self.baseline_estimate,
            self.removed_component,
            self.valid_mask,
        )
        if any(len(array) != n for array in arrays):
            raise ValueError("all signal-filter result arrays must have matching lengths")
