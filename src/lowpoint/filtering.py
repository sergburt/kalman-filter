"""Explicit, disabled-by-default biosignal conditioning filters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from .models import SignalFilterConfig, SignalFilterResult
from .preprocessing import odd_sample_count, prepare_signal


def _prepare_for_mode(
    signal: ArrayLike, phase_mode: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], str]:
    raw = np.asarray(signal, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if raw.size < 5:
        raise ValueError("signal must contain at least 5 samples")
    raw = raw.copy()
    valid = np.isfinite(raw)
    if np.count_nonzero(valid) < 3:
        raise ValueError("signal must contain at least 3 finite samples")

    if phase_mode == "causal":
        if not valid[0]:
            raise ValueError(
                "causal conditioning requires the first sample to be finite; "
                "a leading gap cannot be filled without future information"
            )
        last_valid = np.maximum.accumulate(np.where(valid, np.arange(raw.size), 0))
        prepared = raw[last_valid]
        gap_policy = "causal forward-fill"
    else:
        prepared, _prepared_mask = prepare_signal(raw)
        gap_policy = "offline linear interpolation"
    return raw, np.asarray(prepared, dtype=np.float64), valid, gap_policy


def _default_sos_padlen(sos: NDArray[np.float64]) -> int:
    zero_count = min(int(np.sum(sos[:, 2] == 0.0)), int(np.sum(sos[:, 5] == 0.0)))
    return 3 * (2 * len(sos) + 1 - zero_count)


def _apply_sos(
    signal: NDArray[np.float64],
    sos: NDArray[np.float64],
    phase_mode: str,
    stage_name: str,
) -> NDArray[np.float64]:
    if phase_mode == "zero_phase":
        required_pad = _default_sos_padlen(sos)
        if signal.size <= required_pad:
            raise ValueError(
                f"{stage_name} zero-phase filtering requires more than "
                f"{required_pad} samples; received {signal.size}"
            )
        filtered = sosfiltfilt(sos, signal)
    else:
        initial_state = sosfilt_zi(sos) * signal[0]
        filtered, _final_state = sosfilt(sos, signal, zi=initial_state)
    filtered = np.asarray(filtered, dtype=np.float64)
    if not np.all(np.isfinite(filtered)):
        raise RuntimeError(f"{stage_name} produced non-finite values")
    return filtered


def _rms(signal: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(signal * signal)))


def _short_record_warning(
    warnings: list[str], sample_count: int, sampling_rate: float, cutoff_hz: float, label: str
) -> None:
    duration = sample_count / sampling_rate
    cycles = duration * cutoff_hz
    if cycles < 3.0:
        warnings.append(
            f"{label} has only {cycles:.2f} cutoff-frequency cycles in this record; "
            "edge/transient behavior may dominate"
        )


def preprocess_signal(
    signal: ArrayLike,
    sampling_rate: float,
    config: SignalFilterConfig | None = None,
) -> SignalFilterResult:
    """Condition a biosignal while preserving raw samples and gap provenance.

    Processing order is mains notch, baseline removal, general high-pass, then
    general low-pass. The returned processed signal is the coordinate system in
    which a subsequent envelope and residual must be interpreted.
    """

    config = config or SignalFilterConfig()
    config.validate(sampling_rate)
    raw, prepared, valid_mask, gap_policy = _prepare_for_mode(signal, config.phase_mode)
    work = prepared.copy()
    baseline_estimate = np.zeros_like(work)
    stages: list[dict[str, Any]] = []
    warnings: list[str] = []
    nyquist = sampling_rate / 2.0

    if config.mains_enabled:
        applied_frequencies: list[float] = []
        skipped_frequencies: list[float] = []
        for harmonic in range(1, config.mains_harmonics + 1):
            frequency = config.mains_frequency_hz * harmonic
            if frequency >= nyquist:
                skipped_frequencies.append(float(frequency))
                continue
            numerator, denominator = iirnotch(
                frequency, config.mains_quality_factor, fs=sampling_rate
            )
            sos = tf2sos(numerator, denominator)
            work = _apply_sos(
                work,
                np.asarray(sos, dtype=np.float64),
                config.phase_mode,
                f"mains notch at {frequency:g} Hz",
            )
            applied_frequencies.append(float(frequency))
        if skipped_frequencies:
            warnings.append(
                "Skipped mains harmonics at/above Nyquist: "
                + ", ".join(f"{frequency:g} Hz" for frequency in skipped_frequencies)
            )
        stages.append(
            {
                "name": "mains_notch",
                "frequencies_hz": applied_frequencies,
                "skipped_frequencies_hz": skipped_frequencies,
                "quality_factor": float(config.mains_quality_factor),
            }
        )

    if config.baseline_method == "median":
        requested_samples = int(round(config.baseline_window_seconds * sampling_rate))
        if requested_samples < 3:
            raise ValueError("median baseline window must contain at least 3 samples")
        window_samples = odd_sample_count(config.baseline_window_seconds, sampling_rate, minimum=3)
        if window_samples > work.size:
            raise ValueError(
                "median baseline window cannot be longer than the signal "
                f"({window_samples} > {work.size} samples)"
            )
        if config.phase_mode == "causal":
            baseline_estimate = (
                pd.Series(work).rolling(window_samples, min_periods=1).median().to_numpy()
            )
            median_alignment = "trailing"
        else:
            baseline_estimate = median_filter(work, size=window_samples, mode="reflect")
            median_alignment = "centered"
        work = work - baseline_estimate
        stages.append(
            {
                "name": "median_baseline_removal",
                "window_samples": int(window_samples),
                "effective_window_seconds": float(window_samples / sampling_rate),
                "alignment": median_alignment,
            }
        )
    elif config.baseline_method == "highpass":
        if config.highpass_enabled:
            warnings.append(
                "Baseline high-pass and general high-pass are both enabled; "
                "their attenuation will compound"
            )
        _short_record_warning(
            warnings,
            work.size,
            sampling_rate,
            config.baseline_cutoff_hz,
            "Baseline high-pass",
        )
        baseline_input = work.copy()
        sos = butter(
            config.baseline_order,
            config.baseline_cutoff_hz,
            btype="highpass",
            fs=sampling_rate,
            output="sos",
        )
        work = _apply_sos(
            work,
            np.asarray(sos, dtype=np.float64),
            config.phase_mode,
            "baseline high-pass",
        )
        baseline_estimate = baseline_input - work
        stages.append(
            {
                "name": "highpass_baseline_removal",
                "cutoff_hz": float(config.baseline_cutoff_hz),
                "design_order_per_pass": int(config.baseline_order),
            }
        )

    if config.highpass_enabled:
        _short_record_warning(
            warnings,
            work.size,
            sampling_rate,
            config.highpass_cutoff_hz,
            "General high-pass",
        )
        sos = butter(
            config.highpass_order,
            config.highpass_cutoff_hz,
            btype="highpass",
            fs=sampling_rate,
            output="sos",
        )
        work = _apply_sos(
            work,
            np.asarray(sos, dtype=np.float64),
            config.phase_mode,
            "general high-pass",
        )
        stages.append(
            {
                "name": "highpass",
                "cutoff_hz": float(config.highpass_cutoff_hz),
                "design_order_per_pass": int(config.highpass_order),
            }
        )

    if config.lowpass_enabled:
        sos = butter(
            config.lowpass_order,
            config.lowpass_cutoff_hz,
            btype="lowpass",
            fs=sampling_rate,
            output="sos",
        )
        work = _apply_sos(
            work,
            np.asarray(sos, dtype=np.float64),
            config.phase_mode,
            "general low-pass",
        )
        stages.append(
            {
                "name": "lowpass",
                "cutoff_hz": float(config.lowpass_cutoff_hz),
                "design_order_per_pass": int(config.lowpass_order),
            }
        )

    removed_component = prepared - work
    zero_phase_active = config.phase_mode == "zero_phase" and bool(stages)
    has_missing_samples = bool(np.any(~valid_mask))
    causal_conditioning = config.phase_mode == "causal" or (not stages and not has_missing_samples)
    diagnostics: dict[str, Any] = {
        "filtering_active": bool(stages),
        "phase_mode": config.phase_mode,
        "causal_conditioning": causal_conditioning,
        "zero_phase_forward_backward": zero_phase_active,
        "effective_order_note": (
            "zero-phase forward/backward filtering doubles each IIR stage's effective order"
            if zero_phase_active
            else "configured IIR orders are single-pass design orders"
        ),
        "gap_policy": gap_policy,
        "interpolated_or_filled_samples": int(valid_mask.size - np.count_nonzero(valid_mask)),
        "stages": stages,
        "warnings": warnings,
        "raw_prepared_rms": _rms(prepared),
        "processed_rms": _rms(work),
        "removed_component_rms": _rms(removed_component),
        "baseline_estimate_rms": _rms(baseline_estimate),
        "target_for_envelope": "processed_signal",
    }
    return SignalFilterResult(
        raw_signal=raw,
        prepared_signal=prepared,
        processed_signal=np.asarray(work, dtype=np.float64),
        baseline_estimate=np.asarray(baseline_estimate, dtype=np.float64),
        removed_component=np.asarray(removed_component, dtype=np.float64),
        valid_mask=valid_mask,
        diagnostics=diagnostics,
        config=config,
    )
