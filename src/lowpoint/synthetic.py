"""Deterministic ECG-like signals with explicit envelope and drift ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticECG:
    time: NDArray[np.float64]
    signal: NDArray[np.float64]
    clean_ecg: NDArray[np.float64]
    baseline_wander: NDArray[np.float64]
    true_lower_envelope: NDArray[np.float64]
    sampling_rate: float
    template_minimum: float


def _periodic_distance(phase: NDArray[np.float64], center: float) -> NDArray[np.float64]:
    return (phase - center + 0.5) % 1.0 - 0.5


def _wave(
    phase: NDArray[np.float64], center: float, width: float, amplitude: float
) -> NDArray[np.float64]:
    distance = _periodic_distance(phase, center)
    return amplitude * np.exp(-0.5 * (distance / width) ** 2)


def ecg_template(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytical P-Q-R-S-T morphology over phase in [0, 1)."""

    return (
        _wave(phase, 0.18, 0.025, 0.12)
        + _wave(phase, 0.370, 0.010, -0.16)
        + _wave(phase, 0.400, 0.012, 1.05)
        + _wave(phase, 0.435, 0.014, -0.28)
        + _wave(phase, 0.68, 0.055, 0.30)
    )


def generate_ecg(
    duration_seconds: float = 12.0,
    sampling_rate: float = 250.0,
    heart_rate_bpm: float = 72.0,
    noise_std: float = 0.015,
    wander_amplitude: float = 0.18,
    negative_spikes: int = 2,
    seed: int = 7,
) -> SyntheticECG:
    """Generate an ECG-like trace and two deliberately different truths.

    ``baseline_wander`` is the additive isoelectric drift. The requested
    ``true_lower_envelope`` is that drift plus the template's recurring S-wave
    minimum. Their offset demonstrates why a lower outline is not ECG baseline.
    """

    if duration_seconds <= 0 or sampling_rate <= 0 or heart_rate_bpm <= 0:
        raise ValueError("duration, sampling rate, and heart rate must be positive")
    sample_count = max(5, int(round(duration_seconds * sampling_rate)))
    time = np.arange(sample_count, dtype=np.float64) / sampling_rate
    period = 60.0 / heart_rate_bpm
    phase = (time / period) % 1.0
    clean_ecg = ecg_template(phase)
    baseline = wander_amplitude * (
        0.72 * np.sin(2.0 * np.pi * 0.23 * time + 0.25)
        + 0.28 * np.sin(2.0 * np.pi * 0.07 * time - 0.8)
    )

    dense_phase = np.linspace(0.0, 1.0, 20_001, endpoint=False)
    template_minimum = float(np.min(ecg_template(dense_phase)))
    true_lower = baseline + template_minimum

    rng = np.random.default_rng(seed)
    signal = baseline + clean_ecg + rng.normal(0.0, noise_std, size=sample_count)
    if negative_spikes > 0 and sample_count > 20:
        candidates = np.arange(10, sample_count - 10)
        selected = rng.choice(candidates, size=min(negative_spikes, candidates.size), replace=False)
        signal[selected] -= 0.55

    return SyntheticECG(
        time=time,
        signal=signal,
        clean_ecg=clean_ecg,
        baseline_wander=baseline,
        true_lower_envelope=true_lower,
        sampling_rate=float(sampling_rate),
        template_minimum=template_minimum,
    )


def generate_positive_biosignal(
    duration_seconds: float = 20.0,
    sampling_rate: float = 100.0,
    noise_std: float = 0.02,
    seed: int = 11,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generic nonnegative oscillations riding on a known lower trend."""

    time = np.arange(int(round(duration_seconds * sampling_rate))) / sampling_rate
    lower = 0.25 * np.sin(2 * np.pi * 0.08 * time) + 0.015 * time
    carrier = 0.55 * (1.0 + np.sin(2 * np.pi * 1.2 * time + 0.2))
    modulation = 0.75 + 0.2 * np.sin(2 * np.pi * 0.03 * time)
    rng = np.random.default_rng(seed)
    signal = lower + carrier * modulation + rng.normal(0.0, noise_std, size=time.size)
    return time.astype(float), signal.astype(float), lower.astype(float)
