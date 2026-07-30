"""Deterministic ECG-like signals with explicit envelope and drift ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticArtifact:
    """Known metadata for an opt-in technical signal-quality artifact."""

    mode: str
    component: NDArray[np.float64]
    event_mask: NDArray[np.bool_]
    dropout_mask: NDArray[np.bool_]
    start_seconds: float
    end_seconds: float
    level_shift: float
    movement_noise_std: float
    dropout_start_seconds: float | None
    dropout_end_seconds: float | None


@dataclass(frozen=True)
class SyntheticECG:
    time: NDArray[np.float64]
    signal: NDArray[np.float64]
    clean_ecg: NDArray[np.float64]
    baseline_wander: NDArray[np.float64]
    true_lower_envelope: NDArray[np.float64]
    sampling_rate: float
    template_minimum: float
    artifact: SyntheticArtifact | None = None


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
    artifact_mode: str = "none",
    artifact_start_seconds: float | None = None,
    artifact_duration_seconds: float = 4.0,
    artifact_level_shift: float = -0.45,
    movement_noise_std: float = 0.12,
    dropout_duration_seconds: float = 0.0,
) -> SyntheticECG:
    """Generate an ECG-like trace and two deliberately different truths.

    ``baseline_wander`` is the additive isoelectric drift. The requested
    ``true_lower_envelope`` is that drift plus the template's recurring S-wave
    minimum. Their offset demonstrates why a lower outline is not ECG baseline.

    ``artifact_mode="sustained_contact_movement"`` adds an opt-in technical
    signal-quality event: a sustained level shift, movement-like faster noise,
    and an optional held-value dropout. It is a simulation of degraded
    acquisition quality, not a diagnosis or identification of a specific
    electrode. The clean truths remain unchanged so contamination-induced
    tracking error remains visible.
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

    artifact = None
    supported_artifact_modes = {"none", "sustained_contact_movement"}
    if artifact_mode not in supported_artifact_modes:
        supported = ", ".join(sorted(supported_artifact_modes))
        raise ValueError(f"artifact_mode must be one of: {supported}")
    if artifact_mode == "sustained_contact_movement":
        artifact_values = (
            artifact_duration_seconds,
            artifact_level_shift,
            movement_noise_std,
            dropout_duration_seconds,
        )
        if not all(np.isfinite(value) for value in artifact_values):
            raise ValueError("artifact controls must be finite")
        if artifact_duration_seconds <= 0:
            raise ValueError("artifact duration must be positive")
        if movement_noise_std < 0 or dropout_duration_seconds < 0:
            raise ValueError("movement noise and dropout duration must be nonnegative")

        event_duration = min(float(artifact_duration_seconds), float(duration_seconds))
        if artifact_start_seconds is None:
            event_start = min(4.0, max(0.0, 0.5 * (duration_seconds - event_duration)))
        else:
            if not np.isfinite(artifact_start_seconds):
                raise ValueError("artifact start must be finite")
            event_start = float(artifact_start_seconds)
        if not 0.0 <= event_start < duration_seconds:
            raise ValueError("artifact start must lie within the generated record")
        event_end = min(float(duration_seconds), event_start + event_duration)
        event_duration = event_end - event_start
        if dropout_duration_seconds > event_duration:
            raise ValueError("dropout duration cannot exceed the artifact event")

        event_mask = (time >= event_start) & (time < event_end)
        if not np.any(event_mask):
            raise ValueError("artifact event does not contain a generated sample")
        original_signal = signal.copy()

        artifact_rng = np.random.default_rng(seed + 1_000_003)
        event_time = time - event_start
        frequency_one = min(4.7, 0.16 * sampling_rate)
        frequency_two = min(10.8, 0.34 * sampling_rate)
        movement = (
            0.62 * np.sin(2.0 * np.pi * frequency_one * event_time + 0.4)
            + 0.30 * np.sin(2.0 * np.pi * frequency_two * event_time + 1.3)
            + 0.35 * artifact_rng.normal(size=sample_count)
        )
        movement *= 0.78 + 0.22 * np.sin(2.0 * np.pi * 0.55 * event_time + 0.2)
        event_movement = movement[event_mask]
        event_movement -= np.mean(event_movement)
        movement_scale = float(np.std(event_movement))
        if movement_scale > 0.0:
            event_movement *= movement_noise_std / movement_scale
        else:
            event_movement.fill(0.0)
        signal[event_mask] += artifact_level_shift + event_movement

        dropout_mask = np.zeros(sample_count, dtype=bool)
        dropout_start: float | None = None
        dropout_end: float | None = None
        if dropout_duration_seconds > 0.0:
            # End the degraded-acquisition event with the held-value dropout so
            # its next sample is the first clean sample. This makes estimator
            # reacquisition after a known invalid interval unambiguous.
            dropout_start = event_end - dropout_duration_seconds
            dropout_end = event_end
            dropout_mask = (time >= dropout_start) & (time < dropout_end) & event_mask
            dropout_indices = np.flatnonzero(dropout_mask)
            if dropout_indices.size:
                held_level = float(baseline[dropout_indices[0]] + artifact_level_shift)
                signal[dropout_mask] = held_level

        artifact = SyntheticArtifact(
            mode=artifact_mode,
            component=np.asarray(signal - original_signal, dtype=np.float64),
            event_mask=event_mask,
            dropout_mask=dropout_mask,
            start_seconds=event_start,
            end_seconds=event_end,
            level_shift=float(artifact_level_shift),
            movement_noise_std=float(movement_noise_std),
            dropout_start_seconds=dropout_start,
            dropout_end_seconds=dropout_end,
        )

    return SyntheticECG(
        time=time,
        signal=signal,
        clean_ecg=clean_ecg,
        baseline_wander=baseline,
        true_lower_envelope=true_lower,
        sampling_rate=float(sampling_rate),
        template_minimum=template_minimum,
        artifact=artifact,
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
