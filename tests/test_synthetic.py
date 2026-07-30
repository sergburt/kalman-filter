import numpy as np

from lowpoint import EnvelopeConfig, estimate_envelope
from lowpoint.synthetic import generate_ecg


def test_default_ecg_is_unchanged_when_artifact_mode_is_explicitly_disabled() -> None:
    default = generate_ecg()
    explicit_default = generate_ecg(
        artifact_mode="none",
        artifact_start_seconds=3.0,
        artifact_duration_seconds=5.0,
        artifact_level_shift=-0.7,
        movement_noise_std=0.25,
        dropout_duration_seconds=0.5,
    )

    np.testing.assert_array_equal(default.signal, explicit_default.signal)
    np.testing.assert_array_equal(default.true_lower_envelope, explicit_default.true_lower_envelope)
    assert default.artifact is None
    assert explicit_default.artifact is None


def test_sustained_contact_movement_event_has_shift_noise_and_optional_dropout() -> None:
    common = {
        "duration_seconds": 12.0,
        "sampling_rate": 250.0,
        "noise_std": 0.0,
        "negative_spikes": 0,
        "seed": 31,
    }
    clean = generate_ecg(**common)
    degraded = generate_ecg(
        **common,
        artifact_mode="sustained_contact_movement",
        artifact_start_seconds=4.0,
        artifact_duration_seconds=4.0,
        artifact_level_shift=-0.5,
        movement_noise_std=0.14,
        dropout_duration_seconds=0.4,
    )

    artifact = degraded.artifact
    assert artifact is not None
    outside_event = ~artifact.event_mask
    np.testing.assert_array_equal(degraded.signal[outside_event], clean.signal[outside_event])
    np.testing.assert_array_equal(degraded.true_lower_envelope, clean.true_lower_envelope)

    moving_samples = artifact.event_mask & ~artifact.dropout_mask
    difference = degraded.signal[moving_samples] - clean.signal[moving_samples]
    assert np.median(difference) < -0.45
    assert np.std(difference) > 0.10

    dropout_values = degraded.signal[artifact.dropout_mask]
    assert abs(dropout_values.size - 100) <= 1
    np.testing.assert_allclose(dropout_values, dropout_values[0])
    assert artifact.dropout_end_seconds == artifact.end_seconds


def test_sustained_event_visibly_moves_kalman_envelope_and_residual() -> None:
    common = {
        "duration_seconds": 12.0,
        "sampling_rate": 250.0,
        "noise_std": 0.015,
        "negative_spikes": 2,
        "seed": 7,
    }
    clean = generate_ecg(**common)
    degraded = generate_ecg(
        **common,
        artifact_mode="sustained_contact_movement",
        artifact_start_seconds=4.0,
        artifact_duration_seconds=4.0,
        artifact_level_shift=-0.55,
        movement_noise_std=0.14,
    )
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=2.0)
    clean_result = estimate_envelope(clean.signal, clean.sampling_rate, config)
    degraded_result = estimate_envelope(degraded.signal, degraded.sampling_rate, config)

    late_event = (degraded.time >= 6.0) & (degraded.time < 8.0)
    envelope_change = degraded_result.approximation - clean_result.approximation
    assert np.median(envelope_change[late_event]) < -0.20
    assert np.std(degraded_result.residual[late_event]) > np.std(
        clean_result.residual[late_event]
    )


def test_known_dropout_exclusion_reacquires_clean_ecg_after_signal_returns() -> None:
    common = {
        "duration_seconds": 12.0,
        "sampling_rate": 250.0,
        "noise_std": 0.015,
        "negative_spikes": 2,
        "seed": 7,
    }
    clean = generate_ecg(**common)
    degraded = generate_ecg(
        **common,
        artifact_mode="sustained_contact_movement",
        artifact_start_seconds=4.0,
        artifact_duration_seconds=4.0,
        artifact_level_shift=-0.55,
        movement_noise_std=0.14,
        dropout_duration_seconds=0.4,
    )
    artifact = degraded.artifact
    assert artifact is not None
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=2.0)
    clean_result = estimate_envelope(clean.signal, clean.sampling_rate, config)
    ungated = estimate_envelope(degraded.signal, degraded.sampling_rate, config)
    gated = estimate_envelope(
        degraded.signal,
        degraded.sampling_rate,
        config,
        valid_mask=~artifact.dropout_mask,
    )

    after_clean_resume = (degraded.time >= artifact.end_seconds + 0.25) & (
        degraded.time < artifact.end_seconds + 0.75
    )
    clean_after_resume = clean_result.approximation[after_clean_resume]
    ungated_error = np.mean(
        np.abs(ungated.approximation[after_clean_resume] - clean_after_resume)
    )
    gated_error = np.mean(
        np.abs(gated.approximation[after_clean_resume] - clean_after_resume)
    )

    assert gated.diagnostics["skipped_measurement_updates"] == np.count_nonzero(
        artifact.dropout_mask
    )
    assert gated.diagnostics["reacquisition_updates"] == 1
    assert not np.any(gated.valid_mask[artifact.dropout_mask])
    assert gated_error < 0.2
    assert gated_error < 0.3 * ungated_error
