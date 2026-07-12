import numpy as np
import pytest

from lowpoint import EnvelopeConfig, estimate_envelope
from lowpoint.synthetic import generate_ecg, generate_positive_biosignal


@pytest.mark.parametrize("method", ["quantile", "minima", "kalman"])
def test_constant_signal_remains_constant(method: str) -> None:
    signal = np.full(600, 2.75)
    result = estimate_envelope(
        signal,
        100.0,
        EnvelopeConfig(method=method, edge_padding_seconds=0.0, kalman_warmup_seconds=0.5),
    )
    np.testing.assert_allclose(result.approximation, signal, atol=2e-5)
    np.testing.assert_allclose(result.residual, 0.0, atol=2e-5)


def test_quantile_translation_and_positive_scale_equivariance() -> None:
    _time, signal, _truth = generate_positive_biosignal(duration_seconds=8.0, sampling_rate=80)
    config = EnvelopeConfig(
        method="quantile",
        quantile=0.08,
        smoothness_hz=0.25,
        edge_padding_seconds=0.0,
        max_iterations=60,
    )
    original = estimate_envelope(signal, 80.0, config).approximation
    transformed = estimate_envelope(3.2 * signal - 7.0, 80.0, config).approximation
    np.testing.assert_allclose(transformed, 3.2 * original - 7.0, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("method", ["quantile", "minima"])
def test_lower_upper_sign_duality(method: str) -> None:
    _time, signal, _truth = generate_positive_biosignal(duration_seconds=6.0, sampling_rate=60)
    common = dict(method=method, quantile=0.07, smoothness_hz=0.2, edge_padding_seconds=0.0)
    upper = estimate_envelope(signal, 60.0, EnvelopeConfig(side="upper", **common))
    mirrored_lower = estimate_envelope(-signal, 60.0, EnvelopeConfig(side="lower", **common))
    np.testing.assert_allclose(upper.approximation, -mirrored_lower.approximation, atol=1e-10)


def test_quantile_objective_is_monotone_and_tail_is_lower() -> None:
    _time, signal, _truth = generate_positive_biosignal(duration_seconds=12.0, sampling_rate=100)
    result = estimate_envelope(
        signal,
        100.0,
        EnvelopeConfig(
            method="quantile",
            quantile=0.08,
            smoothness_hz=0.2,
            edge_padding_seconds=0.0,
            max_iterations=80,
        ),
    )
    assert result.diagnostics["objective_monotone"]
    assert result.diagnostics["empirical_tail_fraction"] < 0.25
    assert np.mean(result.approximation) < np.mean(signal)


def test_higher_quantile_produces_a_higher_curve() -> None:
    _time, signal, _truth = generate_positive_biosignal(
        duration_seconds=10.0, sampling_rate=100, noise_std=0.02
    )
    lower = estimate_envelope(
        signal,
        100.0,
        EnvelopeConfig(method="quantile", quantile=0.05, smoothness_hz=0.2),
    )
    higher = estimate_envelope(
        signal,
        100.0,
        EnvelopeConfig(method="quantile", quantile=0.10, smoothness_hz=0.2),
    )
    assert np.all(higher.approximation >= lower.approximation)
    assert (
        higher.diagnostics["empirical_tail_fraction"] > lower.diagnostics["empirical_tail_fraction"]
    )


def test_physical_cutoff_is_stable_across_sampling_rates() -> None:
    low_time, low_signal, _truth = generate_positive_biosignal(
        duration_seconds=10.0, sampling_rate=100, noise_std=0.0
    )
    high_time, high_signal, _truth = generate_positive_biosignal(
        duration_seconds=10.0, sampling_rate=200, noise_std=0.0
    )
    config = EnvelopeConfig(method="quantile", quantile=0.05, smoothness_hz=0.2)
    low_result = estimate_envelope(low_signal, 100.0, config)
    high_result = estimate_envelope(high_signal, 200.0, config)
    high_on_low_grid = np.interp(low_time, high_time, high_result.approximation)
    np.testing.assert_allclose(high_on_low_grid, low_result.approximation, atol=1e-3)


def test_one_deep_impulse_has_bounded_effect_on_quantile_smoother() -> None:
    _time, signal, _truth = generate_positive_biosignal(duration_seconds=10.0, sampling_rate=100)
    config = EnvelopeConfig(
        method="quantile",
        quantile=0.05,
        smoothness_hz=0.2,
        edge_padding_seconds=0.0,
        max_iterations=60,
    )
    clean = estimate_envelope(signal, 100.0, config).approximation
    contaminated = signal.copy()
    contaminated[500] -= 50.0
    robust = estimate_envelope(contaminated, 100.0, config).approximation
    assert np.max(np.abs(robust - clean)) < 0.15


def test_minima_spline_contacts_returned_guarded_supports() -> None:
    demo = generate_ecg(duration_seconds=6.0, negative_spikes=0)
    result = estimate_envelope(
        demo.signal,
        demo.sampling_rate,
        EnvelopeConfig(method="minima", minima_window_seconds=0.7, guard_seconds=0.04),
    )
    assert len(result.support_indices) >= 4
    np.testing.assert_allclose(
        result.approximation[result.support_indices], result.support_values, atol=1e-12
    )


def test_kalman_is_prefix_causal_after_same_warmup() -> None:
    _time, signal, _truth = generate_positive_biosignal(duration_seconds=10.0, sampling_rate=100)
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=1.0)
    full = estimate_envelope(signal, 100.0, config).approximation
    prefix = estimate_envelope(signal[:600], 100.0, config).approximation
    np.testing.assert_allclose(full[:600], prefix, atol=1e-12)


def test_nan_result_preserves_original_validity_mask() -> None:
    signal = np.sin(np.linspace(0, 10, 500))
    signal[100:110] = np.nan
    result = estimate_envelope(signal, 100.0, EnvelopeConfig(edge_padding_seconds=0.0))
    assert np.all(np.isfinite(result.approximation))
    assert not np.any(result.valid_mask[100:110])
    assert result.diagnostics["interpolated_sample_count"] == 10
