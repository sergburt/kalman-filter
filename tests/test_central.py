import numpy as np

from lowpoint import (
    EnvelopeConfig,
    centered_rolling_median,
    estimate_envelope,
    symmetric_kalman_trend,
)
from lowpoint.synthetic import generate_positive_biosignal


def test_symmetric_kalman_treats_positive_and_negative_deviations_equally() -> None:
    _time, signal, _truth = generate_positive_biosignal(
        duration_seconds=8.0,
        sampling_rate=100.0,
    )
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=1.0)

    positive, diagnostics = symmetric_kalman_trend(signal, 100.0, config)
    negative, _negative_diagnostics = symmetric_kalman_trend(-signal, 100.0, config)

    np.testing.assert_allclose(negative, -positive, atol=1e-12)
    assert diagnostics["innovation_weight_below"] == 0.5
    assert diagnostics["innovation_weight_above"] == 0.5
    assert diagnostics["clinical_baseline"] is False


def test_central_kalman_lies_above_asymmetric_lower_tracker() -> None:
    _time, signal, _truth = generate_positive_biosignal(
        duration_seconds=10.0,
        sampling_rate=100.0,
    )
    config = EnvelopeConfig(method="kalman", quantile=0.05, kalman_warmup_seconds=1.0)

    lower = estimate_envelope(signal, 100.0, config).approximation
    central, _diagnostics = symmetric_kalman_trend(signal, 100.0, config)

    assert np.mean(central - lower) > 0.2


def test_central_comparisons_ignore_known_invalid_sample_values() -> None:
    _time, signal, _truth = generate_positive_biosignal(
        duration_seconds=8.0,
        sampling_rate=100.0,
    )
    valid_mask = np.ones(signal.size, dtype=bool)
    valid_mask[350:400] = False
    hostile = signal.copy()
    hostile[350:400] = -100_000.0
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=1.0)

    reference_kalman, kalman_diagnostics = symmetric_kalman_trend(
        signal,
        100.0,
        config,
        valid_mask=valid_mask,
    )
    hostile_kalman, _diagnostics = symmetric_kalman_trend(
        hostile,
        100.0,
        config,
        valid_mask=valid_mask,
    )
    reference_median, median_diagnostics = centered_rolling_median(
        signal,
        100.0,
        0.8,
        valid_mask=valid_mask,
    )
    hostile_median, _diagnostics = centered_rolling_median(
        hostile,
        100.0,
        0.8,
        valid_mask=valid_mask,
    )

    np.testing.assert_allclose(hostile_kalman, reference_kalman, atol=1e-12)
    np.testing.assert_allclose(hostile_median, reference_median, atol=1e-12)
    assert kalman_diagnostics["skipped_measurement_updates"] == 50
    assert kalman_diagnostics["reacquisition_updates"] == 1
    assert median_diagnostics["excluded_sample_count"] == 50
    assert median_diagnostics["predictive"] is False
