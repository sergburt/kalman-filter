"""Exact per-sample computation traces for the mathematical UI panel."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lowpoint.kalman import kalman_step_trace
from lowpoint.models import EnvelopeConfig


def _signal(sample_count: int = 500) -> np.ndarray:
    time = np.arange(sample_count, dtype=np.float64) / 100.0
    return 0.2 * np.sin(2.0 * np.pi * 0.35 * time) + 0.04 * np.sin(
        2.0 * np.pi * 3.0 * time
    )


def test_kalman_trace_matches_update_equations_and_output() -> None:
    signal = _signal()
    config = EnvelopeConfig(
        method="kalman",
        quantile=0.1,
        kalman_warmup_seconds=0.5,
    )

    approximation, trace, _diagnostics = kalman_step_trace(signal, 100.0, config)
    index = 173

    assert trace.measurement_used[index]
    assert trace.predicted_level[index] == pytest.approx(
        trace.previous_level[index] + trace.dt * trace.previous_rate[index]
    )
    assert trace.innovation[index] == pytest.approx(
        trace.measurement[index] - trace.prior_level[index]
    )
    expected_weight = 1.0 - config.quantile if trace.innovation[index] < 0 else config.quantile
    assert trace.tail_weight[index] == pytest.approx(expected_weight)
    assert trace.effective_measurement_variance[index] == pytest.approx(
        config.kalman_measurement_variance / expected_weight
    )
    assert trace.innovation_variance[index] == pytest.approx(
        trace.prior_covariance_00[index]
        + trace.effective_measurement_variance[index]
    )
    assert trace.posterior_level[index] == pytest.approx(
        trace.prior_level[index]
        + trace.gain_level[index] * trace.clipped_innovation[index]
    )
    assert approximation[index] == pytest.approx(
        trace.location + trace.scale * trace.posterior_level[index]
    )


def test_kalman_trace_marks_invalid_sample_as_prediction_only() -> None:
    signal = _signal()
    valid = np.ones(signal.size, dtype=bool)
    valid[220:240] = False
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=0.5)

    approximation, trace, diagnostics = kalman_step_trace(
        signal,
        100.0,
        config,
        valid_mask=valid,
    )

    index = 225
    assert not trace.measurement_used[index]
    assert np.isnan(trace.innovation[index])
    assert np.isnan(trace.gain_level[index])
    assert trace.posterior_level[index] == pytest.approx(trace.predicted_level[index])
    assert approximation[index] == pytest.approx(
        trace.location + trace.scale * trace.predicted_level[index]
    )
    assert diagnostics["skipped_measurement_updates"] == 20
    assert trace.reacquisition[240]
    assert trace.prior_rate[240] == 0.0


def test_symmetric_trace_uses_equal_weights_for_both_innovation_signs() -> None:
    signal = _signal()
    config = EnvelopeConfig(method="kalman", kalman_warmup_seconds=0.5)
    symmetric_config = replace(config, quantile=0.5)

    _approximation, trace, _diagnostics = kalman_step_trace(
        signal,
        100.0,
        symmetric_config,
    )

    positive = trace.innovation > 0
    negative = trace.innovation < 0
    assert np.any(positive)
    assert np.any(negative)
    np.testing.assert_allclose(trace.tail_weight[positive], 0.5)
    np.testing.assert_allclose(trace.tail_weight[negative], 0.5)
