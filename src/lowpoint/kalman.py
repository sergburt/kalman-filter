"""Causal asymmetric Kalman lower-tail tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .models import EnvelopeConfig
from .preprocessing import robust_location_scale
from .support import guarded_block_minima


@dataclass(frozen=True)
class KalmanStepTrace:
    """Exact normalized terms used by every Kalman predict/update step."""

    previous_level: NDArray[np.float64]
    previous_rate: NDArray[np.float64]
    predicted_level: NDArray[np.float64]
    predicted_rate: NDArray[np.float64]
    prior_level: NDArray[np.float64]
    prior_rate: NDArray[np.float64]
    prior_covariance_00: NDArray[np.float64]
    prior_covariance_01: NDArray[np.float64]
    prior_covariance_11: NDArray[np.float64]
    measurement: NDArray[np.float64]
    innovation: NDArray[np.float64]
    tail_weight: NDArray[np.float64]
    effective_measurement_variance: NDArray[np.float64]
    innovation_variance: NDArray[np.float64]
    innovation_limit: NDArray[np.float64]
    clipped_innovation: NDArray[np.float64]
    gain_level: NDArray[np.float64]
    gain_rate: NDArray[np.float64]
    posterior_level: NDArray[np.float64]
    posterior_rate: NDArray[np.float64]
    measurement_used: NDArray[np.bool_]
    reacquisition: NDArray[np.bool_]
    location: float
    scale: float
    dt: float
    quantile: float
    process_variance: float
    measurement_variance: float
    innovation_clip: float


def asymmetric_kalman(
    y: NDArray[np.float64],
    sampling_rate: float,
    config: EnvelopeConfig,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
    trace_out: dict[str, KalmanStepTrace] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
    """Track a low *expectile* online with a local-linear-trend state model.

    Positive innovations are assigned the small tail weight ``tau``; negative
    innovations receive weight ``1-tau``. This makes the tracker asymmetric but
    does not make it an exact quantile estimator. Invalid samples receive a
    prediction-only step. At the first valid sample after a gap, the stale trend
    velocity is cleared and normalized covariance is conservatively restored so
    the filter can reacquire rather than extrapolate gap-time drift. The validity
    mask must come from an external signal-quality decision; this function does
    not detect artifacts. These distinctions are reported in diagnostics and the
    UI.
    """

    if valid_mask is None:
        valid = np.ones(y.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim != 1 or valid.size != y.size:
            raise ValueError("valid_mask must be one-dimensional and match y")

    warmup = min(y.size, max(1, int(round(config.kalman_warmup_seconds * sampling_rate))))
    # Keep the online recursion prefix-invariant: only the explicitly declared
    # initialization interval may influence the fixed robust normalization.
    warmup_values = y[:warmup][valid[:warmup]]
    normalization_used_invalid_fallback = warmup_values.size == 0
    if normalization_used_invalid_fallback:
        warmup_values = y[:warmup]
    location, scale = robust_location_scale(warmup_values)
    work = (y - location) / scale
    dt = 1.0 / sampling_rate
    transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    observation = np.array([1.0, 0.0], dtype=np.float64)
    acceleration_variance = config.kalman_process_variance
    process_noise = acceleration_variance * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64
    )
    identity = np.eye(2, dtype=np.float64)
    reacquisition_level_variance_floor = 0.25

    initial_values = work[:warmup][valid[:warmup]]
    if initial_values.size == 0:
        initial_values = work[:warmup]
    initial_level = float(np.quantile(initial_values, config.quantile))
    state = np.array([initial_level, 0.0], dtype=np.float64)
    covariance = np.diag([1.0, 1.0]).astype(np.float64)
    estimate = np.empty_like(work)
    clipped_innovations = 0
    skipped_measurement_updates = 0
    reacquisition_updates = 0
    trace_float_fields = (
        {
            name: np.full(work.size, np.nan, dtype=np.float64)
            for name in (
                "previous_level",
                "previous_rate",
                "predicted_level",
                "predicted_rate",
                "prior_level",
                "prior_rate",
                "prior_covariance_00",
                "prior_covariance_01",
                "prior_covariance_11",
                "measurement",
                "innovation",
                "tail_weight",
                "effective_measurement_variance",
                "innovation_variance",
                "innovation_limit",
                "clipped_innovation",
                "gain_level",
                "gain_rate",
                "posterior_level",
                "posterior_rate",
            )
        }
        if trace_out is not None
        else None
    )
    trace_measurement_used = (
        np.zeros(work.size, dtype=bool) if trace_out is not None else None
    )
    trace_reacquisition = np.zeros(work.size, dtype=bool) if trace_out is not None else None

    for index, sample in enumerate(work):
        if trace_float_fields is not None:
            trace_float_fields["previous_level"][index] = state[0]
            trace_float_fields["previous_rate"][index] = state[1]
        if index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        if trace_float_fields is not None:
            trace_float_fields["predicted_level"][index] = state[0]
            trace_float_fields["predicted_rate"][index] = state[1]

        if not valid[index]:
            skipped_measurement_updates += 1
            if trace_float_fields is not None:
                trace_float_fields["prior_level"][index] = state[0]
                trace_float_fields["prior_rate"][index] = state[1]
                trace_float_fields["prior_covariance_00"][index] = covariance[0, 0]
                trace_float_fields["prior_covariance_01"][index] = covariance[0, 1]
                trace_float_fields["prior_covariance_11"][index] = covariance[1, 1]
                trace_float_fields["measurement"][index] = sample
                trace_float_fields["posterior_level"][index] = state[0]
                trace_float_fields["posterior_rate"][index] = state[1]
            estimate[index] = state[0]
            continue
        if index > 0 and not valid[index - 1]:
            reacquisition_updates += 1
            if trace_reacquisition is not None:
                trace_reacquisition[index] = True
            state[1] = 0.0
            covariance = np.diag(
                [
                    max(float(covariance[0, 0]), reacquisition_level_variance_floor),
                    float(covariance[1, 1]),
                ]
            )

        if trace_float_fields is not None:
            trace_float_fields["prior_level"][index] = state[0]
            trace_float_fields["prior_rate"][index] = state[1]
            trace_float_fields["prior_covariance_00"][index] = covariance[0, 0]
            trace_float_fields["prior_covariance_01"][index] = covariance[0, 1]
            trace_float_fields["prior_covariance_11"][index] = covariance[1, 1]
            trace_float_fields["measurement"][index] = sample
        innovation = float(sample - observation @ state)
        tail_weight = (1.0 - config.quantile) if innovation < 0 else config.quantile
        effective_measurement_variance = config.kalman_measurement_variance / max(tail_weight, 1e-6)
        innovation_variance = float(
            observation @ covariance @ observation.T + effective_measurement_variance
        )
        limit = config.kalman_innovation_clip * np.sqrt(innovation_variance)
        clipped = float(np.clip(innovation, -limit, limit))
        if clipped != innovation:
            clipped_innovations += 1

        gain = covariance @ observation / innovation_variance
        if trace_float_fields is not None and trace_measurement_used is not None:
            trace_measurement_used[index] = True
            trace_float_fields["innovation"][index] = innovation
            trace_float_fields["tail_weight"][index] = tail_weight
            trace_float_fields["effective_measurement_variance"][index] = (
                effective_measurement_variance
            )
            trace_float_fields["innovation_variance"][index] = innovation_variance
            trace_float_fields["innovation_limit"][index] = limit
            trace_float_fields["clipped_innovation"][index] = clipped
            trace_float_fields["gain_level"][index] = gain[0]
            trace_float_fields["gain_rate"][index] = gain[1]
        state = state + gain * clipped
        update = identity - np.outer(gain, observation)
        covariance = (
            update @ covariance @ update.T + np.outer(gain, gain) * effective_measurement_variance
        )
        if trace_float_fields is not None:
            trace_float_fields["posterior_level"][index] = state[0]
            trace_float_fields["posterior_rate"][index] = state[1]
        estimate[index] = state[0]

    approximation = estimate * scale + location
    if trace_out is not None:
        assert trace_float_fields is not None
        assert trace_measurement_used is not None
        assert trace_reacquisition is not None
        trace_out["trace"] = KalmanStepTrace(
            **trace_float_fields,
            measurement_used=trace_measurement_used,
            reacquisition=trace_reacquisition,
            location=float(location),
            scale=float(scale),
            dt=float(dt),
            quantile=float(config.quantile),
            process_variance=float(config.kalman_process_variance),
            measurement_variance=float(config.kalman_measurement_variance),
            innovation_clip=float(config.kalman_innovation_clip),
        )
    support_indices, support_values, support_diagnostics = guarded_block_minima(
        y,
        sampling_rate,
        config.minima_window_seconds,
        config.minima_overlap,
        config.guard_seconds,
        config.outlier_sigma,
        valid_mask=valid_mask,
    )
    diagnostics: dict[str, Any] = {
        "algorithm": "causal_asymmetric_local_linear_kalman",
        "target_type": "expectile-like (not an exact quantile)",
        "iterations": 1,
        "converged": True,
        "causal": warmup <= 1,
        "causal_after_initialization": True,
        "algorithmic_lookahead_samples": max(0, int(warmup) - 1),
        "warmup_samples": int(warmup),
        "normalization_scope": "initialization_window",
        "normalization_valid_samples": int(np.count_nonzero(valid[:warmup])),
        "normalization_used_invalid_fallback": normalization_used_invalid_fallback,
        "clipped_innovations": int(clipped_innovations),
        "measurement_updates": int(np.count_nonzero(valid)),
        "skipped_measurement_updates": int(skipped_measurement_updates),
        "reacquisition_updates": int(reacquisition_updates),
        "reacquisition_policy": (
            "reset local slope and restore normalized covariance before first valid update"
        ),
        "reacquisition_level_variance_floor": reacquisition_level_variance_floor,
        "normalization_location": location,
        "normalization_scale": scale,
        **support_diagnostics,
    }
    return approximation, support_indices, support_values, diagnostics


def kalman_step_trace(
    y: NDArray[np.float64],
    sampling_rate: float,
    config: EnvelopeConfig,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], KalmanStepTrace, dict[str, Any]]:
    """Return the approximation plus the exact terms used at each sample."""

    trace_out: dict[str, KalmanStepTrace] = {}
    approximation, _, _, diagnostics = asymmetric_kalman(
        y,
        sampling_rate,
        config,
        valid_mask=valid_mask,
        trace_out=trace_out,
    )
    return approximation, trace_out["trace"], diagnostics
