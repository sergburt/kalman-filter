"""Causal asymmetric Kalman lower-tail tracker."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .models import EnvelopeConfig
from .preprocessing import robust_location_scale
from .support import guarded_block_minima


def asymmetric_kalman(
    y: NDArray[np.float64],
    sampling_rate: float,
    config: EnvelopeConfig,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
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

    for index, sample in enumerate(work):
        if index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise

        if not valid[index]:
            skipped_measurement_updates += 1
            estimate[index] = state[0]
            continue
        if index > 0 and not valid[index - 1]:
            reacquisition_updates += 1
            state[1] = 0.0
            covariance = np.diag(
                [
                    max(float(covariance[0, 0]), reacquisition_level_variance_floor),
                    float(covariance[1, 1]),
                ]
            )

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
        state = state + gain * clipped
        update = identity - np.outer(gain, observation)
        covariance = (
            update @ covariance @ update.T + np.outer(gain, gain) * effective_measurement_variance
        )
        estimate[index] = state[0]

    approximation = estimate * scale + location
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
