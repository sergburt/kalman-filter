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
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
    """Track a low *expectile* online with a local-linear-trend state model.

    Positive innovations are assigned the small tail weight ``tau``; negative
    innovations receive weight ``1-tau``. This makes the tracker asymmetric but
    does not make it an exact quantile estimator. The distinction is reported in
    diagnostics and the UI.
    """

    location, scale = robust_location_scale(y)
    work = (y - location) / scale
    dt = 1.0 / sampling_rate
    transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    observation = np.array([1.0, 0.0], dtype=np.float64)
    acceleration_variance = config.kalman_process_variance
    process_noise = acceleration_variance * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64
    )
    identity = np.eye(2, dtype=np.float64)

    warmup = min(y.size, max(1, int(round(config.kalman_warmup_seconds * sampling_rate))))
    initial_level = float(np.quantile(work[:warmup], config.quantile))
    state = np.array([initial_level, 0.0], dtype=np.float64)
    covariance = np.diag([1.0, 1.0]).astype(np.float64)
    estimate = np.empty_like(work)
    clipped_innovations = 0

    for index, sample in enumerate(work):
        if index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise

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
    )
    diagnostics: dict[str, Any] = {
        "algorithm": "causal_asymmetric_local_linear_kalman",
        "target_type": "expectile-like (not an exact quantile)",
        "iterations": 1,
        "converged": True,
        "causal": True,
        "algorithmic_lookahead_samples": 0,
        "warmup_samples": int(warmup),
        "clipped_innovations": int(clipped_innovations),
        "normalization_location": location,
        "normalization_scale": scale,
        **support_diagnostics,
    }
    return approximation, support_indices, support_values, diagnostics
