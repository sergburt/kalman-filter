"""Stable public pipeline shared by the UI and CLI."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .kalman import asymmetric_kalman
from .metrics import envelope_metrics
from .minima import minima_spline
from .models import SOFTWARE_VERSION, EnvelopeConfig, EnvelopeResult
from .preprocessing import prepare_signal
from .quantile import quantile_smooth


def estimate_envelope(
    signal: ArrayLike,
    sampling_rate: float,
    config: EnvelopeConfig | None = None,
) -> EnvelopeResult:
    """Estimate a lower or upper signal envelope.

    Missing values are interpolated for numerical continuity while ``valid_mask``
    records which values were genuinely observed. The raw input is never filtered
    or mutated in place.
    """

    config = config or EnvelopeConfig()
    config.validate(sampling_rate)
    y, valid_mask = prepare_signal(signal)

    # All implementations solve a lower-tail problem. Mirroring provides exact
    # lower/upper symmetry and avoids duplicated, divergent algorithm branches.
    work = y if config.side == "lower" else -y
    estimator: Any
    if config.method == "quantile":
        estimator = quantile_smooth
    elif config.method == "minima":
        estimator = minima_spline
    else:
        estimator = asymmetric_kalman

    approximation_work, support_indices, support_values_work, diagnostics = estimator(
        work, sampling_rate, config
    )
    if config.side == "lower":
        approximation = approximation_work
        support_values = support_values_work
    else:
        approximation = -approximation_work
        support_values = -support_values_work

    residual = y - approximation
    diagnostics.update(
        envelope_metrics(
            y,
            approximation,
            sampling_rate,
            config.quantile,
            config.side,
            valid_mask,
        )
    )
    diagnostics.update(
        {
            "sampling_rate_hz": float(sampling_rate),
            "software_version": SOFTWARE_VERSION,
            "method": config.method,
            "side": config.side,
            "research_only": True,
            "warning": (
                "A numerical lower envelope is not an ECG isoelectric baseline; "
                "do not use it as clinical baseline correction without separate validation."
            ),
        }
    )
    return EnvelopeResult(
        approximation=np.asarray(approximation, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        valid_mask=valid_mask,
        support_indices=np.asarray(support_indices, dtype=np.int64),
        support_values=np.asarray(support_values, dtype=np.float64),
        diagnostics=diagnostics,
        config=config,
    )
