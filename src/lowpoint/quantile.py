"""Offline convex lower-quantile smoothing with a Whittaker penalty."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, solveh_banded
from scipy.ndimage import percentile_filter
from scipy.sparse import csc_matrix, diags

from .models import EnvelopeConfig
from .preprocessing import robust_location_scale
from .support import guarded_block_minima


def cutoff_to_lambda(sampling_rate: float, cutoff_hz: float) -> float:
    """Map a nominal -3 dB Whittaker cutoff to a second-difference penalty.

    The mapping is exact for a symmetric least-squares Whittaker smoother. For
    pinball loss it remains a useful, sampling-rate-aware smoothness convention.
    """

    omega = 2.0 * math.pi * cutoff_hz / sampling_rate
    # ``4*sin(omega/2)^2`` is algebraically equal to ``2-2*cos(omega)`` but
    # avoids catastrophic cancellation when the cutoff is tiny relative to fs.
    response_term = max(4.0 * math.sin(omega / 2.0) ** 2, np.finfo(float).eps)
    return float((math.sqrt(2.0) - 1.0) / (response_term * response_term))


def _difference_penalty(n: int) -> csc_matrix:
    if n < 3:
        return csc_matrix((n, n), dtype=np.float64)
    d2 = diags(
        diagonals=(np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)),
        offsets=(0, 1, 2),
        shape=(n - 2, n),
        format="csc",
    )
    return (d2.T @ d2).tocsc()


def _objective(
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    quantile: float,
    epsilon: float,
    smoothing_lambda: float,
) -> float:
    residual = y - z
    smooth_abs = np.sqrt(residual * residual + epsilon * epsilon)
    fidelity = np.sum(0.5 * smooth_abs + (quantile - 0.5) * residual)
    # Computing z.T @ D2.T @ D2 @ z suffers catastrophic cancellation for
    # extremely smooth curves. The mathematically identical sum of squared
    # second differences is nonnegative by construction.
    second_differences = np.diff(z, n=2)
    roughness = 0.5 * smoothing_lambda * float(np.dot(second_differences, second_differences))
    return float(fidelity + roughness)


def _solve_whittaker_system(
    weights: NDArray[np.float64],
    rhs: NDArray[np.float64],
    penalty: csc_matrix,
    smoothing_lambda: float,
    stage: str,
) -> tuple[NDArray[np.float64], float]:
    """Solve the symmetric positive-definite pentadiagonal IRLS system."""

    n = weights.size
    banded = np.zeros((3, n), dtype=np.float64)
    banded[0] = weights + smoothing_lambda * penalty.diagonal(0)
    banded[1, : n - 1] = smoothing_lambda * penalty.diagonal(-1)
    banded[2, : n - 2] = smoothing_lambda * penalty.diagonal(-2)

    # Uniform scaling keeps band entries near unity without changing the
    # solution and delays overflow/loss of the data-fidelity diagonal.
    matrix_scale = max(float(np.max(np.abs(banded))), 1.0)
    try:
        solution = solveh_banded(
            banded / matrix_scale,
            rhs / matrix_scale,
            lower=True,
            check_finite=False,
        )
    except LinAlgError as exc:
        raise RuntimeError(
            f"Quantile-smoother {stage} is numerically ill-conditioned at the requested "
            "bandwidth. Increase smoothness_hz or downsample the signal before fitting."
        ) from exc

    solution = np.asarray(solution, dtype=np.float64)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(f"Quantile-smoother {stage} produced non-finite values")

    equation_residual = weights * solution + smoothing_lambda * (penalty @ solution) - rhs
    residual_scale = (
        np.linalg.norm(rhs)
        + np.linalg.norm(weights * solution)
        + smoothing_lambda * np.linalg.norm(penalty @ solution)
        + np.finfo(np.float64).eps
    )
    relative_residual = float(np.linalg.norm(equation_residual) / residual_scale)
    if not np.isfinite(relative_residual) or relative_residual > 1e-5:
        raise RuntimeError(
            f"Quantile-smoother {stage} failed its numerical residual check "
            f"({relative_residual:.3g}). Increase smoothness_hz or downsample the signal."
        )
    return solution, relative_residual


def quantile_smooth(
    y: NDArray[np.float64],
    sampling_rate: float,
    config: EnvelopeConfig,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
    """Estimate a smooth lower conditional quantile.

    The optimized (smoothed) convex objective is

        sum rho_tau(y_i - z_i) + lambda/2 * ||D2 z||^2,

    with ``rho`` represented by a differentiable absolute-value approximation.
    Iteratively reweighted symmetric-banded solves implement a majorization step.
    """

    location, scale = robust_location_scale(y)
    record_duration = y.size / sampling_rate
    cutoff_cycles_in_record = config.smoothness_hz * record_duration
    if float(np.ptp(y)) <= np.finfo(np.float64).eps * max(1.0, abs(location)):
        support_indices, support_values, support_diagnostics = guarded_block_minima(
            y,
            sampling_rate,
            config.minima_window_seconds,
            config.minima_overlap,
            config.guard_seconds,
            config.outlier_sigma,
            valid_mask=valid_mask,
        )
        return (
            y.copy(),
            support_indices,
            support_values,
            {
                "algorithm": "penalized_quantile_whittaker",
                "iterations": 0,
                "converged": True,
                "relative_change": 0.0,
                "nominal_smoothing_lambda": cutoff_to_lambda(sampling_rate, config.smoothness_hz),
                "nominal_cutoff_cycles_in_record": cutoff_cycles_in_record,
                "linear_solver": "constant_signal_shortcut",
                "max_linear_system_relative_residual": 0.0,
                "normalization_location": location,
                "normalization_scale": scale,
                "edge_padding_samples": 0,
                "objective_initial": 0.0,
                "objective_final": 0.0,
                "objective_monotone": True,
                "constant_signal_shortcut": True,
                **support_diagnostics,
            },
        )

    if cutoff_cycles_in_record < 1.0:
        minimum_bandwidth = sampling_rate / y.size
        raise ValueError(
            "smoothness_hz must provide at least one nominal cutoff cycle in the observed "
            f"record; use at least {minimum_bandwidth:.6g} Hz for {y.size} samples at "
            f"{sampling_rate:g} Hz, or downsample before fitting"
        )
    normalized = (y - location) / scale

    pad = min(y.size - 1, int(round(config.edge_padding_seconds * sampling_rate)))
    if pad > 0:
        work = np.pad(normalized, (pad, pad), mode="reflect")
    else:
        work = normalized.copy()

    n = work.size
    penalty = _difference_penalty(n)
    smoothing_lambda = cutoff_to_lambda(sampling_rate, config.smoothness_hz)

    # A local percentile gives a stable starting point; the final solution is
    # determined by the global penalized objective, not by this window.
    init_window = max(3, int(round(sampling_rate / config.smoothness_hz)))
    init_window = min(init_window, n if n % 2 else max(3, n - 1))
    if init_window % 2 == 0:
        init_window += 1
    initial_target = percentile_filter(
        work,
        percentile=100.0 * config.quantile,
        size=init_window,
        mode="reflect",
    )
    z, initial_solver_residual = _solve_whittaker_system(
        np.ones(n, dtype=np.float64),
        initial_target,
        penalty,
        smoothing_lambda,
        "initialization",
    )
    solver_residuals = [initial_solver_residual]

    epsilon = float(config.smoothing_epsilon)
    objective_history = [_objective(work, z, config.quantile, epsilon, smoothing_lambda)]
    converged = False
    relative_change = float("inf")

    for _iteration in range(1, config.max_iterations + 1):
        residual = work - z
        weights = 0.5 / np.sqrt(residual * residual + epsilon * epsilon)
        rhs = weights * work + (config.quantile - 0.5)
        candidate, solver_residual = _solve_whittaker_system(
            weights,
            rhs,
            penalty,
            smoothing_lambda,
            "IRLS solve",
        )
        solver_residuals.append(solver_residual)

        # The quadratic majorizer normally decreases the objective. Backtracking
        # makes that guarantee explicit in the presence of floating-point error.
        previous_objective = objective_history[-1]
        candidate_objective = _objective(
            work, candidate, config.quantile, epsilon, smoothing_lambda
        )
        if candidate_objective > previous_objective * (1.0 + 1e-10):
            step = 0.5
            accepted = False
            while step >= 1.0 / 128.0:
                trial = z + step * (candidate - z)
                trial_objective = _objective(
                    work, trial, config.quantile, epsilon, smoothing_lambda
                )
                if trial_objective <= previous_objective * (1.0 + 1e-10):
                    candidate = trial
                    candidate_objective = trial_objective
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                candidate = z.copy()
                candidate_objective = previous_objective

        denominator = max(float(np.linalg.norm(z)), np.finfo(float).eps)
        relative_change = float(np.linalg.norm(candidate - z) / denominator)
        z = candidate
        objective_history.append(candidate_objective)
        if relative_change <= config.tolerance:
            converged = True
            break

    if pad > 0:
        z = z[pad:-pad]
    approximation = z * scale + location

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
        "algorithm": "penalized_quantile_whittaker",
        "iterations": len(objective_history) - 1,
        "converged": converged,
        "relative_change": relative_change,
        "nominal_smoothing_lambda": smoothing_lambda,
        "nominal_cutoff_cycles_in_record": cutoff_cycles_in_record,
        "linear_solver": "symmetric_banded_cholesky",
        "max_linear_system_relative_residual": max(solver_residuals),
        "normalization_location": location,
        "normalization_scale": scale,
        "edge_padding_samples": int(pad),
        "objective_initial": objective_history[0],
        "objective_final": objective_history[-1],
        "objective_monotone": bool(np.all(np.diff(objective_history) <= 1e-8)),
        **support_diagnostics,
    }
    return approximation, support_indices, support_values, diagnostics
