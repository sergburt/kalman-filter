"""Reproducible synthetic benchmark for the v0.1 estimators.

Run from the repository root:

    python scripts/run_validation.py

The script writes machine-readable metrics and a visual summary to ``artifacts``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from lowpoint import EnvelopeConfig, estimate_envelope
from lowpoint.metrics import truth_metrics
from lowpoint.preprocessing import robust_location_scale
from lowpoint.quantile import _difference_penalty, cutoff_to_lambda
from lowpoint.synthetic import generate_ecg, generate_positive_biosignal

ROOT = Path(__file__).resolve().parents[1]


def symmetric_whittaker(signal: np.ndarray, sampling_rate: float, cutoff_hz: float) -> np.ndarray:
    """Conventional middle smoother used only as a benchmark."""

    location, scale = robust_location_scale(signal)
    normalized = (signal - location) / scale
    penalty = _difference_penalty(signal.size)
    smoothing_lambda = cutoff_to_lambda(sampling_rate, cutoff_hz)
    estimate = spsolve(eye(signal.size, format="csc") + smoothing_lambda * penalty, normalized)
    return np.asarray(estimate) * scale + location


def evaluate(
    scenario: str,
    signal: np.ndarray,
    truth: np.ndarray,
    sampling_rate: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    methods: dict[str, EnvelopeConfig | None] = {
        "middle_whittaker": None,
        "quantile_tau_0.05": EnvelopeConfig(method="quantile", quantile=0.05, smoothness_hz=0.25),
        "quantile_tau_0.01": EnvelopeConfig(
            method="quantile", quantile=0.01, smoothness_hz=0.25, max_iterations=90
        ),
        "guarded_minima": EnvelopeConfig(
            method="minima", minima_window_seconds=0.85, guard_seconds=0.012
        ),
        "causal_kalman": EnvelopeConfig(method="kalman", quantile=0.05),
    }
    rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    for name, config in methods.items():
        start = time.perf_counter()
        if config is None:
            approximation = symmetric_whittaker(signal, sampling_rate, 0.25)
            elapsed = time.perf_counter() - start
            tail_fraction = float(np.mean(signal < approximation))
            converged = True
        else:
            result = estimate_envelope(signal, sampling_rate, config)
            approximation = result.approximation
            elapsed = time.perf_counter() - start
            tail_fraction = result.diagnostics["empirical_tail_fraction"]
            converged = result.diagnostics["converged"]
        curves[name] = approximation
        metrics = truth_metrics(approximation, truth)
        central = slice(
            int(sampling_rate), max(int(sampling_rate) + 1, len(signal) - int(sampling_rate))
        )
        central_metrics = truth_metrics(approximation[central], truth[central])
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "method": name,
                "samples": len(signal),
                "sampling_rate_hz": sampling_rate,
                "runtime_ms": 1000.0 * elapsed,
                "tail_fraction": tail_fraction,
                "converged": converged,
                **metrics,
                "central_truth_rmse": central_metrics["truth_rmse"],
            }
        )
    return rows, curves


def main() -> None:
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    all_rows: list[dict[str, object]] = []
    plot_cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []

    for seed in range(10):
        time_axis, signal, truth = generate_positive_biosignal(
            duration_seconds=20.0, sampling_rate=100.0, noise_std=0.02, seed=seed
        )
        if seed % 2 == 1:
            rng = np.random.default_rng(seed + 10_000)
            spike_locations = rng.choice(np.arange(20, len(signal) - 20), size=4, replace=False)
            signal[spike_locations] -= 1.5
            scenario = "generic_with_negative_impulses"
        else:
            scenario = "generic_clean_noise"
        rows, curves = evaluate(scenario, signal, truth, 100.0, seed)
        all_rows.extend(rows)
        if seed == 0:
            plot_cases.append(("Generic one-sided biosignal", time_axis, signal, truth, curves))

    for seed in range(10):
        demo = generate_ecg(
            duration_seconds=20.0,
            sampling_rate=250.0,
            heart_rate_bpm=72.0,
            noise_std=0.015,
            negative_spikes=3 if seed % 2 else 0,
            seed=seed,
        )
        scenario = "ecg_with_negative_impulses" if seed % 2 else "ecg_clean_noise"
        rows, curves = evaluate(
            scenario, demo.signal, demo.true_lower_envelope, demo.sampling_rate, seed
        )
        all_rows.extend(rows)
        if seed == 0:
            plot_cases.append(
                (
                    "ECG-like recurring troughs",
                    demo.time,
                    demo.signal,
                    demo.true_lower_envelope,
                    curves,
                )
            )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_dir / "validation_metrics.csv", index=False)
    summary = (
        frame.groupby(["scenario", "method"], as_index=False)
        .agg(
            runs=("seed", "count"),
            rmse_mean=("truth_rmse", "mean"),
            rmse_std=("truth_rmse", "std"),
            bias_mean=("truth_bias", "mean"),
            tail_fraction_mean=("tail_fraction", "mean"),
            runtime_ms_median=("runtime_ms", "median"),
            convergence_rate=("converged", "mean"),
        )
        .sort_values(["scenario", "rmse_mean"])
    )
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(len(plot_cases), 1, figsize=(14, 7.5), sharex=False)
    axes_array = np.atleast_1d(axes)
    colors = {
        "middle_whittaker": "#7c3aed",
        "quantile_tau_0.05": "#e11d48",
        "quantile_tau_0.01": "#f97316",
        "guarded_minima": "#0891b2",
        "causal_kalman": "#64748b",
    }
    for axis, (title, time_axis, signal, truth, curves) in zip(axes_array, plot_cases, strict=True):
        selection = time_axis <= min(8.0, time_axis[-1])
        axis.plot(
            time_axis[selection], signal[selection], color="#334155", linewidth=0.8, label="signal"
        )
        axis.plot(
            time_axis[selection], truth[selection], color="#16a34a", linewidth=2.2, label="truth"
        )
        for name in [
            "middle_whittaker",
            "quantile_tau_0.05",
            "quantile_tau_0.01",
            "guarded_minima",
        ]:
            axis.plot(
                time_axis[selection],
                curves[name][selection],
                color=colors[name],
                linewidth=1.5,
                label=name,
            )
        axis.set_title(title)
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.2)
    axes_array[-1].set_xlabel("Time (s)")
    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output_dir / "synthetic_validation.png", dpi=170)
    plt.close(figure)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
