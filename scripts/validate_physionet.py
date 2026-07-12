"""Small, streamed PhysioNet smoke validation without bulk dataset downloads.

This script evaluates additive-drift tracking, not a nonexistent public
"lower-envelope ground truth". It streams two short, public ECG segments with
WFDB and records exact source intervals in the output table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from lowpoint import EnvelopeConfig, estimate_envelope

ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[dict[str, object]]:
    return [
        {
            "dataset": "MIT-BIH Arrhythmia Database 1.0.0",
            "record": "100",
            "pn_dir": "mitdb/1.0.0",
            "start_s": 300,
            "duration_s": 20,
            "channel": 0,
            "annotation": "atr",
            "source_url": "https://physionet.org/content/mitdb/1.0.0/",
        },
        {
            "dataset": "QT Database 1.0.0",
            "record": "sel100",
            "pn_dir": "qtdb/1.0.0",
            "start_s": 600,
            "duration_s": 20,
            "channel": 0,
            "annotation": "q1c",
            "source_url": "https://physionet.org/content/qtdb/1.0.0/",
        },
    ]


def _stream(spec: dict[str, object]) -> tuple[np.ndarray, float, np.ndarray, int]:
    header = wfdb.rdheader(str(spec["record"]), pn_dir=str(spec["pn_dir"]))
    sampling_rate = float(header.fs)
    start = int(round(float(spec["start_s"]) * sampling_rate))
    stop = start + int(round(float(spec["duration_s"]) * sampling_rate))
    signal, metadata = wfdb.rdsamp(
        str(spec["record"]),
        pn_dir=str(spec["pn_dir"]),
        sampfrom=start,
        sampto=stop,
        channels=[int(spec["channel"])],
    )
    annotation = wfdb.rdann(
        str(spec["record"]),
        str(spec["annotation"]),
        pn_dir=str(spec["pn_dir"]),
        sampfrom=start,
        sampto=stop,
        shift_samps=True,
    )
    if float(metadata["fs"]) != sampling_rate:
        raise RuntimeError("WFDB header and streamed metadata sampling rates differ")
    # Both selected intervals use ``N`` for the QRS/R fiducial. QT ``q1c`` also
    # contains P/QRS/T boundary and peak events; those must not be treated as QRS
    # centers when calculating a QRS-window amplitude metric.
    qrs_mask = np.asarray(annotation.symbol) == "N"
    qrs_samples = np.asarray(annotation.sample, dtype=int)[qrs_mask]
    return signal[:, 0].astype(float), sampling_rate, qrs_samples, len(annotation.sample)


def _qrs_peak_to_peak_error(
    original: np.ndarray,
    corrected: np.ndarray,
    annotations: np.ndarray,
    sampling_rate: float,
) -> float:
    half_window = max(1, int(round(0.08 * sampling_rate)))
    errors: list[float] = []
    for center in annotations:
        start = max(0, int(center) - half_window)
        stop = min(original.size, int(center) + half_window + 1)
        if stop - start < 3:
            continue
        original_range = float(np.ptp(original[start:stop]))
        corrected_range = float(np.ptp(corrected[start:stop]))
        errors.append(abs(corrected_range - original_range))
    return float(np.mean(errors)) if errors else float("nan")


def main() -> None:
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    methods = {
        "quantile_tau_0.05": EnvelopeConfig(method="quantile", quantile=0.05, smoothness_hz=0.35),
        "guarded_minima": EnvelopeConfig(
            method="minima", minima_window_seconds=0.85, guard_seconds=0.012
        ),
        "causal_kalman": EnvelopeConfig(method="kalman", quantile=0.05),
    }
    result_rows: list[dict[str, object]] = []
    plot_rows: list[
        tuple[dict[str, object], np.ndarray, float, np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ] = []

    for spec in _records():
        signal, sampling_rate, qrs_annotations, annotation_event_count = _stream(spec)
        time_axis = np.arange(signal.size, dtype=float) / sampling_rate
        drift = 0.22 * np.sin(2 * np.pi * 0.18 * time_axis + 0.3) + 0.06 * np.sin(
            2 * np.pi * 0.05 * time_axis - 0.7
        )
        drifted = signal + drift
        plotted: dict[str, np.ndarray] = {}
        for method_name, config in methods.items():
            started = time.perf_counter()
            base_result = estimate_envelope(signal, sampling_rate, config)
            drifted_result = estimate_envelope(drifted, sampling_rate, config)
            elapsed = time.perf_counter() - started
            estimated_drift = drifted_result.approximation - base_result.approximation
            corrected = drifted - estimated_drift
            error = corrected - signal
            correlation = float(np.corrcoef(signal, corrected)[0, 1])
            qrs_error = _qrs_peak_to_peak_error(signal, corrected, qrs_annotations, sampling_rate)
            result_rows.append(
                {
                    **spec,
                    "sampling_rate_hz": sampling_rate,
                    "samples": signal.size,
                    "annotation_events": annotation_event_count,
                    "qrs_fiducials": qrs_annotations.size,
                    "method": method_name,
                    "runtime_two_fits_ms": elapsed * 1000,
                    "drift_rmse_mv": float(np.sqrt(np.mean((estimated_drift - drift) ** 2))),
                    "corrected_waveform_rmse_mv": float(np.sqrt(np.mean(error * error))),
                    "corrected_waveform_prd_percent": float(
                        100 * np.linalg.norm(error) / np.linalg.norm(signal - np.mean(signal))
                    ),
                    "corrected_correlation": correlation,
                    "qrs_peak_to_peak_mae_mv": qrs_error,
                    "raw_tail_fraction": base_result.diagnostics["empirical_tail_fraction"],
                    "converged": base_result.diagnostics["converged"]
                    and drifted_result.diagnostics["converged"],
                }
            )
            plotted[method_name] = estimated_drift
        plot_rows.append((spec, time_axis, sampling_rate, signal, drift, plotted))

    frame = pd.DataFrame(result_rows)
    frame.to_csv(output_dir / "physionet_validation.csv", index=False)
    manifest = {
        "purpose": "additive-drift tracking smoke test; not clinical validation",
        "license": "Open Data Commons Attribution License v1.0",
        "rows": result_rows,
    }
    (output_dir / "physionet_validation.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(len(plot_rows), 1, figsize=(14, 7), sharex=False)
    axes_array = np.atleast_1d(axes)
    colors = {
        "quantile_tau_0.05": "#e11d48",
        "guarded_minima": "#0891b2",
        "causal_kalman": "#64748b",
    }
    for axis, (spec, time_axis, _sampling_rate, _signal, drift, plotted) in zip(
        axes_array, plot_rows, strict=True
    ):
        axis.plot(time_axis, drift, color="#16a34a", linewidth=2.5, label="known added drift")
        for method_name, estimate in plotted.items():
            axis.plot(
                time_axis,
                estimate,
                color=colors[method_name],
                linewidth=1.5,
                label=method_name,
            )
        axis.set_title(f"{spec['dataset']} — record {spec['record']}, channel {spec['channel']}")
        axis.set_ylabel("Drift (mV)")
        axis.grid(alpha=0.2)
    axes_array[-1].set_xlabel("Segment time (s)")
    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_dir / "physionet_validation.png", dpi=170)
    plt.close(figure)

    selected = [
        "dataset",
        "record",
        "method",
        "drift_rmse_mv",
        "corrected_waveform_prd_percent",
        "corrected_correlation",
        "qrs_peak_to_peak_mae_mv",
    ]
    print(frame[selected].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
