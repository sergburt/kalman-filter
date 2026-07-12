"""Repeatable command-line processing for CSV biosignals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .filtering import preprocess_signal
from .io import export_frame, infer_sampling_rate, read_table
from .models import EnvelopeConfig, SignalFilterConfig
from .pipeline import estimate_envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approximate a biosignal lower/upper envelope")
    parser.add_argument("input", type=Path, help="Input CSV/TSV file")
    parser.add_argument("output", type=Path, help="Output CSV file")
    parser.add_argument("--signal-column", required=True)
    parser.add_argument("--time-column", help="Strictly increasing timestamps in seconds")
    parser.add_argument("--sampling-rate", type=float)
    parser.add_argument("--method", choices=["quantile", "minima", "kalman"], default="quantile")
    parser.add_argument("--side", choices=["lower", "upper"], default="lower")
    parser.add_argument("--quantile", type=float, default=0.05)
    parser.add_argument("--smoothness-hz", type=float, default=0.35)
    parser.add_argument("--window-seconds", type=float, default=0.80)
    parser.add_argument("--guard-seconds", type=float, default=0.012)
    parser.add_argument("--kalman-process-variance", type=float, default=2e-4)
    parser.add_argument("--kalman-measurement-variance", type=float, default=0.08)
    parser.add_argument("--kalman-innovation-clip", type=float, default=4.0)
    parser.add_argument("--kalman-warmup-seconds", type=float, default=2.0)

    conditioning = parser.add_argument_group("optional signal conditioning")
    conditioning.add_argument(
        "--filter-phase",
        choices=["zero-phase", "causal"],
        default="zero-phase",
        help="zero-phase is offline forward/backward processing; causal is single-pass",
    )
    conditioning.add_argument(
        "--mains-hz", type=float, help="enable a mains notch at this fundamental frequency"
    )
    conditioning.add_argument("--mains-q", type=float, default=30.0)
    conditioning.add_argument("--mains-harmonics", type=int, default=1)
    conditioning.add_argument(
        "--highpass-hz", type=float, help="enable a general high-pass at this cutoff"
    )
    conditioning.add_argument("--highpass-order", type=int, default=2)
    conditioning.add_argument(
        "--lowpass-hz", type=float, help="enable a general low-pass at this cutoff"
    )
    conditioning.add_argument("--lowpass-order", type=int, default=4)
    conditioning.add_argument(
        "--baseline-method", choices=["none", "median", "highpass"], default="none"
    )
    conditioning.add_argument("--baseline-window-seconds", type=float, default=0.8)
    conditioning.add_argument("--baseline-cutoff-hz", type=float, default=0.5)
    conditioning.add_argument("--baseline-order", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = read_table(args.input)
    if args.signal_column not in frame:
        raise SystemExit(f"Unknown signal column: {args.signal_column!r}")
    signal = frame[args.signal_column].to_numpy(dtype=float)

    if args.time_column:
        if args.time_column not in frame:
            raise SystemExit(f"Unknown time column: {args.time_column!r}")
        time = frame[args.time_column].to_numpy(dtype=float)
        inferred_rate, jitter = infer_sampling_rate(time)
        if jitter > 0.01:
            raise SystemExit(
                f"Time-column interval jitter is {100 * jitter:.2f}%; resample to a uniform grid"
            )
        if args.sampling_rate is not None:
            relative_difference = abs(args.sampling_rate - inferred_rate) / inferred_rate
            if relative_difference > 0.01:
                raise SystemExit(
                    "--sampling-rate differs from the time-column estimate by more than 1%"
                )
            sampling_rate = args.sampling_rate
        else:
            sampling_rate = inferred_rate
    elif args.sampling_rate is not None:
        sampling_rate = args.sampling_rate
        time = np.arange(signal.size, dtype=float) / sampling_rate
    else:
        raise SystemExit("Provide --sampling-rate or --time-column")

    config = EnvelopeConfig(
        method=args.method,
        side=args.side,
        quantile=args.quantile,
        smoothness_hz=args.smoothness_hz,
        minima_window_seconds=args.window_seconds,
        guard_seconds=args.guard_seconds,
        kalman_process_variance=args.kalman_process_variance,
        kalman_measurement_variance=args.kalman_measurement_variance,
        kalman_innovation_clip=args.kalman_innovation_clip,
        kalman_warmup_seconds=args.kalman_warmup_seconds,
    )
    filter_config = SignalFilterConfig(
        phase_mode=args.filter_phase.replace("-", "_"),
        mains_enabled=args.mains_hz is not None,
        mains_frequency_hz=50.0 if args.mains_hz is None else args.mains_hz,
        mains_quality_factor=args.mains_q,
        mains_harmonics=args.mains_harmonics,
        highpass_enabled=args.highpass_hz is not None,
        highpass_cutoff_hz=0.5 if args.highpass_hz is None else args.highpass_hz,
        highpass_order=args.highpass_order,
        lowpass_enabled=args.lowpass_hz is not None,
        lowpass_cutoff_hz=40.0 if args.lowpass_hz is None else args.lowpass_hz,
        lowpass_order=args.lowpass_order,
        baseline_method=args.baseline_method,
        baseline_window_seconds=args.baseline_window_seconds,
        baseline_cutoff_hz=args.baseline_cutoff_hz,
        baseline_order=args.baseline_order,
    )
    filter_result = preprocess_signal(signal, sampling_rate, filter_config)
    result = estimate_envelope(
        filter_result.processed_signal,
        sampling_rate,
        config,
        valid_mask=filter_result.valid_mask,
    )
    output = export_frame(
        time,
        filter_result.raw_signal,
        result.approximation,
        result.residual,
        result.valid_mask,
        processed_signal=filter_result.processed_signal,
        baseline_estimate=filter_result.baseline_estimate,
        removed_component=filter_result.removed_component,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    diagnostics = {
        "conditioning_config": filter_config.to_dict(),
        "conditioning_diagnostics": filter_result.diagnostics,
        "envelope_config": config.to_dict(),
        "envelope_diagnostics": result.diagnostics,
        "pipeline": {
            "envelope_target": "conditioned_signal",
            "residual_definition": "conditioned_signal - envelope_on_conditioned",
        },
    }
    print(json.dumps(diagnostics, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
