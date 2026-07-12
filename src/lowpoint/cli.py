"""Repeatable command-line processing for CSV biosignals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import export_frame, infer_sampling_rate, read_table
from .models import EnvelopeConfig
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
    )
    result = estimate_envelope(signal, sampling_rate, config)
    output = export_frame(time, signal, result.approximation, result.residual, result.valid_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(json.dumps(result.diagnostics, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
