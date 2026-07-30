from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lowpoint.cli import main


def _write_fixture(path: Path) -> None:
    path.write_text(
        "time_s,lead\n0.00,1.0\n0.01,0.8\n0.02,1.2\n0.03,0.7\n0.04,1.1\n0.05,0.9\n",
        encoding="utf-8",
    )


def test_cli_processes_consistent_time_and_rate(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    _write_fixture(source)

    assert (
        main(
            [
                str(source),
                str(output),
                "--signal-column",
                "lead",
                "--time-column",
                "time_s",
                "--sampling-rate",
                "100",
                "--smoothness-hz",
                "25",
            ]
        )
        == 0
    )
    assert output.exists()
    assert list(pd.read_csv(output).columns) == [
        "time_s",
        "raw_signal",
        "conditioned_signal",
        "estimated_baseline",
        "removed_component",
        "envelope_on_conditioned",
        "conditioned_minus_envelope",
        "original_sample_valid",
        "estimator_sample_valid",
    ]


def test_cli_rejects_rate_inconsistent_with_time_column(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    _write_fixture(source)

    with pytest.raises(SystemExit, match="differs"):
        main(
            [
                str(source),
                str(tmp_path / "output.csv"),
                "--signal-column",
                "lead",
                "--time-column",
                "time_s",
                "--sampling-rate",
                "250",
            ]
        )


def test_cli_runs_conditioning_and_full_kalman_controls(tmp_path: Path) -> None:
    sampling_rate = 250.0
    time = np.arange(1000, dtype=float) / sampling_rate
    raw = (
        0.25 * np.sin(2 * np.pi * 0.2 * time)
        + np.sin(2 * np.pi * 3.0 * time)
        + 0.08 * np.sin(2 * np.pi * 50.0 * time)
    )
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    pd.DataFrame({"time_s": time, "lead": raw}).to_csv(source, index=False)

    assert (
        main(
            [
                str(source),
                str(output),
                "--signal-column",
                "lead",
                "--time-column",
                "time_s",
                "--method",
                "kalman",
                "--filter-phase",
                "causal",
                "--mains-hz",
                "50",
                "--lowpass-hz",
                "40",
                "--baseline-method",
                "median",
                "--baseline-window-seconds",
                "0.4",
                "--kalman-process-variance",
                "0.0003",
                "--kalman-measurement-variance",
                "0.1",
                "--kalman-innovation-clip",
                "3.5",
                "--kalman-warmup-seconds",
                "0",
            ]
        )
        == 0
    )
    processed = pd.read_csv(output)
    assert not np.allclose(processed["raw_signal"], processed["conditioned_signal"])
    np.testing.assert_allclose(
        processed["conditioned_minus_envelope"],
        processed["conditioned_signal"] - processed["envelope_on_conditioned"],
    )
