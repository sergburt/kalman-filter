from pathlib import Path

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
                "1",
            ]
        )
        == 0
    )
    assert output.exists()


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
