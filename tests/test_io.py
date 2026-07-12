from io import StringIO

import numpy as np
import pytest

from lowpoint.io import infer_sampling_rate, numeric_columns, read_table


def test_read_table_autodetects_delimiter() -> None:
    frame = read_table(StringIO("time;lead\n0;1.2\n0.01;1.4\n0.02;1.1\n"))
    assert numeric_columns(frame) == ["time", "lead"]


def test_sampling_rate_and_jitter() -> None:
    rate, jitter = infer_sampling_rate(np.arange(100) / 250.0)
    assert rate == pytest.approx(250.0)
    assert jitter < 1e-12
