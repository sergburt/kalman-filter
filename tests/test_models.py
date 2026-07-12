import pytest

from lowpoint.models import EnvelopeConfig


def test_default_config_is_valid() -> None:
    EnvelopeConfig().validate(250.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantile", 0.0),
        ("quantile", 0.5),
        ("smoothness_hz", 0.0),
        ("minima_overlap", 1.0),
        ("guard_seconds", -0.1),
        ("kalman_process_variance", 0.0),
        ("edge_padding_seconds", float("nan")),
        ("minima_window_seconds", float("inf")),
        ("kalman_measurement_variance", float("nan")),
    ],
)
def test_invalid_config_is_rejected(field: str, value: float) -> None:
    config = EnvelopeConfig(**{field: value})
    with pytest.raises(ValueError):
        config.validate(250.0)


def test_cutoff_above_nyquist_is_rejected() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        EnvelopeConfig(smoothness_hz=60.0).validate(100.0)
