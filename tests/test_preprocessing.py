import numpy as np
import pytest

from lowpoint.preprocessing import prepare_signal, robust_location_scale


def test_missing_values_are_interpolated_and_masked() -> None:
    signal = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, 5.0])
    prepared, mask = prepare_signal(signal)
    np.testing.assert_allclose(prepared, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(mask, [False, True, False, True, False, True])


def test_constant_signal_has_positive_fallback_scale() -> None:
    location, scale = robust_location_scale(np.ones(20))
    assert location == 1.0
    assert scale > 0


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.arange(4.0)])
def test_invalid_shape_or_length_is_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        prepare_signal(bad)
