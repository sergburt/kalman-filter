import numpy as np
import pytest

from lowpoint import EnvelopeConfig, SignalFilterConfig, estimate_envelope, preprocess_signal


def _tone_amplitudes(
    signal: np.ndarray,
    sampling_rate: float,
    frequencies: list[float],
    trim_seconds: float,
) -> np.ndarray:
    trim = int(round(trim_seconds * sampling_rate))
    indices = np.arange(trim, len(signal) - trim)
    time = indices / sampling_rate
    columns: list[np.ndarray] = []
    for frequency in frequencies:
        columns.extend(
            [
                np.sin(2 * np.pi * frequency * time),
                np.cos(2 * np.pi * frequency * time),
            ]
        )
    columns.extend([np.ones_like(time), time - np.mean(time)])
    coefficients = np.linalg.lstsq(np.column_stack(columns), signal[indices], rcond=None)[0]
    return np.asarray(
        [
            np.hypot(coefficients[2 * index], coefficients[2 * index + 1])
            for index in range(len(frequencies))
        ]
    )


def test_disabled_conditioning_is_identity_and_does_not_mutate() -> None:
    rng = np.random.default_rng(20260712)
    signal = rng.normal(size=1000)
    original = signal.copy()
    result = preprocess_signal(signal, 250.0, SignalFilterConfig())
    np.testing.assert_array_equal(signal, original)
    np.testing.assert_array_equal(result.prepared_signal, original)
    np.testing.assert_array_equal(result.processed_signal, original)
    np.testing.assert_array_equal(result.removed_component, 0.0)
    assert not result.diagnostics["filtering_active"]


def test_zero_phase_gaps_use_linear_interpolation_and_preserve_mask() -> None:
    signal = np.array([np.nan, 1.0, np.nan, 3.0, np.inf, 5.0, np.nan])
    result = preprocess_signal(signal, 100.0, SignalFilterConfig())
    np.testing.assert_allclose(result.prepared_signal, [1, 1, 2, 3, 4, 5, 5])
    np.testing.assert_array_equal(result.valid_mask, [False, True, False, True, False, True, False])


def test_external_validity_mask_reaches_envelope_metrics() -> None:
    signal = np.sin(np.linspace(0, 8 * np.pi, 1000))
    signal[100:120] = np.nan
    conditioned = preprocess_signal(signal, 100.0, SignalFilterConfig())
    envelope = estimate_envelope(
        conditioned.processed_signal,
        100.0,
        EnvelopeConfig(smoothness_hz=0.3),
        valid_mask=conditioned.valid_mask,
    )
    assert envelope.diagnostics["interpolated_sample_count"] == 20
    assert not np.any(envelope.valid_mask[100:120])


@pytest.mark.parametrize("phase_mode", ["zero_phase", "causal"])
@pytest.mark.parametrize("mains_frequency", [50.0, 60.0])
def test_mains_notch_suppresses_fundamental_and_harmonic(
    phase_mode: str, mains_frequency: float
) -> None:
    sampling_rate = 500.0
    time = np.arange(int(12 * sampling_rate)) / sampling_rate
    signal = (
        0.8 * np.sin(2 * np.pi * 7 * time + 0.2)
        + 0.5 * np.sin(2 * np.pi * mains_frequency * time - 0.4)
        + 0.25 * np.sin(2 * np.pi * 2 * mains_frequency * time + 0.8)
    )
    result = preprocess_signal(
        signal,
        sampling_rate,
        SignalFilterConfig(
            phase_mode=phase_mode,
            mains_enabled=True,
            mains_frequency_hz=mains_frequency,
            mains_quality_factor=30.0,
            mains_harmonics=2,
        ),
    )
    frequencies = [7.0, mains_frequency, 2 * mains_frequency]
    before = _tone_amplitudes(signal, sampling_rate, frequencies, 2.0)
    after = _tone_amplitudes(result.processed_signal, sampling_rate, frequencies, 2.0)
    ratios = after / before
    assert 0.98 <= ratios[0] <= 1.02
    assert ratios[1] < 0.02
    assert ratios[2] < 0.02


@pytest.mark.parametrize("phase_mode", ["zero_phase", "causal"])
def test_highpass_suppresses_slow_tone(phase_mode: str) -> None:
    sampling_rate = 250.0
    time = np.arange(int(30 * sampling_rate)) / sampling_rate
    signal = 0.7 * np.sin(2 * np.pi * 0.2 * time + 0.31) + 0.4 * np.sin(2 * np.pi * 5 * time - 0.7)
    result = preprocess_signal(
        signal,
        sampling_rate,
        SignalFilterConfig(
            phase_mode=phase_mode,
            highpass_enabled=True,
            highpass_cutoff_hz=1.0,
            highpass_order=4,
        ),
    )
    before = _tone_amplitudes(signal, sampling_rate, [0.2, 5.0], 5.0)
    after = _tone_amplitudes(result.processed_signal, sampling_rate, [0.2, 5.0], 5.0)
    ratios = after / before
    assert ratios[0] < 0.01
    assert 0.98 <= ratios[1] <= 1.02


@pytest.mark.parametrize("phase_mode", ["zero_phase", "causal"])
def test_lowpass_suppresses_fast_tone(phase_mode: str) -> None:
    sampling_rate = 250.0
    time = np.arange(int(30 * sampling_rate)) / sampling_rate
    signal = 0.4 * np.sin(2 * np.pi * 5 * time - 0.7) + 0.3 * np.sin(2 * np.pi * 80 * time + 0.31)
    result = preprocess_signal(
        signal,
        sampling_rate,
        SignalFilterConfig(
            phase_mode=phase_mode,
            lowpass_enabled=True,
            lowpass_cutoff_hz=20.0,
            lowpass_order=4,
        ),
    )
    before = _tone_amplitudes(signal, sampling_rate, [5.0, 80.0], 5.0)
    after = _tone_amplitudes(result.processed_signal, sampling_rate, [5.0, 80.0], 5.0)
    ratios = after / before
    assert 0.98 <= ratios[0] <= 1.02
    assert ratios[1] < 0.01


@pytest.mark.parametrize(("phase_mode", "maximum_nrmse"), [("zero_phase", 0.02), ("causal", 0.12)])
def test_highpass_baseline_recovery(phase_mode: str, maximum_nrmse: float) -> None:
    sampling_rate = 250.0
    time = np.arange(int(60 * sampling_rate)) / sampling_rate
    drift = 0.6 * np.sin(2 * np.pi * 0.15 * time + 0.31) + 0.2 * np.sin(
        2 * np.pi * 0.03 * time - 0.4
    )
    content = 0.25 * np.sin(2 * np.pi * 8 * time + 0.7) + 0.12 * np.sin(2 * np.pi * 13 * time - 1.1)
    signal = drift + content
    result = preprocess_signal(
        signal,
        sampling_rate,
        SignalFilterConfig(
            phase_mode=phase_mode,
            baseline_method="highpass",
            baseline_cutoff_hz=0.5,
            baseline_order=4,
        ),
    )
    trim = int(5 * sampling_rate)
    error = result.baseline_estimate[trim:-trim] - drift[trim:-trim]
    nrmse = np.sqrt(np.mean(error * error)) / np.std(drift[trim:-trim])
    assert nrmse < maximum_nrmse
    np.testing.assert_allclose(
        result.processed_signal + result.baseline_estimate, signal, atol=5e-12
    )


def test_centered_median_baseline_recovery() -> None:
    sampling_rate = 250.0
    time = np.arange(int(60 * sampling_rate)) / sampling_rate
    drift = 0.6 * np.sin(2 * np.pi * 0.15 * time + 0.31) + 0.2 * np.sin(
        2 * np.pi * 0.03 * time - 0.4
    )
    content = 0.25 * np.sin(2 * np.pi * 8 * time + 0.7) + 0.12 * np.sin(2 * np.pi * 13 * time - 1.1)
    signal = drift + content
    result = preprocess_signal(
        signal,
        sampling_rate,
        SignalFilterConfig(baseline_method="median", baseline_window_seconds=1.0),
    )
    trim = int(5 * sampling_rate)
    error = result.baseline_estimate[trim:-trim] - drift[trim:-trim]
    nrmse = np.sqrt(np.mean(error * error)) / np.std(drift[trim:-trim])
    assert nrmse < 0.08
    np.testing.assert_allclose(
        result.processed_signal + result.baseline_estimate, signal, atol=5e-12
    )


def test_causal_chain_is_prefix_invariant_with_an_interior_gap() -> None:
    rng = np.random.default_rng(20260712)
    signal = rng.normal(size=5000)
    signal[1000:1020] = np.nan
    prefix_length = 3187
    config = SignalFilterConfig(
        phase_mode="causal",
        mains_enabled=True,
        mains_frequency_hz=50.0,
        mains_quality_factor=25.0,
        mains_harmonics=2,
        highpass_enabled=True,
        highpass_cutoff_hz=0.7,
        highpass_order=3,
        lowpass_enabled=True,
        lowpass_cutoff_hz=40.0,
        lowpass_order=5,
    )
    full = preprocess_signal(signal, 250.0, config)
    prefix = preprocess_signal(signal[:prefix_length], 250.0, config)
    np.testing.assert_allclose(
        full.processed_signal[:prefix_length], prefix.processed_signal, rtol=1e-12, atol=1e-12
    )


def test_causal_trailing_median_is_prefix_invariant() -> None:
    rng = np.random.default_rng(20260712)
    signal = rng.normal(size=5000)
    prefix_length = 3187
    config = SignalFilterConfig(
        phase_mode="causal", baseline_method="median", baseline_window_seconds=0.4
    )
    full = preprocess_signal(signal, 250.0, config)
    prefix = preprocess_signal(signal[:prefix_length], 250.0, config)
    np.testing.assert_allclose(
        full.processed_signal[:prefix_length], prefix.processed_signal, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        full.baseline_estimate[:prefix_length], prefix.baseline_estimate, rtol=0, atol=0
    )


def test_causal_conditioning_rejects_a_leading_gap() -> None:
    signal = np.arange(100.0)
    signal[0] = np.nan
    with pytest.raises(ValueError, match="first sample"):
        preprocess_signal(signal, 100.0, SignalFilterConfig(phase_mode="causal"))


def test_zero_phase_lowpass_preserves_impulse_location_and_symmetry() -> None:
    signal = np.zeros(4001)
    center = 2000
    signal[center] = 1.0
    result = preprocess_signal(
        signal,
        250.0,
        SignalFilterConfig(lowpass_enabled=True, lowpass_cutoff_hz=20.0, lowpass_order=4),
    )
    assert int(np.argmax(result.processed_signal)) == center
    np.testing.assert_allclose(
        result.processed_signal[center - 500 : center],
        result.processed_signal[center + 1 : center + 501][::-1],
        atol=2e-10,
    )


@pytest.mark.parametrize(
    "config",
    [
        SignalFilterConfig(phase_mode="invalid"),  # type: ignore[arg-type]
        SignalFilterConfig(baseline_method="invalid"),  # type: ignore[arg-type]
        SignalFilterConfig(mains_enabled=True, mains_quality_factor=0),
        SignalFilterConfig(mains_enabled=True, mains_harmonics=1.5),  # type: ignore[arg-type]
        SignalFilterConfig(highpass_enabled=True, highpass_cutoff_hz=10, highpass_order=0),
        SignalFilterConfig(lowpass_enabled=True, lowpass_cutoff_hz=float("nan")),
        SignalFilterConfig(
            highpass_enabled=True,
            highpass_cutoff_hz=30,
            lowpass_enabled=True,
            lowpass_cutoff_hz=20,
        ),
    ],
)
def test_invalid_filter_config_is_rejected(config: SignalFilterConfig) -> None:
    with pytest.raises(ValueError):
        config.validate(100.0)


def test_notch_at_nyquist_is_rejected_but_high_harmonics_are_skipped() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SignalFilterConfig(mains_enabled=True, mains_frequency_hz=50).validate(100.0)

    result = preprocess_signal(
        np.sin(np.linspace(0, 100, 1000)),
        250.0,
        SignalFilterConfig(mains_enabled=True, mains_frequency_hz=50, mains_harmonics=4),
    )
    stage = result.diagnostics["stages"][0]
    assert stage["frequencies_hz"] == [50.0, 100.0]
    assert stage["skipped_frequencies_hz"] == [150.0, 200.0]


def test_zero_phase_short_signal_and_oversized_median_raise() -> None:
    with pytest.raises(ValueError, match="requires more than"):
        preprocess_signal(
            np.arange(10.0),
            250.0,
            SignalFilterConfig(lowpass_enabled=True, lowpass_cutoff_hz=40),
        )
    with pytest.raises(ValueError, match="longer than the signal"):
        preprocess_signal(
            np.arange(100.0),
            100.0,
            SignalFilterConfig(baseline_method="median", baseline_window_seconds=2.0),
        )
