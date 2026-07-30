"""Streamlit research application for LowPoint Biosignals."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lowpoint.filtering import preprocess_signal
from lowpoint.io import export_frame, infer_sampling_rate, numeric_columns, read_table
from lowpoint.metrics import truth_metrics
from lowpoint.models import EnvelopeConfig, SignalFilterConfig
from lowpoint.pipeline import estimate_envelope
from lowpoint.synthetic import generate_ecg

METHOD_LABELS = {
    "Robust lower quantile (recommended)": "quantile",
    "Guarded block minima + PCHIP (literal)": "minima",
    "Experimental causal Kalman (expectile-like)": "kalman",
}

SYNTHETIC_SCENARIOS = {
    "Default ECG (noise + isolated impulses)": "none",
    "Sustained contact / movement artifact": "sustained_contact_movement",
}

APP_RESULT_SCHEMA_VERSION = 2


def _jsonable(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _load_uploaded_data(uploaded_file: object) -> tuple[pd.DataFrame, list[str]]:
    frame = read_table(uploaded_file)
    columns = numeric_columns(frame)
    if not columns:
        raise ValueError("The file has no numeric columns.")
    return frame, columns


def _decimation_indices(length: int, maximum: int = 25_000) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum).astype(int))


def _clamp_float_widget(key: str, default: float, minimum: float, maximum: float) -> float:
    bounded_default = float(np.clip(default, minimum, maximum))
    if key in st.session_state:
        bounded = float(np.clip(float(st.session_state[key]), minimum, maximum))
        if bounded != st.session_state[key]:
            st.session_state[key] = bounded
        return bounded
    return bounded_default


def _plot_result(bundle: dict[str, object]) -> go.Figure:
    time = np.asarray(bundle["time"])
    raw_signal = np.asarray(bundle["raw_signal"])
    filter_result = bundle["filter_result"]
    conditioned_signal = np.asarray(filter_result.processed_signal)
    filtering_active = bool(filter_result.diagnostics["filtering_active"])
    result = bundle["result"]
    truth = bundle.get("truth")
    synthetic_baseline = bundle.get("synthetic_baseline")
    synthetic_artifact = bundle.get("synthetic_artifact")
    simulated_dropout_excluded = bool(bundle.get("simulated_dropout_excluded", False))
    dropout_gating_confirmed = bool(
        simulated_dropout_excluded
        and "skipped_measurement_updates" in result.diagnostics
        and "reacquisition_updates" in result.diagnostics
    )
    index = _decimation_indices(time.size)

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.30, 0.46, 0.24],
        subplot_titles=(
            "Raw and conditioned signal",
            "Conditioned signal and envelope",
            "Conditioned signal minus envelope",
        ),
    )
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=raw_signal[index],
            name="Raw signal",
            line={"color": "#64748b", "width": 1},
        ),
        row=1,
        col=1,
    )
    if filtering_active:
        figure.add_trace(
            go.Scattergl(
                x=time[index],
                y=conditioned_signal[index],
                name="Conditioned signal",
                line={"color": "#0369a1", "width": 1.4},
            ),
            row=1,
            col=1,
        )
    if filter_result.config.baseline_method != "none":
        figure.add_trace(
            go.Scattergl(
                x=time[index],
                y=np.asarray(filter_result.baseline_estimate)[index],
                name="Estimated baseline removed",
                line={"color": "#7c3aed", "width": 1.5, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=conditioned_signal[index],
            name="Envelope input (conditioned)",
            line={"color": "#334155", "width": 1},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=result.approximation[index],
            name="Estimated envelope",
            line={"color": "#e11d48", "width": 2.5},
        ),
        row=2,
        col=1,
    )
    if truth is not None:
        truth_array = np.asarray(truth)
        figure.add_trace(
            go.Scattergl(
                x=time[index],
                y=truth_array[index],
                name="True lower envelope",
                line={"color": "#16a34a", "width": 2, "dash": "dash"},
            ),
            row=2,
            col=1,
        )
    if synthetic_baseline is not None:
        baseline_array = np.asarray(synthetic_baseline)
        figure.add_trace(
            go.Scattergl(
                x=time[index],
                y=baseline_array[index],
                name="Isoelectric drift (different target)",
                line={"color": "#7c3aed", "width": 1.5, "dash": "dot"},
            ),
            row=2,
            col=1,
        )
    if result.support_indices.size:
        supports = result.support_indices
        figure.add_trace(
            go.Scatter(
                x=time[supports],
                y=result.support_values,
                mode="markers",
                name="Guarded low-point supports",
                marker={"color": "#f59e0b", "size": 7, "symbol": "diamond"},
            ),
            row=2,
            col=1,
        )
    residual_display = np.asarray(result.residual).copy()
    if dropout_gating_confirmed and synthetic_artifact is not None:
        residual_display[synthetic_artifact.dropout_mask] = np.nan
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=residual_display[index],
            name="Residual",
            line={"color": "#0369a1", "width": 1.2},
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    figure.add_hline(y=0, line={"color": "#94a3b8", "width": 1}, row=3, col=1)
    if synthetic_artifact is not None:
        for row in range(1, 4):
            event_options: dict[str, object] = {}
            if row == 1:
                event_options = {
                    "annotation_text": "Simulated contact / movement event",
                    "annotation_position": "top left",
                }
            figure.add_vrect(
                x0=synthetic_artifact.start_seconds,
                x1=synthetic_artifact.end_seconds,
                fillcolor="#f97316",
                opacity=0.10,
                line_width=0,
                row=row,
                col=1,
                **event_options,
            )
            if synthetic_artifact.dropout_start_seconds is not None:
                dropout_options: dict[str, object] = {}
                if row == 1:
                    dropout_options = {
                        "annotation_text": (
                            "Known-invalid dropout: prediction only"
                            if dropout_gating_confirmed
                            else "Optional flatline / dropout"
                        ),
                        "annotation_position": "bottom left",
                    }
                figure.add_vrect(
                    x0=synthetic_artifact.dropout_start_seconds,
                    x1=synthetic_artifact.dropout_end_seconds,
                    fillcolor="#475569",
                    opacity=0.16,
                    line_width=0,
                    row=row,
                    col=1,
                    **dropout_options,
                )
    figure.update_xaxes(title_text="Time (s)", row=3, col=1)
    figure.update_yaxes(title_text="Amplitude", row=1, col=1)
    figure.update_yaxes(title_text="Amplitude", row=2, col=1)
    figure.update_yaxes(title_text="Residual", row=3, col=1)
    figure.update_layout(
        height=900,
        margin={"l": 50, "r": 20, "t": 65, "b": 45},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0},
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="LowPoint Biosignal Lab", page_icon="〽️", layout="wide")
    st.title("LowPoint Biosignal Lab")
    st.caption("Smooth approximation at the lower edge of ECG and other sampled biosignals")
    st.warning(
        "Research prototype — not a medical device. A lower envelope through ECG troughs is "
        "not the isoelectric baseline and must not be used as clinical baseline correction "
        "without a separate, validated ECG-specific method.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("Input")
        source = st.radio("Signal source", ["Built-in ECG example", "Upload CSV / TSV"])

    demo = None
    frame = None
    jitter = 0.0
    rate_mismatch = 0.0
    exclude_synthetic_dropout = False
    if source == "Built-in ECG example":
        with st.sidebar:
            duration = st.slider("Duration (s)", 5.0, 30.0, 12.0, 1.0)
            sampling_rate = st.select_slider(
                "Sampling rate (Hz)", [100.0, 250.0, 360.0, 500.0], value=250.0
            )
            heart_rate = st.slider("Heart rate (bpm)", 45.0, 150.0, 72.0, 1.0)
            noise = st.slider("Noise standard deviation", 0.0, 0.08, 0.015, 0.005)
            spikes = st.slider("Negative impulse artifacts", 0, 8, 2, 1)
            scenario_label = st.selectbox(
                "Built-in ECG scenario",
                list(SYNTHETIC_SCENARIOS),
                key="synthetic_scenario",
                help=(
                    "The sustained scenario is a technical signal-quality simulation. It is not "
                    "a diagnosis or guaranteed identification of a particular electrode."
                ),
            )
            artifact_mode = SYNTHETIC_SCENARIOS[scenario_label]
            if artifact_mode == "sustained_contact_movement":
                st.caption(
                    "Adds an abrupt level shift and several seconds of movement-like faster "
                    "noise. This demonstrates estimator response to degraded acquisition quality."
                )
                artifact_start_max = max(0.5, duration - 2.25)
                artifact_start_default = _clamp_float_widget(
                    "artifact_start_widget",
                    min(4.0, artifact_start_max),
                    0.0,
                    artifact_start_max,
                )
                artifact_start = st.slider(
                    "Artifact event start (s)",
                    min_value=0.0,
                    max_value=float(artifact_start_max),
                    value=artifact_start_default,
                    step=0.25,
                    key="artifact_start_widget",
                )
                artifact_duration_max = duration - artifact_start
                artifact_duration_default = _clamp_float_widget(
                    "artifact_duration_widget",
                    min(4.0, artifact_duration_max),
                    2.0,
                    artifact_duration_max,
                )
                artifact_duration = st.slider(
                    "Sustained event duration (s)",
                    min_value=2.0,
                    max_value=float(artifact_duration_max),
                    value=artifact_duration_default,
                    step=0.25,
                    key="artifact_duration_widget",
                )
                artifact_shift = st.slider(
                    "Abrupt level shift",
                    min_value=-0.80,
                    max_value=0.80,
                    value=-0.45,
                    step=0.05,
                    key="artifact_shift_widget",
                )
                movement_noise = st.slider(
                    "Movement-like noise standard deviation",
                    min_value=0.02,
                    max_value=0.30,
                    value=0.12,
                    step=0.01,
                    key="movement_noise_widget",
                )
                include_dropout = st.toggle(
                    "Include short flatline / dropout",
                    value=True,
                    key="synthetic_dropout",
                )
                if include_dropout:
                    dropout_duration_max = min(1.0, artifact_duration)
                    dropout_duration_default = _clamp_float_widget(
                        "dropout_duration_widget",
                        0.35,
                        0.10,
                        dropout_duration_max,
                    )
                    dropout_duration = st.slider(
                        "Flatline / dropout duration (s)",
                        min_value=0.10,
                        max_value=float(dropout_duration_max),
                        value=dropout_duration_default,
                        step=0.05,
                        key="dropout_duration_widget",
                    )
                    exclude_synthetic_dropout = st.toggle(
                        "Exclude known dropout from Kalman updates",
                        value=True,
                        key="exclude_synthetic_dropout",
                        help=(
                            "Uses the generator's exact dropout mask for this demonstration. "
                            "It does not detect signal loss in real data or identify an electrode."
                        ),
                    )
                else:
                    dropout_duration = 0.0
            else:
                artifact_start = None
                artifact_duration = 4.0
                artifact_shift = -0.45
                movement_noise = 0.12
                dropout_duration = 0.0
        demo = generate_ecg(
            duration,
            sampling_rate,
            heart_rate,
            noise,
            negative_spikes=spikes,
            artifact_mode=artifact_mode,
            artifact_start_seconds=artifact_start,
            artifact_duration_seconds=artifact_duration,
            artifact_level_shift=artifact_shift,
            movement_noise_std=movement_noise,
            dropout_duration_seconds=dropout_duration,
        )
        time = demo.time
        signal = demo.signal
        source_name = "synthetic_ecg"
        input_identity = {
            "source": source_name,
            "duration": duration,
            "sampling_rate": sampling_rate,
            "heart_rate": heart_rate,
            "noise": noise,
            "negative_spikes": spikes,
            "scenario": artifact_mode,
        }
        if demo.artifact is not None:
            input_identity["artifact"] = {
                "start_seconds": demo.artifact.start_seconds,
                "end_seconds": demo.artifact.end_seconds,
                "level_shift": demo.artifact.level_shift,
                "movement_noise_std": demo.artifact.movement_noise_std,
                "dropout_start_seconds": demo.artifact.dropout_start_seconds,
                "dropout_end_seconds": demo.artifact.dropout_end_seconds,
                "exclude_from_kalman_updates": exclude_synthetic_dropout,
            }
    else:
        with st.sidebar:
            uploaded = st.file_uploader("Data file", type=["csv", "tsv", "txt"])
        if uploaded is None:
            st.info("Upload a delimited file to configure its time and signal columns.")
            return
        uploaded_bytes = uploaded.getvalue()
        try:
            frame, columns = _load_uploaded_data(uploaded_bytes)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(str(exc))
            return
        with st.sidebar:
            signal_column = st.selectbox("Signal column", columns)
            time_options = [
                "(sample index)",
                *[column for column in columns if column != signal_column],
            ]
            time_column = st.selectbox(
                "Time column (seconds)",
                time_options,
                help="Convert millisecond or microsecond timestamps to seconds before upload.",
            )
            if time_column == "(sample index)":
                sampling_rate = st.number_input("Sampling rate (Hz)", min_value=0.01, value=250.0)
                time = np.arange(len(frame), dtype=float) / sampling_rate
            else:
                try:
                    inferred_rate, jitter = infer_sampling_rate(
                        frame[time_column].to_numpy(dtype=float)
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    return
                sampling_rate = st.number_input(
                    "Sampling rate (Hz)", min_value=0.01, value=float(inferred_rate), format="%.6f"
                )
                rate_mismatch = abs(float(sampling_rate) - inferred_rate) / inferred_rate
                time = frame[time_column].to_numpy(dtype=float)
            signal = frame[signal_column].to_numpy(dtype=float)
            source_name = uploaded.name
            input_identity = {
                "source": source_name,
                "content_sha256": hashlib.sha256(uploaded_bytes).hexdigest(),
                "size_bytes": len(uploaded_bytes),
                "rows": len(frame),
                "signal_column": signal_column,
                "time_column": time_column,
                "sampling_rate": sampling_rate,
            }
        if jitter > 0.01:
            st.warning(
                f"Timestamp interval jitter is {100 * jitter:.2f}%. The core assumes uniform "
                "sampling; resample the data before interpreting smoothness in hertz."
            )
        if rate_mismatch > 0.01:
            st.warning(
                "The entered sampling rate differs from the time-column estimate by more than 1%. "
                "The algorithm will use the entered rate, so bandwidth and time may be "
                "inconsistent."
            )

    nyquist = float(sampling_rate) / 2.0
    frequency_min = max(1e-6, nyquist * 1e-5)
    frequency_max = nyquist * 0.999
    minimum_quantile_bandwidth = max(0.001, float(sampling_rate) / len(signal))

    with st.sidebar:
        st.divider()
        st.header("Signal conditioning (optional)")
        conditioning_enabled = st.toggle(
            "Enable conditioning",
            value=False,
            key="conditioning_enabled",
            help=(
                "All stages are bypassed unless this master switch is enabled. "
                "The envelope is fitted to the resulting conditioned signal."
            ),
        )
        phase_label = st.selectbox(
            "Filter phase",
            ["Zero phase (offline)", "Causal (real-time)"],
            key="filter_phase",
            disabled=not conditioning_enabled,
            help=(
                "Zero phase filters forward and backward, so it has no phase shift but uses "
                "future samples. Causal mode is single-pass and suitable for streaming."
            ),
        )
        phase_mode = {
            "Zero phase (offline)": "zero_phase",
            "Causal (real-time)": "causal",
        }[phase_label]

        with st.expander("Mains interference"):
            mains_mode = st.selectbox(
                "Notch frequency",
                ["Off", "50 Hz", "60 Hz", "Custom"],
                key="mains_mode",
                disabled=not conditioning_enabled,
            )
            if mains_mode == "Custom":
                mains_default = _clamp_float_widget(
                    "mains_frequency_widget", 50.0, frequency_min, frequency_max
                )
                mains_frequency = st.number_input(
                    "Custom frequency (Hz)",
                    min_value=frequency_min,
                    max_value=frequency_max,
                    value=mains_default,
                    format="%.3f",
                    key="mains_frequency_widget",
                    disabled=not conditioning_enabled,
                )
            else:
                mains_frequency = 60.0 if mains_mode == "60 Hz" else 50.0
            mains_q = st.number_input(
                "Notch quality factor Q",
                min_value=1.0,
                max_value=200.0,
                value=30.0,
                step=1.0,
                disabled=not conditioning_enabled or mains_mode == "Off",
                help="Higher Q makes a narrower notch.",
            )
            mains_harmonics = st.number_input(
                "Harmonics",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                disabled=not conditioning_enabled or mains_mode == "Off",
                help="Harmonics at or above Nyquist are skipped and reported.",
            )
            if conditioning_enabled and mains_mode != "Off" and float(mains_frequency) >= nyquist:
                st.warning(
                    f"{float(mains_frequency):g} Hz is not below the {nyquist:g} Hz Nyquist "
                    "frequency. Increase the sampling rate or disable/change the notch."
                )

        with st.expander("High-pass and low-pass"):
            highpass_selected = st.toggle(
                "High-pass filter",
                value=False,
                key="highpass_selected",
                disabled=not conditioning_enabled,
                help="Suppresses slow components below the cutoff.",
            )
            highpass_default = _clamp_float_widget(
                "highpass_cutoff_widget", 0.5, frequency_min, frequency_max
            )
            highpass_cutoff = st.number_input(
                "High-pass cutoff (Hz)",
                min_value=frequency_min,
                max_value=frequency_max,
                value=highpass_default,
                format="%.4f",
                key="highpass_cutoff_widget",
                disabled=not conditioning_enabled or not highpass_selected,
            )
            highpass_order = st.selectbox(
                "High-pass order (per pass)",
                list(range(1, 9)),
                index=1,
                disabled=not conditioning_enabled or not highpass_selected,
            )

            lowpass_selected = st.toggle(
                "Low-pass filter",
                value=False,
                key="lowpass_selected",
                disabled=not conditioning_enabled,
                help="Suppresses fast components above the cutoff.",
            )
            lowpass_default = _clamp_float_widget(
                "lowpass_cutoff_widget", 40.0, frequency_min, frequency_max
            )
            lowpass_cutoff = st.number_input(
                "Low-pass cutoff (Hz)",
                min_value=frequency_min,
                max_value=frequency_max,
                value=lowpass_default,
                format="%.3f",
                key="lowpass_cutoff_widget",
                disabled=not conditioning_enabled or not lowpass_selected,
            )
            lowpass_order = st.selectbox(
                "Low-pass order (per pass)",
                list(range(1, 9)),
                index=3,
                disabled=not conditioning_enabled or not lowpass_selected,
            )
            st.caption("Zero-phase processing doubles each IIR stage's effective order.")

        with st.expander("Baseline removal"):
            baseline_label = st.selectbox(
                "Method",
                ["None", "Moving median", "High-pass"],
                key="baseline_method",
                disabled=not conditioning_enabled,
                help=(
                    "Baseline removal changes the signal coordinate system. It is distinct from "
                    "the lower-envelope estimate and is not automatically a clinical ECG baseline."
                ),
            )
            baseline_method = {
                "None": "none",
                "Moving median": "median",
                "High-pass": "highpass",
            }[baseline_label]
            baseline_window = st.number_input(
                "Median window (s)",
                min_value=0.001,
                value=0.80,
                step=0.05,
                format="%.3f",
                disabled=not conditioning_enabled or baseline_method != "median",
            )
            baseline_cutoff_default = _clamp_float_widget(
                "baseline_cutoff_widget", 0.5, frequency_min, frequency_max
            )
            baseline_cutoff = st.number_input(
                "Baseline high-pass cutoff (Hz)",
                min_value=frequency_min,
                max_value=frequency_max,
                value=baseline_cutoff_default,
                format="%.4f",
                key="baseline_cutoff_widget",
                disabled=not conditioning_enabled or baseline_method != "highpass",
            )
            baseline_order = st.selectbox(
                "Baseline high-pass order (per pass)",
                list(range(1, 9)),
                index=1,
                disabled=not conditioning_enabled or baseline_method != "highpass",
            )
            if conditioning_enabled and baseline_method == "highpass" and highpass_selected:
                st.warning(
                    "Baseline high-pass and general high-pass are both selected; their "
                    "attenuation will compound. Usually choose one."
                )

        st.divider()
        st.header("Approximation")
        method_label = st.selectbox("Method", list(METHOD_LABELS), index=0, key="envelope_method")
        method = METHOD_LABELS[method_label]
        side = st.selectbox("Envelope side", ["lower", "upper"], index=0)
        tau_label = "Asymmetry τ" if method == "kalman" else "Tail level τ"
        quantile = st.slider(
            tau_label,
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            key="tail_level",
            help=(
                "For the quantile method, about τ of samples should lie beyond the fitted edge. "
                "For Kalman mode, τ controls asymmetric innovation weighting but is not an exact "
                "quantile target."
            ),
            disabled=method == "minima",
        )
        smoothness_default = _clamp_float_widget(
            "smoothness_hz_widget",
            0.35,
            minimum_quantile_bandwidth,
            nyquist * 0.99,
        )
        smoothness_hz = st.number_input(
            "Nominal envelope bandwidth (Hz)",
            min_value=minimum_quantile_bandwidth,
            max_value=nyquist * 0.99,
            value=smoothness_default,
            format="%.3f",
            key="smoothness_hz_widget",
            disabled=method != "quantile",
            help=(
                "At least one nominal cutoff cycle must fit in the observed record. For slower "
                "envelopes, use a longer record or downsample first."
            ),
        )
        window_seconds = st.number_input(
            "Low-point support block width (s)", min_value=0.02, value=0.80, step=0.05
        )
        guard_seconds = st.number_input(
            "Negative-spike guard (s)", min_value=0.0, value=0.012, step=0.002, format="%.3f"
        )
        with st.expander("Quantile advanced settings"):
            max_iterations = st.slider(
                "Quantile iterations", 5, 100, 60, 5, disabled=method != "quantile"
            )
            edge_padding = st.number_input(
                "Optional edge reflection (s)",
                0.0,
                10.0,
                0.0,
                0.25,
                help="Disabled by default because reflected ECG creates fictitious edge beats.",
                disabled=method != "quantile",
            )

        defaults = EnvelopeConfig()
        if method == "kalman":
            with st.expander("Kalman controls", expanded=True):
                kalman_q = st.number_input(
                    "Initialization-normalized process variance Q",
                    min_value=1e-10,
                    value=defaults.kalman_process_variance,
                    format="%.8f",
                    help=(
                        "Local-trend acceleration variance after fixed robust normalization from "
                        "the declared warm-up interval. With zero warm-up, the scale fallback is "
                        "one signal unit. Larger Q follows changes faster and admits more noise."
                    ),
                )
                kalman_r = st.number_input(
                    "Initialization-normalized measurement variance R",
                    min_value=1e-8,
                    value=defaults.kalman_measurement_variance,
                    format="%.6f",
                    help=(
                        "Measurement variance in the same fixed initialization-normalized units. "
                        "Larger R produces a smoother, slower track."
                    ),
                )
                kalman_clip = st.number_input(
                    "Innovation clip (standard deviations)",
                    min_value=0.1,
                    value=defaults.kalman_innovation_clip,
                    step=0.25,
                    help="Limits how strongly a single sample can change the state estimate.",
                )
                kalman_warmup = st.number_input(
                    "Initialization warm-up (s)",
                    min_value=0.0,
                    value=defaults.kalman_warmup_seconds,
                    step=0.25,
                    help=(
                        "The initial level is estimated from this leading interval, which creates "
                        "startup lookahead. Set to 0 for one-sample causal initialization."
                    ),
                )
        else:
            kalman_q = defaults.kalman_process_variance
            kalman_r = defaults.kalman_measurement_variance
            kalman_clip = defaults.kalman_innovation_clip
            kalman_warmup = defaults.kalman_warmup_seconds
        process = st.button("Run pipeline", type="primary", width="stretch", key="run_pipeline")

    filter_config = SignalFilterConfig(
        phase_mode=phase_mode if conditioning_enabled else "zero_phase",
        mains_enabled=conditioning_enabled and mains_mode != "Off",
        mains_frequency_hz=float(mains_frequency),
        mains_quality_factor=float(mains_q),
        mains_harmonics=int(mains_harmonics),
        highpass_enabled=conditioning_enabled and highpass_selected,
        highpass_cutoff_hz=float(highpass_cutoff),
        highpass_order=int(highpass_order),
        lowpass_enabled=conditioning_enabled and lowpass_selected,
        lowpass_cutoff_hz=float(lowpass_cutoff),
        lowpass_order=int(lowpass_order),
        baseline_method=baseline_method if conditioning_enabled else "none",
        baseline_window_seconds=float(baseline_window),
        baseline_cutoff_hz=float(baseline_cutoff),
        baseline_order=int(baseline_order),
    )
    config = EnvelopeConfig(
        method=method,
        side=side,
        quantile=quantile,
        smoothness_hz=float(smoothness_hz),
        max_iterations=max_iterations,
        edge_padding_seconds=float(edge_padding),
        minima_window_seconds=float(window_seconds),
        guard_seconds=float(guard_seconds),
        kalman_process_variance=float(kalman_q),
        kalman_measurement_variance=float(kalman_r),
        kalman_innovation_clip=float(kalman_clip),
        kalman_warmup_seconds=float(kalman_warmup),
    )
    run_signature = json.dumps(
        {
            "app_result_schema_version": APP_RESULT_SCHEMA_VERSION,
            "input": input_identity,
            "conditioning": filter_config.to_dict(),
            "envelope": config.to_dict(),
        },
        sort_keys=True,
    )

    if process:
        st.session_state.pop("result_bundle", None)
        try:
            with st.spinner("Conditioning signal and estimating envelope…"):
                filter_result = preprocess_signal(signal, float(sampling_rate), filter_config)
                estimator_valid_mask = np.asarray(filter_result.valid_mask).copy()
                simulated_dropout_excluded = bool(
                    method == "kalman"
                    and exclude_synthetic_dropout
                    and demo is not None
                    and demo.artifact is not None
                    and np.any(demo.artifact.dropout_mask)
                )
                if simulated_dropout_excluded:
                    estimator_valid_mask &= ~demo.artifact.dropout_mask
                result = estimate_envelope(
                    filter_result.processed_signal,
                    float(sampling_rate),
                    config,
                    valid_mask=estimator_valid_mask,
                )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return
        pipeline_causal = bool(
            method == "kalman"
            and result.diagnostics.get("causal", False)
            and filter_result.diagnostics["causal_conditioning"]
        )
        pipeline_causal_after_initialization = bool(
            method == "kalman"
            and result.diagnostics.get("causal_after_initialization", False)
            and filter_result.diagnostics["causal_conditioning"]
        )
        bundle: dict[str, object] = {
            "time": np.asarray(time),
            "raw_signal": np.asarray(signal),
            "filter_result": filter_result,
            "result": result,
            "source_name": source_name,
            "input_identity": input_identity,
            "run_signature": run_signature,
            "pipeline_causal": pipeline_causal,
            "pipeline_causal_after_initialization": pipeline_causal_after_initialization,
            "simulated_dropout_excluded": simulated_dropout_excluded,
            "dropout_gating_confirmed": bool(
                not simulated_dropout_excluded
                or (
                    "skipped_measurement_updates" in result.diagnostics
                    and "reacquisition_updates" in result.diagnostics
                )
            ),
        }
        if (
            demo is not None
            and side == "lower"
            and not filter_result.diagnostics["filtering_active"]
        ):
            bundle["truth"] = demo.true_lower_envelope
            bundle["synthetic_baseline"] = demo.baseline_wander
            result.diagnostics.update(truth_metrics(result.approximation, demo.true_lower_envelope))
        if demo is not None and demo.artifact is not None:
            bundle["synthetic_artifact"] = demo.artifact
        st.session_state["result_bundle"] = bundle

    if "result_bundle" not in st.session_state:
        st.info("Choose the parameters and run the approximation.")
        return

    bundle = st.session_state["result_bundle"]
    if bundle.get("run_signature") != run_signature:
        st.info(
            "Input, parameters, or the result schema changed. "
            "Run the approximation to refresh the result."
        )
        return
    result = bundle["result"]
    diagnostics = result.diagnostics
    filter_result = bundle["filter_result"]
    filter_diagnostics = filter_result.diagnostics
    st.subheader("Result")
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Tail beyond curve", f"{100 * diagnostics['empirical_tail_fraction']:.2f}%"
    )
    if result.config.method == "quantile":
        metric_columns[1].metric("Target tail", f"{100 * diagnostics['target_tail_fraction']:.2f}%")
        metric_columns[2].metric("Coverage error", f"{100 * diagnostics['coverage_error']:.2f} pp")
    elif result.config.method == "minima":
        metric_columns[1].metric("Block width", f"{result.config.minima_window_seconds:.3g} s")
        metric_columns[2].metric("Rejected supports", diagnostics["rejected_support_count"])
    else:
        metric_columns[1].metric("Asymmetry τ", f"{result.config.quantile:.2f}")
        metric_columns[2].metric("Clipped innovations", diagnostics["clipped_innovations"])
    metric_columns[3].metric("Support points", f"{len(result.support_indices)}")
    if "truth_rmse" in diagnostics:
        metric_columns[4].metric("Known-truth RMSE", f"{diagnostics['truth_rmse']:.4f}")
    else:
        metric_columns[4].metric("Pinball loss", f"{diagnostics['pinball_loss']:.5f}")

    if filter_diagnostics["filtering_active"]:
        stage_names = [stage["name"] for stage in filter_diagnostics["stages"]]
        conditioning_columns = st.columns(3)
        conditioning_columns[0].metric("Conditioning stages", len(stage_names))
        conditioning_columns[1].metric(
            "Phase mode", str(filter_diagnostics["phase_mode"]).replace("_", " ")
        )
        conditioning_columns[2].metric(
            "Removed-component RMS", f"{filter_diagnostics['removed_component_rms']:.5g}"
        )
        st.caption("Processing order: " + " → ".join(stage_names))
        st.info(
            "The envelope and residual below are defined on the conditioned signal, not the raw "
            "signal. Raw and conditioned coordinates are shown separately."
        )
    if filter_result.config.baseline_method != "none":
        st.warning(
            "Baseline removal changes the approximation target. The displayed envelope is fitted "
            "after baseline removal; it must not be interpreted as an envelope of the raw signal."
        )
    if filter_diagnostics["interpolated_or_filled_samples"]:
        st.warning(
            f"Filled {filter_diagnostics['interpolated_or_filled_samples']} missing/non-finite "
            f"sample(s) using {filter_diagnostics['gap_policy']}. The original validity mask is "
            "preserved in diagnostics and the CSV export."
        )
    for warning in filter_diagnostics["warnings"]:
        st.warning(warning)

    if "synthetic_artifact" in bundle:
        st.info(
            "The orange region is a simulated signal-quality event, not a diagnostic finding or "
            "proof that a specific electrode caused the change. Its clean synthetic truth stays "
            "unchanged so the envelope and residual response to contamination is visible."
        )
        if bundle.get("simulated_dropout_excluded", False):
            skipped_updates = diagnostics.get("skipped_measurement_updates")
            reacquisition_updates = diagnostics.get("reacquisition_updates")
            dropout_gating_confirmed = bool(
                bundle.get(
                    "dropout_gating_confirmed",
                    skipped_updates is not None and reacquisition_updates is not None,
                )
            )
            exclusion_columns = st.columns(3)
            exclusion_columns[0].metric(
                "Excluded dropout samples",
                skipped_updates if skipped_updates is not None else "Unavailable",
            )
            exclusion_columns[1].metric(
                "Kalman reacquisitions",
                reacquisition_updates if reacquisition_updates is not None else "Unavailable",
            )
            exclusion_columns[2].metric(
                "Dropout update policy",
                "Prediction only" if dropout_gating_confirmed else "Refresh required",
            )
            if dropout_gating_confirmed:
                st.success(
                    "The darker dropout interval uses the generator's known-invalid mask: Kalman "
                    "measurement updates are skipped, the local trend is reacquired at the first "
                    "clean sample, and the residual is intentionally blank in that interval. This "
                    "is a controlled demonstration, not automatic signal-quality detection."
                )
                if filter_diagnostics["filtering_active"]:
                    st.warning(
                        "Conditioning ran before dropout exclusion, so filter transients can still "
                        "carry some dropout influence. Disable conditioning for the clearest "
                        "Kalman reacquisition demonstration."
                    )
            else:
                st.warning(
                    "This retained result was produced without the new Kalman dropout diagnostics, "
                    "so prediction-only gating cannot be confirmed. Restart the Streamlit server "
                    "if it was hot-reloaded, then run the approximation again."
                )
        elif (
            exclude_synthetic_dropout
            and result.config.method != "kalman"
            and np.any(bundle["synthetic_artifact"].dropout_mask)
        ):
            st.info(
                "Known-dropout exclusion is a Kalman measurement-update demonstration. The "
                "current offline approximation method only shows the shaded dropout interval."
            )

    if result.config.method == "kalman":
        lookahead_samples = int(diagnostics["algorithmic_lookahead_samples"])
        lookahead_seconds = lookahead_samples / float(sampling_rate)
        st.info(
            "Kalman mode estimates an asymmetric expectile-like track, not an exact quantile. "
            "Use the offline quantile method when calibrated tail coverage matters."
        )
        if bundle["pipeline_causal"]:
            st.success(
                "The conditioning + Kalman approximation path is causal with the current "
                "settings. Guarded support markers and full-record summary metrics remain "
                "offline diagnostics."
            )
        elif bundle["pipeline_causal_after_initialization"]:
            st.warning(
                "The approximation path is causal only after initialization. The warm-up uses "
                f"{lookahead_samples} future sample(s), or {lookahead_seconds:.4g} s, to set the "
                "initial lower level. Set warm-up to 0 for one-sample initialization."
            )
        else:
            st.warning(
                "The approximation path is not causal with the current settings. Zero-phase "
                "conditioning and offline gap interpolation use future samples; choose causal "
                "conditioning and remove startup lookahead for strict streaming behavior."
            )
    elif result.config.method == "quantile" and "truth" in bundle:
        st.info(
            "This curve is a robust statistical lower tail. The synthetic S trough is very narrow, "
            "so a 5% curve intentionally lies above its geometric minimum. Select guarded minima "
            "for one trough per cycle, or investigate a smaller τ with stronger artifact checks."
        )
    st.plotly_chart(_plot_result(bundle), width="stretch", config={"displaylogo": False})

    output_frame = export_frame(
        np.asarray(bundle["time"]),
        np.asarray(filter_result.raw_signal),
        result.approximation,
        result.residual,
        filter_result.valid_mask,
        processed_signal=np.asarray(filter_result.processed_signal),
        baseline_estimate=np.asarray(filter_result.baseline_estimate),
        removed_component=np.asarray(filter_result.removed_component),
        estimator_valid_mask=result.valid_mask,
    )
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download processed CSV",
        output_frame.to_csv(index=False).encode("utf-8"),
        file_name="lowpoint_result.csv",
        mime="text/csv",
        width="stretch",
    )
    manifest = {
        "source": bundle["source_name"],
        "input": bundle["input_identity"],
        "conditioning_config": filter_result.config.to_dict(),
        "envelope_config": result.config.to_dict(),
        "conditioning_diagnostics": filter_diagnostics,
        "envelope_diagnostics": diagnostics,
        "pipeline": {
            "envelope_target": "conditioned_signal",
            "residual_definition": "conditioned_signal - envelope_on_conditioned",
            "approximation_path_causal": bundle["pipeline_causal"],
            "approximation_path_causal_after_initialization": bundle[
                "pipeline_causal_after_initialization"
            ],
            "support_markers_and_summary_metrics_streaming_ready": False,
            "simulated_dropout_exclusion_requested": bool(
                bundle.get("simulated_dropout_excluded", False)
            ),
            "simulated_dropout_excluded_from_kalman_updates": bool(
                bundle.get("simulated_dropout_excluded", False)
                and bundle.get("dropout_gating_confirmed", False)
            ),
        },
    }
    download_right.download_button(
        "Download reproducibility manifest",
        json.dumps(manifest, indent=2, default=_jsonable),
        file_name="lowpoint_manifest.json",
        mime="application/json",
        width="stretch",
    )
    with st.expander("Full diagnostics and interpretation"):
        st.json(manifest)
        st.markdown(
            "**Interpretation:** `envelope_on_conditioned` is a numerical lower/upper outline of "
            "`conditioned_signal`. `conditioned_minus_envelope` is their explicit residual. "
            "`raw_signal`, `estimated_baseline`, `removed_component`, and the original validity "
            "mask are exported separately so these coordinate systems cannot be confused. The "
            "residual must not be called clinically baseline-corrected ECG without independent "
            "physiological validation."
        )


if __name__ == "__main__":
    main()
