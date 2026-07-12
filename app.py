"""Streamlit research application for LowPoint Biosignals."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lowpoint.io import export_frame, infer_sampling_rate, numeric_columns, read_table
from lowpoint.metrics import truth_metrics
from lowpoint.models import EnvelopeConfig
from lowpoint.pipeline import estimate_envelope
from lowpoint.synthetic import generate_ecg

METHOD_LABELS = {
    "Robust lower quantile (recommended)": "quantile",
    "Guarded block minima + PCHIP (literal)": "minima",
    "Experimental causal Kalman (expectile-like)": "kalman",
}


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


def _plot_result(bundle: dict[str, object]) -> go.Figure:
    time = np.asarray(bundle["time"])
    signal = np.asarray(bundle["signal"])
    result = bundle["result"]
    truth = bundle.get("truth")
    baseline = bundle.get("baseline")
    index = _decimation_indices(time.size)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.68, 0.32],
        subplot_titles=("Signal and approximation", "Signal minus approximation"),
    )
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=signal[index],
            name="Observed signal",
            line={"color": "#334155", "width": 1},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=result.approximation[index],
            name="Estimated envelope",
            line={"color": "#e11d48", "width": 2.5},
        ),
        row=1,
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
            row=1,
            col=1,
        )
    if baseline is not None:
        baseline_array = np.asarray(baseline)
        figure.add_trace(
            go.Scattergl(
                x=time[index],
                y=baseline_array[index],
                name="Isoelectric drift (different target)",
                line={"color": "#7c3aed", "width": 1.5, "dash": "dot"},
            ),
            row=1,
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
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scattergl(
            x=time[index],
            y=result.residual[index],
            name="Residual",
            line={"color": "#0369a1", "width": 1.2},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line={"color": "#94a3b8", "width": 1}, row=2, col=1)
    figure.update_xaxes(title_text="Time (s)", row=2, col=1)
    figure.update_yaxes(title_text="Amplitude", row=1, col=1)
    figure.update_yaxes(title_text="Residual", row=2, col=1)
    figure.update_layout(
        height=720,
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
    if source == "Built-in ECG example":
        with st.sidebar:
            duration = st.slider("Duration (s)", 5.0, 30.0, 12.0, 1.0)
            sampling_rate = st.select_slider(
                "Sampling rate (Hz)", [100.0, 250.0, 360.0, 500.0], value=250.0
            )
            heart_rate = st.slider("Heart rate (bpm)", 45.0, 150.0, 72.0, 1.0)
            noise = st.slider("Noise standard deviation", 0.0, 0.08, 0.015, 0.005)
            spikes = st.slider("Negative impulse artifacts", 0, 8, 2, 1)
        demo = generate_ecg(duration, sampling_rate, heart_rate, noise, negative_spikes=spikes)
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
        }
    else:
        with st.sidebar:
            uploaded = st.file_uploader("Data file", type=["csv", "tsv", "txt"])
        if uploaded is None:
            st.info("Upload a delimited file to configure its time and signal columns.")
            return
        try:
            frame, columns = _load_uploaded_data(uploaded)
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
                "file_id": str(getattr(uploaded, "file_id", "")),
                "size_bytes": uploaded.size,
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

    with st.sidebar:
        st.divider()
        st.header("Approximation")
        method_label = st.selectbox("Method", list(METHOD_LABELS), index=0)
        method = METHOD_LABELS[method_label]
        side = st.selectbox("Envelope side", ["lower", "upper"], index=0)
        quantile = st.slider(
            "Tail level τ",
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            help="For the quantile method, about τ of samples should lie beyond the fitted edge.",
            disabled=method == "minima",
        )
        smoothness_hz = st.number_input(
            "Nominal envelope bandwidth (Hz)",
            min_value=0.001,
            max_value=float(sampling_rate / 2 * 0.99),
            value=min(0.35, float(sampling_rate / 10)),
            format="%.3f",
            disabled=method != "quantile",
        )
        window_seconds = st.number_input(
            "Low-point block width (s)", min_value=0.02, value=0.80, step=0.05
        )
        guard_seconds = st.number_input(
            "Negative-spike guard (s)", min_value=0.0, value=0.012, step=0.002, format="%.3f"
        )
        with st.expander("Advanced settings"):
            max_iterations = st.slider("Quantile iterations", 5, 100, 60, 5)
            edge_padding = st.number_input(
                "Optional edge reflection (s)",
                0.0,
                10.0,
                0.0,
                0.25,
                help="Disabled by default because reflected ECG creates fictitious edge beats.",
            )
            kalman_q = st.number_input(
                "Kalman process variance", min_value=1e-8, value=2e-4, format="%.6f"
            )
            kalman_r = st.number_input(
                "Kalman measurement variance", min_value=1e-6, value=0.08, format="%.4f"
            )
        process = st.button("Run approximation", type="primary", use_container_width=True)

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
    )
    run_signature = json.dumps(
        {"input": input_identity, "config": config.to_dict()}, sort_keys=True
    )

    if process:
        try:
            with st.spinner("Estimating envelope…"):
                result = estimate_envelope(signal, float(sampling_rate), config)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return
        bundle: dict[str, object] = {
            "time": np.asarray(time),
            "signal": np.asarray(signal),
            "result": result,
            "source_name": source_name,
            "run_signature": run_signature,
        }
        if demo is not None and side == "lower":
            bundle["truth"] = demo.true_lower_envelope
            bundle["baseline"] = demo.baseline_wander
            result.diagnostics.update(truth_metrics(result.approximation, demo.true_lower_envelope))
        st.session_state["result_bundle"] = bundle

    if "result_bundle" not in st.session_state:
        st.info("Choose the parameters and run the approximation.")
        return

    bundle = st.session_state["result_bundle"]
    if bundle.get("run_signature") != run_signature:
        st.info("Input or parameters changed. Run the approximation to refresh the result.")
        return
    result = bundle["result"]
    diagnostics = result.diagnostics
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

    if result.config.method == "kalman":
        st.info(
            "The Kalman mode is causal and estimates an asymmetric expectile-like track, not an "
            "exact quantile. Use the offline quantile method when calibrated tail coverage matters."
        )
    elif result.config.method == "quantile" and "truth" in bundle:
        st.info(
            "This curve is a robust statistical lower tail. The synthetic S trough is very narrow, "
            "so a 5% curve intentionally lies above its geometric minimum. Select guarded minima "
            "for one trough per cycle, or investigate a smaller τ with stronger artifact checks."
        )
    st.plotly_chart(_plot_result(bundle), use_container_width=True, config={"displaylogo": False})

    output_frame = export_frame(
        np.asarray(bundle["time"]),
        np.asarray(bundle["signal"]),
        result.approximation,
        result.residual,
        result.valid_mask,
    )
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download processed CSV",
        output_frame.to_csv(index=False).encode("utf-8"),
        file_name="lowpoint_result.csv",
        mime="text/csv",
        use_container_width=True,
    )
    manifest = {
        "source": bundle["source_name"],
        "config": result.config.to_dict(),
        "diagnostics": diagnostics,
    }
    download_right.download_button(
        "Download reproducibility manifest",
        json.dumps(manifest, indent=2, default=_jsonable),
        file_name="lowpoint_manifest.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("Full diagnostics and interpretation"):
        st.json(manifest)
        st.markdown(
            "**Interpretation:** `envelope` is a numerical lower/upper outline. "
            "`signal_minus_envelope` is provided for research workflows, but it should not be "
            "called clinically baseline-corrected ECG. For physiological baseline-wander removal, "
            "an ECG-specific QRS-gated PR/TP estimator and morphology validation are required."
        )


if __name__ == "__main__":
    main()
