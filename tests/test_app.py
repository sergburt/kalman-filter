import json

import numpy as np
from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception
    return app


def _number_input(app: AppTest, label: str):
    return next(widget for widget in app.number_input if widget.label == label)


def _plot_trace_names(app: AppTest) -> set[str]:
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    return {str(trace["name"]) for trace in spec["data"] if "name" in trace}


def _plot_trace(app: AppTest, name: str) -> dict[str, object]:
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    return next(trace for trace in spec["data"] if trace.get("name") == name)


def test_default_pipeline_selects_kalman_and_runs_with_known_truth() -> None:
    app = _app()
    assert app.selectbox("envelope_method").value == (
        "Experimental causal Kalman (expectile-like)"
    )
    app.button("run_pipeline").click().run()

    assert not app.exception
    assert [heading.value for heading in app.subheader] == [
        "Result",
        "Interactive formula walkthrough",
    ]
    assert "Known-truth RMSE" in {metric.label for metric in app.metric}
    bundle = app.session_state["result_bundle"]
    assert bundle["result"].config.method == "kalman"
    assert bundle["central_kalman_trend"] is not None
    assert bundle["rolling_median_trend"] is not None
    assert len(app.get("plotly_chart")) == 1


def test_sustained_contact_scenario_runs_and_records_signal_quality_event() -> None:
    app = _app()
    app.selectbox("synthetic_scenario").set_value("Sustained contact / movement artifact").run()
    app.selectbox("envelope_method").set_value(
        "Experimental causal Kalman (expectile-like)"
    ).run()
    app.button("run_pipeline").click().run()

    assert not app.exception
    bundle = app.session_state["result_bundle"]
    assert bundle["input_identity"]["scenario"] == "sustained_contact_movement"
    assert "synthetic_artifact" in bundle
    artifact = bundle["synthetic_artifact"]
    assert np.count_nonzero(artifact.event_mask) >= 3 * 250
    assert np.count_nonzero(artifact.dropout_mask) > 0
    assert bundle["simulated_dropout_excluded"] is True
    result = bundle["result"]
    assert result.diagnostics["skipped_measurement_updates"] == np.count_nonzero(
        artifact.dropout_mask
    )
    assert result.diagnostics["reacquisition_updates"] == 1
    assert not np.any(result.valid_mask[artifact.dropout_mask])
    assert bundle["central_kalman_trend"] is not None
    assert bundle["rolling_median_trend"] is not None
    assert bundle["central_kalman_diagnostics"]["skipped_measurement_updates"] == np.count_nonzero(
        artifact.dropout_mask
    )
    assert bundle["rolling_median_diagnostics"]["excluded_sample_count"] == np.count_nonzero(
        artifact.dropout_mask
    )
    trace_names = _plot_trace_names(app)
    assert "Asymmetric lower Kalman envelope" in trace_names
    assert "Symmetric central Kalman trend" in trace_names
    assert "Centered rolling median (offline)" in trace_names
    assert any(
        "simulated signal-quality event" in str(message.value) for message in app.info
    )
    assert any("known-invalid mask" in str(message.value) for message in app.success)
    assert len(app.get("plotly_chart")) == 1


def test_kalman_comparison_visibility_toggles_are_independent() -> None:
    app = _app()
    app.selectbox("envelope_method").set_value(
        "Experimental causal Kalman (expectile-like)"
    ).run()
    app.button("run_pipeline").click().run()

    app.toggle("show_asymmetric_kalman").set_value(False)
    app.toggle("show_central_kalman").set_value(True)
    app.toggle("show_rolling_median").set_value(False)
    app.run()

    assert not app.exception
    bundle = app.session_state["result_bundle"]
    assert bundle["comparison_visibility"] == {
        "asymmetric_kalman": False,
        "central_kalman": True,
        "rolling_median": False,
    }
    trace_names = _plot_trace_names(app)
    assert "Asymmetric lower Kalman envelope" not in trace_names
    assert "Symmetric central Kalman trend" in trace_names
    assert "Centered rolling median (offline)" not in trace_names


def test_formula_slider_marks_dropout_and_switches_to_offline_median_details() -> None:
    app = _app()
    app.selectbox("synthetic_scenario").set_value("Sustained contact / movement artifact").run()
    app.selectbox("envelope_method").set_value(
        "Experimental causal Kalman (expectile-like)"
    ).run()
    app.button("run_pipeline").click().run()

    bundle = app.session_state["result_bundle"]
    dropout_index = int(np.flatnonzero(bundle["synthetic_artifact"].dropout_mask)[3])
    app.slider("formula_sample_index").set_value(dropout_index).run()

    assert not app.exception
    bundle = app.session_state["result_bundle"]
    assert bundle["formula_selected_index"] == dropout_index
    assert bundle["primary_kalman_trace"].measurement_used[dropout_index] == np.bool_(False)
    marker = _plot_trace(app, "Selected formula sample")
    assert marker["x"] == [bundle["time"][dropout_index]]
    assert any(
        "prediction-only step" in str(message.value)
        and "measurement update is skipped" in str(message.value)
        for message in app.warning
    )

    app.radio("formula_estimator").set_value("Symmetric central Kalman").run()

    assert not app.exception
    assert any(
        "central Kalman is symmetric" in str(message.value) for message in app.caption
    )

    app.radio("formula_estimator").set_value("Centered rolling median").run()

    assert not app.exception
    assert app.session_state["result_bundle"]["formula_selected_estimator"] == (
        "Centered rolling median"
    )
    assert any(
        "offline and non-predictive" in str(message.value) for message in app.warning
    )
    assert any(
        "Sorted valid values (compact view)" in str(message.value)
        for message in app.markdown
    )


def test_retained_preupdate_kalman_result_does_not_raise_missing_diagnostics() -> None:
    app = _app()
    app.selectbox("synthetic_scenario").set_value("Sustained contact / movement artifact").run()
    app.selectbox("envelope_method").set_value(
        "Experimental causal Kalman (expectile-like)"
    ).run()
    app.button("run_pipeline").click().run()

    bundle = app.session_state["result_bundle"]
    bundle["result"].diagnostics.pop("skipped_measurement_updates")
    bundle["result"].diagnostics.pop("reacquisition_updates")
    bundle["dropout_gating_confirmed"] = False
    app.run()

    assert not app.exception
    assert any(
        "without the new Kalman dropout diagnostics" in str(message.value)
        for message in app.warning
    )


def test_conditioning_changes_target_and_hides_raw_truth_metric() -> None:
    app = _app()
    app.toggle("conditioning_enabled").set_value(True).run()
    app.toggle("lowpass_selected").set_value(True)
    app.selectbox("baseline_method").set_value("Moving median")
    app.run()
    app.button("run_pipeline").click().run()

    assert not app.exception
    metric_labels = {metric.label for metric in app.metric}
    assert "Conditioning stages" in metric_labels
    assert "Known-truth RMSE" not in metric_labels
    warning_text = " ".join(str(warning.value) for warning in app.warning)
    assert "Baseline removal changes the approximation target" in warning_text
    assert len(app.get("plotly_chart")) == 1


def test_causal_kalman_exposes_warmup_and_reports_causal_approximation_path() -> None:
    app = _app()
    app.toggle("conditioning_enabled").set_value(True).run()
    app.selectbox("filter_phase").set_value("Causal (real-time)")
    app.selectbox("envelope_method").set_value("Experimental causal Kalman (expectile-like)")
    app.run()
    _number_input(app, "Initialization warm-up (s)").set_value(0.0)
    app.run()
    app.button("run_pipeline").click().run()

    assert not app.exception
    assert any(
        "conditioning + Kalman approximation path is causal" in str(message.value)
        for message in app.success
    )
    assert "Clipped innovations" in {metric.label for metric in app.metric}


def test_disabled_conditioning_cannot_retain_hidden_causal_gap_policy() -> None:
    app = _app()
    app.toggle("conditioning_enabled").set_value(True).run()
    app.selectbox("filter_phase").set_value("Causal (real-time)").run()
    app.toggle("conditioning_enabled").set_value(False).run()
    app.button("run_pipeline").click().run()

    assert not app.exception
    filter_result = app.session_state["result_bundle"]["filter_result"]
    assert filter_result.config.phase_mode == "zero_phase"


def test_failed_filter_validation_clears_previous_result() -> None:
    app = _app()
    app.button("run_pipeline").click().run()
    assert "result_bundle" in app.session_state.filtered_state

    app.select_slider[0].set_value(100.0)
    app.toggle("conditioning_enabled").set_value(True)
    app.run()
    app.selectbox("mains_mode").set_value("50 Hz").run()
    app.button("run_pipeline").click().run()

    assert not app.exception
    assert any("strictly between 0 and Nyquist" in str(error.value) for error in app.error)
    assert "result_bundle" not in app.session_state.filtered_state
