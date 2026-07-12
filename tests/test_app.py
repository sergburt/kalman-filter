from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception
    return app


def _number_input(app: AppTest, label: str):
    return next(widget for widget in app.number_input if widget.label == label)


def test_default_pipeline_runs_with_known_truth() -> None:
    app = _app()
    app.button("run_pipeline").click().run()

    assert not app.exception
    assert [heading.value for heading in app.subheader] == ["Result"]
    assert "Known-truth RMSE" in {metric.label for metric in app.metric}
    assert len(app.get("plotly_chart")) == 1


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
