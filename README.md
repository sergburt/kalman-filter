# LowPoint Biosignal Lab

LowPoint Biosignal Lab estimates a smooth **lower or upper envelope** of a uniformly sampled
biosignal. It was designed for the requirement “approximate at the lowest points, not through
the middle,” with ECG as the motivating example.

> **Research software, not a medical device.** An ECG lower envelope through Q/S troughs is not
> the physiological isoelectric baseline. Optional numerical baseline removal is also not a
> QRS-gated PR/TP estimator. Do not describe `conditioned_signal - envelope_on_conditioned` as
> clinically baseline-corrected ECG without separate physiological and clinical validation.

## What is implemented

- **Penalized quantile smoother (default):** a smooth low conditional quantile using pinball
  loss and a second-difference Whittaker penalty. This is the robust generic choice.
- **Guarded minima + PCHIP:** a literal, explainable curve through one spike-guarded minimum per
  overlapping time block. This is appropriate when recurring trough amplitude is the target.
- **Asymmetric Kalman tracker:** an expectile-like local-linear tracker with exposed normalized
  process variance (Q), measurement variance (R), innovation clipping, and initialization
  warm-up. It is causal after initialization; the default 2 s warm-up uses future startup samples.
- **Optional signal conditioning, disabled by default:** 50/60 Hz or custom mains notch with
  harmonics, Butterworth high-pass and low-pass stages, and moving-median or high-pass numerical
  baseline removal. Every stage can use offline zero-phase or single-pass causal processing.
- CSV/TSV upload, synthetic ECG demo, interactive Plotly inspection, explicit raw/conditioned
  coordinates, support points, tail calibration, diagnostics, CSV export, and a reproducibility
  manifest.
- An opt-in synthetic **sustained contact / movement artifact** scenario with an abrupt level
  shift, several seconds of faster movement-like noise, and an optional short held-value
  flatline/dropout. This is a technical signal-quality simulation, not a diagnosis or guaranteed
  identification of a particular electrode.
- In experimental Kalman mode, independently visible comparison curves on the main graph:
  the asymmetric lower/upper track, a symmetric central Kalman trend, and a centered rolling
  median. The median is an offline non-predictive summary, and neither central curve is a
  clinical ECG baseline.
- An interactive formula walkthrough below the graphs: a sample-index/time slider marks the
  chosen point and shows both symbolic equations and the exact substituted values for the
  asymmetric Kalman, central Kalman, or centered median. Known-invalid samples are shown as
  prediction-only rather than as fabricated Kalman updates.
- A reusable Python API and CLI, deterministic synthetic generators, 87 automated tests, and a
  reproducible validation benchmark.

The full mathematical, biomedical, validation, risk, and implementation discussion is in
[`DEVELOPMENT_REPORT.md`](DEVELOPMENT_REPORT.md).

## Quick start

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are recommended.

```powershell
uv sync --extra validation --extra dev
uv run streamlit run app.py
```

Then open the local address shown by Streamlit. The built-in ECG example requires no data file.

To inspect the sustained-event response, choose **Sustained contact / movement artifact** under
**Built-in ECG scenario**, adjust its event controls if desired, select the asymmetric Kalman
method, and run the pipeline. **Exclude known dropout from Kalman updates** is enabled by default
when the optional dropout is present. It uses the simulation's exact known mask: samples in the
darker interval get prediction-only Kalman steps, then the stale local slope is cleared and state
uncertainty is conservatively restored for reacquisition at the first clean sample. The residual
plot is blank over excluded samples. This demonstrates handling of an externally supplied
signal-quality flag; it does not detect signal loss or identify an electrode in real data.

The orange plot region marks the full event and the darker region marks the optional dropout at
its end. Compare both **Estimated envelope** and **Conditioned signal minus envelope** across the
shaded interval and immediately after clean ECG resumes. Disable the exclusion toggle to compare
the deliberately ungated behavior.

When **Experimental causal Kalman** is selected, open **Graph comparison lines** to independently
show or hide the asymmetric envelope, symmetric central Kalman, and centered rolling median. The
central Kalman shares the Q/R responsiveness settings with the asymmetric tracker. Only the
rolling median adds a window control; a longer window gives a smoother offline centerline. The
visibility switches update the graph without rerunning the approximation.

Below the graph, use **Selected sample index** to move the dark vertical guide and point marker.
Choose one of the three estimators under **Estimator to explain**:

- The asymmetric and central Kalman views show the level/rate prediction, normalized measurement,
  innovation, weighting, effective measurement variance, gain, clipping, and final level. The
  asymmetric view explains its direction-dependent weight; the central view fixes both directions
  at 0.5.
- Selecting a known simulated dropout reports a prediction-only step and omits the innovation and
  gain instead of pretending that the held signal was measured.
- The centered median view shows the window bounds, excluded invalid samples, compact sorted
  values, and the resulting median. It is explicitly offline and non-predictive because the
  centered window can include later samples.

The panel explains numerical signal processing, not diagnosis, electrode identification, or a
clinical ECG baseline. Graph-click synchronization is not required; the index/time slider is the
reliable interaction.

### Streamlit Community Cloud

Deploy `app.py` from the repository root and keep both `pyproject.toml` and `uv.lock` committed.
`streamlit` and `plotly` are default project dependencies because Community Cloud synchronizes the
default lockfile environment; optional extras are not installed during a normal deployment.

Run the tests and benchmark:

```powershell
uv run pytest -q
uv run python scripts/run_validation.py
```

## Python API

```python
import numpy as np
from lowpoint import (
    EnvelopeConfig,
    SignalFilterConfig,
    estimate_envelope,
    preprocess_signal,
)

fs = 250.0
t = np.arange(2500) / fs
signal = 0.2 * np.sin(2 * np.pi * 0.2 * t) + np.maximum(0, np.sin(2 * np.pi * 1.2 * t))

conditioning = SignalFilterConfig(
    # Every stage is off unless explicitly enabled.
    phase_mode="zero_phase",
    mains_enabled=True,
    mains_frequency_hz=50.0,
    mains_quality_factor=30.0,
    mains_harmonics=2,
    lowpass_enabled=True,
    lowpass_cutoff_hz=40.0,
    lowpass_order=4,
)
conditioned = preprocess_signal(signal, fs, conditioning)

result = estimate_envelope(
    conditioned.processed_signal,
    fs,
    EnvelopeConfig(
        method="quantile",
        quantile=0.05,
        smoothness_hz=0.35,
    ),
    valid_mask=conditioned.valid_mask,
)

lower_curve = result.approximation
conditioned_minus_curve = result.residual
raw_signal = conditioned.raw_signal
removed_component = conditioned.removed_component
print(result.diagnostics)
```

`SignalFilterConfig()` with no enabled stages is the identity on a finite signal. The fixed stage
order is mains notch, baseline removal, general high-pass, then general low-pass. The envelope is
always fitted to `processed_signal`, so its residual and amplitude coordinate are those of the
conditioned signal, not the raw signal.

Offline `zero_phase` mode uses forward/backward filtering: it removes phase shift but uses future
samples and doubles each IIR stage's effective order. `causal` mode is single-pass and uses a
trailing median where requested. Offline gaps are linearly interpolated; causal gaps are
forward-filled and a leading gap is rejected. In both cases the original finite-sample provenance
is retained in `valid_mask`.

### Kalman startup and controls

The Kalman configuration defaults are `kalman_process_variance=2e-4`,
`kalman_measurement_variance=0.08`, `kalman_innovation_clip=4.0`, and
`kalman_warmup_seconds=2.0`. (Q) and (R) act after fixed robust amplitude normalization derived
only from the declared warm-up interval. The warm-up sets the initial lower level, creating
`warmup_samples - 1` samples of startup lookahead. Set `kalman_warmup_seconds=0` and select causal
conditioning for a strictly causal conditioning-plus-Kalman **approximation path**. With zero
warm-up, normalization has only one sample and therefore falls back to a scale of one signal unit;
Q/R then depend on the raw amplitude unit. A hostile future suffix cannot alter earlier Kalman
estimates. Guarded support markers and full-record summary metrics remain offline diagnostics, and
the offline quantile and minima methods are not streaming estimators.

In Kalman mode, invalid samples skip the measurement update. The state and covariance are
predicted through the gap; at the first valid sample, stale local-trend velocity is cleared and a
conservative normalized level-variance floor is restored before updating. Diagnostics report
`skipped_measurement_updates`, `reacquisition_updates`, and the reacquisition policy. These masks
must come from acquisition metadata or a separately validated signal-quality rule; the Kalman
tracker does not infer electrode status.

The symmetric central Kalman uses the same local-linear state model, initialization, innovation
clipping, and invalid-sample gating as the asymmetric tracker, but assigns equal 0.5 weights to
positive and negative innovations. It is a central trend, not a lower envelope. The centered
rolling median ignores invalid samples in its window and may interpolate its trend across a fully
invalid window; because it uses future samples, it is neither causal nor predictive.

For UI explanation, the Kalman implementation can optionally return a per-sample trace containing
the exact prior, predicted state, covariance terms, measurement validity, innovation, asymmetric
weight, effective measurement variance, gain, clipped innovation, posterior, and reacquisition
decision. The trace is generated by the same recursion as the plotted output, so the formula panel
does not reverse-engineer or approximate these values afterward.

For quantile mode, at least one nominal `smoothness_hz` cutoff cycle must fit in the observed
record. The UI enforces this record-length-dependent lower bound. The core raises a clear error for
slower requests; use a longer record or downsample before fitting. The solver uses a symmetric
banded Cholesky system, a nonnegative squared-second-difference objective calculation, and a
linear-system residual check to prevent silent corruption in ill-conditioned regimes.

## CLI

```powershell
uv run lowpoint input.csv output.csv `
  --signal-column ECG `
  --time-column time_s `
  --filter-phase zero-phase `
  --mains-hz 50 `
  --mains-q 30 `
  --mains-harmonics 2 `
  --lowpass-hz 40 `
  --lowpass-order 4 `
  --method quantile `
  --quantile 0.05 `
  --smoothness-hz 0.35
```

If no time column exists, pass `--sampling-rate 250`. Time columns are interpreted as seconds.
Inputs must represent effectively uniform sampling; the app warns when timestamp jitter exceeds
1%, and the CLI rejects it.

Conditioning is bypassed when its enabling flags are absent. CLI controls include
`--filter-phase`, `--mains-hz`, `--mains-q`, `--mains-harmonics`, `--highpass-hz`,
`--highpass-order`, `--lowpass-hz`, `--lowpass-order`, `--baseline-method`,
`--baseline-window-seconds`, `--baseline-cutoff-hz`, and `--baseline-order`. Kalman mode also
exposes `--kalman-process-variance`, `--kalman-measurement-variance`,
`--kalman-innovation-clip`, and `--kalman-warmup-seconds`.

The output columns deliberately encode their coordinate system:

| Column | Definition |
|---|---|
| `time_s` | Input time in seconds or generated sample time |
| `raw_signal` | Unmodified input values, including original missing values |
| `conditioned_signal` | Finite signal presented to the envelope estimator |
| `estimated_baseline` | Numerical baseline stage estimate; zero when that stage is disabled |
| `removed_component` | Finite prepared input minus the final conditioned signal across all stages |
| `envelope_on_conditioned` | Lower/upper approximation fitted in conditioned coordinates |
| `conditioned_minus_envelope` | `conditioned_signal - envelope_on_conditioned` |
| `original_sample_valid` | Whether the corresponding raw input sample was finite |
| `estimator_sample_valid` | Whether the sample was eligible for estimator updates and metrics |
| `symmetric_central_kalman` | Equal-deviation Kalman centerline; present for Kalman runs |
| `centered_rolling_median` | Offline centered median comparison; present for Kalman runs |

The Streamlit manifest additionally records the uploaded-file SHA-256 identity, input selections,
conditioning and envelope configurations, both diagnostic sets, and approximation-path causality
flags. Filled samples remain marked invalid and are excluded from low-point support selection.

## Choosing a mode

| Intended output | Use | Important limitation |
|---|---|---|
| Robust smooth lower outline | `method="quantile"` | It estimates a statistical quantile, not a hard boundary. |
| Trajectory through recurring lowest events | `method="minima"` | Window length defines which event is selected; artifacts can still win. |
| Immediate causal estimate | `method="kalman"` with causal conditioning and zero warm-up | Tail coverage is not calibrated as a true quantile; lag and filter startup transients remain. |
| ECG isoelectric/baseline-wander estimate | **Not implemented** | Requires QRS exclusion and PR/TP evidence; use a separately validated pipeline. |

For a generic lower outline, start with `quantile=0.05` and set `smoothness_hz` to the highest
frequency the envelope is allowed to follow. For a recurring ECG trough trajectory, set the
minima window near one beat period and keep the impulse guard shorter than the physiological
trough (the default is 12 ms).

Enable conditioning only for a specified acquisition problem. A notch is appropriate for known
mains interference; high-pass, low-pass, and numerical baseline removal can alter clinically
meaningful morphology and change the envelope target. Validate those choices separately for the
sensor, sampling rate, waveform, and downstream measurement.

## Project layout

```text
app.py                       Streamlit research UI
src/lowpoint/filtering.py    Optional conditioning and provenance
src/lowpoint/                Envelope core, I/O, metrics, CLI
tests/                       Unit and property-style regression tests
scripts/run_validation.py    Reproducible synthetic comparison
artifacts/                   Generated benchmark tables and figure
DEVELOPMENT_REPORT.md        Comprehensive design and implementation report
```

Raw PhysioNet or patient data is intentionally not committed. The workspace is inside OneDrive;
review information-governance requirements before placing identifiable biomedical data here.
