# LowPoint Biosignal Lab

LowPoint Biosignal Lab estimates a smooth **lower or upper envelope** of a uniformly sampled
biosignal. It was designed for the requirement “approximate at the lowest points, not through
the middle,” with ECG as the motivating example.

> **Research software, not a medical device.** An ECG lower envelope through Q/S troughs is not
> the physiological isoelectric baseline. Do not describe `signal - envelope` as clinically
> baseline-corrected ECG without a separate QRS-gated PR/TP estimator and clinical validation.

## What is implemented

- **Penalized quantile smoother (default):** a smooth low conditional quantile using pinball
  loss and a second-difference Whittaker penalty. This is the robust generic choice.
- **Guarded minima + PCHIP:** a literal, explainable curve through one spike-guarded minimum per
  overlapping time block. This is appropriate when recurring trough amplitude is the target.
- **Asymmetric causal Kalman tracker:** a zero-look-ahead local-linear tracker. It is explicitly
  labeled expectile-like; an ordinary Gaussian Kalman filter is not an exact quantile filter.
- CSV/TSV upload, synthetic ECG demo, interactive Plotly inspection, support points, tail
  calibration, diagnostics, CSV export, and a reproducibility manifest.
- A reusable Python API and CLI, deterministic synthetic generators, automated tests, and a
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
from lowpoint import EnvelopeConfig, estimate_envelope

fs = 250.0
t = np.arange(2500) / fs
signal = 0.2 * np.sin(2 * np.pi * 0.2 * t) + np.maximum(0, np.sin(2 * np.pi * 1.2 * t))

result = estimate_envelope(
    signal,
    fs,
    EnvelopeConfig(
        method="quantile",
        quantile=0.05,
        smoothness_hz=0.35,
    ),
)

lower_curve = result.approximation
distance_above_curve = result.residual
print(result.diagnostics)
```

The input may contain short `NaN` gaps. They are interpolated for numerical continuity and
reported by `result.valid_mask`; the transformation is never hidden.

## CLI

```powershell
uv run lowpoint input.csv output.csv `
  --signal-column ECG `
  --time-column time_s `
  --method quantile `
  --quantile 0.05 `
  --smoothness-hz 0.35
```

If no time column exists, pass `--sampling-rate 250`. Time columns are interpreted as seconds.
Inputs must represent effectively uniform sampling; the app warns when timestamp jitter exceeds
1%, and the CLI rejects it.

## Choosing a mode

| Intended output | Use | Important limitation |
|---|---|---|
| Robust smooth lower outline | `method="quantile"` | It estimates a statistical quantile, not a hard boundary. |
| Trajectory through recurring lowest events | `method="minima"` | Window length defines which event is selected; artifacts can still win. |
| Immediate causal estimate | `method="kalman"` | Tail coverage is not calibrated as a true quantile and causal lag remains. |
| ECG isoelectric/baseline-wander estimate | **Not implemented** | Requires QRS exclusion and PR/TP evidence; use a separately validated pipeline. |

For a generic lower outline, start with `quantile=0.05` and set `smoothness_hz` to the highest
frequency the envelope is allowed to follow. For a recurring ECG trough trajectory, set the
minima window near one beat period and keep the impulse guard shorter than the physiological
trough (the default is 12 ms).

## Project layout

```text
app.py                       Streamlit research UI
src/lowpoint/                Scientific core, I/O, metrics, CLI
tests/                       Unit and property-style regression tests
scripts/run_validation.py    Reproducible synthetic comparison
artifacts/                   Generated benchmark tables and figure
DEVELOPMENT_REPORT.md        Comprehensive design and implementation report
```

Raw PhysioNet or patient data is intentionally not committed. The workspace is inside OneDrive;
review information-governance requirements before placing identifiable biomedical data here.
