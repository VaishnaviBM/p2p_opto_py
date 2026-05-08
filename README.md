# optoTCSF

**Optogenetic Temporal Contrast Sensitivity Framework**

A pip-installable Python GUI toolkit for optogenetics–vision research.

## Installation

### Prerequisites

- **Python 3.9 or later** — check with `python --version`
- **pip**
- **git**

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/VaishnaviBM/p2p_opto_py.git
cd p2p_opto_py
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. Install the package and its dependencies**
```bash
pip install .
```

This automatically installs:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.23 | Numerical arrays, vectorised ODE integration |
| `scipy` | ≥ 1.10 | Curve fitting (`least_squares`), interpolation, Weibull fitting |
| `matplotlib` | ≥ 3.7 | All plots and figures |
| `PyQt5` | ≥ 5.15 | GUI framework |
| `pandas` | ≥ 1.5 | CSV export |

**4. (Optional) Video Simulation support**
```bash
pip install imageio-ffmpeg scikit-image
```

| Package | Purpose |
|---------|---------|
| `imageio-ffmpeg` | Reading and writing MP4/AVI/MOV video files |
| `scikit-image` | Frame resizing |

**5. (Optional) GPU acceleration for Video Simulation**

Install one of the following if you have a CUDA-capable GPU:
```bash
pip install cupy-cuda12x   # CUDA 12.x — fastest, recommended
# or
pip install torch           # PyTorch CUDA fallback
```

The app detects whichever backend is available at runtime and displays it in the Processing panel. Falls back to NumPy (CPU) if neither is installed.

## Launch GUI

```bash
optoTCSF
# or
python -m optoTCSF
```

## Notes

- No data files are needed to get started — builtin opsins (ChR2, ReaChR, ChrimsonR, CsChrimson, bReaChES, ChRmine) are bundled with the package.
- User-saved opsins are stored locally in `~/.optoTCSF/user_opsins.json` and persist across sessions.
- `.mat`, CSV, and JSON experiment files for the TCSF Estimator and Opto TCSF Prediction modules are loaded at runtime via the GUI.

---

## Modules

### 1 · Opsin Simulator

**Simulate** photocurrents for any built-in or user-defined opsin:

- Select from built-in opsins: **ChR2, ReaChR, ChrimsonR, CsChrimson, bReaChES, ChRmine**
- Set irradiance (W/mm²), temporal frequency (Hz), wavelength (nm), holding potential (mV)
- Run a TF sweep (multiple frequencies overlaid)
- View all 15 model parameters on the Parameters tab

**Fit** from patch-clamp CSV data:

- CSV format: two columns `time (s), current (nA)`
- Specify stimulus ON/OFF times, irradiance, wavelength
- Runs `scipy.optimize.least_squares` on the 4-state photocycle ODE
- Save fitted parameters to the persistent user library (`~/.optoTCSF/user_opsins.json`)

**Model**: 4-state photocycle (Bansal et al. 2021)
```
dC1/dt = Gd1·O1 + Gr·C2 − Ga1(ϕ)·C1
dO1/dt = Ga1(ϕ)·C1 + Gb(ϕ)·O2 − (Gd1 + Gf(ϕ))·O1
dO2/dt = Ga2(ϕ)·C2 + Gf(ϕ)·O1 − (Gd2 + Gb(ϕ))·O2
dC2/dt = Gd2·O2 − (Gr + Ga2(ϕ))·C2
I = g0_ph · (O1 + γ·O2) · (V − E)
```

---

### 2 · TCSF Estimator

Estimates **log-sensitivity surfaces** over TF × SF space from psychophysics data.

**Data loading:**
- Load `.mat` files directly (struct and MCOS/dictionary formats supported)
- Load CSV or JSON trial data with columns `[TF, SF, contrast_level, response]`
- Multiple files per subject merged automatically (v1 + v2 blocks combined)
- Multi-subject loading with per-subject identity tracking (no silent merging across subjects)

**Estimation:**
- Sliding-window Weibull fitting across a TF × SF grid
- Configurable grid size (N TF bins, N SF bins) and window size (log units)

**Sensitivity scaling (checkbox):**
- **Checked** — independent peak alignment: each opsin surface shifted to match the baseline at the peak-SF index (preserves surface shape)
- **Unchecked** — physiological scale factors from `Data/LoadMats/*/scaling.mat` applied per opsin (ChRmine as reference), then all opto surfaces globally shifted so ChRmine aligns with baseline at lowest TF × lowest SF

**Multi-subject plot mode:**
- *Average across subjects* — mean surface per condition with ±SEM shaded error bands on TF slices
- *Individual subplots per subject* — one tab per subject, each with a 3-D surface (top) and 2-D TF slices (bottom) in a vertical splitter

**2-D TF slice selection:**
- Slices taken at first, middle, and highest SF value present in v2 (fixed-SF) blocks
- X-axis anchored at the minimum TF; consistent Y-axis range across panels

**Exports:** sensitivity table as CSV `[TF, SF, Condition, Sensitivity]`

---

### 3 · Opto TCSF Prediction

Predicts the optogenetic TCSF from first principles — **no psychophysics required**.

**Baseline source:**
- Load from `.mat` file(s) — same multi-file list interface as the TCSF Estimator
- Load from CSV or JSON

**Pipeline:**
1. Load a neurotypical baseline TCSF
2. Select one or more opsins to predict
3. Run the opsin photocurrent model at each TF → compute per-TF attenuation via decision rule
4. Apply linking hypothesis: `S_opto(tf, sf) = S_base(tf, sf) + log10(attenuation(tf))`

**Decision rules (linking hypothesis):**
- *Probability Summation* (default) — weighted temporal summation
- *Max−Min*, *Max* — amplitude-based rules
- *1st-order leaky integrator* + rectification
- *2nd-order Butterworth* filter + rectification

**Colors:** each opsin has a fixed unique color; user-added opsins get deterministic colors derived from the opsin name (always consistent regardless of list order or size).

**Exports:** predicted sensitivity as CSV `[TF, SF, Condition, PredictedSensitivity]`

---

### 4 · Video Simulation

Applies the 4-state opsin photocurrent model to every pixel of an input video, producing an opsin-mediated output video.

**Pipeline** (based on `opto_simulation.m`):
1. Read video → grayscale → resize by fraction → normalise [0, 1]
2. Upsample each pixel's luminance trace from video fps to upsample fps (default 5000 fps, dt = 0.2 ms)
3. Prepend background-luminance padding
4. Run 4-state Euler integration across all pixels simultaneously (vectorised)
5. Downsample back to sampling frame rate, normalise output to [0, 1]
6. Write scaled output video at sampling FR × playback FR multiplier

**GPU acceleration:**
- Automatically uses **CuPy** (CUDA) if installed — drop-in numpy replacement on GPU
- Falls back to **PyTorch CUDA** if CuPy is unavailable
- Falls back to **NumPy** (multi-threaded via OpenBLAS) otherwise
- Active backend shown in the Processing panel

**Controls:**
- **Resize factor** — entered as a fraction (e.g. `1/8`, `3/4`); validated against video dimensions to ensure integer pixel output; suggests nearby valid fractions if invalid
- **Upsample FPS** — temporal resolution for ODE integration (default 5000 fps)
- **Pad duration** — background luminance pre-stimulus padding (seconds)
- **Chunk size** — number of pixels processed per batch (trade memory vs. speed)
- **Sampling FR** — read-only display, matches the input video frame rate
- **Playback FR multiplier** — applied to both input and output preview and saved video (default 4×)
- **Input frame rate** — auto-detected from video metadata, manually overridable

**Preview:** side-by-side input and output panels with play/pause/stop and frame scrubbing. Input preview appears as soon as frames are read (before integration completes).

**Save:** exports output as MP4 via ffmpeg at `sampling_fps × playback_multiplier`.

---

## Utilities

### mat2csv.py

Standalone CLI to convert qCSF `.mat` experiment files to CSV:

```bash
python mat2csv.py <file1.mat> [file2.mat ...]   # explicit files
python mat2csv.py output/ --recursive            # whole folder
python mat2csv.py output/*.mat -o csvs/          # custom output dir
```

Output columns: `sID, condition, opsin, block, fixed_freq, TF, SF, trial, contrast, contrast_level, response`

---

## Data Formats

| Module | Input | Output |
|--------|-------|--------|
| Opsin Simulator | CSV `[time_s, current_nA]` | `~/.optoTCSF/user_opsins.json` |
| TCSF Estimator | `.mat` / CSV / JSON trial data | CSV `[TF, SF, Condition, Sensitivity]` |
| Opto TCSF Prediction | `.mat` / CSV / JSON baseline | CSV `[TF, SF, Condition, PredictedSensitivity]` |
| Video Simulation | MP4 / AVI / MOV video | MP4 opsin-processed video |

---

## References

- Bansal, H. et al. (2021). *A generalizable model for photocurrent kinetics and neural responses to complex optogenetic stimulation.* J. Neural Eng. https://doi.org/10.1088/1741-2552/ac1175
- Lesmes, L. A. et al. (2010). *Bayesian adaptive estimation of the contrast sensitivity function.* J. Vis. https://doi.org/10.1167/10.3.17
