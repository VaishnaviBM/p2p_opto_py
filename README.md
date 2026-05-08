# optoTCSF

**Optogenetic Temporal Contrast Sensitivity Framework**

A pip-installable Python GUI toolkit for optogenetics–vision research.

## Installation

```bash
pip install .
# With PsychoPy experiment support:
pip install ".[experiment]"
```

## Launch GUI

```bash
optoTCSF
# or
python -m optoTCSF
```

---

## Modules

### 1 · Opsin Simulator

**Simulate** photocurrents for any built-in or user-defined opsin:

- Select from 7 built-in opsins: **ChR2, ReaChR, ChrimsonR, CsChrimson, bReaChES, ChRmine, MCO**
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

### 2 · Psychophysics Experiment

Launches a **qCSF-based 2AFC orientation-discrimination experiment** via PsychoPy.

| Condition | Fixed | Varied |
|-----------|-------|--------|
| `baseline_v1` | TF | SF |
| `baseline_v2` | SF | TF |
| `opto_v1` | TF | SF + opsin filter |
| `opto_v2` | SF | TF + opsin filter |

Output: JSON files in the specified output directory (one per TF/SF run), with columns `tf, sf, contrast, contrast_level, response`.

Requires `psychopy` (install with `pip install ".[experiment]"`).

---

### 3 · TCSF Estimator

Estimates **sensitivity surfaces** over TF × SF space from psychophysics data.

- Load JSON output from Module 2 or CSV files with `[TF, SF, contrast_level, response]`
- Sliding-window Weibull fitting (window size in log units)
- **Scaled**: preserves relative sensitivity differences across opsin conditions
- **Unscaled**: normalises all conditions to the control at lowest TF
- Exports sensitivity table as CSV
- 3-D surface plot and per-TF slice plots

---

### 4 · Opto TCSF Prediction

Predicts the optogenetic TCSF from first principles — **no psychophysics required**.

**Pipeline:**
1. Load (or specify) a neurotypical baseline TCSF
2. Select one or more opsins
3. Run the opsin photocurrent model at each TF → compute per-TF response via decision rule
4. Apply the linking hypothesis: `S_opto(tf, sf) = S_base(tf, sf) + log10(attenuation(tf))`

**Decision rules (linking hypothesis):**
- *Probability Summation* (default) — weighted temporal summation
- *Max−Min*, *Max* — amplitude-based rules
- *1st-order leaky integrator* + rectification
- *2nd-order Butterworth* filter + rectification
- *BashKeyboard* — frequency-domain weighting

**Fit linking model**: Optimise β parameters to minimise MSE between predicted and measured opsin TCSF curves (requires psychophysics data in Module 3).

---

## Data Formats

| Module | Input format | Output format |
|--------|-------------|---------------|
| Opsin Simulator | CSV `[time_s, current_nA]` | `~/.optoTCSF/user_opsins.json` |
| Experiment | Config JSON → displays | JSON `[{tf, sf, contrast, response}]` |
| TCSF Estimator | JSON / CSV trial data | CSV `[TF, SF, Condition, Sensitivity]` |
| Opto TCSF | JSON baseline + opsin library | CSV `[TF, SF, Condition, PredictedSensitivity]` |

---

## References

- Bansal, H. et al. (2021). *A generalizable model for photocurrent kinetics and neural responses to complex optogenetic stimulation.* J. Neural Eng. https://doi.org/10.1088/1741-2552/ac1175
- Lesmes, L. A. et al. (2010). *Bayesian adaptive estimation of the contrast sensitivity function.* J. Vis. https://doi.org/10.1167/10.3.17
- van Hateren, J. H. & Lamb, T. D. (2006). *The photocurrent response of human cones is fast and monophasic.* BMC Neurosci. https://doi.org/10.1186/1471-2202-7-34
