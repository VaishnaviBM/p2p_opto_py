"""
PsychoPy-based qCSF experiment runner.

Launched as a subprocess with a JSON config file as the argument.

Config keys:
    sID, cond, opsin, frame_rate, view_dist, screen_width,
    n_trials, stim_dur (ms), debug, out_dir, fixed_freq.

Conditions:
    baseline_v1 – fixed TF, measure spatial CSF
    baseline_v2 – fixed SF, measure temporal CSF
    opto_v1     – fixed TF + opsin filter
    opto_v2     – fixed SF + opsin filter

Stimulus: 2AFC orientation discrimination (±45°) of a Gabor/grating patch.
Response: left arrow (-45°), right arrow (+45°).
"""

import sys
import json
import time
import datetime
import numpy as np
from pathlib import Path


def run_experiment(config: dict):
    sID = config["sID"]
    cond = config["cond"]
    opsin_name = config.get("opsin", "")
    frame_rate = float(config.get("frame_rate", 120))
    view_dist_cm = float(config.get("view_dist", 57.0))
    screen_width_cm = float(config.get("screen_width", 70.0))
    n_trials = int(config.get("n_trials", 50))
    stim_dur_ms = float(config.get("stim_dur", 800.0))
    debug = bool(config.get("debug", False))
    out_dir = Path(config.get("out_dir", "."))
    fixed_freq = float(config.get("fixed_freq", 5.0))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine experiment type
    is_opto = "opto" in cond
    fix_temporal = "v1" in cond   # True -> fix TF, vary SF; False -> fix SF, vary TF

    if fix_temporal:
        fixed_TF = fixed_freq
        SF_list = np.logspace(np.log10(0.5), np.log10(16), 5)
    else:
        fixed_SF = fixed_freq
        TF_list = np.logspace(np.log10(1.5), np.log10(20), 5)

    # --- Import PsychoPy (optional dep) ---
    try:
        from psychopy import visual, core, event, sound
        HAS_PSYCHOPY = True
    except ImportError:
        HAS_PSYCHOPY = False
        print("[WARNING] PsychoPy not installed. Running in headless simulation mode.")

    # --- Load opsin parameters ---
    opsin_params = None
    if is_opto and opsin_name:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from optoTCSF.core.opsin_model import load_all_opsins
        all_opsins = load_all_opsins()
        opsin_params = all_opsins.get(opsin_name)
        if opsin_params is None:
            print(f"[ERROR] Unknown opsin '{opsin_name}'.")
            return

    # --- Setup qCSF ---
    from optoTCSF.core.qcsf_algo import setup_qcsf, pretrial, posttrial, get_current_estimate

    if fix_temporal:
        vary_freqs = SF_list
        priors = (75, 5, 2.5, 0.1)   # peak_sens, peak_SF, bw, trunc
    else:
        vary_freqs = TF_list
        priors = (75, 1, 2.5, 0.1)   # peak_sens, peak_TF, bw, trunc

    trial_records = []

    # --- Pixel/degree conversion ---
    screen_width_px = 1920  # default; adjust if known
    px_per_deg = screen_width_px / (2 * np.degrees(np.arctan(screen_width_cm / 2 / view_dist_cm)))
    dt_frame = 1.0 / frame_rate  # seconds per frame

    print(f"[INFO] Starting {cond} | Subject: {sID} | Opsin: {opsin_name}")
    print(f"[INFO] px/deg = {px_per_deg:.2f}, dt = {dt_frame*1000:.2f} ms")

    if HAS_PSYCHOPY and not debug:
        _run_psychopy(
            sID, cond, opsin_name, opsin_params, is_opto, fix_temporal,
            fixed_freq, vary_freqs, priors,
            n_trials, stim_dur_ms, frame_rate, px_per_deg, dt_frame,
            out_dir, trial_records,
        )
    else:
        _run_headless(
            sID, cond, opsin_name, opsin_params, is_opto, fix_temporal,
            fixed_freq, vary_freqs, priors,
            n_trials, out_dir, trial_records,
        )


def _opsin_filter_frame(opsin_params, irradiance, lambda_nm, V, dt_ms, grating_1d, tf_hz, t_stim_ms):
    """Apply opsin filter to a 1-D grating contrast profile.

    Returns normalised luminance values in [0, 1] for the current frame.
    We precompute the opsin response for a sinusoidal stimulus at this TF
    and use a contrast-level lookup table to map each pixel.
    """
    from optoTCSF.core.opsin_model import get_opsin_current_euler
    n_contrasts = 256
    c_list = np.linspace(-1.0, 1.0, n_contrasts)
    lum_bank = np.zeros(n_contrasts)

    for ci, c in enumerate(c_list):
        # Single-pixel sinusoidal stimulus: Irr * (1 + c*sin) / 2
        bins_per_ms = int(1.0 / dt_ms)
        pad_dur_ms = 500.0
        pad = irradiance * 0.5 * np.ones(int(pad_dur_ms * bins_per_ms))
        t_s = np.arange(0, t_stim_ms + dt_ms, dt_ms)
        stim = irradiance * (1.0 + c * np.sin(2 * np.pi * tf_hz * t_s / 1000.0)) / 2.0
        stim_full = np.concatenate([pad, stim])
        I = get_opsin_current_euler(opsin_params, V, lambda_nm, stim_full, dt_ms)
        I_stim = I[int(pad_dur_ms * bins_per_ms):]
        lum_bank[ci] = np.mean(I_stim)

    # Normalise bank to [0, 1]
    lum_min = lum_bank.min()
    lum_max = lum_bank.max()
    if lum_max > lum_min:
        lum_bank = (lum_bank - lum_min) / (lum_max - lum_min)
    else:
        lum_bank[:] = 0.5

    # Map each pixel contrast to normalised luminance
    idx = np.searchsorted(c_list, np.clip(grating_1d, -1, 1))
    idx = np.clip(idx, 0, n_contrasts - 1)
    return lum_bank[idx]


def _run_psychopy(
    sID, cond, opsin_name, opsin_params, is_opto, fix_temporal,
    fixed_freq, vary_freqs, priors,
    n_trials, stim_dur_ms, frame_rate, px_per_deg, dt_frame,
    out_dir, trial_records,
):
    from psychopy import visual, core, event
    from optoTCSF.core.qcsf_algo import setup_qcsf, pretrial, posttrial

    win = visual.Window(
        size=[1920, 1080], fullscr=True, screen=0,
        color=[-1, -1, -1], colorSpace="rgb",
        units="deg", waitBlanking=True,
    )
    win.setMouseVisible(False)

    # Stimuli
    fix_cross = visual.ShapeStim(
        win, vertices=((-0.3, 0), (0.3, 0), (0, 0), (0, -0.3), (0, 0.3)),
        lineWidth=2, lineColor="white",
    )

    irr = 0.001
    V = -60.0
    lam = opsin_params.peak_lambda if opsin_params else 590.0
    dt_ms = 0.1

    stim_dur_s = stim_dur_ms / 1000.0
    n_frames = int(stim_dur_s * frame_rate)
    t_arr = np.linspace(0, stim_dur_ms, n_frames, endpoint=False)

    clock = core.Clock()

    for freq_idx, freq in enumerate(vary_freqs):
        if fix_temporal:
            sf = freq; tf = fixed_freq
        else:
            tf = freq; sf = fixed_freq

        state = setup_qcsf(fix_temporal=fix_temporal, priors=priors)

        for trial in range(n_trials):
            state = pretrial(state)
            contrast = state.next_contrast
            VF = freq  # spatial or temporal vary-freq

            # Orientation 2AFC
            orient = np.random.choice([-45.0, 45.0])
            CR = 1 if orient == -45 else 2

            # Generate grating frames
            frames = []
            for k, t_ms in enumerate(t_arr):
                if fix_temporal:
                    lum_val = contrast * np.cos(2 * np.pi * sf * 0)  # static SF pattern
                    temporal_mod = np.sin(2 * np.pi * tf * t_ms / 1000.0)
                    frame_contrast = contrast * temporal_mod
                else:
                    frame_contrast = contrast * np.sin(2 * np.pi * tf * t_ms / 1000.0)

                grating = visual.GratingStim(
                    win, tex="sin", mask="gauss", sf=sf, ori=orient,
                    contrast=frame_contrast, size=8,
                )
                frames.append(grating)

            # Present stimulus
            clock.reset()
            for frame_stim in frames:
                frame_stim.draw()
                win.flip()

            # Inter-stimulus
            win.flip()

            # Collect response (timeout 3 s)
            event.clearEvents()
            keys = event.waitKeys(maxWait=3.0, keyList=["left", "right", "escape", "q"])
            if keys is None:
                continue
            if "escape" in keys or "q" in keys:
                break
            response = 1 if (("left" in keys and CR == 1) or ("right" in keys and CR == 2)) else 0

            state = posttrial(state, VF, contrast, response)
            trial_records.append({
                "trial": trial,
                "tf": float(tf), "sf": float(sf),
                "contrast": float(contrast),
                "contrast_level": float(-np.log10(contrast + 1e-12)),
                "response": int(response),
            })

            # Brief ISI
            fix_cross.draw(); win.flip()
            core.wait(0.3)

    win.close()
    _save_results(sID, cond, opsin_name, trial_records, out_dir)


def _run_headless(
    sID, cond, opsin_name, opsin_params, is_opto, fix_temporal,
    fixed_freq, vary_freqs, priors,
    n_trials, out_dir, trial_records,
):
    """Headless simulation (no display). Used for debug/testing."""
    from optoTCSF.core.qcsf_algo import setup_qcsf, pretrial, posttrial, get_current_estimate

    print("[HEADLESS] Running simulated experiment.")

    for freq in vary_freqs:
        if fix_temporal:
            sf = freq; tf = fixed_freq
        else:
            tf = freq; sf = fixed_freq

        state = setup_qcsf(fix_temporal=fix_temporal, priors=priors)

        for trial in range(n_trials):
            state = pretrial(state)
            contrast = state.next_contrast
            VF = freq
            orient = np.random.choice([-45.0, 45.0])
            CR = 1 if orient == -45 else 2

            # Simulated observer – random with p_correct = 0.75
            response = int(np.random.random() < 0.75)

            state = posttrial(state, VF, contrast, response)
            trial_records.append({
                "trial": trial,
                "tf": float(tf), "sf": float(sf),
                "contrast": float(contrast),
                "contrast_level": float(-np.log10(contrast + 1e-12)),
                "response": int(response),
            })
            print(f"  freq={freq:.2f}, c={contrast:.4f}, resp={response}")

    _save_results(sID, cond, opsin_name, trial_records, out_dir)


def _save_results(sID, cond, opsin_name, trial_records, out_dir):
    ts = datetime.datetime.now().strftime("%y%m%d_%H%M")
    suffix = f"_{opsin_name}" if opsin_name else ""
    fname = out_dir / f"{sID}_qCSF_{cond}{suffix}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(trial_records, f, indent=2)
    print(f"[SAVED] {fname}  ({len(trial_records)} trials)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: psychopy_runner.py <config.json>")
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)
    config = json.loads(cfg_path.read_text())
    run_experiment(config)
