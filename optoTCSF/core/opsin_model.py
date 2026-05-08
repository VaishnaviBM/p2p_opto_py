"""
4-state photocurrent model for microbial opsins.
Ref: Bansal et al. (2021) https://iopscience.iop.org/article/10.1088/1741-2552/ac1175
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8


@dataclass
class OpsinParams:
    Gd1: float = 0.02
    Gd2: float = 0.013
    Gr: float = 5.9e-4
    g0_ph: float = 110.0
    phim: float = 2.1e15
    k1: float = 0.2
    k2: float = 0.01
    Gf0: float = 0.0027
    Gb0: float = 0.0005
    kf: float = 0.001
    kb: float = 0.0
    gamma: float = 0.05
    p: float = 0.8
    q: float = 1.0
    E: float = 5.64
    peak_lambda: float = 590.0
    name: str = ""
    description: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keys})


def load_builtin_opsins() -> dict:
    json_path = Path(__file__).parent.parent / "data" / "builtin_opsins.json"
    with open(json_path) as f:
        data = json.load(f)
    return {name: OpsinParams.from_dict({**params, "name": name})
            for name, params in data.items()}


BUILTIN_OPSINS = load_builtin_opsins()


def lux_to_irradiance(lux: float, lambda_um: float) -> float:
    """Convert luminance in lux to irradiance in W/mm2."""
    V = 1.019 * np.exp(-285.4 * (lambda_um - 0.559) ** 2)
    return lux / (V * 683)


def _compute_phi(irr: float, lambda_nm: float) -> float:
    return max(0.0, irr) * (lambda_nm * 1e-9) / (H_PLANCK * C_LIGHT)


def _rate_functions(P: OpsinParams, phi: float, epsilon: float = 1.0):
    pp = phi ** P.p if phi > 0 else 0.0
    pq = phi ** P.q if phi > 0 else 0.0
    phim_p = P.phim ** P.p
    phim_q = P.phim ** P.q

    Ga1 = epsilon * P.k1 * pp / (pp + phim_p) if (pp + phim_p) > 0 else 0.0
    Ga2 = epsilon * P.k2 * pp / (pp + phim_p) if (pp + phim_p) > 0 else 0.0
    Gf = P.Gf0 + (epsilon * P.kf * pq / (pq + phim_q) if (pq + phim_q) > 0 else 0.0)
    Gb = P.Gb0 + (epsilon * P.kb * pq / (pq + phim_q) if (pq + phim_q) > 0 else 0.0)
    return Ga1, Ga2, Gf, Gb


def _stable_dt(params: OpsinParams, irr: float, lambda_nm: float, dt: float) -> float:
    """Return a numerically stable dt for Euler integration.

    The Euler method is stable when dt < 1 / max_rate. We estimate the
    max rate from the model parameters at the given irradiance level.
    """
    phi = _compute_phi(irr, lambda_nm)
    Ga1, Ga2, Gf, Gb = _rate_functions(params, phi)
    max_rate = max(params.Gd1, params.Gd2, params.Gr, Ga1, Ga2, Gf, Gb, 1e-6)
    dt_stable = 0.5 / max_rate   # safety factor 0.5
    return min(dt, dt_stable)


def get_opsin_current_euler(
    params: OpsinParams,
    V: float,
    lambda_nm: float,
    Irr: np.ndarray,
    dt: float,
    auto_substep: bool = True,
) -> np.ndarray:
    """Compute opsin photocurrent using Euler integration.

    Args:
        params: Opsin parameters.
        V: Holding potential in mV.
        lambda_nm: Stimulus wavelength in nm.
        Irr: Irradiance array in W/mm2.
        dt: Requested time step in ms.
        auto_substep: If True, automatically sub-step when kinetics are very fast.

    Returns:
        I_opsin: Photocurrent array in pA, same length as Irr.
    """
    # Determine required sub-step for stability
    if auto_substep and len(Irr) > 0:
        dt_use = _stable_dt(params, float(np.max(np.abs(Irr))), lambda_nm, dt)
        n_sub = max(1, int(np.ceil(dt / dt_use)))
        dt_use = dt / n_sub
    else:
        dt_use = dt
        n_sub = 1

    C1, C2, O1, O2 = 1.0, 0.0, 0.0, 0.0
    N = len(Irr)
    I_opsin = np.zeros(N)

    for i in range(N - 1):
        irr_i = float(Irr[i])
        # Sub-step loop for numerical stability
        for _ in range(n_sub):
            phi = _compute_phi(irr_i, lambda_nm)
            Ga1, Ga2, Gf, Gb = _rate_functions(params, phi)

            dC1 = (params.Gd1 * O1 + params.Gr * C2 - Ga1 * C1) * dt_use
            dO1 = (Ga1 * C1 + Gb * O2 - (params.Gd1 + Gf) * O1) * dt_use
            dO2 = (Ga2 * C2 + Gf * O1 - (params.Gd2 + Gb) * O2) * dt_use
            dC2 = (params.Gd2 * O2 - (params.Gr + Ga2) * C2) * dt_use

            C1 = max(0.0, C1 + dC1)
            O1 = max(0.0, O1 + dO1)
            O2 = max(0.0, O2 + dO2)
            C2 = max(0.0, C2 + dC2)

            # Re-normalise to maintain constraint C1+O1+O2+C2=1
            total = C1 + O1 + O2 + C2
            if total > 0:
                C1 /= total; O1 /= total; O2 /= total; C2 /= total

        f_phi = O1 + params.gamma * O2
        I_opsin[i] = params.g0_ph * f_phi * (V - params.E)

    return I_opsin


def simulate_sinusoidal(
    params: OpsinParams,
    irradiance: float,
    freq_hz: float,
    lambda_nm: float,
    V: float = -60.0,
    stim_dur_ms: float = 1000.0,
    pad_dur_ms: float = 2000.0,
    dt: float = 0.1,
) -> tuple:
    """Simulate opsin current driven by a sinusoidal irradiance stimulus.

    Returns:
        t: Time vector (ms), stimulus duration only.
        stim: Irradiance stimulus (W/mm2).
        I_opsin: Photocurrent (pA), same length as t.
    """
    bins = int(1.0 / dt)
    t_stim = np.arange(0, stim_dur_ms + dt, dt)
    pad = irradiance * 0.5 * np.ones(int(pad_dur_ms * bins))
    stim = irradiance * (1.0 + np.sin(2 * np.pi * freq_hz * t_stim / 1000.0)) / 2.0
    stim_padded = np.concatenate([pad, stim])

    I_full = get_opsin_current_euler(params, V, lambda_nm, stim_padded, dt)

    pad_pts = int(pad_dur_ms * bins)
    I_stim = I_full[pad_pts: pad_pts + len(t_stim) - 1]
    return t_stim[:-1], stim[:-1], I_stim


def simulate_step(
    params: OpsinParams,
    irradiance: float,
    lambda_nm: float,
    V: float = -60.0,
    stim_on_ms: float = 200.0,
    stim_off_ms: float = 500.0,
    pad_ms: float = 100.0,
    dt: float = 0.1,
) -> tuple:
    """Simulate opsin current driven by a rectangular (step) stimulus.

    Returns:
        t: Time vector (ms), full trace.
        stim: Irradiance array (W/mm2).
        I_opsin: Photocurrent (pA), same length as t.
    """
    total_ms = pad_ms + stim_on_ms + stim_off_ms
    t = np.arange(0, total_ms, dt)
    stim = np.zeros(len(t))
    on_idx = int(pad_ms / dt)
    off_idx = int((pad_ms + stim_on_ms) / dt)
    stim[on_idx:off_idx] = irradiance
    I_opsin = get_opsin_current_euler(params, V, lambda_nm, stim, dt)
    return t, stim, I_opsin


def get_scale_factor(y: np.ndarray, dt: float, pad_pts: int) -> tuple:
    """Compute offset and scale factor to normalize a photocurrent trace to [-1, 1] range."""
    y0 = y[pad_pts]
    segment = y[pad_pts:]
    a = np.max(segment)
    b = np.min(segment)
    denom = max(a - y0, y0 - b)
    scale = 1.0 / denom if denom > 0 else 1.0
    return y0, scale


# ---------------------------------------------------------------------------
# Fitting from patch-clamp CSV data (step stimulus)
# ---------------------------------------------------------------------------

def _opsin_odes(t, y, stim_on, stim_off, phi_val, P_list):
    """ODE RHS for solve_ivp: 4-state model with step stimulus."""
    C1, O1, O2, C2 = y
    total = C1 + O1 + O2 + C2
    if abs(total - 1) > 1e-6:
        C1 /= total; O1 /= total; O2 /= total; C2 /= total

    Gd1, Gd2, Gr, g0, phi_m, k1, k2, Gf0, Gb0, kf, kb, gamma, p, q = P_list
    epsilon = 1.0

    phi = phi_val if (stim_on < t <= stim_off) else 0.0

    pp = phi ** p if phi > 0 else 0.0
    pq = phi ** q if phi > 0 else 0.0
    pm_p = phi_m ** p
    pm_q = phi_m ** q

    Ga1 = epsilon * k1 * pp / (pp + pm_p) if (pp + pm_p) > 0 else 0.0
    Ga2 = epsilon * k2 * pp / (pp + pm_p) if (pp + pm_p) > 0 else 0.0
    Gf = Gf0 + (epsilon * kf * pq / (pq + pm_q) if (pq + pm_q) > 0 else 0.0)
    Gb = Gb0 + (epsilon * kb * pq / (pq + pm_q) if (pq + pm_q) > 0 else 0.0)

    dC1 = Gd1 * O1 + Gr * C2 - Ga1 * C1
    dO1 = Ga1 * C1 + Gb * O2 - (Gd1 + Gf) * O1
    dO2 = Ga2 * C2 + Gf * O1 - (Gd2 + Gb) * O2
    dC2 = Gd2 * O2 - (Gr + Ga2) * C2
    return [dC1, dO1, dO2, dC2]


def _predict_step_current(
    t_data: np.ndarray,
    stim_on: float,
    stim_off: float,
    phi: float,
    V: float,
    E_rev: float,
    params_vec: np.ndarray,
) -> np.ndarray:
    """Predict opsin current for step stimulus at given time points."""
    P_list = list(params_vec)
    g0 = P_list[3]
    gamma = P_list[11]

    sol = solve_ivp(
        fun=lambda t, y: _opsin_odes(t, y, stim_on, stim_off, phi, P_list),
        t_span=(t_data[0], t_data[-1]),
        y0=[1.0, 0.0, 0.0, 0.0],
        method="RK45",
        t_eval=t_data,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    O1 = sol.y[1]
    O2 = sol.y[2]
    g = g0 * (O1 + gamma * O2)
    return g * (V - E_rev)


def fit_opsin_from_csv(
    csv_path: str,
    stim_on_s: float,
    stim_off_s: float,
    irradiance_W_mm2: float,
    lambda_nm: float,
    V_hold: float = -60.0,
    E_rev: float = 0.0,
    progress_callback=None,
) -> OpsinParams:
    """Fit 4-state opsin model to patch-clamp photocurrent data.

    CSV format: two columns [time_in_seconds, current_in_nA].
    Returns fitted OpsinParams.
    """
    # Auto-detect header: try parsing first row as floats; skip it if it fails
    with open(csv_path) as _f:
        _first = _f.readline()
    try:
        [float(v) for v in _first.split(",")]
        _skip = 0
    except ValueError:
        _skip = 1

    data = np.loadtxt(csv_path, delimiter=",", skiprows=_skip)
    if data.ndim == 1:
        data = data.reshape(-1, 2)

    # solve_ivp requires t_eval to be strictly increasing; sort by time
    order = np.argsort(data[:, 0])
    data = data[order]
    # Drop duplicate timestamps that would cause solve_ivp to error
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[unique_idx]

    # Convert time to ms so the ODE runs in ms → fitted rates come out in ms⁻¹,
    # exactly matching the Euler simulator's units. No conversion needed anywhere.
    t_data = data[:, 0] * 1000.0          # s → ms
    stim_on_ms = stim_on_s * 1000.0
    stim_off_ms = stim_off_s * 1000.0
    I_data = data[:, 1] * 1000.0          # nA → pA

    phi = irradiance_W_mm2 * (lambda_nm * 1e-9) / (H_PLANCK * C_LIGHT)

    # Initial guess in ms⁻¹ (original s⁻¹ values divided by 1000)
    # [Gd1,  Gd2,   Gr,    g0,   phim,   k1,   k2,   Gf0,     Gb0,   kf,    kb,    gamma,  p,    q]
    x0 = np.array([0.01, 0.005, 0.00289, 50.0, 2e13, 0.7, 0.6, 0.009975, 7.82e-9, 0.02, 0.001, 8.74e-4, 1.0, 1.0])
    lb = np.zeros(len(x0))
    lb[12] = 0.1  # p min
    lb[13] = 0.1  # q min

    call_count = [0]

    def residuals(params):
        call_count[0] += 1
        if progress_callback and call_count[0] % 20 == 0:
            progress_callback(call_count[0])
        try:
            I_pred = _predict_step_current(t_data, stim_on_ms, stim_off_ms, phi, V_hold, E_rev, params)
            return I_pred - I_data
        except Exception:
            return np.ones_like(I_data) * 1e6

    result = least_squares(
        residuals,
        x0,
        bounds=(lb, np.inf),
        method="trf",
        max_nfev=3000,
        x_scale="jac",
        verbose=0,
    )
    p = result.x

    fitted = OpsinParams(
        Gd1=p[0], Gd2=p[1], Gr=p[2], g0_ph=p[3], phim=p[4],
        k1=p[5], k2=p[6], Gf0=p[7], Gb0=p[8], kf=p[9], kb=p[10],
        gamma=p[11], p=p[12], q=p[13], E=E_rev,
        peak_lambda=lambda_nm,
    )
    return fitted, t_data, I_data


def predict_step_from_params(
    params: OpsinParams,
    t_data_ms: np.ndarray,
    stim_on_s: float,
    stim_off_s: float,
    irradiance_W_mm2: float,
    V_hold: float = -60.0,
) -> np.ndarray:
    """Compute predicted current trace for a step-stimulus experiment.

    t_data_ms is in ms (as returned by fit_opsin_from_csv).
    stim_on_s / stim_off_s are in seconds (from the UI spin boxes) and are
    converted to ms here. All rates in params are in ms⁻¹.
    """
    phi = irradiance_W_mm2 * (params.peak_lambda * 1e-9) / (H_PLANCK * C_LIGHT)
    pvec = np.array([
        params.Gd1, params.Gd2, params.Gr, params.g0_ph, params.phim,
        params.k1, params.k2, params.Gf0, params.Gb0, params.kf, params.kb,
        params.gamma, params.p, params.q,
    ])
    return _predict_step_current(
        t_data_ms, stim_on_s * 1000.0, stim_off_s * 1000.0,
        phi, V_hold, params.E, pvec,
    )


# ---------------------------------------------------------------------------
# User-defined opsin parameter storage
# ---------------------------------------------------------------------------

def _user_opsin_file() -> Path:
    p = Path.home() / ".optoTCSF" / "user_opsins.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_user_opsin(params: OpsinParams):
    fpath = _user_opsin_file()
    try:
        existing = json.loads(fpath.read_text()) if fpath.exists() else {}
    except Exception:
        existing = {}
    existing[params.name] = params.to_dict()
    fpath.write_text(json.dumps(existing, indent=2))


def delete_user_opsin(name: str) -> bool:
    """Remove a user-defined opsin by name. Returns True if it was found and deleted."""
    fpath = _user_opsin_file()
    if not fpath.exists():
        return False
    try:
        existing = json.loads(fpath.read_text())
    except Exception:
        return False
    if name not in existing:
        return False
    del existing[name]
    fpath.write_text(json.dumps(existing, indent=2))
    return True


def load_all_opsins() -> dict:
    """Return merged dict of builtin + user-defined opsins."""
    opsins = dict(BUILTIN_OPSINS)
    fpath = _user_opsin_file()
    if fpath.exists():
        try:
            user_data = json.loads(fpath.read_text())
            for name, d in user_data.items():
                opsins[name] = OpsinParams.from_dict({**d, "name": name})
        except Exception:
            pass
    return opsins
