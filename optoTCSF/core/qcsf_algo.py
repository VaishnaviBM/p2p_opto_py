"""
Quick CSF (qCSF) Bayesian adaptive algorithm.
Port of Lesmes & Lu (2010) qCSF MATLAB code.
"""

import numpy as np
from dataclasses import dataclass, field


def find_qcsf(freq: np.ndarray, log_gain, log_center, octave_width, log_trunc) -> np.ndarray:
    """Compute log-sensitivity from truncated log-parabola CSF model.

    Args:
        freq: log10(frequency) values.
        log_gain: log10(peak sensitivity).
        log_center: log10(peak frequency in cpd).
        octave_width: bandwidth in octaves.
        log_trunc: log10(low-frequency truncation level).

    Returns:
        log_csf: log-sensitivity values.
    """
    tau_decay = 0.5
    K = np.log10(tau_decay)
    log_width = (10.0 ** octave_width) * np.log10(2) / 2.0

    freq = np.atleast_1d(freq)
    log_gain = np.atleast_1d(log_gain)[..., np.newaxis] if np.ndim(log_gain) > 0 else log_gain
    log_center = np.atleast_1d(log_center)[..., np.newaxis] if np.ndim(log_center) > 0 else log_center
    log_width_arr = (np.atleast_1d(log_width)[..., np.newaxis]
                     if np.ndim(log_width) > 0 else log_width)
    log_trunc_arr = np.atleast_1d(log_trunc)[..., np.newaxis] if np.ndim(log_trunc) > 0 else log_trunc

    logP = log_gain + K * ((freq - log_center) / log_width_arr) ** 2

    lin_trunc = 10.0 ** log_trunc_arr
    trunc_half = log_gain - lin_trunc

    left = (logP < trunc_half) & (freq < log_center)
    log_csf = np.where(left, trunc_half, logP)
    log_csf = np.where(log_csf < 0, 0.0, log_csf)

    return log_csf.squeeze() if log_csf.ndim > 1 else log_csf


@dataclass
class QCSFState:
    """Complete state for the qCSF adaptive procedure."""
    # Stimulus space
    contrasts: np.ndarray = field(default_factory=lambda: np.array([]))
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))

    # Parameter grid
    gain: np.ndarray = field(default_factory=lambda: np.array([]))
    center: np.ndarray = field(default_factory=lambda: np.array([]))
    width: np.ndarray = field(default_factory=lambda: np.array([]))
    trunc: np.ndarray = field(default_factory=lambda: np.array([]))

    # Bayesian prior/posterior
    prior: np.ndarray = field(default_factory=lambda: np.array([]))
    prior0: np.ndarray = field(default_factory=lambda: np.array([]))

    # Psychometric function parameters
    guess_rate: float = 0.5
    lapse_rate: float = 0.05

    # Sampling settings
    prior_samples: int = 500
    opt_percentile: float = 10.0

    # Trial tracking
    trial: int = 0
    est_csf_history: list = field(default_factory=list)
    est_sensitivity_history: list = field(default_factory=list)
    correct_responses: list = field(default_factory=list)
    incorrect_responses: list = field(default_factory=list)
    history: list = field(default_factory=list)

    # Next stimulus
    next_frequency: float = 1.0
    next_contrast: float = 0.1


def setup_qcsf(
    fix_temporal: bool = True,
    priors: tuple = (75, 1, 2.5, 0.1),
    contrast_levels: int = 1024,
    frequency_levels: int = 12,
    min_contrast: float = 0.001,
    max_contrast: float = 1.0,
    min_freq: float = 0.25,
    max_freq: float = 16.0,
) -> QCSFState:
    """Initialize the qCSF state.

    Args:
        fix_temporal: If True, fix TF and vary SF (tCSF mode); else fix SF, vary TF.
        priors: (peak_gain, peak_freq_cpd, bandwidth_octaves, low_freq_trunc).
    """
    state = QCSFState()
    state.guess_rate = 0.5
    state.lapse_rate = 0.05
    state.prior_samples = 500
    state.opt_percentile = 10.0

    state.contrasts = np.logspace(
        np.log10(min_contrast), np.log10(max_contrast), contrast_levels
    )
    state.frequencies = np.logspace(
        np.log10(min_freq), np.log10(max_freq), frequency_levels
    )

    n_params = [29, 28, 27, 26]
    state.gain = np.linspace(np.log10(2), np.log10(2000), n_params[0])
    state.center = np.linspace(np.log10(0.2), np.log10(20), n_params[1])
    state.width = np.linspace(np.log10(1), np.log10(9), n_params[2])
    state.trunc = np.linspace(np.log10(0.02), np.log10(2), n_params[3])

    state.prior = _build_prior(state, priors)
    state.prior0 = state.prior.copy()
    return state


def _sech_fit_weight(param_vec: np.ndarray, guess: float, confidence: float) -> float:
    """Find hyperbolic-secant weight to match desired confidence level."""
    from scipy.optimize import fminbound

    def cost(w):
        prior_1d = 1.0 / np.cosh(w * (param_vec - guess))
        prior_1d /= prior_1d.sum()
        H = -np.sum(prior_1d * np.log(prior_1d + 1e-300))
        H_flat = np.log(len(param_vec))
        H_target = (1.0 - confidence) * H_flat
        return (H - H_target) ** 2

    return fminbound(cost, 0.01, 100.0, xtol=1e-4)


def _build_prior(state: QCSFState, priors: tuple) -> np.ndarray:
    """Build 4-D joint prior as product of sech marginals."""
    prior_modes = np.log10(np.array(priors, dtype=float))
    confidence = 0.1

    wg = _sech_fit_weight(state.gain, prior_modes[0], confidence)
    wc = _sech_fit_weight(state.center, prior_modes[1], confidence)
    ww = _sech_fit_weight(state.width, prior_modes[2], confidence)
    wt = _sech_fit_weight(state.trunc, prior_modes[3], confidence)

    G, C, W, T = np.meshgrid(state.gain, state.center, state.width, state.trunc,
                              indexing="ij")
    prior = (
        (1.0 / np.cosh(wg * (G - prior_modes[0])))
        * (1.0 / np.cosh(wc * (C - prior_modes[1])))
        * (1.0 / np.cosh(ww * (W - prior_modes[2])))
        * (1.0 / np.cosh(wt * (T - prior_modes[3])))
    )
    prior /= prior.sum()
    return prior


def _csf_probability(log_freq: float, log_contrast: float, log_csf_vals: np.ndarray,
                     guess_rate: float, lapse_rate: float) -> np.ndarray:
    """Compute psychometric probability of correct response."""
    log_tau = -log_csf_vals
    Pc = np.minimum(
        1.0 - lapse_rate,
        guess_rate + (1.0 - guess_rate) * (1.0 - np.exp(-10.0 ** (2.0 * (log_contrast - log_tau))))
    )
    return Pc


def _marginalize(prior: np.ndarray, axes: list) -> np.ndarray:
    result = prior.copy()
    for ax in sorted(axes, reverse=True):
        result = result.sum(axis=ax)
    return result


def _analyze_posterior(state: QCSFState) -> np.ndarray:
    """Compute posterior mean CSF parameters."""
    prior = state.prior
    g_hat = np.dot(state.gain, _marginalize(prior, [1, 2, 3]))
    c_hat = np.dot(state.center, _marginalize(prior, [0, 2, 3]))
    w_hat = np.dot(state.width, _marginalize(prior, [0, 1, 3]))
    t_hat = np.dot(state.trunc, _marginalize(prior, [0, 1, 2]))
    return np.array([g_hat, c_hat, w_hat, t_hat])


def _sample_from_prior(prior: np.ndarray, n_samples: int) -> np.ndarray:
    """Draw samples from the 4-D prior (flattened and sampled)."""
    flat = prior.ravel()
    flat /= flat.sum()
    indices = np.random.choice(len(flat), size=n_samples, replace=True, p=flat)
    return np.array(np.unravel_index(indices, prior.shape)).T  # (n_samples, 4)


def pretrial(state: QCSFState) -> QCSFState:
    """Compute the next optimal stimulus (information-gain criterion)."""
    n_samples = state.prior_samples
    p_tile = state.opt_percentile / 100.0
    p_tile = np.clip(p_tile, 0.01, 0.99)

    idx = _sample_from_prior(state.prior, n_samples)
    sGain = state.gain[idx[:, 0]]
    sCenter = state.center[idx[:, 1]]
    sWidth = state.width[idx[:, 2]]
    sTrunc = state.trunc[idx[:, 3]]

    freq_arr = np.log10(state.frequencies)
    contrast_arr = np.log10(state.contrasts)
    FF, CC = np.meshgrid(freq_arr, contrast_arr, indexing="ij")
    n_stim = len(freq_arr) * len(contrast_arr)
    FF_flat = FF.ravel()
    CC_flat = CC.ravel()

    # For each stimulus, compute information gain over sampled CSFs
    I = np.zeros(n_stim)
    for si in range(n_stim):
        log_freq = FF_flat[si]
        log_c = CC_flat[si]
        log_csf_s = find_qcsf(log_freq, sGain, sCenter, sWidth, sTrunc)
        Pc = _csf_probability(log_freq, log_c, log_csf_s, state.guess_rate, state.lapse_rate)
        mean_Pc = np.mean(Pc)
        H_mean = _entropy(mean_Pc)
        H_each = _entropy(Pc)
        I[si] = H_mean - np.mean(H_each)

    sorted_idx = np.argsort(I)[::-1]
    top_k = max(1, int(np.ceil(p_tile * n_stim)))
    best_rand = sorted_idx[:top_k][np.random.randint(top_k)]

    fi = best_rand // len(contrast_arr)
    ci = best_rand % len(contrast_arr)

    state.next_frequency = state.frequencies[fi]
    state.next_contrast = state.contrasts[ci]
    return state


def posttrial(
    state: QCSFState,
    frequency: float,
    contrast: float,
    response: int,
) -> QCSFState:
    """Update the posterior given the observer's response."""
    state.trial += 1
    log_freq = np.log10(frequency)
    log_c = np.log10(contrast)

    G, C, W, T = np.meshgrid(state.gain, state.center, state.width, state.trunc,
                              indexing="ij")
    log_csf_grid = find_qcsf(log_freq, G.ravel(), C.ravel(), W.ravel(), T.ravel())
    Pc_grid = _csf_probability(log_freq, log_c, log_csf_grid,
                               state.guess_rate, state.lapse_rate / state.guess_rate)
    Pc_grid = Pc_grid.reshape(state.prior.shape)

    if response:
        likelihood = Pc_grid
    else:
        likelihood = 1.0 - Pc_grid

    posterior = state.prior * likelihood
    total = posterior.sum()
    if total > 0:
        posterior /= total
    state.prior = posterior

    est_csf = _analyze_posterior(state)
    state.est_csf_history.append(est_csf)

    log_freqs = np.log10(state.frequencies)
    est_sens = find_qcsf(log_freqs, est_csf[0], est_csf[1], est_csf[2], est_csf[3])
    state.est_sensitivity_history.append(est_sens)

    jitter = 0.02
    plot_pt = [log10_freq + jitter * np.random.randn()
               for log10_freq in [log_freq]]
    plot_sens = -log_c + jitter * np.random.randn()
    if response:
        state.correct_responses.append((plot_pt[0], plot_sens))
    else:
        state.incorrect_responses.append((plot_pt[0], plot_sens))

    state.history.append((frequency, contrast, response))
    return state


def get_current_estimate(state: QCSFState) -> tuple:
    """Return (est_csf_params, est_sensitivity) for the current posterior."""
    est_csf = _analyze_posterior(state)
    log_freqs = np.log10(state.frequencies)
    est_sens = find_qcsf(log_freqs, est_csf[0], est_csf[1], est_csf[2], est_csf[3])
    return est_csf, est_sens


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


# ---------------------------------------------------------------------------
# Weibull psychometric function fitting for TCSF estimation
# ---------------------------------------------------------------------------

def weibull(x: np.ndarray, t: float, b: float = 1.0) -> np.ndarray:
    """Weibull psychometric function P(correct | contrast c).
    x: contrast values (linear scale).
    t: threshold (contrast at ~0.816 correct for b=1).
    b: slope (default 1 gives steep enough).
    """
    return 1.0 - np.exp(-((x / t) ** b))


def fit_weibull(intensities: np.ndarray, responses: np.ndarray, init_t: float = 0.15) -> float:
    """Fit Weibull threshold to binary response data.

    Args:
        intensities: Stimulus contrast values.
        responses: Binary responses (0 or 1).
        init_t: Initial guess for threshold.

    Returns:
        threshold: Fitted threshold t.
    """
    from scipy.optimize import minimize_scalar

    def neg_log_lik(log_t):
        t = 10.0 ** log_t
        p = 0.5 + 0.5 * weibull(intensities, t)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return -np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))

    result = minimize_scalar(neg_log_lik, bounds=(-4, 0), method="bounded")
    return 10.0 ** result.x


def estimate_sensitivity_grid(
    data: np.ndarray,
    ntf: int = 12,
    nsf: int = 12,
    tf_range: tuple = (0.3, 2.9),
    sf_range: tuple = (-1.4, 3.0),
    win_size: float = 1.0,
) -> tuple:
    """Estimate log-sensitivity on a TF x SF grid using sliding-window Weibull fitting.

    Args:
        data: Array with columns [TF, SF, log10_contrast_level, response].
        ntf, nsf: Grid dimensions.
        tf_range: (min, max) in log(TF).
        sf_range: (min, max) in log(SF).
        win_size: Window radius in log units.

    Returns:
        tf: TF grid centres (Hz).
        sf: SF grid centres (cpd).
        S: Sensitivity matrix (ntf x nsf).
    """
    tf_list = np.linspace(tf_range[0], tf_range[1], ntf + 1)
    sf_list = np.linspace(sf_range[0], sf_range[1], nsf + 1)

    tf = np.exp((tf_list[:-1] + tf_list[1:]) / 2)
    sf = np.exp((sf_list[:-1] + sf_list[1:]) / 2)

    log_tf = np.log(data[:, 0])
    log_sf = np.log(data[:, 1])
    contrasts = 0.1 ** data[:, 2]
    responses = data[:, 3].astype(bool)

    S = np.zeros((ntf, nsf))
    for i in range(ntf):
        for j in range(nsf):
            mask = ((log_tf - tf_list[i]) ** 2 + (log_sf - sf_list[j]) ** 2) <= win_size ** 2
            if mask.sum() > 3:
                try:
                    t = fit_weibull(contrasts[mask], responses[mask].astype(float))
                    S[i, j] = -np.log10(t)
                except Exception:
                    S[i, j] = np.nan
    return tf, sf, S
