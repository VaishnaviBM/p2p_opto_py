"""
Decision rules (linking hypothesis) for predicting perceptual attenuation
from opsin photocurrent modulation.
Ref: attenuation_models.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union
from scipy.signal import butter, lfilter


@dataclass
class DecisionParams:
    rule: str = "Probability Summation"
    # Probability Summation
    beta: list = field(default_factory=lambda: [1.0, -0.5])
    # 1st-order leaky integrator
    tau: float = 10.0
    # 2nd-order Butterworth
    fc: float = 8.0
    # BashKeyboard (frequency-domain weighting)
    cutoff: float = 7.935
    steepness: float = 0.5845

    def to_dict(self):
        return {
            "rule": self.rule,
            "beta": list(self.beta),
            "tau": self.tau,
            "fc": self.fc,
            "cutoff": self.cutoff,
            "steepness": self.steepness,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def decision_rule(y: np.ndarray, dt: float, params: DecisionParams) -> float:
    """Apply the specified decision rule to a photocurrent trace.

    Args:
        y: Normalised photocurrent trace (dimensionless).
        dt: Time step in ms.
        params: Decision rule parameters.

    Returns:
        Scalar response (proxy for perceived stimulus strength).
    """
    rule = params.rule

    if rule == "Probability Summation":
        beta = params.beta
        pos = y >= 0
        yy = np.where(pos, beta[0] * y, beta[1] * y)
        return float(dt * np.sum(yy))

    elif rule == "Max-Min":
        y2 = 0.5 * (y + 1.0)
        return float(np.max(y2) - np.min(y2))

    elif rule == "Max":
        y2 = 0.5 * (y + 1.0)
        return float(np.max(y2))

    elif rule == "1OrderFilter":
        alpha = dt / (params.tau + dt)
        y_filt = lfilter([alpha], [1.0, -(1.0 - alpha)], y)
        pos = y_filt >= 0
        yy = np.where(pos, params.beta[0] * y_filt, params.beta[1] * y_filt)
        return float(dt * np.sum(yy))

    elif rule == "2OrderFilter":
        fs = 1000.0 / dt
        b, a = butter(2, params.fc / (fs / 2.0), btype="low")
        y_filt = lfilter(b, a, y)
        pos = y_filt >= 0
        yy = np.where(pos, params.beta[0] * y_filt, params.beta[1] * y_filt)
        return float(dt * np.sum(yy))

    elif rule == "BashKeyboard":
        N = len(y)
        fs = 1000.0 / dt
        f = np.arange(N) * (fs / N)
        Y_fft = np.fft.fft(y)
        weights = 1.0 / (1.0 + np.exp(-params.steepness * (f - params.cutoff)))
        power = np.abs(Y_fft) ** 2
        return float(np.sum(power * weights))

    else:
        raise ValueError(f"Unknown decision rule: {rule}")


AVAILABLE_RULES = [
    "Probability Summation",
    "Max-Min",
    "Max",
    "1OrderFilter",
    "2OrderFilter",
    "BashKeyboard",
]
