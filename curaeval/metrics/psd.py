"""PSD total-spectral-power score (arXiv:2605.01544), reimplemented from the paper.

Specification as stated in the paper:
  X_d(Omega) = sum_t x_d(t) e^{-j Omega t},  Omega = 2*pi*k/T
  P_{i,d}(Omega) proportional to |X_d(Omega)|^2
  P_i(Omega) = sum_{d=0..2} P_{i,d}(Omega)
  W_i = sum_Omega P_i(Omega)
Signal: raw EE position in 3D Cartesian space. No windowing, no normalization,
no frequency-band restriction are specified by the paper. Lower W = better;
retain demos with W below the (1-rho) quantile.

Note: by Parseval, sum_Omega |X(Omega)|^2 = T * sum_t |x(t)|^2, so the score
on raw positions is dominated by the DC term (mean position magnitude) and
scales with episode length T. The "position" signal follows the paper;
"velocity" applies the same score to end-effector velocity (Velocity-PSD), and
"position_centered" removes the mean first (analysis only).
"""
from __future__ import annotations

import numpy as np

from .base import Demo, DemoMetric


def _total_spectral_power(x: np.ndarray) -> float:
    """x: (T, D). Verbatim W = sum over dims and all DFT bins of |X|^2."""
    X = np.fft.fft(x, axis=0)
    return float(np.sum(np.abs(X) ** 2))


class PSDMetric(DemoMetric):
    higher_is_better = False  # lower total power = smoother = better

    def __init__(self, signal: str = "position"):
        # "position": as in the paper; "velocity": Velocity-PSD;
        # "position_centered": mean position removed first.
        assert signal in ("position", "position_centered", "velocity")
        self.signal = signal
        self.name = f"psd[{signal}]"

    def score(self, demos: list[Demo]) -> np.ndarray:
        out = np.empty(len(demos))
        for i, d in enumerate(demos):
            x = np.asarray(d.ee_pos, dtype=np.float64)
            if self.signal == "position_centered":
                x = x - x.mean(axis=0, keepdims=True)
            elif self.signal == "velocity":
                x = np.diff(x, axis=0) / d.dt
            out[i] = _total_spectral_power(x)
        return out
