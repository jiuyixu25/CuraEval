"""RINSE-SAL spectral-arc-length smoothness metric (arXiv:2604.23000).

Specification as stated in the paper and its appendix:
  speed signal v[0:T-1] = ||diff(ee_pos)|| / dt
  V_k = FFT(v) at f_k = k / (T * dt), one-sided k = 0 .. T//2
  L_k = log(|V_k| + eps),  eps = 1e-5
  SAL = - sum_{k=1}^{T/2} sqrt( (f_k - f_{k-1})^2 + (L_k - L_{k-1})^2 )
Higher (closer to zero) = smoother = better. All frequencies up to Nyquist,
no pre-filtering, "no task-specific tuning".

Note: unlike classic SPARC (Balasubramanian et al.,
https://github.com/siva82kb/SPARC), RINSE-SAL does not normalize the frequency axis or the
magnitude spectrum, so the two terms under the sqrt carry different units
(Hz vs log-magnitude); the score is therefore sensitive to trajectory
duration and speed scale. The verbatim variant is the
primary one; `classic_sparc` exists for analysis only.
"""
from __future__ import annotations

import numpy as np

from .base import Demo, DemoMetric

EPS = 1e-5


def rinse_sal(speed: np.ndarray, dt: float) -> float:
    T = len(speed)
    V = np.abs(np.fft.fft(speed))
    k = np.arange(T // 2 + 1)
    f = k / (T * dt)
    L = np.log(V[: T // 2 + 1] + EPS)
    df = np.diff(f)
    dL = np.diff(L)
    return float(-np.sum(np.sqrt(df**2 + dL**2)))


def classic_sparc(speed: np.ndarray, dt: float, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Reference SPARC (normalized magnitude, adaptive cutoff) for analysis."""
    n = max(len(speed), 256)
    V = np.abs(np.fft.rfft(speed, n=n))
    f = np.fft.rfftfreq(n, d=dt)
    V = V / (V.max() + 1e-12)
    sel = f <= fc
    f_sel, V_sel = f[sel], V[sel]
    above = np.nonzero(V_sel >= amp_th)[0]
    if len(above) >= 2:
        f_sel, V_sel = f_sel[: above[-1] + 1], V_sel[: above[-1] + 1]
    fr = f_sel / (f_sel[-1] + 1e-12)
    return float(-np.sum(np.sqrt(np.diff(fr) ** 2 + np.diff(V_sel) ** 2)))


class SALMetric(DemoMetric):
    higher_is_better = True  # less negative = smoother

    def __init__(self, variant: str = "rinse"):
        assert variant in ("rinse", "classic_sparc")
        self.variant = variant
        self.name = f"sal[{variant}]"

    def score(self, demos: list[Demo]) -> np.ndarray:
        out = np.empty(len(demos))
        for i, d in enumerate(demos):
            v = np.linalg.norm(np.diff(np.asarray(d.ee_pos, dtype=np.float64), axis=0), axis=1) / d.dt
            fn = rinse_sal if self.variant == "rinse" else classic_sparc
            out[i] = fn(v, d.dt)
        return out
