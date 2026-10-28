"""Geometric Entropy H_G dataset-diversity metric (arXiv:2606.20871).

Specification as stated in the paper:
  1. arc-length resample each trajectory to T=50 waypoints (linear interp)
  2. anchor: p'_k = p_k - p_T   (terminal point to origin)
  3. rotate into target frame: p''_k = R_T^T p'_k   (identity if no target)
  4. scale: p~_k = p''_k / (||p_1 - p_T|| + eps),  eps = 1e-9
  5. descriptor xi = flatten -> R^150; stack M demos -> X in R^{M x 150}
  6. mean-center X, SVD -> eigenvalues lambda_j of the covariance
     p_j = lambda_j / sum_k lambda_k
     d_eff = exp( -sum_{p_j>0} p_j ln p_j )
     H_G = ln d_eff + ln( sum_j lambda_j )
  Multi-phase: H_G^avg = weighted mean over phases by sample counts.

Convention note: the paper says "eigenvalues from SVD of the mean-centered
descriptor matrix" without fixing lambda = s^2 vs s^2/(M-1). We default to
covariance eigenvalues s^2/(M-1); `eig_convention="s2"` switches to s^2.

Per-demonstration wrapper (ours, not part of the paper): the leave-one-out
contribution
  score_i = H_G(all) - H_G(all minus i),
i.e. how much demonstration i adds to dataset diversity. Direction, fixed in
advance: higher contribution = keep (the rising side of the paper's inverted-U
relation between H_G and success).
"""
from __future__ import annotations

import numpy as np

from .base import Demo, DemoMetric

T_RESAMPLE = 50
EPS = 1e-9


def _resample_arclength(p: np.ndarray, T: int = T_RESAMPLE) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < EPS:  # degenerate: no motion
        return np.repeat(p[:1], T, axis=0)
    grid = np.linspace(0.0, s[-1], T)
    return np.stack([np.interp(grid, s, p[:, d]) for d in range(p.shape[1])], axis=1)


def descriptor(p: np.ndarray, R_T: np.ndarray | None = None) -> np.ndarray:
    q = _resample_arclength(np.asarray(p, dtype=np.float64))
    q = q - q[-1]
    if R_T is not None:
        q = q @ R_T  # (R_T^T q^T)^T
    q = q / (np.linalg.norm(q[0]) + EPS)  # ||p_1 - p_T|| after anchoring = ||q[0]||
    return q.reshape(-1)


def h_g(descriptors: np.ndarray, eig_convention: str = "cov") -> float:
    X = descriptors - descriptors.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s**2 / (len(X) - 1) if eig_convention == "cov" else s**2
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("-inf")
    p = lam / lam.sum()
    d_eff = np.exp(-np.sum(p * np.log(p)))
    return float(np.log(d_eff) + np.log(lam.sum()))


class GeometricEntropyLOO(DemoMetric):
    """Per-demo leave-one-out diversity contribution (our wrapper)."""

    higher_is_better = True
    name = "geom_entropy[loo]"

    def __init__(self, eig_convention: str = "cov"):
        self.eig_convention = eig_convention

    def score(self, demos: list[Demo]) -> np.ndarray:
        D = np.stack([descriptor(d.ee_pos) for d in demos])
        full = h_g(D, self.eig_convention)
        out = np.empty(len(demos))
        for i in range(len(demos)):
            out[i] = full - h_g(np.delete(D, i, axis=0), self.eig_convention)
        return out
