"""FAKTUAL signature-kernel entropy (arXiv:2603.11634), reimplemented from the paper.

"Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for
Robotics Datasets", Sirigiri, de Lara, Agia, Shkurti and Ramos, 2026.
FAKTUAL = FAst trajectory Kernel enTropy cUration for imitation Learning.

Specification as stated in the paper:
  1. Each demonstration d_i is turned into a (multivariate) path X_i(t).
     The paper concatenates per-timestep state/action with (optionally ViT-)
     embedded observations and appends a time coordinate for injectivity
     (Appendix A). We use the end-effector path ee_pos (T, 3k) for k arms
     with a normalized time channel (a path in R^4 for one arm); the paper
     gives no fixed low-dim recipe.
  2. Truncated signature transform at level L:
       phi^L_{s,t}(x) = (1, \\int dx, ..., \\int...\\int dx(u_1)(x)...(x)dx(u_L))
     level-k term = the k-th iterated integral, a tensor in (R^d)^{(x)k}.
     The paper finds that the best L depends on the dataset (Sec. VIII,
     Fig. 12); we use depth 3.
  3. Signature kernel = truncated inner product of signatures:
       k^sig(x,y) = <phi^L(x), phi^L(y)>   (level-0 term contributes 1)
  4. Gram matrix normalized to unit diagonal:
       K^sig_{i,j} = k^sig(d_i,d_j),  K^sig_{i,i} = 1
     i.e. cosine normalization  k(x,y)/sqrt(k(x,x)k(y,y))  in signature space.
  5. Signature Shannon entropy (Def. 3), the von Neumann entropy:
       H^sig = -sum_{lambda in Lambda} lambda * ln(lambda)
     where Lambda = eigenvalues of K^sig / n. Since K is PSD with unit
     diagonal, trace(K/n)=1 and the eigenvalues are >=0 and sum to 1. This is
     the Renyi/von Neumann kernel entropy (RKE, NeurIPS'23) specialized to
     the signature kernel.
  6. H^sig is a dataset-level scalar (like h_g). The paper's own curation is
     a two-step greedy: greedily take an entropy-maximizing subset of size
     m*p (p in [0.5, 0.9]), then a determinant-maximizing subset of size
     m*(1-p), and take their union (Sec. VI, Appendix E).

Implementation notes:
  * The truncated signature is computed exactly in numpy via Chen's identity
    over piecewise-linear segments (segment increment delta -> level-k =
    delta^{(x)k}/k!; concatenation -> truncated tensor product). For low path
    dimension d and shallow depth L the feature length is (d^{L+1}-1)/(d-1)
    (d=4, L=3 -> 85), so the signature features are materialized and
    K = Phi_hat @ Phi_hat.T is built directly.
  * Per-demonstration wrapper (ours, as GeometricEntropyLOO): the
    leave-one-out contribution score_i = H^sig(all) - H^sig(all \\ i), i.e.
    how much demonstration i adds to dataset diversity. Direction, fixed in
    advance: higher contribution = keep, following the paper's report that
    success rises with diversity. The K/n normalization makes H(all) a
    constant offset across i, so the ranking is exactly a ranking of
    H(all \\ i).
  * greedy_entropy_select() implements the paper's entropy-greedy selection
    (forward greedy, entropy step only).
"""
from __future__ import annotations

import numpy as np

from .base import Demo, DemoMetric

EPS = 1e-12
DEFAULT_DEPTH = 3


# --------------------------------------------------------------------------- #
# Truncated signature transform (exact, piecewise-linear, pure numpy)
# --------------------------------------------------------------------------- #
def _segment_signature(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """Exact signature of a straight segment with increment `delta` in R^d,
    truncated at `depth`. Level k = delta^{(x)k}/k!, flattened row-major.
    Returns [level0 (len 1), level1 (len d), ..., level_depth (len d^depth)]."""
    levels = [np.array([1.0])]
    cur = np.array([1.0])
    fact = 1.0
    for k in range(1, depth + 1):
        cur = np.outer(cur, delta).ravel()  # delta^{(x)k}
        fact *= k
        levels.append(cur / fact)
    return levels


def _chen_product(A: list[np.ndarray], B: list[np.ndarray], depth: int, d: int) -> list[np.ndarray]:
    """Truncated tensor-algebra product of two signatures (Chen's identity):
    C_m = sum_{i+j=m} A_i (x) B_j."""
    C = []
    for m in range(depth + 1):
        acc = np.zeros(d ** m)
        for i in range(m + 1):
            acc = acc + np.outer(A[i], B[m - i]).ravel()
        C.append(acc)
    return C


def _path_signature(path: np.ndarray, depth: int) -> list[np.ndarray]:
    """Truncated signature of a discrete path (T,d), treated as piecewise-linear."""
    d = path.shape[1]
    deltas = np.diff(path, axis=0)
    if len(deltas) == 0:  # single point -> trivial signature (1,0,0,...)
        return [np.array([1.0])] + [np.zeros(d ** k) for k in range(1, depth + 1)]
    sig = _segment_signature(deltas[0], depth)
    for delta in deltas[1:]:
        sig = _chen_product(sig, _segment_signature(delta, depth), depth, d)
    return sig


def signature_feature(path: np.ndarray, depth: int = DEFAULT_DEPTH) -> np.ndarray:
    """Flattened truncated signature feature vector (concatenated levels 0..L)."""
    return np.concatenate(_path_signature(np.asarray(path, dtype=np.float64), depth))


# --------------------------------------------------------------------------- #
# Path preparation from a Demo
# --------------------------------------------------------------------------- #
def _prepare_path(
    demo: Demo,
    time_mode: str = "normalized",
    resample_T: int | None = None,
) -> np.ndarray:
    """ee_pos (T,3) -> path, optionally resampled and time-augmented.

    time_mode: "normalized" (t in [0,1], paper's injectivity coordinate),
               "seconds" (t = k*dt), or "none" (pure spatial, reparam-invariant).
    resample_T: uniform index subsample/interp to this many points (None = raw).
    """
    p = np.asarray(demo.ee_pos, dtype=np.float64)
    T = len(p)
    if resample_T is not None and T != resample_T:
        grid = np.linspace(0.0, 1.0, resample_T)
        base = np.linspace(0.0, 1.0, T)
        p = np.stack([np.interp(grid, base, p[:, d]) for d in range(p.shape[1])], axis=1)
        T = resample_T
    if time_mode == "none":
        return p
    if time_mode == "seconds":
        t = np.arange(T, dtype=np.float64) * float(demo.dt)
    elif time_mode == "normalized":
        t = np.linspace(0.0, 1.0, T)
    else:
        raise ValueError(f"unknown time_mode {time_mode!r}")
    return np.concatenate([t[:, None], p], axis=1)


# --------------------------------------------------------------------------- #
# Signature-kernel Gram matrix + entropy
# --------------------------------------------------------------------------- #
def gram_matrix(
    demos: list[Demo],
    depth: int = DEFAULT_DEPTH,
    time_mode: str = "normalized",
    resample_T: int | None = None,
    include_level0: bool = True,
) -> np.ndarray:
    """Normalized signature-kernel Gram matrix K (unit diagonal), shape (n,n).

    include_level0=True is the verbatim signature inner product (leading "1"
    term kept). It is shared by every path, so after the unit-diagonal cosine
    normalization it acts like a bias that inflates all pairwise similarities
    and *compresses* the entropy's dynamic range. Setting it False drops that
    constant term (analysis only).
    """
    feats = np.stack([
        signature_feature(_prepare_path(d, time_mode, resample_T), depth) for d in demos
    ])
    if not include_level0:
        feats = feats[:, 1:]
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, EPS)
    K = feats @ feats.T
    return np.clip(K, -1.0, 1.0)


def kernel_entropy(K: np.ndarray) -> float:
    """Signature Shannon/von-Neumann entropy from a unit-diagonal Gram matrix:
    eigenvalues of K/n, H = -sum lambda ln lambda."""
    n = K.shape[0]
    if n == 0:
        return float("-inf")
    if n == 1:
        return 0.0
    w = np.linalg.eigvalsh(K / n)
    w = w[w > EPS]                 # drop numerical zeros/negatives of the PSD matrix
    if w.size == 0:
        return 0.0
    w = w / w.sum()               # guard drift; trace(K/n)=1 in exact arithmetic
    return float(-np.sum(w * np.log(w)))


def faktual_entropy(
    demos: list[Demo],
    depth: int = DEFAULT_DEPTH,
    time_mode: str = "normalized",
    resample_T: int | None = None,
    include_level0: bool = True,
) -> float:
    """Dataset-level FAKTUAL signature entropy."""
    return kernel_entropy(gram_matrix(demos, depth, time_mode, resample_T, include_level0))


# --------------------------------------------------------------------------- #
# Per-demo LOO wrapper (mirrors GeometricEntropyLOO)
# --------------------------------------------------------------------------- #
class FAKTUALMetric(DemoMetric):
    """Per-demo leave-one-out signature-entropy diversity contribution.

    score_i = H^sig(all) - H^sig(all \\ i). Higher = demo adds more diversity =
    keep. Deterministic, no RNG.
    """

    higher_is_better = True

    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        time_mode: str = "normalized",
        resample_T: int | None = None,
        include_level0: bool = True,
    ):
        self.depth = depth
        self.time_mode = time_mode
        self.resample_T = resample_T
        self.include_level0 = include_level0
        self.name = f"faktual[loo,L{depth}]"

    def score(self, demos: list[Demo]) -> np.ndarray:
        K = gram_matrix(demos, self.depth, self.time_mode, self.resample_T, self.include_level0)
        n = len(demos)
        full = kernel_entropy(K)
        out = np.empty(n)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            keep[i] = False
            out[i] = full - kernel_entropy(K[np.ix_(keep, keep)])
            keep[i] = True
        return out


# --------------------------------------------------------------------------- #
# The paper's entropy-greedy selection
# --------------------------------------------------------------------------- #
def greedy_entropy_select(
    demos: list[Demo],
    k: int,
    depth: int = DEFAULT_DEPTH,
    time_mode: str = "normalized",
    resample_T: int | None = None,
    include_level0: bool = True,
) -> np.ndarray:
    """Forward-greedy entropy-maximizing subset of size k (the paper's Sec. VI
    curation, entropy step only; the full recipe also unions a determinant-
    maximizing subset). Returns selected indices in selection order.

    O(k * n * eig(k)). Deterministic (ties resolved by lowest index)."""
    K = gram_matrix(demos, depth, time_mode, resample_T, include_level0)
    n = len(demos)
    k = min(max(1, k), n)
    selected: list[int] = []
    remaining = list(range(n))
    while len(selected) < k:
        best_j, best_H = remaining[0], -np.inf
        for j in remaining:
            idx = selected + [j]
            H = kernel_entropy(K[np.ix_(idx, idx)])
            if H > best_H:
                best_H, best_j = H, j
        selected.append(best_j)
        remaining.remove(best_j)
    return np.array(selected)
