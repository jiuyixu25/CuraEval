"""Property tests of the SigEnt (FAKTUAL, arXiv:2603.11634) score on
synthetic trajectories: directions and documented properties only.
"""
import numpy as np
import pytest

from curaeval.metrics.base import Demo
from curaeval.metrics.faktual import (
    FAKTUALMetric,
    faktual_entropy,
    greedy_entropy_select,
    signature_feature,
)

RNG = np.random.default_rng(0)
DT = 0.05  # 20 Hz


def min_jerk(p0, p1, T=120):
    s = np.linspace(0, 1, T)[:, None]
    shape = 10 * s**3 - 15 * s**4 + 6 * s**5
    return np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * shape


def make_demo(noise=0.0, p0=(0.1, 0.2, 0.3), p1=(0.5, 0.1, 0.2), T=120, seed=None):
    rng = np.random.default_rng(seed)
    p = min_jerk(p0, p1, T)
    if noise > 0:
        p = p + rng.normal(0, noise, p.shape)
    return Demo(ee_pos=p, dt=DT)


def test_faktual_diverse_beats_identical():
    # As test_hg_diverse_beats_identical: a varied set must
    # carry more signature entropy than a near-identical set.
    identical = [make_demo(0.0005, seed=i) for i in range(20)]
    diverse = [make_demo(0.0005, p1=tuple(RNG.uniform(0.2, 0.8, 3)), seed=i) for i in range(20)]
    assert faktual_entropy(diverse) > faktual_entropy(identical)


def test_faktual_identical_near_zero_entropy():
    # All (near-)identical -> Gram ~ all-ones -> one dominant eigenvalue -> H ~ 0.
    identical = [make_demo(0.0, seed=0) for _ in range(15)]
    diverse = [make_demo(0.0, p1=tuple(RNG.uniform(0.2, 0.8, 3)), seed=i) for i in range(15)]
    h_id = faktual_entropy(identical)
    assert h_id < 0.05
    assert faktual_entropy(diverse) > 10 * max(h_id, 1e-6)


def test_faktual_loo_flags_outlier():
    # As test_hg_loo_flags_outlier: the lone outlier adds the most
    # marginal diversity -> highest leave-one-out contribution.
    base = [make_demo(0.0005, seed=i) for i in range(15)]
    outlier = make_demo(0.0005, p0=(0.9, 0.9, 0.9), p1=(0.1, 0.8, 0.9), seed=99)
    scores = FAKTUALMetric().score(base + [outlier])
    assert np.argmax(scores) == len(base)


def test_faktual_loo_direction_and_select_api():
    m = FAKTUALMetric()
    assert m.higher_is_better is True
    demos = [make_demo(0.0005, p1=tuple(RNG.uniform(0.2, 0.8, 3)), seed=i) for i in range(10)]
    keep = m.select(demos, retention=0.5)
    assert len(keep) == 5 and len(set(keep.tolist())) == 5


def test_faktual_deterministic():
    demos = [make_demo(0.001, p1=tuple(RNG.uniform(0.2, 0.8, 3)), seed=i) for i in range(8)]
    m = FAKTUALMetric()
    assert np.allclose(m.score(demos), m.score(demos))


def test_signature_feature_length():
    # concatenated levels 0..L for path in R^d has length (d^{L+1}-1)/(d-1).
    # ee_pos (3) + normalized time channel -> d=4; depth 3 -> 1+4+16+64 = 85.
    from curaeval.metrics.faktual import _prepare_path

    path = _prepare_path(make_demo(0.0), time_mode="normalized")
    assert path.shape[1] == 4
    feat = signature_feature(path, depth=3)
    assert feat.shape == (85,)
    assert feat[0] == pytest.approx(1.0)  # level-0 term is always 1


def test_greedy_select_prefers_spread():
    # The paper's greedy selection: from 3 tight clusters, greedy entropy-max should
    # cover distinct clusters rather than pile into one.
    clusters = [(0.2, 0.2, 0.2), (0.8, 0.2, 0.5), (0.3, 0.8, 0.7)]
    demos = []
    for c in clusters:
        for j in range(5):
            demos.append(make_demo(0.001, p1=c, seed=hash((c, j)) % 10000))
    sel = greedy_entropy_select(demos, k=3)
    picked_clusters = {i // 5 for i in sel}
    assert len(picked_clusters) == 3, "greedy entropy-max should span all 3 clusters"
