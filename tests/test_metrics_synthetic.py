"""Property tests of the scores on synthetic trajectories.

They check directions and documented properties, not agreement with each
paper's own numbers.
"""
import numpy as np
import pytest

from curaeval.metrics import (
    Demo,
    EpisodeLength,
    GeometricEntropyLOO,
    MeanSquaredJerk,
    PathInefficiency,
    PSDMetric,
    SALMetric,
)
from curaeval.metrics.geometric_entropy import descriptor, h_g
from curaeval.metrics.psd import _total_spectral_power

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


def test_sal_prefers_smooth():
    smooth, noisy = make_demo(0.0), make_demo(0.004, seed=1)
    s = SALMetric("rinse").score([smooth, noisy])
    assert s[0] > s[1], "smooth demo must have less-negative SAL"


def test_psd_velocity_variant_prefers_smooth():
    smooth, noisy = make_demo(0.0), make_demo(0.004, seed=1)
    s = PSDMetric("velocity").score([smooth, noisy])
    assert s[0] < s[1], "velocity-PSD must separate smooth from noisy"


def test_psd_verbatim_is_noise_blind():
    # PSD on raw positions is dominated by the DC term (Parseval): 4 mm of
    # high-frequency noise moves W by <1%, with indeterminate sign, while
    # position offsets move it by orders of magnitude.
    smooth, noisy = make_demo(0.0), make_demo(0.004, seed=1)
    s = PSDMetric("position").score([smooth, noisy])
    assert abs(s[1] - s[0]) / s[0] < 0.01, "noise term should be negligible vs DC"


def test_psd_parseval_dc_dominance():
    # The note in psd.py: PSD on raw positions equals T * sum |x|^2
    # (Parseval), so it is dominated by the DC term.
    x = make_demo(0.0).ee_pos
    W = _total_spectral_power(x)
    assert np.isclose(W, len(x) * np.sum(x**2), rtol=1e-9)
    far = make_demo(0.0, p0=(1.1, 1.2, 1.3), p1=(1.5, 1.1, 1.2))
    s = PSDMetric("position").score([make_demo(0.0), far])
    assert s[1] > s[0], "identical shape farther from origin scores worse (DC term)"


def test_jerk_prefers_smooth():
    smooth, noisy = make_demo(0.0), make_demo(0.004, seed=1)
    s = MeanSquaredJerk().score([smooth, noisy])
    assert s[0] < s[1]


def test_path_inefficiency_baseline():
    straight = make_demo(0.0)
    detour_p = np.concatenate([min_jerk((0.1, 0.2, 0.3), (0.4, 0.6, 0.5), 60),
                               min_jerk((0.4, 0.6, 0.5), (0.5, 0.1, 0.2), 60)])
    s = PathInefficiency().score([straight, Demo(ee_pos=detour_p, dt=DT)])
    assert s[0] == pytest.approx(1.0, abs=0.05)
    assert s[1] > 1.5


def test_hg_diverse_beats_identical():
    identical = [make_demo(0.0005, seed=i) for i in range(20)]
    diverse = [make_demo(0.0005, p1=tuple(RNG.uniform(0.2, 0.8, 3)), seed=i) for i in range(20)]
    hg_id = h_g(np.stack([descriptor(d.ee_pos) for d in identical]))
    hg_div = h_g(np.stack([descriptor(d.ee_pos) for d in diverse]))
    assert hg_div > hg_id


def test_hg_scale_invariance():
    demos = [make_demo(0.001, p1=tuple(RNG.uniform(0.2, 0.8, 3)), seed=i) for i in range(15)]
    D1 = np.stack([descriptor(d.ee_pos) for d in demos])
    D2 = np.stack([descriptor(d.ee_pos * 10.0) for d in demos])
    assert h_g(D1) == pytest.approx(h_g(D2), rel=1e-6)


def test_hg_loo_flags_outlier():
    base = [make_demo(0.0005, seed=i) for i in range(15)]
    outlier = make_demo(0.0005, p0=(0.9, 0.9, 0.9), p1=(0.1, 0.8, 0.9), seed=99)
    scores = GeometricEntropyLOO().score(base + [outlier])
    assert np.argmax(scores) == len(base), "outlier must have max LOO diversity contribution"


def test_resample_length():
    d = make_demo(0.0, T=333)
    assert descriptor(d.ee_pos).shape == (150,)


def test_episode_length_direction():
    s = EpisodeLength().score([make_demo(T=100), make_demo(T=200)])
    assert s[0] < s[1]


def test_select_api():
    demos = [make_demo(noise=0.001 * i, seed=i) for i in range(10)]
    keep = SALMetric("rinse").select(demos, retention=0.5)
    assert len(keep) == 5 and len(set(keep.tolist())) == 5
