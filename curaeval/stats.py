"""Statistics of the comparison protocol.

The unit of analysis is the training seed: rollouts of one trained policy are
not independent samples of what a selection does, so every comparison uses
per-seed success rates.

* cell means carry a Student-t 95% interval over training seeds;
* differences use two-sided Welch t tests and Welch 95% intervals;
* Holm's step-down correction is applied within a fixed family of comparisons.

Intervals are pointwise and unclipped; they are not simultaneous intervals.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy import stats


def summarize(values):
    """Mean over training seeds with a Student-t 95% interval.

    ``values`` maps training seed to success rate.
    """
    a = np.asarray(list(values.values()), dtype=float)
    n = len(a)
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    half = float(stats.t.ppf(.975, n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def welch(a, b):
    """Two-sided Welch comparison of the mean of ``a`` minus the mean of ``b``."""
    x = np.array(list(a.values()), dtype=float)
    y = np.array(list(b.values()), dtype=float)
    vx = float(np.var(x, ddof=1) / len(x))
    vy = float(np.var(y, ddof=1) / len(y))
    delta = float(x.mean() - y.mean())
    se = math.sqrt(vx + vy)
    if se == 0:
        raise ValueError("no between-seed variance; the Welch test is undefined")
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    t = delta / se
    p = float(2 * stats.t.sf(abs(t), df))
    half = float(stats.t.ppf(.975, df) * se)
    return {"delta": delta, "ci95_low": delta - half, "ci95_high": delta + half,
            "p_raw": p, "standard_error": se, "welch_df": df,
            "n_condition": len(x), "n_control": len(y)}


def holm(comparisons):
    """Holm step-down correction, in place.

    ``comparisons`` is a list of ``(name, comparison_dict)``; each dict gains
    ``p_holm`` and ``family_size``. Ties in raw p are broken by name.
    """
    ordered = sorted(comparisons, key=lambda item: (item[1]["p_raw"], item[0]))
    m = len(ordered)
    previous = 0.0
    for rank, (_name, comparison) in enumerate(ordered):
        previous = max(previous, min(1.0, (m - rank) * comparison["p_raw"]))
        comparison["p_holm"] = previous
        comparison["family_size"] = m


def fmt(x, signed=False):
    """Three decimals, rounded half-up from the unrounded value."""
    value = Decimal(str(round(x, 12))).quantize(Decimal(".001"), rounding=ROUND_HALF_UP)
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def p_text(p):
    return "<.001" if p < .001 else f"{p:.3f}".removeprefix("0")
