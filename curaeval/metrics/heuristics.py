"""Simple heuristic scores, used as baselines.

Directions, fixed in advance:
  episode length: shorter is better (hesitation and retries inflate length)
  path inefficiency (path length / straight-line distance): lower is better
  mean squared jerk: lower is better
"""
from __future__ import annotations

import numpy as np

from .base import Demo, DemoMetric


class EpisodeLength(DemoMetric):
    higher_is_better = False
    name = "heuristic[length]"

    def score(self, demos: list[Demo]) -> np.ndarray:
        return np.array([len(d.ee_pos) for d in demos], dtype=np.float64)


class PathInefficiency(DemoMetric):
    higher_is_better = False
    name = "heuristic[path_ineff]"

    def score(self, demos: list[Demo]) -> np.ndarray:
        out = np.empty(len(demos))
        for i, d in enumerate(demos):
            p = np.asarray(d.ee_pos, dtype=np.float64)
            path = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
            chord = np.linalg.norm(p[-1] - p[0])
            out[i] = path / (chord + 1e-9)
        return out


class MeanSquaredJerk(DemoMetric):
    higher_is_better = False
    name = "heuristic[jerk]"

    def score(self, demos: list[Demo]) -> np.ndarray:
        out = np.empty(len(demos))
        for i, d in enumerate(demos):
            p = np.asarray(d.ee_pos, dtype=np.float64)
            jerk = np.diff(p, n=3, axis=0) / d.dt**3
            out[i] = float(np.mean(np.sum(jerk**2, axis=1))) if len(jerk) else 0.0
        return out
