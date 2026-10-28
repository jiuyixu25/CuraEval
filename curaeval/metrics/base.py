"""Common interface for demonstration-quality metrics.

Every metric consumes a list of Demo and returns one scalar per demo.
`higher_is_better` fixes the filtering direction; every direction was fixed
before any policy was trained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Demo:
    """Trajectory container shared by all scores.

    ee_pos: (T, 3k) end-effector positions in meters, for k arms.
    dt: control period in seconds (robomimic: 0.05 = 20 Hz).
    ee_quat: (T, 4k) optional end-effector orientations (quaternions).
    gripper: (T, 2k) optional gripper joint positions.
    """

    ee_pos: np.ndarray
    dt: float
    ee_quat: np.ndarray | None = None
    gripper: np.ndarray | None = None


class DemoMetric:
    name: str = "base"
    higher_is_better: bool = True

    def score(self, demos: list[Demo]) -> np.ndarray:
        """Return shape-(M,) per-demo scores. Deterministic, no RNG."""
        raise NotImplementedError

    def rank(self, demos: list[Demo]) -> np.ndarray:
        """Indices sorted best-first under the metric's own direction."""
        s = self.score(demos)
        order = np.argsort(-s if self.higher_is_better else s, kind="stable")
        return order

    def select(self, demos: list[Demo], retention: float) -> np.ndarray:
        """Indices of the retained top `retention` fraction (best-first)."""
        k = max(1, int(round(retention * len(demos))))
        return self.rank(demos)[:k]
