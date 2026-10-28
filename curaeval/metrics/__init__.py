"""Demonstration scores evaluated in the paper.

``METRICS`` is the exact roster that produced every selection mask; each key is
also the prefix of its masks (``psd_25``, ``hlen_50``, ...). ``PAPER_NAMES``
maps the keys to the names used in the paper.
"""
from .base import Demo, DemoMetric
from .faktual import FAKTUALMetric, faktual_entropy
from .geometric_entropy import GeometricEntropyLOO, descriptor, h_g
from .heuristics import EpisodeLength, MeanSquaredJerk, PathInefficiency
from .psd import PSDMetric
from .sal import SALMetric

METRICS = {
    "psd": PSDMetric("position"),
    "psdvel": PSDMetric("velocity"),
    "sal": SALMetric("rinse"),
    "geomloo": GeometricEntropyLOO(),
    "faktual": FAKTUALMetric(),  # arXiv:2603.11634 signature-kernel entropy (LOO)
    "hlen": EpisodeLength(),
    "hpath": PathInefficiency(),
    "hjerk": MeanSquaredJerk(),
}

PAPER_NAMES = {
    "psd": "PSD",
    "psdvel": "Velocity-PSD",
    "sal": "SAL",
    "geomloo": "HG-WT (LOO)",
    "geomloo_greedy": "HG-WT (greedy)",
    "faktual": "SigEnt (LOO)",
    "faktual_greedy": "SigEnt (greedy)",
    "hlen": "Length",
    "hpath": "Path inefficiency",
    "hjerk": "Jerk",
    "cupid": "CUPID",
    "random": "Random",
    "none": "All data",
}

__all__ = [
    "Demo",
    "DemoMetric",
    "PSDMetric",
    "SALMetric",
    "GeometricEntropyLOO",
    "descriptor",
    "h_g",
    "EpisodeLength",
    "PathInefficiency",
    "MeanSquaredJerk",
    "FAKTUALMetric",
    "faktual_entropy",
    "METRICS",
    "PAPER_NAMES",
]
