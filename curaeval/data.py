"""Reading robomimic demonstrations and reading/writing selection masks.

A selection is stored the way robomimic expects it: as a ``mask/<key>`` dataset
of demonstration names inside the hdf5 file, which robomimic then trains on via
``config.train.hdf5_filter_key``.
"""
from __future__ import annotations

import h5py
import numpy as np

from curaeval.metrics.base import Demo

DT = 0.05  # robomimic datasets are recorded at 20 Hz
TIERS = ("better", "okay", "worse")


def _demo_keys(f):
    return sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))


def load_demos(path):
    """Load end-effector paths, in demo-index order.

    Bimanual tasks (transport) get both arms' paths concatenated along the
    feature axis; every metric in the roster is dimension-generic, so this needs
    no per-metric special casing.

    Returns (demos, names).
    """
    demos, names = [], []
    with h5py.File(path, "r") as f:
        keys = _demo_keys(f)
        arms = [a for a in ("robot0", "robot1") if f"obs/{a}_eef_pos" in f["data"][keys[0]]]
        for n in keys:
            g = f["data"][n]
            cat = lambda field: np.concatenate(  # noqa: E731
                [g[f"obs/{a}_{field}"][:] for a in arms], axis=1
            ).astype(np.float64)
            demos.append(
                Demo(
                    ee_pos=cat("eef_pos"),
                    ee_quat=cat("eef_quat"),
                    gripper=cat("gripper_qpos"),
                    dt=DT,
                )
            )
            names.append(n)
    return demos, names


def write_mask(f, key, selected_names):
    """Write (or overwrite) ``mask/<key>`` in an open, writable hdf5 file."""
    if "mask" not in f:
        f.create_group("mask")
    if key in f["mask"]:
        del f["mask"][key]
    f["mask"].create_dataset(key, data=np.array(selected_names, dtype="S"))


def read_mask(path, key):
    """Demonstration names stored under ``mask/<key>``."""
    with h5py.File(path, "r") as f:
        return [x.decode() for x in f["mask"][key][:]]


def tiers_of(path, names):
    """Operator tier (better / okay / worse) of each named demonstration.

    robomimic's multi-human (MH) datasets ship these tiers as masks.
    """
    with h5py.File(path, "r") as f:
        tier = {}
        for t in TIERS:
            for d in f["mask"][t][:]:
                tier[d.decode()] = t
    return np.array([tier.get(n, "okay") for n in names])
