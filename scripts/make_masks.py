"""Turn demonstration scores into robomimic filter_key masks.

For each score and retention, keep the top-retention demonstrations and write
their names as ``mask/<score>_<retention>`` in the robomimic hdf5 (robomimic
trains on a filter_key subset). Also writes size-matched random controls
``mask/random_<retention>_s<seed>``. Adding masks is non-destructive: only
``mask/`` entries are created; ``data/`` is untouched. Run it on a copy of the
dataset that you will train on.

Needs only numpy, scipy and h5py (no torch, no GPU).

CUPID masks come from cupid_port/ and the greedy diversity masks from
make_faktual_greedy.py / make_geom_greedy.py.

Usage:
  python scripts/make_masks.py --dataset data_masked/square_mh.hdf5 \
      --retentions 0.25 0.5 0.75 --seeds 0 1 2
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
from curaeval.data import load_demos, write_mask  # noqa: E402
from curaeval.metrics import METRICS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--retentions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()

    demos, names = load_demos(a.dataset)
    n = len(demos)
    print(f"{a.dataset}: {n} demos")

    # selections per (score, retention)
    sel = {}  # (name, ret) -> set of indices
    for ret in a.retentions:
        for mname, metric in METRICS.items():
            sel[(mname, ret)] = set(metric.select(demos, ret).tolist())
        for seed in a.seeds:
            k = max(1, int(round(ret * n)))
            rng = np.random.default_rng(seed)
            sel[(f"random_s{seed}", ret)] = set(rng.choice(n, k, replace=False).tolist())

    with h5py.File(a.dataset, "a") as f:
        for ret in a.retentions:
            tag = int(round(ret * 100))
            for mname in METRICS:
                write_mask(f, f"{mname}_{tag}", [names[i] for i in sorted(sel[(mname, ret)])])
            for seed in a.seeds:
                write_mask(f, f"random_{tag}_s{seed}", [names[i] for i in sorted(sel[(f'random_s{seed}', ret)])])
        added = [k for k in f["mask"].keys()]
    print(f"masks written; mask/ now has {len(added)} keys")


if __name__ == "__main__":
    main()
