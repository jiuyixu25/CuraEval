"""HG-WT masks by forward-greedy maximisation of H_G.

H_G (arXiv:2606.20871) scores a whole dataset. ``geomloo`` in curaeval.metrics
ranks single demonstrations by their leave-one-out (LOO) contribution, which
is our construction, not the paper's (see curaeval/metrics/geometric_entropy.py).
This script grows the subset greedily instead, adding at each step the
demonstration that maximises H_G of the selected set, the selection rule
FAKTUAL uses for its own entropy (make_faktual_greedy.py). Writes
``geomloo_greedy_<retention>``.

Deterministic (no seed; ties go to the lowest index). CPU only.

Usage:
  python scripts/make_geom_greedy.py --dataset data_masked/square_mh.hdf5 --retentions 0.25
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
from curaeval.data import load_demos, write_mask  # noqa: E402
from curaeval.metrics.geometric_entropy import descriptor, h_g  # noqa: E402


def greedy_hg_select(demos, k, eig_convention="cov"):
    """Forward-greedy H_G-maximizing subset of size k, mirroring
    curaeval.metrics.faktual.greedy_entropy_select.

    H_G needs >=2 descriptors to have a covariance, so the first two picks are
    seeded by the pair with the largest H_G; after that each step adds the demo
    that maximises H_G of the selected set.
    """
    D = np.stack([descriptor(d.ee_pos) for d in demos])
    n = len(demos)
    k = min(max(1, k), n)

    # seed: best pair (greedy from a single point is undefined for a covariance)
    best_pair, best_h = (0, 1), -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            h = h_g(D[[i, j]], eig_convention)
            if h > best_h:
                best_h, best_pair = h, (i, j)
    selected = list(best_pair)
    remaining = [i for i in range(n) if i not in selected]

    while len(selected) < k:
        best_j, best_h = remaining[0], -np.inf
        for j in remaining:
            h = h_g(D[selected + [j]], eig_convention)
            if h > best_h:
                best_h, best_j = h, j
        selected.append(best_j)
        remaining.remove(best_j)
    return np.array(selected[:k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--retentions", type=float, nargs="+", default=[0.25])
    a = p.parse_args()

    demos, names = load_demos(a.dataset)
    n = len(demos)
    print(f"{a.dataset}: {n} demos")

    with h5py.File(a.dataset, "a") as f:
        for ret in a.retentions:
            k = max(1, int(round(ret * n)))
            idx = greedy_hg_select(demos, k)
            tag = int(round(ret * 100))
            key = f"geomloo_greedy_{tag}"
            write_mask(f, key, [names[i] for i in sorted(idx.tolist())])
            print(f"  {key}: selected {len(idx)}/{n} (greedy H_G-max)")


if __name__ == "__main__":
    main()
