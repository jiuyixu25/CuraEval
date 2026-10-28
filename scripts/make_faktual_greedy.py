"""SigEnt (FAKTUAL) masks by forward-greedy entropy maximisation.

``faktual`` in curaeval.metrics ranks demonstrations by their leave-one-out
(LOO) contribution to the signature-kernel entropy. FAKTUAL's paper (Sec. VI,
App. E) selects greedily instead: each step adds the demonstration that
maximises the entropy of the selected set (the paper then adds a
determinant-maximising subset, which is omitted here). Writes
``faktual_greedy_<retention>``.

Deterministic (no seed): the greedy subset is fixed, and training seeds vary
only the policy training. CPU only.

Usage:
  python scripts/make_faktual_greedy.py --dataset data_masked/square_mh.hdf5 \
      --retentions 0.25 0.5 0.75
"""
import argparse
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
from curaeval.data import load_demos, write_mask  # noqa: E402
from curaeval.metrics.faktual import greedy_entropy_select  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--retentions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    a = p.parse_args()

    demos, names = load_demos(a.dataset)
    n = len(demos)
    print(f"{a.dataset}: {n} demos")

    with h5py.File(a.dataset, "a") as f:
        for ret in a.retentions:
            k = max(1, int(round(ret * n)))
            idx = greedy_entropy_select(demos, k)
            tag = int(round(ret * 100))
            key = f"faktual_greedy_{tag}"
            write_mask(f, key, [names[i] for i in sorted(idx.tolist())])
            print(f"  {key}: selected {len(idx)}/{n} (greedy entropy-max)")


if __name__ == "__main__":
    main()
