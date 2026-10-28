#!/usr/bin/env python
"""Turn cupid_influence.py score json into robomimic filter_key masks.

Writes `mask/<prefix>_<tag>` (e.g. mask/cupid_25, cupid_50, cupid_75) into the
hdf5, following the convention of scripts/make_masks.py: top-`retention` demos
by score (higher = keep), k = max(1, round(ret * n)), names stored as an
S-dtype dataset sorted by demo index.  Non-destructive (touches only mask/),
but run it on the masked copy used for training
(e.g. $CURAEVAL_WORK/data_masked/square_mh.hdf5).

Pure numpy/h5py; no GPU needed.

Example:
  python cupid_port/make_cupid_masks.py \
    --scores  $CURAEVAL_WORK/cupid/scores_square_rankmerged.json \
    --dataset $CURAEVAL_WORK/data_masked/square_mh.hdf5 \
    --retentions 0.25 0.5 0.75
Then train:  scripts/run_dp.py --filter_key cupid_50 ...
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import h5py
import numpy as np


def write_mask(f, key, selected_names):
    """Same as curaeval.data.write_mask (kept here so the port runs on its own)."""
    if "mask" not in f:
        f.create_group("mask")
    if key in f["mask"]:
        del f["mask"][key]
    f["mask"].create_dataset(key, data=np.array(selected_names, dtype="S"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, help="json from cupid_influence.py")
    p.add_argument("--dataset", required=True,
                   help="hdf5 to write mask/ keys into (the masked copy)")
    p.add_argument("--retentions", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75])
    p.add_argument("--exp_key", default=None,
                   help="score variant (default: the json's canonical_exp_key "
                        "= sum_of_sum-net-all)")
    p.add_argument("--prefix", default="cupid",
                   help="mask key prefix -> mask/<prefix>_<tag>")
    a = p.parse_args()

    with open(a.scores) as fh:
        blob = json.load(fh)
    exp_key = a.exp_key or blob["canonical_exp_key"]
    if exp_key not in blob["scores"]:
        raise KeyError(f"exp_key {exp_key!r} not in scores json "
                       f"(have {sorted(blob['scores'])[:6]}...)")
    names = list(blob["demos"])
    scores = np.asarray(blob["scores"][exp_key], dtype=np.float64)
    assert len(names) == len(scores)
    n = len(names)

    with h5py.File(a.dataset, "a") as f:
        h5_names = set(f["data"].keys())
        missing = [x for x in names if x not in h5_names]
        if missing:
            raise KeyError(
                f"{len(missing)} scored demos not present in {a.dataset} "
                f"(e.g. {missing[:5]}); wrong dataset for these scores?")
        if len(h5_names) != n:
            print(f"[make_cupid_masks] note: dataset has {len(h5_names)} demos, "
                  f"scores cover {n} (scored subset will be masked)")

        # rank: higher influence = keep (CUPID higher_is_better).
        order = np.argsort(-scores, kind="stable")
        written = []
        for ret in a.retentions:
            tag = int(round(ret * 100))
            k = max(1, int(round(ret * n)))          # scripts/make_masks.py rule
            sel_idx = sorted(order[:k].tolist())     # write in demo-index order
            key = f"{a.prefix}_{tag}"
            write_mask(f, key, [names[i] for i in sel_idx])
            cutoff = scores[order[k - 1]]
            written.append(key)
            print(f"[make_cupid_masks] mask/{key}: kept {k}/{n} "
                  f"(score cutoff {cutoff:.6g}, exp_key {exp_key})")
        total = len(f["mask"].keys())
    print(f"[make_cupid_masks] done; mask/ now has {total} keys "
          f"(+{written})")


if __name__ == "__main__":
    main()
