"""Rank-average CUPID scores across scoring checkpoints -> one merged score json.

CUPID scores one checkpoint at a time. One mask set, shared by all training
seeds like the masks of every other score, is built by averaging the per-demo
ranks of the canonical variant across the scoring checkpoints (three in the
paper). The output has the schema make_cupid_masks.py reads; pointing
make_cupid_masks.py at a single checkpoint's json gives that checkpoint's masks.

Usage: rank_merge_cupid.py --scores a.json b.json c.json --out merged.json
"""
import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    blobs = [json.load(open(s)) for s in a.scores]
    key = blobs[0]["canonical_exp_key"]
    demos = blobs[0]["demos"]
    for b in blobs[1:]:
        assert b["demos"] == demos, "seed jsons cover different demo lists"
        assert b["canonical_exp_key"] == key
    ranks = []
    for b in blobs:
        s = np.asarray(b["scores"][key], dtype=np.float64)
        order = np.argsort(np.argsort(s))  # rank 0..n-1, higher score = higher rank
        ranks.append(order.astype(np.float64))
    merged = np.mean(ranks, axis=0)

    out = {
        "schema": "cupid_port.v1+rankmerge",
        "canonical_exp_key": key,
        "demos": demos,
        "scores": {key: merged.tolist()},
        "config": {"merged_from": a.scores, "rule": "mean per-demo rank of "
                   f"{key} across {len(blobs)} scoring seeds; higher = keep"},
    }
    open(a.out, "w").write(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {len(demos)} demos, key={key}, "
          f"seed-rank Spearman pairwise=" +
          ",".join(f"{np.corrcoef(ranks[i], ranks[j])[0,1]:.3f}"
                   for i in range(len(ranks)) for j in range(i + 1, len(ranks))))


if __name__ == "__main__":
    main()
