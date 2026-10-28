"""Write a run manifest for the Slurm array jobs: one line per run, ``task filter_key seed``.

The manifest is the grid tasks x filter keys x training seeds, where the
filter keys are 'none' (all data), '<score>_<retention>' for every score and
retention, and 'random_<retention>_s0', the size-matched random subset. The
paper trained a part of this grid (see the README); CUPID masks
('--scores cupid') exist only after cluster/cupid.slurm.

Usage:
  python cluster/gen_manifest.py --tasks square --out cluster/manifest_square.txt
  sbatch --array=0-<N-1>%10 --export=ALL,MANIFEST=cluster/manifest_square.txt cluster/train_dp.slurm
"""
import argparse

SCORES = ["psd", "psdvel", "sal", "hlen", "hjerk", "geomloo", "faktual", "faktual_greedy"]
RETENTIONS = [25, 50, 75]
SEEDS = [1, 2, 3]


def grid(tasks, scores, retentions, seeds):
    lines = [f"{t} none {s}" for t in tasks for s in seeds]
    for t in tasks:
        for r in retentions:
            for key in [f"{m}_{r}" for m in scores] + [f"random_{r}_s0"]:
                lines += [f"{t} {key} {s}" for s in seeds]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["can", "square"])
    p.add_argument("--scores", nargs="+", default=SCORES)
    p.add_argument("--retentions", type=int, nargs="+", default=RETENTIONS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--out", default="cluster/manifest.txt")
    a = p.parse_args()

    lines = grid(a.tasks, a.scores, a.retentions, a.seeds)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} runs to {a.out}  (sbatch --array=0-{len(lines) - 1}%10)")


if __name__ == "__main__":
    main()
