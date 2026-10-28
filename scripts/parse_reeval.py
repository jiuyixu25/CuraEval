"""Collect the evaluation logs of cluster/reeval.slurm into a run table.

Each log holds the evaluation of one final checkpoint (``last.pth``): the
header line ``name=<run> nroll=<n> horizon=<h> evalseed=<s>`` and robomimic's
``Average Rollout Stats``. ``--tsv`` writes one line per run,
``<run name> <success rate> <mean episode length>``, the input of
scripts/compare.py; the mean success of each condition is printed.

Usage:
  python scripts/parse_reeval.py 'logs/reeval_*.log' --tsv runs_square_dp.tsv
"""
import argparse
import glob
import json
import re
from collections import defaultdict

HEADER = re.compile(r"name=(\S+)\s+nroll=(\d+)\s+horizon=(\d+)\s+evalseed=(\d+)")
STATS = re.compile(r"Average Rollout Stats\s*(\{.*?\})", re.S)
SEED = re.compile(r"(.+)_seed\d+")


def parse_log(path):
    """``(run name, rollouts, rollout stats or None)``, or None without a header."""
    with open(path, errors="ignore") as f:
        text = f.read()
    header = HEADER.search(text)
    if not header:
        return None
    stats = None
    m = STATS.search(text)
    if m:
        try:
            stats = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return header.group(1), int(header.group(2)), stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log files or glob patterns")
    p.add_argument("--tsv", help="write the per-run table to this file")
    a = p.parse_args()

    paths = sorted({q for g in a.logs for q in glob.glob(g)})
    runs, missing = {}, []
    for parsed in filter(None, map(parse_log, paths)):
        name, nroll, stats = parsed
        if stats is None:
            missing.append(name)
            continue
        if name in runs:
            print(f"  note: {name} appears in several logs; keeping the last one")
        runs[name] = (nroll, stats)
    print(f"{len(paths)} logs: {len(runs)} runs with results, {len(missing)} without")
    if missing:
        print("  without results: " + ", ".join(sorted(missing)))

    if a.tsv:
        with open(a.tsv, "w") as f:
            for name in sorted(runs):
                stats = runs[name][1]
                f.write(f"{name}\t{stats['Success_Rate']}\t{stats['Horizon']}\n")
        print(f"wrote {len(runs)} runs to {a.tsv}")

    groups = defaultdict(list)
    for name, (nroll, stats) in runs.items():
        m = SEED.fullmatch(name)
        groups[m.group(1) if m else name].append((stats["Success_Rate"], nroll))
    print(f"\n{'condition':40s}{'seeds':>6s}{'episodes':>10s}{'success':>9s}")
    for cond in sorted(groups):
        rates = [r for r, _ in groups[cond]]
        episodes = sum(n for _, n in groups[cond])
        print(f"{cond:40s}{len(rates):6d}{episodes:10d}{sum(rates) / len(rates):9.3f}")


if __name__ == "__main__":
    main()
