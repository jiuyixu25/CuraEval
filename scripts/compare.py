"""Compare selections with the size-matched random subset and with all data.

Input: run tables written by ``scripts/parse_reeval.py --tsv``, one line per
trained policy, ``<run name> <success rate> [...]``. Run names are
``<policy>_<task>_<filter key>_seed<k>`` with policy ``dp`` or ``bc`` and
filter key ``none`` (all data), ``<score>_<retention>`` or
``random_<retention>_s<subset seed>``.

For each policy and task, every selection is compared with the random subset
of the same size (subset seed ``--random-subset``), and every subset, that
random subset included, with all data. Comparisons are two-sided Welch tests
over training seeds, and Holm's correction is applied within each of the two
families. Random subsets with other subset seeds are compared with all data
only and stay outside the families (raw p values).

Usage:
  python scripts/compare.py runs_square_dp.tsv [more.tsv ...] [--out statistics.tsv]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root
from curaeval.stats import fmt, holm, p_text, summarize, welch  # noqa: E402

RUN = re.compile(r"(dp|bc)_(can|square|transport)_(.+)_seed(\d+)")
RANDOM = re.compile(r"random_(\d+)_s(\d+)")
SUBSET = re.compile(r"(.+)_(\d+)")
FIELDS = ["n_control", "delta", "ci95_low", "ci95_high", "p_raw", "p_holm",
          "family_size", "welch_df"]


def read_runs(paths):
    """``{(policy, task): {filter_key: {training_seed: success}}}``."""
    runs = {}
    for path in paths:
        for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            m = RUN.fullmatch(fields[0])
            if not m:
                raise ValueError(f"{path}:{lineno}: unrecognised run name {fields[0]!r}")
            policy, task, key, seed = m.groups()
            rate = float(fields[1])
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{path}:{lineno}: success {rate} outside [0, 1]")
            per_seed = runs.setdefault((policy, task), {}).setdefault(key, {})
            if int(seed) in per_seed:
                raise ValueError(f"{path}:{lineno}: duplicate run {fields[0]}")
            per_seed[int(seed)] = rate
    return runs


def describe(key):
    """Filter key -> (score, retention in percent, random subset seed or None)."""
    if key == "none":
        return "none", 100, None
    m = RANDOM.fullmatch(key)
    if m:
        return "random", int(m.group(1)), int(m.group(2))
    m = SUBSET.fullmatch(key)
    if not m:
        raise ValueError(f"unrecognised filter key {key!r}")
    return m.group(1), int(m.group(2)), None


def compare(values, random_subset=0):
    """Cells of one policy and task with both comparisons and Holm p values."""
    if "none" not in values:
        raise ValueError("no all-data runs (filter key 'none')")
    cells = {}
    for key in values:
        score, retention, subset = describe(key)
        cell = {"key": key, "score": score, "retention": retention,
                "family": "primary", **summarize(values[key])}
        if key != "none":
            cell["vs_all"] = welch(values[key], values["none"])
        if subset is not None and subset != random_subset:
            cell["family"] = "extra random subset"
        elif score not in ("none", "random"):
            control = f"random_{retention}_s{random_subset}"
            if control in values:
                cell["vs_random"] = welch(values[key], values[control])
            else:
                print(f"  note: no {control} runs; {key} is compared with all data only")
        cells[key] = cell
    primary = {k: c for k, c in cells.items() if c["family"] == "primary"}
    for family in ("vs_random", "vs_all"):
        holm([(k, c[family]) for k, c in primary.items() if family in c])
    return cells


def order(cell):
    return (cell["retention"] != 100, cell["retention"], cell["score"] == "random", cell["key"])


def print_cells(policy, task, cells):
    sizes = {f: sum(f in c and "p_holm" in c[f] for c in cells.values())
             for f in ("vs_random", "vs_all")}
    print(f"\n{policy} {task}: {len(cells)} cells; Holm families: "
          f"{sizes['vs_random']} vs random, {sizes['vs_all']} vs all data")
    print(f"{'filter key':24s}{'n':>3s}{'success':>9s}   {'vs random':>15s}   {'vs all data':>15s}")
    for c in sorted(cells.values(), key=order):
        cols = []
        for family in ("vs_random", "vs_all"):
            d = c.get(family)
            p = d and d.get("p_holm", d["p_raw"])
            cols.append(f"{fmt(d['delta'], True)} {p_text(p):>6s}" if d else "")
        print(f"{c['key']:24s}{c['n']:3d}{fmt(c['mean']):>9s}   {cols[0]:>15s}   {cols[1]:>15s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run tables from scripts/parse_reeval.py --tsv")
    p.add_argument("--random-subset", type=int, default=0,
                   help="subset seed of the random control (default 0)")
    p.add_argument("--out", help="write every cell and comparison to this TSV")
    a = p.parse_args()

    records = []
    for (policy, task), values in sorted(read_runs(a.runs).items()):
        cells = compare(values, a.random_subset)
        print_cells(policy, task, cells)
        for c in sorted(cells.values(), key=order):
            r = {"policy": policy, "task": task, "filter_key": c["key"], "score": c["score"],
                 "retention": c["retention"], "family": c["family"], "n": c["n"],
                 "success": c["mean"], "success_sd": c["sd"],
                 "success_ci95_low": c["ci95_low"], "success_ci95_high": c["ci95_high"]}
            for family in ("vs_random", "vs_all"):
                d = c.get(family) or {}
                r.update({f"{family}_{f}": d.get(f, "") for f in FIELDS})
            records.append(r)
    if a.out:
        with open(a.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {len(records)} rows to {a.out}")


if __name__ == "__main__":
    main()
