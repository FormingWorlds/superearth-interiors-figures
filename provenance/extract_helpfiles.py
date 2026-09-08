"""Reduce PROTEUS runtime helpfiles to the columns the figure scripts read.

The figure scripts for the contraction grid, the redox sweep and the volatile
inventory read ``runtime_helpfile.csv`` of individual PROTEUS runs. The full
helpfiles carry more than 300 columns per time step; this script copies only the
columns those scripts use, for the runs they use, into ``data/helpfiles/<run>/``
at the repository root, together with each run's configuration file
``init_coupler.toml`` and its termination status ``status`` (as ``status.txt``).
Every row is kept and every value is copied as the original text, so the
reduced files reproduce the figures exactly.

Usage
-----
    python provenance/extract_helpfiles.py --grid DIR

``DIR`` holds one folder per run with a tab-separated ``runtime_helpfile.csv``
(the archived PROTEUS output of the paper grid).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "helpfiles")

# Columns read by fig06_contraction_grid/compute_shrinking.py.
COLS_S1 = [
    "Time", "R_int", "R_obs", "R_core", "Phi_global", "T_magma", "T_surf",
    "T_cmb", "R_solvus", "RF_depth", "P_surf", "M_mantle_solid",
    "M_mantle_liquid", "gravity", "fO2_shift_IW_derived",
]
# Columns read by fig07_redox/plot_redox.py from the fixed-fugacity runs.
SPECIES = ["H2O", "H2", "CO2", "CO", "O2", "CH4", "N2", "S2", "SO2", "H2S"]
COLS_S2_FIXED = ["Phi_global", "P_surf"] + [f"{s}_vmr" for s in SPECIES] + [
    "H2O_kg_total", "CO2_kg_total", "CO_kg_total", "O2_kg_total",
]
# Columns read by fig07_redox/plot_redox.py from the oxygen-conserving twins.
COLS_S2_OXAUTH = ["Phi_global", "fO2_shift_IW_derived"]
# Columns read by fig08_volatiles/plot_volatiles.py.
COLS_VOL = ["R_int", "Phi_global", "P_surf"]

IW_TOKENS = ["m6", "m5", "m4", "m3", "m2", "m1", "0", "p1", "p2", "p3", "p4", "p5"]


def run_specs():
    """Return {run name: sorted column list} for every run the figures use."""
    spec = {}
    for m in (1, 3, 5, 10):
        spec[f"S1_m{m}_dyn_IW4"] = set(COLS_S1) | set(COLS_VOL)
        spec[f"S1_m{m}_stat_IW4"] = set(COLS_S1)
    for tok in IW_TOKENS:
        spec[f"S2_m5_IW{tok}_fixed"] = set(COLS_S2_FIXED)
        spec[f"S2_m5_IW{tok}_oxauth"] = set(COLS_S2_OXAUTH)
    for t in ("T1", "T3", "T4"):
        for m in (3, 5, 10):
            spec[f"S3_{t}_m{m}"] = set(COLS_VOL)
    return {k: sorted(v) for k, v in spec.items()}


def extract(src, dst, cols):
    """Copy the named columns of a tab-separated helpfile, all rows, text intact."""
    if os.path.exists(dst) and os.path.samefile(src, dst):
        raise ValueError(f"{src}: source and destination are the same file")
    with open(src, newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"{src}: empty helpfile")
    header = rows[0]
    missing = [c for c in cols if c not in header]
    if missing:
        raise KeyError(f"{src}: missing columns {missing}")
    keep = [c for c in header if c in cols]  # original column order
    idx = [header.index(c) for c in keep]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(keep)
        for r in rows[1:]:
            w.writerow([r[j] if j < len(r) else "" for j in idx])
    return len(rows) - 1, len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", required=True, help="directory of archived PROTEUS runs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not os.path.isdir(args.grid):
        raise SystemExit(f"{args.grid} is not a directory")
    for run, cols in run_specs().items():
        src = os.path.join(args.grid, run, "runtime_helpfile.csv")
        dst = os.path.join(args.out, run, "runtime_helpfile.csv")
        if not os.path.isfile(src):
            raise SystemExit(f"{src} not found; the figures need all {len(run_specs())} runs")
        n, k = extract(src, dst, cols)
        shutil.copyfile(os.path.join(args.grid, run, "init_coupler.toml"),
                        os.path.join(args.out, run, "init_coupler.toml"))
        shutil.copyfile(os.path.join(args.grid, run, "status"),
                        os.path.join(args.out, run, "status.txt"))
        print(f"{run:22s} {n:6d} rows  {k:2d} columns")


if __name__ == "__main__":
    main()
