"""Assemble the structural-contraction time series for the S1 run set.

The S1 set is the headline experiment of the paper: planet mass {1, 3, 5, 10}
:math:`M_\\oplus` crossed with interior-structure treatment {dynamic, static}, all
else fixed (mass-scaled volatiles, IW+4, Sun at 1 AU). The dynamic runs re-solve
the interior structure as the mantle crystallises, so their surface radius
contracts with the melt-to-solid density increase; the static twins freeze the
structure at the initial state and isolate that contraction as the difference.

This script reads each run's ``runtime_helpfile.csv`` (tab-separated, despite the
extension) and writes a compact ``data/cache_shrinking.npz`` holding the
interior-radius, melt-fraction, magma-temperature, and rheological-front time
series that ``plot_shrinking.py`` replots. Run folders are matched by the pattern
``S1_m{mass}_{dyn|stat}_IW4`` under the runs directory; missing or still-running
folders are skipped with a note so the cache can be rebuilt incrementally.

Usage
-----
    python compute_shrinking.py [--runs-dir DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS = os.path.join(os.path.dirname(HERE), "data", "helpfiles")
DEFAULT_OUT = os.path.join(HERE, "data", "cache_shrinking.npz")

R_EARTH = 6.371e6      # m
M_EARTH = 5.972e24     # kg

# Helpfile columns pulled into the cache (clean names only; the helpfile also
# has slash-bearing ratio columns that are not needed here).
FIELDS = [
    "Time", "R_int", "R_obs", "R_core", "Phi_global", "T_magma", "T_surf",
    "T_cmb", "R_solvus", "RF_depth", "P_surf", "M_mantle_solid",
    "M_mantle_liquid", "gravity", "fO2_shift_IW_derived",
]

RUN_RE = re.compile(r"S1_m(?P<mass>[0-9]+(?:p[0-9]+)?)_(?P<struct>dyn|stat)")


def _read_helpfile(path):
    """Read a PROTEUS runtime helpfile into a name-keyed column dict.

    Parameters
    ----------
    path : str
        Path to ``runtime_helpfile.csv`` (tab-separated).

    Returns
    -------
    dict of str -> numpy.ndarray
        One float array per requested field in :data:`FIELDS` that is present in
        the header. Rows with a blank or non-numeric entry in a field are kept as
        NaN for that field so array lengths stay aligned across fields.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if len(rows) < 2:
        return {}
    header = rows[0]
    idx = {c: i for i, c in enumerate(header)}
    out = {}
    for field in FIELDS:
        if field not in idx:
            continue
        j = idx[field]
        vals = []
        for r in rows[1:]:
            if len(r) <= j or r[j] == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(r[j]))
                except ValueError:
                    vals.append(np.nan)
        out[field] = np.asarray(vals, dtype=float)
    return out


def _parse_run(name):
    """Return (mass_in_Mearth, 'dyn'|'stat') parsed from a run-folder name."""
    m = RUN_RE.search(name)
    if not m:
        return None, None
    mass = float(m.group("mass").replace("p", "."))
    return mass, m.group("struct")


def gather_real(runs_dir):
    """Collect S1 run time series from helpfiles under ``runs_dir``.

    Returns
    -------
    dict
        Cache dict with flat keys ``<run>__<field>`` plus ``<run>__mass`` and
        ``<run>__struct``, and a ``runs`` array listing the run names found.
    """
    cache = {}
    runs = []
    patt = os.path.join(runs_dir, "S1_m*_*")
    for folder in sorted(glob.glob(patt)):
        name = os.path.basename(folder)
        mass, struct = _parse_run(name)
        if mass is None:
            continue
        hp = os.path.join(folder, "runtime_helpfile.csv")
        if not os.path.isfile(hp):
            print(f"  skip {name}: no runtime_helpfile.csv yet")
            continue
        cols = _read_helpfile(hp)
        if "Time" not in cols or "R_int" not in cols:
            print(f"  skip {name}: helpfile missing Time/R_int")
            continue
        for field, arr in cols.items():
            cache[f"{name}__{field}"] = arr
        cache[f"{name}__mass"] = np.array(mass)
        cache[f"{name}__struct"] = np.array(struct)
        runs.append(name)
        n = len(cols["Time"])
        phi = cols.get("Phi_global")
        phitxt = f"  Phi {phi[0]:.2f}->{np.nanmin(phi):.3g}" if phi is not None else ""
        print(f"  + {name}: {n} rows{phitxt}")
    cache["runs"] = np.array(runs)
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"Reading S1 helpfiles under {args.runs_dir}")
    cache = gather_real(args.runs_dir)
    if len(cache.get("runs", [])) == 0:
        print("No S1 run helpfiles found; point --runs-dir at the archived runs.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **cache)
    print(f"Wrote {args.out} with {len(cache.get('runs', []))} runs.")


if __name__ == "__main__":
    main()
