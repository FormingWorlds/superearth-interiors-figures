"""Structural robustness (S5/S6) for results 4.5.

The crystallisation contraction of the fiducial 5 Mearth reference is varied
one axis at a time and the total contraction (measured from the molten peak) is
compared across the set. Two panels:

(a) total contraction against core-mass fraction: the contraction is driven by
    the silicate shell, so a more iron-rich planet, with less mantle to
    crystallise, contracts less.
(b) total contraction for the host-star / irradiation and initial-superheat
    variations, against the fiducial reference band: the host star and the
    instellation leave the contraction unchanged (the environment sets the floor
    surface temperature, annotated, not the contraction), while the apparent
    superheat spread reflects the molten-peak reference, the solidified radius
    being superheat-independent. Tidal heating holds the interior molten and is
    noted separately.

Values are read from robustness_summary.csv (the near-final lower bounds quoted
in Results 4.5, extracted from the S5/S6 runs).

Usage
-----
    python plot_robustness.py [--csv CSV] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE), "style")
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use("white", font="mono")
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 6.8, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "lines.linewidth": 1.5,
})

DEF_CSV = os.path.join(HERE, "data", "robustness_summary.csv")
FIDUCIAL = 10.2  # fiducial 5 Mearth, Sun at 1 AU, near-final lower bound [per cent]


def _read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(filter(lambda ln: not ln.startswith("#"), f)):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEF_CSV)
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()

    rows = _read(args.csv)
    cmf = [r for r in rows if r["group"] == "cmf"]
    env = [r for r in rows if r["group"] in ("env", "init")]

    # ax_b (host-star and superheat) is drawn into the top panel and ax_a
    # (core-mass fraction) into the bottom, so the panel order (a) then (b)
    # matches the order Section 4.4 discusses them.
    fig, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True)

    # core-mass fraction against the shared contraction axis (axes swapped so
    # both panels read on one horizontal contraction scale).
    cmf_frac = [float(r["x"]) for r in cmf]
    cmf_con = [float(r["contraction_pct"]) for r in cmf]
    ax_a.plot(cmf_con, cmf_frac, "o-", color=ix.STRATA["magma"], ms=6, lw=1.6, zorder=3)
    for con, frac in zip(cmf_con, cmf_frac):
        ax_a.annotate(f"{con:.1f}%", (con, frac), textcoords="offset points",
                      xytext=(6, 3), ha="left", fontsize=7.2, color=ix.NEUTRALS["ink"])
    ax_a.axvline(FIDUCIAL, color=ix.NEUTRALS["ink"], lw=0.8, ls="--", zorder=0)
    ax_a.text(FIDUCIAL, 0.63, "fiducial ", fontsize=6.4, va="center", ha="right",
              rotation=90, color=ix.NEUTRALS["ink"])
    EARTH_CMF = 0.325  # Earth-like core-mass fraction
    ax_a.axhline(EARTH_CMF, color=ix.NEUTRALS["ink"], lw=0.8, ls=":", zorder=0)
    ax_a.text(7.3, EARTH_CMF, " Earth", fontsize=6.4, va="bottom", ha="left",
              color=ix.NEUTRALS["ink"])
    ax_a.set_ylabel("Core-mass fraction")
    ax_a.set_ylim(0.15, 0.77)
    ax_a.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    # (b) environment and initial-state variations, on the same contraction axis
    labels = [r["label"] for r in env]
    vals = [float(r["contraction_pct"]) for r in env]
    notes = [r.get("note", "") for r in env]
    ypos = np.arange(len(env))[::-1]
    cols = [ix.STRATA["cobalt"] if r["group"] == "env" else ix.STRATA["amber"] for r in env]
    ax_b.axvline(FIDUCIAL, color=ix.NEUTRALS["ink"], lw=0.8, ls="--", zorder=0)
    ax_b.scatter(vals, ypos, c=cols, s=42, zorder=3)
    # notes sit to the left of their markers, on a masking patch so they read cleanly
    # over the gridlines and the fiducial line
    _mask = dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9, edgecolor="none")
    for v, y, n in zip(vals, ypos, notes):
        if not n:
            continue
        txt = n.replace("Tsurf ", r"$T_\mathrm{surf}$ = ") if n.startswith("Tsurf") else n
        ax_b.annotate(txt, (v, y), textcoords="offset points", xytext=(-8, -1),
                      ha="right", va="center", fontsize=6.2, zorder=5,
                      color=ix.NEUTRALS["graphite"], bbox=_mask)
    ax_b.set_yticks(ypos)
    # incline the category labels so they take less horizontal space, freeing the
    # left margin so the plot area fills the column width
    ax_b.set_yticklabels(labels, fontsize=6.6, rotation=28, ha="right",
                         rotation_mode="anchor")
    ax_a.set_xlabel("Contraction [%]")
    ax_b.set_ylim(-0.6, len(env) + 0.3)
    ax_b.set_xlim(7.2, 11.7)          # shared axis; notes sit to the left of the markers
    ax_b.set_xticks([8, 9, 10, 11])

    titles = ("Host star and superheat", "Core-mass fraction")
    pos = (dict(x=0.04, ha="left"), dict(x=0.96, ha="right"))
    for ax, lab, ttl, p in zip((ax_b, ax_a), "ab", titles, pos):
        ix.panel_label(ax, f"({lab}) {ttl}", fontsize=9, **p)

    for ax in (ax_a, ax_b):
        ix.set_mono_ticks(ax)

    fig.tight_layout(pad=0.6, h_pad=1.0)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "robustness.pdf")
    fig.savefig(out)
    print(f"Wrote {out}")
    print(f"  CMF: {dict(zip([r['label'] for r in cmf], cmf_con))}")
    print(f"  env/init: {dict(zip(labels, vals))}; fiducial {FIDUCIAL}%")


if __name__ == "__main__":
    main()
