"""Volatile inventory and the high-pressure regime (S3) for Results 4.4.

Four volatile-inventory treatments at fixed oxidation state (IW+4), dynamic
interior structure, across planet mass. The treatments span a volatile ladder:

    T1  fixed-absolute      (volatile-poor; S3_T1)
    T2  mass-scaled fiducial (the S1 headline runs)
    T3  hydrogen-enriched   (H ~25x the fiducial; S3_T3)
    T4  volatile-rich       (H and C enriched; S3_T4)

The message: the volatile inventory sets the atmosphere thickness, and a thick
atmosphere throttles the surface heat loss so that the most volatile-rich, most
massive interiors retain a deep magma ocean at radiative equilibrium rather than
crystallising. Incomplete crystallisation then leaves a smaller structural
contraction, tying 4.4 back to the contraction result of 4.2.

Three panels against planet mass:
    (a) total crystallisation contraction (from the molten peak)
    (b) crystallisation-floor melt fraction (how far solidification proceeds)
    (c) surface pressure at the floor (the volatile ladder), log scale

Reads the helpfiles by column name.

Usage
-----
    python plot_volatiles.py [--data DIR] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt

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

DEF_DATA = os.path.join(os.path.dirname(HERE), "data", "helpfiles")

# treatment -> {mass: run folder}
TREATMENTS = {
    "T1 fixed-absolute":   {3: "S3_T1_m3", 5: "S3_T1_m5", 10: "S3_T1_m10"},
    "T2 mass-scaled":      {1: "S1_m1_dyn_IW4", 3: "S1_m3_dyn_IW4",
                            5: "S1_m5_dyn_IW4", 10: "S1_m10_dyn_IW4"},
    "T3 hydrogen-rich":    {3: "S3_T3_m3", 5: "S3_T3_m5", 10: "S3_T3_m10"},
    "T4 volatile-rich":    {3: "S3_T4_m3", 5: "S3_T4_m5", 10: "S3_T4_m10"},
}
COLORS = {
    "T1 fixed-absolute": ix.STRATA["gold"],
    "T2 mass-scaled":    ix.NEUTRALS["ink"],
    "T3 hydrogen-rich":  ix.STRATA["cobalt"],
    "T4 volatile-rich":  ix.STRATA["magma"],
}


def _load(run, data, cols):
    hf = os.path.join(data, run, "runtime_helpfile.csv")
    with open(hf) as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        idx = {c: header.index(c) for c in cols}
        mx = max(idx.values())
        out = []
        for row in r:
            if not row or len(row) <= mx:
                continue
            try:
                out.append({c: float(row[idx[c]]) for c in cols})
            except ValueError:
                continue
    return out


def _metrics(run, data):
    rows = _load(run, data, ["R_int", "Phi_global", "P_surf"])
    if not rows:
        return None
    rmax = max(x["R_int"] for x in rows)
    e = rows[-1]
    return dict(contr=100 * (rmax - e["R_int"]) / rmax,
                phi=e["Phi_global"], psurf=e["P_surf"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()

    data = {}
    for t, runs in TREATMENTS.items():
        data[t] = {}
        for m, run in runs.items():
            mt = _metrics(run, args.data)
            if mt:
                data[t][m] = mt

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.7), sharex=True)
    ax_c, ax_p, ax_ps = axes

    for t, series in data.items():
        ms = sorted(series)
        c = COLORS[t]
        ax_c.plot(ms, [series[m]["contr"] for m in ms], "o-", color=c, ms=5, label=t)
        ax_p.plot(ms, [series[m]["phi"] for m in ms], "o-", color=c, ms=5, label=t)
        ax_ps.plot(ms, [series[m]["psurf"] for m in ms], "o-", color=c, ms=5, label=t)

    ax_c.set_ylabel("Contraction [%]")
    ax_c.set_ylim(2, 12)
    # Direction cue: crystallisation contraction is higher on this axis, the
    # opposite sense to the interior-radius panels of Figures 5 and 6, where a
    # contracting model moves down. Three short arrows in the clear lower-left
    # band mark the upward direction a model takes as it contracts.
    for xa in (1.3, 2.4, 3.5):
        ax_c.annotate("", xy=(xa, 6.3), xytext=(xa, 3.3),
                      arrowprops=dict(arrowstyle="-|>", color=ix.NEUTRALS["fog"],
                                      lw=1.1, shrinkA=0, shrinkB=0))
    ax_c.text(2.4, 6.7, "contraction", fontsize=6.6, style="italic",
              color=ix.NEUTRALS["fog"], ha="center", va="bottom")

    ax_p.set_ylabel(r"Floor melt fraction $\Phi$")
    ax_p.set_ylim(0, 0.9)

    ax_ps.set_xlabel(r"Planet mass [$M_\oplus$]")  # shared x-axis; label on the lowest panel
    ax_ps.set_ylabel("Surface pressure [bar]")
    ax_ps.set_yscale("log")
    ax_ps.set_ylim(1e2, 1e5)

    for ax in axes:
        ax.set_xlim(0.5, 10.8)
        ax.set_xticks([1, 3, 5, 10])
    # panel sublabels in the manuscript-wide style
    titles = ("Contraction", "Residual melt", "Surface pressure")
    # (a) contraction peaks at the top-left (all masses near 11 per cent at 1 M_E),
    # so its label goes top-right, clear of the curves; (b) and (c) stay top-left.
    label_kw = {0: dict(x=0.96, ha="right")}
    for i, (ax, lab, ttl) in enumerate(zip(axes, "abc", titles)):
        ix.panel_label(ax, f"({lab}) {ttl}", fontsize=8.5, **label_kw.get(i, {}))

    # one shared treatment legend above the panels, touching no data; two columns
    # so the four treatments fit the single-column width
    handles, labels_ = ax_c.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, frameon=False, fontsize=8, columnspacing=1.4,
               handletextpad=0.5)

    for ax in axes:
        ix.set_mono_ticks(ax)

    fig.tight_layout(pad=0.6, h_pad=0.8, rect=(0, 0, 1, 0.95))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "volatiles.pdf")
    fig.savefig(out)
    print(f"Wrote {out}")
    for t, series in data.items():
        for m in sorted(series):
            s = series[m]
            print(f"  {t:20s} m{m:<2d} contr={s['contr']:5.2f}% phi={s['phi']:.3f} Psurf={s['psurf']:.0f}bar")


if __name__ == "__main__":
    main()
