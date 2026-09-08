"""Redox axis (S2) for results 4.3: oxidation state and the free-oxygen budget.

The 5 Mearth reference planet is run at a sweep of imposed iron-wuestite (IW)
offsets in two redox treatments on otherwise identical setups:
  - fixed-fugacity: the surface oxygen fugacity is buffered at the imposed IW
    offset throughout the evolution (oxidation state imposed as a boundary
    condition);
  - oxygen-authoritative: the total oxygen content of each fixed-fugacity twin
    is read out and conserved as an elemental inventory, and the surface
    fugacity (a derived IW offset) evolves freely.

The oxidising extreme IW+6 is excluded from both treatments: the fixed-fugacity
mode draws an unbounded O2 atmosphere there (mass non-conservation), so the
physical sweep spans IW-6..+5.

Two panels:
(a) atmospheric speciation (volume mixing ratio of the main C-H-O-S species)
    against imposed IW, at a common crystallisation snapshot (Phi = 0.40, the
    rheological front of Section 4.1), isolating the redox response from the
    differing crystallisation states of the runs: the reduced (CO, H2, S2) to
    oxidised (CO2, O2, SO2) transition, with sulfur flipping from S2 at the
    reducing end to SO2-dominated at the oxidising end.
(b) on the left axis, the derived IW offset of the oxygen-authoritative twins
    against the imposed IW of the fixed-fugacity twins, with the 1:1 line: at
    equal conserved oxygen the two treatments recover the same oxidation state,
    validating their equivalence; closure loosens at both extremes. On the
    right (log) axis, the free-oxygen budget (conserved oxygen inventory)
    against imposed IW: monotonic and steepening sharply toward the oxidising
    end, where the fixed-fugacity treatment ultimately breaks down (IW+6,
    excluded).

The derived IW offset and the oxygen budget are read at the initial outgassing
equilibrium (t=0), where they are written; the speciation is the interpolated
state at the common melt fraction.

Usage
-----
    python plot_redox.py [--s2dir DIR] [--outdir DIR] [--phi PHI]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE), "style")
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use("white", font="mono")
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 6.8, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.borderpad": 0.3, "legend.labelspacing": 0.25, "legend.framealpha": 0.9,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "lines.linewidth": 1.5,
    "axes.linewidth": 0.9, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

DEF_S2 = os.path.join(os.path.dirname(HERE), "data", "helpfiles")
# The S2 helpfiles supply the reducing half of the speciation panel and the round-trip
# and oxygen-budget panels. Their SO2 mixing ratio is floored at 1e-30, so the oxidising
# half of the speciation panel (IW0..+5, where sulfur speciation matters) reads the
# window file below, a separate set of runs of the same configurations with sulfur
# speciation resolved; the two agree at the IW0 seam. Both are the 5 Mearth S2 sweep at
# the common Phi=0.40 snapshot.
DEF_RERUN = os.path.join(HERE, "data", "phi040_window.csv")
# imposed IW offsets and run-name tokens; IW+6 excluded (mass-conservation breakdown)
IW_TOKENS = [("m6", -6), ("m5", -5), ("m4", -4), ("m3", -3), ("m2", -2), ("m1", -1),
             ("0", 0), ("p1", 1), ("p2", 2), ("p3", 3), ("p4", 4), ("p5", 5)]
# main C-H-N-O-S atmospheric species and the oxygen mass fraction of each O-bearing
# C-H-O volatile (used for the conserved-oxygen budget; mantle oxides and sulfur
# species excluded from the budget). N2 and the sulfur species enter the speciation
# panel but not the free-oxygen budget.
SPECIES = ["H2O", "H2", "CO2", "CO", "O2", "CH4", "N2", "S2", "SO2", "H2S"]
# math-formatted species labels for the legend (subscripted stoichiometry)
SPECIES_MATH = {"H2O": r"H$_2$O", "H2": r"H$_2$", "CO2": r"CO$_2$", "CO": r"CO",
                "O2": r"O$_2$", "CH4": r"CH$_4$", "N2": r"N$_2$", "S2": r"S$_2$",
                "SO2": r"SO$_2$", "H2S": r"H$_2$S"}
O_MASSFRAC = {"H2O": 0.88808, "CO2": 0.72708, "CO": 0.57119, "O2": 1.0}
PHI_SNAP = 0.40  # common crystallisation snapshot (rheological front)


def _load(path):
    """Return (header, data_rows) of a tab-separated helpfile."""
    rows = list(csv.reader(open(path), delimiter="\t"))
    return rows[0], rows[1:]


def _interp_at_phi(path, phi_target, cols):
    """Interpolate helpfile columns at a target global melt fraction.

    Phi_global decreases monotonically with time; the requested columns are
    linearly interpolated against it. Returns a name->float dict, or None if
    phi_target lies outside the run's reached range.
    """
    hdr, data = _load(path)
    jp = hdr.index("Phi_global")
    idx = {c: hdr.index(c) for c in cols if c in hdr}
    phis, series = [], {c: [] for c in cols}
    for r in data:
        try:
            ph = float(r[jp])
        except (ValueError, IndexError):
            continue
        phis.append(ph)
        for c in cols:
            try:
                series[c].append(float(r[idx[c]]))
            except (ValueError, IndexError, KeyError):
                series[c].append(np.nan)
    phis = np.array(phis)
    if phis.size < 2 or phi_target < phis.min() or phi_target > phis.max():
        return None
    order = np.argsort(phis)  # ascending Phi for np.interp
    xs = phis[order]
    return {c: float(np.interp(phi_target, xs, np.array(series[c])[order])) for c in cols}


def _load_rerun(path):
    """Interpolate the staged corrected re-run window at Phi=0.40.

    Returns {config_name: {column: value}} for every S2_m5_IW*_{fixed,oxauth}
    config in the window CSV, linearly interpolated against Phi_global.
    """
    if not os.path.isfile(path):
        return {}
    rows = list(csv.reader(open(path)))
    hdr = rows[0]
    ci = {c: i for i, c in enumerate(hdr)}
    by = {}
    for r in rows[1:]:
        by.setdefault(r[0], []).append(r)
    out = {}
    for name, rs in by.items():
        phi = np.array([float(x[ci["Phi_global"]]) for x in rs])
        order = np.argsort(phi)
        d = {}
        for c in hdr[2:]:
            v = np.array([float(x[ci[c]]) if x[ci[c]] not in ("", "nan") else np.nan
                          for x in rs])[order]
            m = ~np.isnan(v)
            d[c] = float(np.interp(PHI_SNAP, phi[order][m], v[m])) if m.sum() >= 2 else np.nan
        out[name] = d
    return out


def _oxygen_budget(path):
    """Conserved oxygen inventory (kg) from the t=0 outgassing equilibrium.

    Sum of O-bearing volatile species totals weighted by their oxygen mass
    fraction (mantle oxides excluded); equals the budget the oxygen-authoritative
    twin conserves.
    """
    hdr, data = _load(path)
    r = data[0]
    tot = 0.0
    for sp, mf in O_MASSFRAC.items():
        c = f"{sp}_kg_total"
        if c in hdr:
            try:
                tot += float(r[hdr.index(c)]) * mf
            except (ValueError, IndexError):
                pass
    return tot


def _derived_iw(path):
    """Derived IW offset of an oxygen-authoritative run at the t=0 equilibrium.

    Panel (b) is the round-trip test at the *initial* outgassing equilibrium, so
    read the first parseable value. The column is written at every step (the derived fugacity climbs as the mantle crystallises
    and outgasses), so reading the last row would return the evolved endpoint,
    not the initial equilibrium; the first row is the t=0 equilibrium value.
    """
    hdr, data = _load(path)
    if "fO2_shift_IW_derived" not in hdr:
        return np.nan
    j = hdr.index("fO2_shift_IW_derived")
    for r in data:
        try:
            return float(r[j])
        except (ValueError, IndexError):
            continue
    return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2dir", default=DEF_S2)
    ap.add_argument("--rerun", default=DEF_RERUN)
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--phi", type=float, default=PHI_SNAP)
    args = ap.parse_args()

    spec_cols = [f"{s}_vmr" for s in SPECIES] + ["P_surf"]
    rerun = _load_rerun(args.rerun)  # corrected oxidising sweep (IW0..+5)
    iw, vmr, psurf, obud = [], {s: [] for s in SPECIES}, [], []
    oxa_iw, oxa_der = [], []
    for tok, val in IW_TOKENS:
        fx = os.path.join(args.s2dir, f"S2_m5_IW{tok}_fixed", "runtime_helpfile.csv")
        # Speciation + surface pressure: oxidising half (IW>=0) from the window file,
        # reducing half from the S2 helpfiles. The conserved-oxygen budget is a t=0
        # equilibrium quantity that differs by under one per cent between the two
        # sets, so it is read from the S2 helpfiles throughout.
        rr = rerun.get(f"S2_m5_IW{tok}_fixed") if val >= 0 else None
        if rr is not None:
            iw.append(val)
            for s in SPECIES:
                vmr[s].append(rr.get(f"{s}_vmr", np.nan))
            psurf.append(rr.get("P_surf", np.nan))
            obud.append(_oxygen_budget(fx) if os.path.isfile(fx) else np.nan)
        elif os.path.isfile(fx):
            snap = _interp_at_phi(fx, args.phi, spec_cols)
            if snap is not None:
                iw.append(val)
                for s in SPECIES:
                    vmr[s].append(snap.get(f"{s}_vmr", np.nan))
                psurf.append(snap["P_surf"])
                obud.append(_oxygen_budget(fx))
        ox = os.path.join(args.s2dir, f"S2_m5_IW{tok}_oxauth", "runtime_helpfile.csv")
        if os.path.isfile(ox):
            d = _derived_iw(ox)
            if np.isfinite(d):
                oxa_iw.append(val)
                oxa_der.append(d)
    iw = np.array(iw, dtype=float)
    psurf = np.array(psurf, dtype=float)
    obud = np.array(obud, dtype=float)

    # Ten distinct series colours: the nine-hue categorical cycle for the first
    # nine species, plus a pale-blue tenth for H2S so no two species share a hue
    # (H2S would otherwise recycle the first, near-black, H2O hue).
    pal = ix.strata_colors(9)
    colors = dict(zip(SPECIES, pal))
    colors["H2S"] = ix.CORE["ice"]
    budget_color = ix.CORE['azure']  # right-axis budget stays azure
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)

    # (a) atmospheric speciation at the common crystallisation snapshot; legend
    # placed above the panel so it never overlaps the species curves. Each species
    # takes one distinct hue; alternating solid/dashed line styles add a second
    # separation among the ten overlapping log curves.
    for k, s in enumerate(SPECIES):
        y = np.array(vmr[s], dtype=float)
        if np.all(~np.isfinite(y)) or np.nanmax(y) <= 1e-5:
            continue
        ls = "-" if k % 2 == 0 else "--"
        ax_a.plot(iw, y, ls, marker="o", color=colors[s], ms=3.6, lw=1.3,
                  label=SPECIES_MATH[s])
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-5, 2.0)
    ax_a.set_ylabel(r"Atmospheric volume mixing ratio")  # x-axis shared with panel (b)
    ax_a.axvline(0.0, color=ix.NEUTRALS["ink"], lw=0.5, ls=":", zorder=0)
    # the species key is drawn as a figure-level legend above the upper panel

    # (b) merged equivalence and budget panel: the oxygen-authoritative round-trip
    # on the left axis (derived vs imposed IW, with the 1:1 line) and the conserved
    # free-oxygen budget on the right axis, both against imposed IW.
    lim = [-6.8, 5.8]
    ax_b.plot(lim, lim, color=ix.NEUTRALS["ink"], lw=0.8, ls="--", zorder=3, label="1:1")
    if oxa_iw:
        ax_b.plot(oxa_iw, oxa_der, "o", color=ix.STRATA["amber"], ms=5.0, zorder=6,
                  label="oxygen-authoritative")
    ax_b.set_ylim(lim)
    ax_b.set_ylabel(r"Derived $\Delta$IW (oxygen-authoritative)")
    ax_b.set_yticks([-6, -4, -2, 0, 2, 4])
    ax_b.set_xlabel(r"Imposed oxygen fugacity $\Delta$IW")
    ax_b.axvline(0.0, color=ix.NEUTRALS["ink"], lw=0.5, ls=":", zorder=0)

    ax_bud = ax_b.twinx()  # right axis: conserved free-oxygen budget (log)
    ax_bud.grid(False)     # keep a single grid (from ax_b), so it is not doubled
    ax_bud.plot(iw, obud, "s-", color=budget_color, ms=3.8, lw=1.3, zorder=5,
                label="free-oxygen budget")
    ax_bud.set_yscale("log")
    ax_bud.set_ylim(2.5e21, 2.5e24)
    # span is under three decades: label 1/2/5 sub-decade ticks for >=4 readable values
    ax_bud.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    ax_bud.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax_bud.set_ylabel(r"Free-oxygen budget [kg]", color=budget_color)
    ax_bud.tick_params(axis="y", colors=budget_color)
    ax_bud.spines["right"].set_color(budget_color)

    # one legend for the merged panel, combining both axes; opaque and on top so
    # it is drawn above the grid rather than behind it
    h_b, l_b = ax_b.get_legend_handles_labels()
    h_o, l_o = ax_bud.get_legend_handles_labels()
    leg_b = ax_bud.legend(h_b + h_o, l_b + l_o, loc="upper left", fontsize=7.6,
                          framealpha=1.0)
    leg_b.set_zorder(20)

    for ax in (ax_a, ax_b):
        ax.set_xlim(-6.8, 5.8)
        ax.set_xticks([-6, -4, -2, 0, 2, 4])
    for ax in (ax_a, ax_b, ax_bud):
        ax.set_axisbelow(True)  # grid below the data lines, markers, and legend
    # panel sublabels in the manuscript-wide style, bottom-right clear of data
    ix.panel_label(ax_a, "(a) Speciation", x=0.96, y=0.05, ha="right", va="bottom")
    ix.panel_label(ax_bud, "(b) Round-trip and oxygen budget", x=0.96, y=0.05,
                   ha="right", va="bottom")

    for ax in (ax_a, ax_b, ax_bud):
        ix.set_mono_ticks(ax)

    # species key as a figure-level legend above the upper panel, two rows of five
    h_a, l_a = ax_a.get_legend_handles_labels()
    fig.legend(h_a, l_a, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=6.6, columnspacing=0.8, handletextpad=0.4, handlelength=1.8,
               frameon=False)
    fig.tight_layout(pad=0.6, h_pad=1.0, rect=(0, 0, 1, 0.925))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "redox.pdf")
    fig.savefig(out)
    print(f"Wrote {out}  (common snapshot Phi={args.phi})")
    print("IW | P_surf(bar) | O_budget(kg) | derivedIW | dominant species")
    der = dict(zip(oxa_iw, oxa_der))
    for i, v in enumerate(iw):
        doms = sorted(((vmr[s][i], s) for s in SPECIES if np.isfinite(vmr[s][i])),
                      reverse=True)[:2]
        ds = ", ".join(f"{s}={x:.2f}" for x, s in doms if x > 0.005)
        d = der.get(v, np.nan)
        print(f"  {v:+.0f}: {psurf[i]:8.0f}  {obud[i]:.3e}  {d:+.2f}  [{ds}]")


if __name__ == "__main__":
    main()
