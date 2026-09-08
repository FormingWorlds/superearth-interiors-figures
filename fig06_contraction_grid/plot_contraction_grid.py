"""Structural contraction of super-Earths as they crystallise: the S1 figure.

The interior-radius panels are referenced to the shared self-consistent baseline:
the dynamic and static runs reach a common interior radius once the forced
structure solve runs at the start, and the static twin then freezes there. The
dynamic run rises slightly above that baseline through a molten-phase relaxation,
then contracts below it as the mantle crystallises. The crystallisation
contraction (panel d) is measured from the molten peak, the maximally molten
structure, and is the same physical effect regardless of the radius-axis anchor.

Four panels, the headline result of the paper:

(a) interior radius against time, dynamic structure (solid) versus static twin
    (dashed), one colour per planet mass; radii normalised to each mass's shared
    baseline. The static twin is flat at 1; the dynamic curve rises slightly above
    1 (the molten-phase relaxation) then falls below 1 as the mantle solidifies.
(b) interior radius against solid fraction (1 - Phi_global), dynamic runs only,
    normalised to the shared baseline, putting the four masses on a common scale.
(c) global melt fraction against time, dynamic (solid) versus static (dashed): the
    crystallisation clock and any structure feedback on the cooling rate.
(d) the total fractional contraction from the molten peak against planet mass, the
    one-number summary of how much a super-Earth shrinks.

The figure replots from the cache ``data/cache_shrinking.npz`` written by
``compute_shrinking.py`` from the S1 helpfiles under ``../data/helpfiles``.

Usage
-----
    python plot_contraction_grid.py [--cache FILE] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

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

R_EARTH = 6.371e6  # m
# The final grid cache is the data behind the published figure.
DEFAULT_CACHE = os.path.join(HERE, "data", "cache_shrinking.npz")
MASS_ORDER = [1.0, 3.0, 5.0, 10.0]


def _smooth_xy(x, y, n_grid=200, frac=0.07):
    """Resample (x, y) onto a uniform grid and apply a moving average.

    The dynamic runs report the interior radius at discrete melt-fraction steps,
    which makes the radius-versus-solid-fraction curves look blocky. Resampling
    onto a uniform solid-fraction grid and smoothing with a short odd-length box
    filter recovers the underlying trend for an overlay, leaving the raw curve
    visible beneath it.

    Parameters
    ----------
    x, y : array_like
        Monotonic-in-``x`` samples (sorted by ``x`` before calling).
    n_grid : int
        Number of points in the uniform resampling grid.
    frac : float
        Smoothing window as a fraction of the grid length.

    Returns
    -------
    xu, ys : ndarray
        Uniform grid and the smoothed values on it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4:
        return x, y
    xu = np.linspace(x.min(), x.max(), n_grid)
    yu = np.interp(xu, x, y)
    win = max(5, int(round(n_grid * frac)))
    win += 1 - win % 2  # force odd
    pad = win // 2
    yp = np.pad(yu, pad, mode="edge")
    ys = np.convolve(yp, np.ones(win) / win, mode="valid")
    return xu, ys


def _load(cache_path):
    """Load the cache into a per-run record dict keyed by run name."""
    z = np.load(cache_path, allow_pickle=True)
    runs = [str(r) for r in z["runs"]]
    rec = {}
    for name in runs:
        d = {"mass": float(z[f"{name}__mass"]),
             "struct": str(z[f"{name}__struct"])}
        for field in ("Time", "R_int", "R_obs", "Phi_global", "T_magma"):
            key = f"{name}__{field}"
            if key in z.files:
                d[field] = np.asarray(z[key], dtype=float)
        rec[name] = d
    return rec


def _by_mass_struct(rec):
    """Index records as {(mass, struct): record}, dropping unparsed runs."""
    out = {}
    for d in rec.values():
        out[(d["mass"], d["struct"])] = d
    return out


def _r0(d):
    """Initial interior radius (first finite R_int) for a record."""
    ri = d["R_int"]
    finite = ri[np.isfinite(ri)]
    return finite[0] if finite.size else np.nan


def _rpeak(d):
    """Molten-peak interior radius (max finite R_int) for a record.

    A coupled run rises from its initial radius to a molten-equilibrium peak
    before crystallisation contracts it; the peak is the reference state for the
    contraction. ``None`` records and all-NaN R_int return NaN.
    """
    if d is None or "R_int" not in d:
        return np.nan
    ri = d["R_int"]
    finite = ri[np.isfinite(ri)]
    return finite.max() if finite.size else np.nan


def _peak_index(ri):
    """Index of the molten peak (argmax over finite R_int)."""
    masked = np.where(np.isfinite(ri), ri, -np.inf)
    return int(np.argmax(masked))


def _baseline_radius(idx, mass):
    """Shared self-consistent baseline radius for a mass.

    Both the dynamic and static runs reach a common interior radius once the
    forced structure solve runs at the start (the static twin then freezes
    there); that shared baseline is the honest anchor for the radius axis, since
    it is the state the two runs actually share. It is recovered as the static
    twin's settled radius (its maximum finite ``R_int``, since the static run
    rises once from the initial-condition value to the baseline and then holds).
    Falls back to the dynamic molten peak if no static twin is present.
    """
    st = idx.get((mass, "stat"))
    if st is not None and "R_int" in st:
        ri = st["R_int"][np.isfinite(st["R_int"])]
        if ri.size:
            return float(np.max(ri))
    dy = idx.get((mass, "dyn"))
    return _rpeak(dy) if dy is not None else np.nan


def _from_baseline(ri, rb):
    """Index of the first row at or after the baseline (drops leading IC rows).

    A run starts at the initial-condition radius (below the baseline), jumps to
    the baseline when the forced structure solve runs, then evolves; the leading
    sub-baseline rows are the pre-solve transient and are trimmed.
    """
    reached = np.where(ri >= 0.99 * rb)[0]
    return int(reached[0]) if reached.size else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()

    if not os.path.isfile(args.cache):
        print(f"No cache at {args.cache}. Run compute_shrinking.py first.")
        return
    rec = _load(args.cache)
    idx = _by_mass_struct(rec)
    masses = [m for m in MASS_ORDER if any(k[0] == m for k in idx)]
    if not masses:
        print("Cache has no recognised S1 runs.")
        return
    # Fixed per-mass hues (warm-to-cool by mass), named so they stay put when the
    # brand cycle changes: 1, 3, 5, 10 M_Earth -> red, ocean, plum, azure.
    colors = dict(zip(MASS_ORDER, [ix.STRATA['amber'], ix.STRATA['cobalt'],
                                   ix.STRATA['plum'], ix.STRATA['gold']]))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) interior radius vs time, dynamic (solid) vs static (dashed). Both curves
    # of a mass are normalised to that mass's shared self-consistent baseline, so
    # the static twin is flat at 1 while the dynamic curve shows its small
    # molten-phase rise above the baseline before contracting below it as the
    # mantle crystallises. Leading initial-condition rows before the baseline are
    # dropped.
    for m in masses:
        c = colors[m]
        rb = _baseline_radius(idx, m)
        for struct, ls in (("dyn", "-"), ("stat", "--")):
            d = idx.get((m, struct))
            if d is None or "Time" not in d:
                continue
            t = d["Time"]
            ri = d["R_int"]
            good = np.isfinite(t) & np.isfinite(ri) & (t > 0)
            t_g, ri_g = t[good], ri[good]
            s = _from_baseline(ri_g, rb)
            if struct == "dyn":
                # raw stepped curve faint, smoothed trend on top, as in (b);
                # smoothing runs in log-time to match the axis.
                ax_a.plot(t_g[s:], ri_g[s:] / rb, ls, color=c, lw=1.2, alpha=0.45)
                lu, yu = _smooth_xy(np.log10(t_g[s:]), ri_g[s:] / rb)
                ax_a.plot(10.0 ** lu, yu, ls, color=c, lw=1.8)
            else:
                ax_a.plot(t_g[s:], ri_g[s:] / rb, ls, color=c, lw=1.5)
    ax_a.axhline(1.0, color=ix.NEUTRALS["ink"], lw=0.6, ls=":", zorder=0)
    # arrow running parallel to (just below) the flat static-structure runs,
    # naming them directly
    ax_a.annotate("", xy=(0.86, 0.90), xytext=(0.50, 0.90),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="->", lw=1.0, color=ix.NEUTRALS["ink"]))
    ax_a.text(0.68, 0.865, "Static runs", transform=ax_a.transAxes,
              fontsize=7.5, ha="center", va="top", color=ix.NEUTRALS["ink"])
    ax_a.set_xscale("log")
    ax_a.set_xlim(left=1.0e3)
    ax_a.set_xlabel("Time [yr]")
    ax_a.set_ylabel(r"Radius from baseline $R_\mathrm{int}/R_\mathrm{base}$")

    # (b) interior radius vs solid fraction (1 - Phi), dynamic only, normalised to
    # the shared baseline: the curve starts at the baseline (1) while molten, rises
    # slightly through the molten-phase relaxation, then contracts as the mantle
    # crystallises. Leading initial-condition rows before the baseline are dropped.
    for m in masses:
        d = idx.get((m, "dyn"))
        if d is None or "Phi_global" not in d:
            continue
        rb = _baseline_radius(idx, m)
        ri = d["R_int"]
        sol_all = 1.0 - d["Phi_global"]
        good = np.isfinite(sol_all) & np.isfinite(ri)
        sol_g, ri_g = sol_all[good], ri[good]
        s = _from_baseline(ri_g, rb)
        sol_g, ri_g = sol_g[s:], ri_g[s:]
        order = np.argsort(sol_g)
        xs, ys = sol_g[order], ri_g[order] / rb
        # raw (blocky) curve faint, smoothed trend overlaid on top to separate masses
        ax_b.plot(xs, ys, "-", color=colors[m], lw=1.2, alpha=0.45)
        xu, yu = _smooth_xy(xs, ys)
        ax_b.plot(xu, yu, "-", color=colors[m], lw=1.8)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.axhline(1.0, color=ix.NEUTRALS["ink"], lw=0.6, ls=":", zorder=0)
    ax_b.set_xlabel(r"Solid fraction $\,1-\Phi$")
    ax_b.set_ylabel(r"Radius from baseline $R_\mathrm{int}/R_\mathrm{base}$")

    # (c) global melt fraction vs time, dynamic (solid) vs static (dashed): the
    # crystallisation clock, and any structure feedback on the cooling rate.
    for m in masses:
        c = colors[m]
        for struct, ls in (("dyn", "-"), ("stat", "--")):
            d = idx.get((m, struct))
            if d is None or "Phi_global" not in d or "Time" not in d:
                continue
            t = d["Time"]
            good = np.isfinite(t) & np.isfinite(d["Phi_global"]) & (t > 0)
            ax_c.plot(t[good], d["Phi_global"][good], ls, color=c, lw=1.5)
    ax_c.set_xscale("log")
    ax_c.set_xlim(left=1.0e3)
    ax_c.set_ylim(0.0, 1.02)
    ax_c.set_xlabel("Time [yr]")
    ax_c.set_ylabel(r"Melt fraction $\Phi$")

    # (d) total contraction from the molten peak vs mass.
    tot_m, tot_c = [], []
    for m in masses:
        d = idx.get((m, "dyn"))
        if d is None:
            continue
        rpeak = _rpeak(d)
        ri = d["R_int"][np.isfinite(d["R_int"])]
        if ri.size == 0 or not np.isfinite(rpeak):
            continue
        tot_m.append(m)
        tot_c.append((rpeak - ri[-1]) / rpeak * 100.0)
    if tot_m:
        ax_d.plot(tot_m, tot_c, "o-", color=ix.NEUTRALS["ink"], ms=6, lw=1.3,
                  zorder=3)
        for m, c in zip(tot_m, tot_c):
            ax_d.scatter([m], [c], s=46, color=colors[m], zorder=4,
                         edgecolor=ix.NEUTRALS["ink"], linewidth=0.6)
    ax_d.set_xlabel(r"Planet mass [$M_\oplus$]")
    ax_d.set_ylabel(r"Contraction from molten peak [%]")
    ax_d.set_xlim(0, 11)
    ax_d.xaxis.set_major_locator(FixedLocator([0, 1, 3, 5, 10]))

    # panel sublabels in the manuscript-wide style: bold "(a) Subtitle" inside
    # the axes on a translucent patch
    titles = ("Contraction in time", "Contraction with solidification",
              "Crystallisation clock", "Mass dependence")
    for ax, lab, ttl in zip((ax_a, ax_b, ax_c, ax_d), "abcd", titles):
        ix.panel_label(ax, f"({lab}) {ttl}", x=0.04, y=0.06, va="bottom",
                       fontsize=8.5)
        ax.margins(x=0.02)

    # legends: structure treatment (linestyle) in panel (a, bottom-left), mass
    # (colour) in panel (b, top-right), each placed where its panel is empty.
    mass_handles = [Line2D([0], [0], color=colors[m], lw=1.8,
                           label=rf"{m:g} $M_\oplus$") for m in masses]
    struct_handles = [
        Line2D([0], [0], color=ix.NEUTRALS["ink"], lw=1.8, ls="-",
               label="dynamic structure"),
        Line2D([0], [0], color=ix.NEUTRALS["ink"], lw=1.8, ls="--",
               label="static structure"),
    ]
    ax_a.legend(handles=struct_handles, loc="center left")
    ax_b.legend(handles=mass_handles, loc="upper right", ncol=2,
                title="Mass", fontsize=6.4)

    # Flag a still-crystallising grid: the contraction magnitudes deepen until
    # every dynamic run reaches the solid end.
    min_phi = [np.nanmin(idx[(m, "dyn")]["Phi_global"])
               for m in masses if (m, "dyn") in idx
               and "Phi_global" in idx[(m, "dyn")]]
    unfinished = max(min_phi) if min_phi else 0.0
    if unfinished > 0.05:
        fig.suptitle(rf"Preliminary: grid still crystallising "
                     rf"(highest $\Phi_\mathrm{{min}}={unfinished:.2f}$); "
                     rf"magnitudes deepen toward $\Phi\!\to\!0$",
                     fontsize=7.2, y=1.005, color=ix.STRATA["amber"])

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ix.set_mono_ticks(ax)

    fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=1.0)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "contraction_grid.pdf")
    fig.savefig(out)
    print(f"Wrote {out}")
    if tot_m:
        print("Contraction from the molten peak (latest crystallisation state):")
        for m, c in zip(tot_m, tot_c):
            print(f"  {m:>4g} Mearth: {c:5.2f} %")


if __name__ == "__main__":
    main()
