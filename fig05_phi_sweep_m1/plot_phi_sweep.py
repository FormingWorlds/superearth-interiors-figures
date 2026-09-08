"""Structural contraction of a crystallising super-Earth: the standalone sweep.

The interior radius of a rocky planet is set by its interior structure, which
responds to the melt fraction through the melt-to-solid density contrast. This
figure plots a standalone zalmoxis structure sweep over mantle melt fraction at
fixed mass and core mass fraction: as the mantle crystallises (solid fraction
1 - Phi grows), the interior radius contracts. This isolates the structural
contraction from the cooling timeline and the atmosphere; the coupled
dynamic-versus-static evolution runs confirm it occurs self-consistently and that
freezing the structure removes it.

Reads ``zalmoxis_phi_sweep_m1_mantle.csv`` (columns surface_T_K, Phi_mantle,
Phi_whole, R_int_m, R_int_RE, R_core_m, P_cmb_GPa, T_cmb_K, P_center_GPa,
T_center_K, core_mass_frac) from ``data/``. The melt fraction on
the horizontal axis is integrated over mantle shells only, mass-weighted, with
the core-mantle boundary taken from the density step between the iron and
silicate equations of state. ``Phi_whole``, the same integral carried over the
whole planet, is retained in the file for reference but is not a melt fraction:
it scores the iron core against the silicate melting curve.

Usage
-----
    python plot_phi_sweep.py [--csv FILE] [--outdir DIR]
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
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.4,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.7,
    "axes.linewidth": 0.9, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

R_EARTH = 6.371e6  # m
# Guide arrow along the contraction curve: the solid-fraction endpoints of the
# chord it follows, and how far clear of the curve the arrow and its label sit,
# in typographic points measured perpendicular to that chord.
TRAJ_SPAN = (0.30, 0.78)
TRAJ_GAP_PT = 7.0
LABEL_GAP_PT = 15.0
# Surface-temperature anchors from 4800 K, where the mantle is molten, down to
# 1300 K, where it is fully solid; the data behind the figure.
DEFAULT_CSV = os.path.join(HERE, "data", "zalmoxis_phi_sweep_m1_mantle.csv")


def _read(csv_path):
    """Read the sweep CSV (skipping comment lines) into a column dict."""
    lines = [ln for ln in open(csv_path) if not ln.lstrip().startswith("#")]
    rows = list(csv.reader(lines))
    idx = {c.strip(): i for i, c in enumerate(rows[0])}
    out = {}
    for name, i in idx.items():
        vals = []
        for r in rows[1:]:
            if len(r) > i and r[i] != "":
                try:
                    vals.append(float(r[i]))
                except ValueError:
                    vals.append(np.nan)
        out[name] = np.asarray(vals, dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()

    d = _read(args.csv)
    phi = d["Phi_mantle"]
    r_int = d["R_int_m"]
    r_core = d["R_core_m"]
    if phi.size < 2:
        print(f"Only {phi.size} sweep points; need at least 2. Sweep still filling?")
        return

    order = np.argsort(-phi)                 # molten (Phi=1) first
    phi, r_int, r_core = phi[order], r_int[order], r_core[order]
    sol = 1.0 - phi                          # solid fraction
    r0 = r_int[0]                            # radius at the most molten state
    has_core = np.isfinite(r_core).any()     # R_core is NaN if not in the solver return
    mantle = r_int - r_core
    mantle0 = mantle[0] if has_core else np.nan

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 5.2), sharex=True)
    c = ix.CORE["magma"]  # interior radius in the PROTEUS warm red

    # (a) normalised interior radius vs solid fraction: the contraction law.
    # Endpoints sit on the axis limits, so markers draw unclipped to stay round.
    ax_a.plot(sol, r_int / r0, "-o", color=c, ms=4, clip_on=False)
    ax_a.set_ylabel(r"Normalised radius $R_\mathrm{int}/R_0$")  # x-axis shared with panel (b)
    ax_a.set_xlim(0, 1)
    # Total-contraction callout: label parked in the bottom-left corner, arrow
    # reaching to the crystallised endpoint but stopping short of its marker.
    total = (r0 - r_int[-1]) / r0 * 100.0
    ax_a.annotate(rf"$-{total:.1f}\%$ at $\Phi={phi[-1]:.2f}$",
                  xy=(sol[-1], r_int[-1] / r0), xycoords="data",
                  xytext=(0.05, 0.10), textcoords="axes fraction",
                  fontsize=8, ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.8,
                                  color=ix.NEUTRALS["ink"], shrinkB=9))

    # (b) absolute interior (and core) radius: the mantle shell thins.
    ax_b.plot(sol, r_int / R_EARTH, "-o", color=c, ms=4, clip_on=False,
              label="Planetary surface")
    if has_core:
        ax_b.plot(sol, r_core / R_EARTH, "-s", color=ix.NEUTRALS["ink"], ms=3.5,
                  clip_on=False, label="Core-mantle boundary")
    ax_b.set_xlabel(r"Mantle solid fraction $\,1-\Phi$")
    ax_b.set_ylabel(r"Interior radius [$R_\oplus$]")
    ax_b.set_xlim(0, 1)
    ax_b.legend(loc="center left")
    # Right axis in km, so the radius change reads directly in absolute units at
    # this single planet mass (1 Earth radius = 6371 km).
    sec_b = ax_b.secondary_yaxis("right",
                                 functions=(lambda v: v * 6371.0, lambda v: v / 6371.0))
    sec_b.set_ylabel(r"Interior radius [$\mathrm{km}$]")
    ix.set_mono_ticks(sec_b)

    # (a) top-right (the curve occupies the top-left); (b) on the left at the
    # height of the 0.95 tick, between the surface curve and the legend
    ix.panel_label(ax_a, "(a) Normalised contraction", x=0.96, ha="right",
                   fontsize=9)
    b_lo, b_hi = ax_b.get_ylim()
    ix.panel_label(ax_b, "(b) Surface and core radii",
                   y=(0.95 - b_lo) / (b_hi - b_lo), fontsize=9)

    for ax in (ax_a, ax_b):
        ix.set_mono_ticks(ax)

    fig.tight_layout(pad=0.6, h_pad=1.0)

    # "Crystallisation trajectory" guide arrow and its label, laid along a chord
    # of the contraction curve and offset just clear of it on the upper side,
    # pointing toward the crystallised end. Built after layout so the offsets are
    # true distances on the page and the drawn slope matches the curve's.
    fig.canvas.draw()
    ends = np.interp(TRAJ_SPAN, sol, r_int / r0)
    d0, d1 = ax_a.transData.transform(np.column_stack([TRAJ_SPAN, ends]))
    unit = (d1 - d0) / np.hypot(*(d1 - d0))
    perp = np.array([-unit[1], unit[0]])
    if perp[1] < 0:                       # keep the offset on the upper side
        perp = -perp
    per_pt = fig.dpi / 72.0
    inv = ax_a.transData.inverted()
    tail = inv.transform(d0 + perp * TRAJ_GAP_PT * per_pt)
    head = inv.transform(d1 + perp * TRAJ_GAP_PT * per_pt)
    ax_a.annotate("", xy=head, xytext=tail, xycoords="data", textcoords="data",
                  arrowprops=dict(arrowstyle="->", lw=1.2, color=c, alpha=0.9))
    traj_ang = np.degrees(np.arctan2(unit[1], unit[0]))
    lx, ly = inv.transform(0.5 * (d0 + d1) + perp * LABEL_GAP_PT * per_pt)
    ax_a.text(lx, ly, "Crystallisation trajectory", fontsize=7.5, color=c,
              ha="center", va="center", rotation=traj_ang, rotation_mode="anchor")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "phi_sweep_m1.pdf")
    fig.savefig(out)
    print(f"Wrote {out}")
    print(f"  molten R0 = {r0/R_EARTH:.4f} R_E, solid (Phi={phi[-1]:.2f}) = "
          f"{r_int[-1]/R_EARTH:.4f} R_E")
    print(f"  total interior contraction: {total:.2f}%")
    if has_core:
        print(f"  mantle-shell thinning: {(mantle0-mantle[-1])/mantle0*100:.2f}% "
              f"({mantle0/R_EARTH:.4f} -> {mantle[-1]/R_EARTH:.4f} R_E)")
    else:
        print("  (R_core absent from solver return; mantle-shell thinning not computed)")


if __name__ == "__main__":
    main()
