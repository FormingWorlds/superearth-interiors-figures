"""Cross-validation of the Aragog energy solver against SPIDER.

Two figures are produced, each saved as vector PDF, both replotted from the
committed array caches under ``data/`` so they reproduce from the repository
alone (numpy + matplotlib only):

1. ``aragog_spider_const.pdf`` - constant-property heat-equation limit. Both
   production solvers relax a linear entropy profile in an insulated mantle
   shell by pure conduction (raised conductivity so the diffusion time is
   integrable). With constant density, heat capacity, and conductivity and the
   analytic temperature relation T(S) = T_ref exp((S - S_ref)/Cp), the entropy
   balance reduces exactly to the classical heat equation, so both solvers can
   be checked against each other and against the analytic T(S). Panels: the
   temperature profile evolution, the residual of each solver against the
   analytic T(S), and the radius-resolved difference between the two solvers.

2. ``aragog_spider_static.pdf`` - realistic magma-ocean state with the PALEOS
   MgSiO3 equation of state. Both solvers read bit-identical PALEOS pressure-
   entropy tables on the identical Adams-Williamson mesh and are evaluated on
   the same magma-ocean entropy profile. The instantaneous fields they compute
   are compared: temperature, melt fraction, density, and the convective heat
   flux. The thermodynamic fields agree to machine precision; the convective
   flux exercises the mixing-length transport and the mushy-zone viscosity
   blending, where the two independent implementations agree to about 1.5 per
   cent.

The caches were computed with the ``aragog`` package and a built SPIDER binary
in the PROTEUS environment (synthesised constant-property SPIDER tables, the
SPIDER runs, and the Aragog runs); the repository README records the
provenance. The plot path needs neither.

Usage
-----
    python plot_aragog_spider_parity.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

# PROTEUS visual identity, vendored under style/ at the repo root so the
# figures reproduce from the repository alone.
HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use('white', font='mono')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'legend.handlelength': 1.3, 'legend.handletextpad': 0.5,
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.25, 'legend.framealpha': 0.85,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'lines.linewidth': 2.0,
    'axes.linewidth': 0.9, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

DATA_DIR = os.path.join(HERE, 'data')
C_SP = ix.STRATA['cobalt']    # SPIDER
C_AR = ix.STRATA['amber']     # Aragog
C_AN = ix.NEUTRALS['slate'] if 'slate' in getattr(ix, 'NEUTRALS', {}) else '0.35'  # analytic


def _panel(ax, text, x=0.04, y=0.94, ha='left', va='top', fontweight='bold', fs=None):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fs if fs is not None else plt.rcParams['axes.titlesize'],
            fontweight=fontweight,
            bbox=dict(boxstyle='round,pad=0.13', facecolor='white', alpha=0.9, edgecolor='none'))


def _mono(ax):
    if hasattr(ix, 'set_mono_ticks'):
        ix.set_mono_ticks(ax)


def _log_resid_axis(ax, lo, hi):
    """Tidy a log y-axis of relative residuals with labelled decades."""
    ax.set_yscale('log')
    decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** decades))
    ax.set_ylim(10.0 ** decades.min(), 10.0 ** decades.max())


# ----------------------------------------------------------------------------
# Figure 1: constant-property heat-equation limit
# ----------------------------------------------------------------------------


def plot_const(outdir):
    d = np.load(os.path.join(DATA_DIR, 'cache_const_anchor.npz'))
    r = d['radius']            # Mm, surface-first
    t = d['times']            # Myr
    spT, arT = d['spT'], d['arT']     # (nt, nr) temperature [K]

    order = np.argsort(r)
    rr = r[order] * 1000.0   # Mm -> km
    xt = FixedLocator([3500, 4500, 5500, 6371])
    # snapshot indices spread across the run
    idx = [0, len(t) // 4, len(t) // 2, len(t) - 1]
    tcol = [ix.STRATA['cobalt'], ix.STRATA['plum'], ix.STRATA['magma'], ix.STRATA['amber']]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))

    # (a) temperature profile evolution, both solvers (they homogenise together)
    ax = axes[0]
    for n, k in enumerate(idx):
        ax.plot(rr, spT[k][order], '-', color=tcol[n], lw=1.9, label=f'{t[k]:.0f} Myr')
        ax.plot(rr, arT[k][order], '--', color='white', lw=1.9)
        ax.plot(rr, arT[k][order], '--', color=tcol[n], lw=1.2)
    ax.set_xlabel(r'Radius [$\mathrm{km}$]')
    ax.set_ylabel(r'$T$ [$\mathrm{K}$]')
    ax.set_xlim(3480, 6371)
    ax.set_ylim(2990, 3280)
    ax.xaxis.set_major_locator(xt)
    ax.yaxis.set_major_locator(FixedLocator([3000, 3100, 3200, 3280]))
    ax.legend(loc='lower left', title='solid SPIDER, dashed Aragog', title_fontsize=7.5,
              ncol=2, columnspacing=1.0)
    _panel(ax, '(a) Temperature', x=0.96, y=0.94, ha='right', va='top', fs=9.5)
    _mono(ax)

    # (b) radius-resolved difference between the two solvers, over time
    ax = axes[1]
    for n, k in enumerate(idx[1:], start=1):
        rel = np.abs(arT[k][order] - spT[k][order]) / spT[k][order]
        ax.plot(rr, rel, '-', color=tcol[n], lw=1.8, label=f'{t[k]:.0f} Myr')
    ax.set_xlabel(r'Radius [$\mathrm{km}$]')
    ax.set_ylabel(r'$|T_\mathrm{Aragog} - T_\mathrm{SPIDER}|\, /\, T_\mathrm{SPIDER}$')
    ax.set_xlim(3480, 6371)
    ax.xaxis.set_major_locator(xt)
    _log_resid_axis(ax, 1e-5, 1e-1)
    ax.legend(loc='upper right')
    _panel(ax, '(b) Relative difference', x=0.04, y=0.08, ha='left', va='bottom', fs=9.5)
    _mono(ax)

    fig.tight_layout(w_pad=1.3)
    for _a in fig.axes:
        _mono(_a)
    out = os.path.join(outdir, 'aragog_spider_const.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', os.path.relpath(out, HERE))


# ----------------------------------------------------------------------------
# Figure 2: realistic PALEOS magma-ocean state
# ----------------------------------------------------------------------------


def plot_static(outdir):
    d = np.load(os.path.join(DATA_DIR, 'cache_static_paleos.npz'))
    r = d['radius']                      # Mm, surface-first
    order = np.argsort(r)
    rr = r[order] * 1000.0               # Mm -> km
    T_sp, T_ar = d['T_sp'][order], d['T_ar'][order]
    phi_sp, phi_ar = d['phi_sp'][order], d['phi_ar'][order]
    Jc_sp, Jc_ar = d['Jconv_sp'][order], d['Jconv_ar'][order]

    # Single row of four panels, sized close to the figure* text width so the
    # figure is placed at ~1:1 and the fonts are not shrunk on inclusion.
    xt = FixedLocator([3500, 4500, 5500, 6371])
    xfmt = FuncFormatter(lambda v, p: f'{v:g}')
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.95))
    fig.subplots_adjust(left=0.062, right=0.997, bottom=0.255, top=0.965, wspace=0.40)
    LBL = 9.0
    for ax in axes:
        ax.set_xlim(rr.min(), rr.max())
        ax.xaxis.set_major_locator(xt)
        ax.xaxis.set_major_formatter(xfmt)
        ax.tick_params(labelsize=7.5)

    TS = 9.0  # panel-title font size

    # (a) temperature
    ax = axes[0]
    ax.plot(rr, T_sp, '-', color=C_SP, label='SPIDER')
    ax.plot(rr, T_ar, '--', color=C_AR, label='Aragog')
    ax.set_ylabel(r'$T$ [$\mathrm{K}$]', fontsize=LBL)
    ax.set_ylim(T_sp.min(), T_sp.max())
    ax.yaxis.set_major_locator(FixedLocator([2500, 3500, 4500, 5200]))
    ax.legend(loc='upper right', fontsize=6.6, handlelength=1.0, borderpad=0.2)
    _panel(ax, '(a) Temperature', x=0.05, y=0.13, va='bottom', fs=TS)
    _mono(ax)

    # (b) melt fraction
    ax = axes[1]
    ax.plot(rr, phi_sp, '-', color=C_SP)
    ax.plot(rr, phi_ar, '--', color=C_AR)
    ax.set_ylabel(r'$\phi$', fontsize=LBL)
    ax.set_ylim(0.4, 1.02)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 0.9, 1.0]))
    _panel(ax, '(b) Melt fraction', x=0.05, y=0.94, va='top', fs=TS)
    _mono(ax)

    # (c) convective heat flux (log)
    ax = axes[2]
    ax.plot(rr, np.abs(Jc_sp), '-', color=C_SP)
    ax.plot(rr, np.abs(Jc_ar), '--', color=C_AR)
    ax.set_yscale('log')
    ax.set_ylabel(r'$|F_\mathrm{conv}|$ [$\mathrm{W\,m^{-2}}$]', fontsize=LBL)
    _panel(ax, '(c) Convective\nflux', x=0.05, y=0.16, va='bottom', fs=TS)
    _mono(ax)

    # (d) relative differences. Extend the floor to 1e-12 so the temperature
    # difference (near machine precision, ~5e-12) is visible rather than
    # falling off the bottom of the axis.
    ax = axes[3]
    relT = np.abs(T_ar - T_sp) / np.abs(T_sp)
    mask = np.abs(Jc_sp) > 1e-2 * np.abs(Jc_sp).max()
    relJ = np.full_like(Jc_sp, np.nan)
    relJ[mask] = np.abs(Jc_ar[mask] - Jc_sp[mask]) / np.abs(Jc_sp[mask])
    ax.plot(rr, relT, '-', color=ix.STRATA['ink'] if 'ink' in ix.STRATA else 'k', label=r'$T$')
    ax.plot(rr, relJ, '-', color=ix.STRATA['magma'], label=r'$F_\mathrm{conv}$')
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e-1)
    ax.set_ylabel(r'$|\Delta X|\,/\,X$', fontsize=LBL)
    ax.yaxis.set_major_locator(FixedLocator([1e-12, 1e-9, 1e-6, 1e-3, 1e-1]))
    ax.legend(loc='center', fontsize=6.6, handlelength=1.0, borderpad=0.2,
              ncol=2, columnspacing=0.8)
    _panel(ax, '(d) Relative\ndifference', x=0.05, y=0.16, va='bottom', fs=TS)
    _mono(ax)

    fig.supxlabel(r'Radius [$\mathrm{km}$]', fontsize=LBL, y=0.055)
    for _a in fig.axes:
        _mono(_a)
    out = os.path.join(outdir, 'aragog_spider_static.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', os.path.relpath(out, HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_const(args.outdir)
    plot_static(args.outdir)


if __name__ == '__main__':
    main()
