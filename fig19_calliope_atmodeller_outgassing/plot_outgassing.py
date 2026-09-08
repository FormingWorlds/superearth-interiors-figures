"""CALLIOPE vs atmodeller outgassing-speciation figure for the Super-Earth paper.

A three-panel comparison of the equilibrium gas speciation predicted by the two
PROTEUS outgassing modules in fixed-fugacity mode, at the Earth bulk-silicate
volatile inventory of Krijt et al. 2023, melt fraction unity:

(a) partial pressures of the eight major C-H-N-S-O species against oxygen
    fugacity (the IW-buffer offset swept from -6 to +6 dex) at fixed inventory,
    2000 K;
(b) the same against total surface pressure, the inventory scaled at fixed
    oxygen fugacity (IW+3.5), 2000 K;
(c) the same against magma temperature, at fixed oxygen fugacity (IW+3.5) and
    fixed total surface pressure (1e4 bar).

Each species has one colour; CALLIOPE is drawn solid and atmodeller dashed.

The figure replots from the committed cache (``data/cache_outgassing.npz``) with
numpy, matplotlib, and the vendored style in ``../style/``; the cache was
computed with CALLIOPE and atmodeller in the PROTEUS environment (see the
repository README).

Usage
-----
    python plot_outgassing.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use('white', font='mono')
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'legend.fontsize': 6.4, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.4,
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.25, 'legend.framealpha': 0.9,
    'xtick.labelsize': 7.6, 'ytick.labelsize': 7.6, 'lines.linewidth': 1.4,
    'axes.linewidth': 0.9, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

CACHE = os.path.join(HERE, 'data', 'cache_outgassing.npz')

# Species colours shared with the redox figure (Fig 6): the extended categorical
# palette assigned in the redox species order, so each species keeps one colour
# across the manuscript (CO2 gold, CO green, and so on). Computed the same way as
# the redox figure so the two stay in step if the palette changes.
_REDOX_ORDER = ['H2O', 'H2', 'CO2', 'CO', 'O2', 'CH4', 'N2', 'S2', 'SO2', 'H2S']
_pal = ix.strata_colors(9)
COL = {sp: _pal[i % len(_pal)] for i, sp in enumerate(_REDOX_ORDER)}
LAB = {'H2O': r'H$_2$O', 'CO2': r'CO$_2$', 'N2': r'N$_2$', 'H2': r'H$_2$',
       'CO': 'CO', 'CH4': r'CH$_4$', 'S2': r'S$_2$', 'SO2': r'SO$_2$'}
ORDER = ['H2O', 'CO2', 'N2', 'H2', 'CO', 'CH4', 'S2', 'SO2']


def _load():
    return np.load(CACHE, allow_pickle=True)['data'].item()


def _panel(ax, text, x=0.035, y=0.965, fs=9, fontweight='bold'):
    ax.text(x, y, text, transform=ax.transAxes, ha='left', va='top', fontsize=fs,
            fontweight=fontweight, zorder=30,
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=1.0, edgecolor='none'))


def _truncate_collapse(x, y):
    """Enforce monotonicity of the fixed-fugacity inventory sweep: mask any
    point that falls more than a decade below the running maximum in pressure.
    At extreme pressure the minor species are almost fully dissolved and the
    ideal-gas solve underflows nonphysically, producing dips and collapses;
    this removes those artifacts while leaving monotonically rising species
    (CO2, CO) intact to the top of the axis. The other panels, where declines
    are physical, are not passed through this filter."""
    y = np.asarray(y, float).copy()
    o = np.argsort(x)
    ys = y[o]
    run = -np.inf
    for i in range(len(ys)):
        if np.isfinite(ys[i]) and ys[i] > 0:
            if ys[i] >= run / 10.0:
                run = max(run, ys[i])
            else:
                ys[i] = np.nan
    y[o] = ys
    return y


def _series(ax, xc, xa, d, ylog=True, truncate=False):
    for sp in ORDER:
        yc, ya = d['cal'][sp], d['atm'][sp]
        if truncate:
            yc, ya = _truncate_collapse(xc, yc), _truncate_collapse(xa, ya)
        ax.plot(xc, yc, '-', color=COL[sp], lw=1.6)
        ax.plot(xa, ya, '--', color=COL[sp], lw=1.25, alpha=0.95)
    if ylog:
        ax.set_yscale('log')


def plot_outgassing(c, outdir):
    a, b, cc = c['a'], c['b'], c['c']
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.5, 3.2))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.33, top=0.95, wspace=0.26)

    # (a) vs oxygen fugacity
    _series(axA, a['diw'], a['diw'], a)
    axA.axvline(c['diw_anchor'], color=ix.NEUTRALS['graphite'], lw=0.8, ls=':', alpha=0.7)
    axA.set_xlim(-6, 6)
    axA.set_ylim(1e-4, 1e4)
    axA.set_xlabel(r'$\log_{10}(f_{\mathrm{O_2}}/\mathrm{IW})$ [dex]')
    axA.set_ylabel(r'Partial pressure [$\mathrm{bar}$]')
    axA.xaxis.set_major_locator(FixedLocator([-6, -3, 0, 3, 6]))
    axA.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    _panel(axA, '(a) vs oxygen fugacity')

    # (b) vs total surface pressure (capped at the reliable ideal-gas regime)
    _series(axB, b['cal_P'], b['atm_P'], b, truncate=True)
    axB.set_xscale('log')
    axB.set_xlim(1, 1e5)
    axB.set_ylim(1e-4, 2e5)
    axB.set_xlabel(r'Surface pressure [$\mathrm{bar}$]')
    axB.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    axB.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    _panel(axB, '(b) vs surface pressure')

    # (c) vs magma temperature at fixed fO2 and surface pressure
    _series(axC, cc['T'], cc['T'], cc)
    axC.set_xlim(1400, 3000)
    axC.set_ylim(1e-4, 1e5)
    axC.set_xlabel(r'$T_\mathrm{magma}$ [$\mathrm{K}$]')
    axC.xaxis.set_major_locator(FixedLocator([1400, 1800, 2200, 2600, 3000]))
    axC.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    _panel(axC, '(c) vs temperature')

    for ax in (axA, axB, axC):
        ix.set_mono_ticks(ax)

    sp_handles = [Line2D([0], [0], color=COL[sp], lw=2.2, label=LAB[sp]) for sp in ORDER]
    code_handles = [
        Line2D([0], [0], color=ix.NEUTRALS['ink'], lw=1.8, ls='-', label='CALLIOPE'),
        Line2D([0], [0], color=ix.NEUTRALS['ink'], lw=1.5, ls='--', label='atmodeller'),
    ]
    leg1 = fig.legend(handles=sp_handles, loc='upper center', ncol=8,
                      bbox_to_anchor=(0.5, 0.17), frameon=False, columnspacing=1.1,
                      handlelength=1.5, fontsize=6.8)
    fig.add_artist(leg1)
    fig.legend(handles=code_handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.085), frameon=False, columnspacing=1.8,
               handlelength=2.2, fontsize=6.8)

    out = os.path.join(outdir, 'calliope_atmodeller_outgassing.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', os.path.relpath(out, HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_outgassing(_load(), args.outdir)


if __name__ == '__main__':
    main()
