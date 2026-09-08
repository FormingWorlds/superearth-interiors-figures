"""CALLIOPE vs atmodeller module-agreement figure for the Super-Earth paper appendix.

One two-panel figure (``calliope_atmodeller.pdf``) selected from the CALLIOPE
module-comparison documentation and adapted to the paper formatting (PROTEUS
style and figure font, vector PDF). Both outgassing modules are run through
the shared authoritative-oxygen entry point at the same Earth bulk-silicate
volatile inventory, so any disagreement reflects the modules' internal choices
(oxygen-fugacity buffer, solubility laws, equilibrium-constant fits) rather than
a difference in the inputs.

(a) Converged oxygen-fugacity offset ``dIW`` from each module across magma
    temperature: CALLIOPE with its current Fischer 2011 buffer, CALLIOPE with the
    legacy O'Neill 2002 buffer, atmodeller with its Hirschmann composite, and the
    offset atmodeller would show if the buffer were the only difference.
(b) The raw cross-module gaps for both CALLIOPE buffers and the residual after
    the analytic buffer offset is removed, against a +/- 0.1 dex solver tolerance.

The figure is replotted from the committed CSV cache under ``data/``
(``fig3_grid.csv``); the live recompute that regenerates the cache is in the
CALLIOPE repository (``scripts/cross_backend``) and needs both modules
installed. Provenance is recorded in the repository README.

Usage
-----
    python plot_calliope_atmodeller.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use('white', font='mono')
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 6.6, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.4,
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.3, 'legend.framealpha': 0.9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'lines.linewidth': 1.4,
    'axes.linewidth': 0.9, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

DATA_DIR = os.path.join(HERE, 'data')
CSV = os.path.join(DATA_DIR, 'fig3_grid.csv')

# PROTEUS module colours; the legacy buffer is neutral grey.
C_CAL = ix.CORE['ocean']    # CALLIOPE, Fischer default
C_ATM = ix.CORE['magma']    # atmodeller, Hirschmann composite
C_LEG = ix.NEUTRALS.get('graphite', '#666')  # legacy O'Neill / raw-gap grey
C_RES = ix.STRATA['plum']  # buffer-corrected residual


def _load():
    """Read the committed CSV cache into column arrays keyed by header name."""
    with open(CSV, newline='') as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


def _panel(ax, text, x=0.04, y=0.95, ha='left', va='top', fontweight='bold', fs=9):
    # Fully opaque box so no gridline shows beneath the panel label.
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            fontweight=fontweight, zorder=20,
            bbox=dict(boxstyle='round,pad=0.13', facecolor='white', alpha=1.0, edgecolor='none'))


def plot_comparison(d, outdir):
    """Two panels sharing the magma-temperature axis: (a) converged dIW per
    module, (b) the cross-module gaps and the buffer-corrected residual."""
    T = d['T_K']
    # atmodeller offset predicted from the buffer difference alone: where it
    # would land if the two modules shared identical chemistry.
    atm_pred = d['dIW_calliope_fischer'] - d['buffer_offset_HminusF_dex']

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 4.8), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.09, top=0.99, hspace=0.10)
    Tticks = [1800, 2000, 2400, 2800, 3000]

    # (a) converged dIW per module
    ax_a.plot(T, d['dIW_calliope_fischer'], '-o', color=C_CAL, ms=4,
              label='CALLIOPE (Fischer, default)')
    ax_a.plot(T, d['dIW_calliope_oneill'], '--o', color=C_LEG, ms=4, alpha=0.85,
              label="CALLIOPE (O'Neill, legacy)")
    ax_a.plot(T, d['dIW_atmodeller'], '-s', color=C_ATM, ms=4,
              label='atmodeller (Hirschmann)')
    ax_a.plot(T, atm_pred, ':x', color=C_ATM, ms=5, alpha=0.7,
              label='atmodeller, buffer-predicted')
    ax_a.set_ylim(2.3, 4.0)
    ax_a.set_ylabel(r'$\Delta\mathrm{IW}$ [dex]')
    ax_a.yaxis.set_major_locator(FixedLocator([2.4, 2.8, 3.2, 3.6, 4.0]))
    ax_a.legend(loc='lower left', fontsize=6.4, framealpha=0.9)
    _panel(ax_a, '(a) Converged offset', x=0.04, ha='left')

    # (b) cross-module gaps and the buffer-corrected residual
    ax_b.axhline(0.0, color=ix.NEUTRALS['ink'], lw=0.7, alpha=0.5)
    for yl in (0.1, -0.1):
        ax_b.axhline(yl, color=C_LEG, ls='--', lw=0.7, alpha=0.6)
    # The band is empty of curve ink above +0.1 dex, so the pair is named once,
    # right-aligned just above the upper line.
    ax_b.text(0.985, 0.115, r'$\pm 0.1$ dex solver tolerance',
              transform=ax_b.get_yaxis_transform(), ha='right', va='bottom',
              fontsize=6.4, color=C_LEG)
    ax_b.plot(T, d['raw_gap_F_dex'], '-o', color=C_CAL, ms=4,
              label='raw gap (Fischer)')
    ax_b.plot(T, d['raw_gap_O_dex'], '--o', color=C_LEG, ms=4, alpha=0.8,
              label="raw gap (O'Neill)")
    ax_b.plot(T, d['residual_after_buffer_F_dex'], '-D', color=C_RES, ms=4,
              label='buffer-corrected residual')
    ax_b.set_ylim(-1.35, 0.52)
    ax_b.set_ylabel(r'$\Delta\mathrm{IW}$ disagreement [dex]')
    ax_b.yaxis.set_major_locator(FixedLocator([-1.2, -0.8, -0.4, 0.0]))
    ax_b.legend(loc='lower left', fontsize=6.4, framealpha=0.9)
    _panel(ax_b, '(b) Module gap', x=0.04, ha='left')

    for ax in (ax_a, ax_b):
        ax.set_xlim(1780, 3020)
        ax.xaxis.set_major_locator(FixedLocator(Tticks))
        ix.set_mono_ticks(ax)
    ax_b.set_xlabel(r'$T_\mathrm{magma}$ [$\mathrm{K}$]')  # shared x-axis; label on the lower panel

    out = os.path.join(outdir, 'calliope_atmodeller.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', os.path.relpath(out, HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_comparison(_load(), args.outdir)


if __name__ == '__main__':
    main()
