"""CHILI validation figure for the Super-Earth paper appendix.

A single three-panel figure (``chili_validation.pdf``) is assembled from the
PROTEUS CHILI intercomparison and adapted to the paper formatting (PROTEUS
style and figure font, radius in km, vector PDF); the eight model curves keep
the Wong colourblind palette so they stay separable:

(a) solidification time against melt fraction for the CHILI Nominal Earth (solid) and
    Nominal Venus (dashed) cases, with the current PROTEUS run overlaid on the
    six-model community ensemble and the earlier submitted PROTEUS-CHILI result;
(b) surface temperature and (c) rheological-front radius against melt fraction
    for the Nominal Earth case; all three panels share the melt-fraction axis.

The community curves carry the volume-weighted melt fraction; PROTEUS reports
the mass-weighted global melt fraction (its ``Phi_global_vol`` is a stub equal
to ``Phi_global`` in the Aragog path), and the two agree to a few per cent over
the magma-ocean range, so the melt-fraction axes stay unit-neutral.

The figure is replotted from the curve cache ``data/cache_chili.npz`` (numpy
and matplotlib only). The cache holds the CHILI benchmark curves
(github.com/projectcuisines/chili) and the PROTEUS Earth and Venus runs; how
it was extracted is described in the repository README.

Usage
-----
    python plot_chili_validation.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use('white', font='mono')
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 6.6, 'legend.handlelength': 1.4, 'legend.handletextpad': 0.4,
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.25, 'legend.framealpha': 0.9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'lines.linewidth': 1.4,
    'axes.linewidth': 0.9, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

DATA_DIR = os.path.join(HERE, 'data')
CACHE = os.path.join(DATA_DIR, 'cache_chili.npz')

def _load():
    """Load the cached CHILI and PROTEUS curves from ``data/cache_chili.npz``."""
    return np.load(CACHE, allow_pickle=True)['data'].item()


# Community-model curve colours from the true PROTEUS extended palette (replacing
# the imported Wong set), one hue per model in first-appearance order. The earlier
# submitted PROTEUS-CHILI reference stays black and the current run keeps its
# highlight colour, so neither is remapped.
COMMUNITY_PAL = [ix.CORE['solar'], ix.CORE['azure'], ix.CORE['verdant'],
                 ix.CORE['ocean'], ix.CORE['magma'], ix.CORE['fog']]
_KEEP_COLORS = {'#000000', '#9C27B0'}


def _recolor_community(c):
    """Remap community-model curve colours onto the PROTEUS extended palette."""
    order = []
    for cv in c['f1']:
        col = cv['color']
        if col not in _KEEP_COLORS and col not in order:
            order.append(col)
    remap = {old: COMMUNITY_PAL[i % len(COMMUNITY_PAL)] for i, old in enumerate(order)}
    for key in ('f1', 'f9'):
        for cv in c[key]:
            if cv['color'] in remap:
                cv['color'] = remap[cv['color']]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _panel(ax, text, x=0.04, y=0.95, ha='left', va='top', fontweight='bold', fs=9):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            fontweight=fontweight,
            bbox=dict(boxstyle='round,pad=0.13', facecolor='white', alpha=0.9, edgecolor='none'))


def plot_chili(c, outdir):
    """Single row of three panels sharing the melt-fraction axis: (a) solidification
    time, (b) surface temperature, (c) rheological-front radius, with a shared
    legend."""
    from matplotlib.lines import Line2D
    m = c['meta']
    _recolor_community(c)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.30, top=0.95, wspace=0.36)
    ax_m, ax_t, ax_r = axes
    neutral = ix.NEUTRALS.get('ink', '#222')

    # (a) solidification time against melt fraction, so all three panels share the
    # melt-fraction abscissa (decreasing left to right as the mantle solidifies).
    for cv in c['f1']:
        ax_m.plot(np.asarray(cv['phi'], float) * 100.0, cv['t'], cv['ls'],
                  color=cv['color'], lw=cv['lw'], alpha=0.85)
    ax_m.plot(np.asarray(c['pe_f1']['phi'], float) * 100.0, c['pe_f1']['t'], '-',
              color=m['ns_color'], lw=m['ns_lw'], zorder=10)
    ax_m.plot(np.asarray(c['pv_f1']['phi'], float) * 100.0, c['pv_f1']['t'], '--',
              color=m['ns_color'], lw=m['ns_lw'], zorder=10)
    ax_m.set_yscale('log')
    ax_m.set_xlim(100, 0)
    ax_m.set_ylim(1e-3, 1e1)
    ax_m.set_xlabel(r'Melt fraction [%]')
    ax_m.set_ylabel(r'Time [$\mathrm{Myr}$]')
    ax_m.xaxis.set_major_locator(FixedLocator([100, 50, 0]))
    ax_m.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    # Earth/Venus line-style key as a proper legend, below the panel title with
    # its top at 10^0 Myr (0.75 of the log axis), where no curve passes.
    ev = [Line2D([0], [0], color=neutral, ls='-', lw=1.4, label='Earth'),
          Line2D([0], [0], color=neutral, ls='--', lw=1.4, label='Venus')]
    ax_m.legend(handles=ev, loc='upper left', bbox_to_anchor=(0.0, 0.76),
                fontsize=6.6, handlelength=1.8, borderpad=0.3, framealpha=0.85)
    _panel(ax_m, '(a) Solidification time')

    # (b) surface temperature, (c) rheological-front radius vs melt fraction
    for cv in c['f9']:
        if cv['ts'] is not None:
            ax_t.plot(cv['ts_phi'], cv['ts'], cv['ls'], color=cv['color'], lw=cv['lw'], alpha=0.85)
        if cv['r_km'] is not None:
            ax_r.plot(cv['r_phi'], cv['r_km'], cv['ls'], color=cv['color'], lw=cv['lw'], alpha=0.85)
    pe = c['pe_f9']
    ax_t.plot(pe['ts_phi'], pe['ts'], '-', color=m['ns_color'], lw=m['ns_lw'], zorder=10)
    if pe['r_km'] is not None:
        ax_r.plot(pe['r_phi'], pe['r_km'], '-', color=m['ns_color'], lw=m['ns_lw'], zorder=10)
    ax_r.axhline(m['R_cmb_km'], color=ix.NEUTRALS['graphite'], ls='--', lw=1.0)
    ax_r.text(0.96, m['R_cmb_km'] - 120, 'core-mantle boundary', transform=ax_r.get_yaxis_transform(),
              fontsize=6.0, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    # present-day solid Earth surface radius, the end state the rheological
    # front should approach once the interior structure is recontracted
    ax_r.axhline(6371.0, color=ix.NEUTRALS['graphite'], ls=':', lw=1.0)
    # Placed below the line and at the left, where the rheological front is
    # near the core-mantle boundary, so the label clears the panel title and
    # every model curve.
    ax_r.text(0.04, 6371.0 - 95, 'present-day solid Earth', transform=ax_r.get_yaxis_transform(),
              fontsize=6.0, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    for a in (ax_t, ax_r):
        a.set_xlim(100, 0)   # melt fraction decreases as the mantle solidifies (time runs left to right)
        # Community curves are volume-weighted; PROTEUS reports the mass-weighted
        # global melt fraction, so the axis label stays unit-neutral and the
        # weighting is noted in the caption (the two agree to a few per cent here).
        a.set_xlabel(r'Melt fraction [%]')
        a.xaxis.set_major_locator(FixedLocator([100, 50, 0]))
    ax_t.set_ylabel(r'$T_\mathrm{surf}$ [$\mathrm{K}$]')
    ax_r.set_ylabel(r'$R_\mathrm{rheo}$ [$\mathrm{km}$]')
    _panel(ax_t, '(b) Surface temperature')
    _panel(ax_r, '(c) Rheological front')

    # shared legend just below the row. Order the models so the submitted
    # PROTEUS-CHILI entry is directly above the current run in the last
    # column (the legend fills column-major over two rows).
    others = [cv for cv in c['f9'] if cv['label'] != 'PROTEUS CHILI']
    chili = [cv for cv in c['f9'] if cv['label'] == 'PROTEUS CHILI']
    handles = [Line2D([0], [0], color=cv['color'], lw=max(cv['lw'], 1.4), ls=cv['ls'], label=cv['label'])
               for cv in (others + chili)]
    handles.append(Line2D([0], [0], color=m['ns_color'], lw=m['ns_lw'], label=m['ns_label']))
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=6.6,
               bbox_to_anchor=(0.5, 0.12), frameon=False, columnspacing=1.3, handlelength=1.6)
    for a in axes:
        ix.set_mono_ticks(a)
    out = os.path.join(outdir, 'chili_validation.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    print('wrote', os.path.relpath(out, HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    c = _load()
    plot_chili(c, args.outdir)


if __name__ == '__main__':
    main()
