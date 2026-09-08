"""Literature comparison of Zalmoxis interior-only mass-radius relations.

Overlays the production Zalmoxis solver (PALEOS equation of state) against four
independent references, all for bare, condensed (no-atmosphere) rocky planets:

1. Zalmoxis with the built-in Seager et al. (2007) equation of state.
2. MAGRATHEA (Huang, Rice & Steffen 2022), an independent standalone code.
3. Noack & Lasbleis (2020) analytic scaling relations (PROTEUS dummy module),
   calibrated for 0.8-2 M_earth and extrapolated above.
4. Zeng et al. (2016) PREM-based tabulated mass-radius relations.

The figure has two rows: the mass-radius relations coloured by core-mass
fraction, and the fractional radius residual of each model against the Zeng
(2016) baseline with a +/- 5 per cent acceptance band.

All inputs are assembled in this repository (Zalmoxis grids and interior
profiles under zalmoxis_grids/, MAGRATHEA output, Zeng tables, and cached
Noack iron fractions), so the figures reproduce from the repository alone with
only numpy, pandas, and matplotlib.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

# PROTEUS visual identity, vendored under style/ at the repo root so the figure
# reproduces from the repository alone.
_STYLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use('white', font='mono')
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'legend.fontsize': 6.8, 'legend.handlelength': 1.8, 'legend.handletextpad': 0.5,
    'legend.borderpad': 0.3, 'legend.labelspacing': 0.25, 'legend.framealpha': 0.9,
    'xtick.labelsize': 7.6, 'ytick.labelsize': 7.6, 'lines.linewidth': 1.5,
    'axes.linewidth': 0.9, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

HERE = os.path.dirname(os.path.abspath(__file__))
# output directory: first command-line argument, else ./figures
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures')
# Physics-code outputs are assembled in-repo so the figures reproduce from the
# repository alone (no PROTEUS/Zalmoxis checkout needed): mass-radius grid
# summaries (zalmoxis_grids/<grid>.csv) and interior profiles
# (zalmoxis_grids/profiles/).
GRIDS = os.path.join(HERE, 'zalmoxis_grids')
EARTH_RADIUS_KM = 6371.0

# Core-mass fractions swept, with display labels and strata colours.
CMF_VALUES = [0.01, 0.325, 0.5, 0.999]
CMF_LABEL = {0.01: 'rock', 0.325: 'Earth-like', 0.5: 'Fe-rich', 0.999: 'iron'}
CMF_COLOUR = {0.01: ix.STRATA['cobalt'], 0.325: ix.STRATA['amber'],
              0.5: ix.STRATA['magma'], 0.999: ix.STRATA['ink']}

# Zeng (2016) tables map onto three of the four core-mass fractions.
ZENG_FILES = {
    0.999: 'zeng2016_massradiusFe.txt',
    0.325: 'zeng2016_massradiusEarthlikeRocky.txt',
    0.01: 'zeng2016_massradiusmgsio3.txt',
}


def _load_grid(name):
    """Load a Zalmoxis grid_summary.csv as {cmf: (mass, radius)} for converged points."""
    path = os.path.join(GRIDS, f'{name}.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    conv = df[df['converged']]
    dropped = sorted(set(df['planet_mass']) - set(conv['planet_mass']))
    if dropped:
        print(f'WARNING: {name}: dropped non-converged masses {dropped}')
    df = conv
    out = {}
    for cmf in CMF_VALUES:
        sub = df[np.isclose(df['core_mass_fraction'], cmf)].sort_values('planet_mass')
        if len(sub):
            out[cmf] = (sub['planet_mass'].to_numpy(), sub['R_earth'].to_numpy())
    return out


def _load_magrathea():
    """Load MAGRATHEA mode-2 output text files as {cmf: (mass, radius)}."""
    out = {}
    for cmf in CMF_VALUES:
        path = os.path.join(HERE, 'magrathea', f'magrathea_cmf_{cmf}.txt')
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path, comments='#')
        if data.size:
            data = np.atleast_2d(data)
            out[cmf] = (data[:, 0], data[:, 1])
    return out


def _load_zeng():
    """Load the Zeng (2016) tables as {cmf: (mass, radius)} within the plotted range."""
    out = {}
    for cmf, fname in ZENG_FILES.items():
        path = os.path.join(HERE, 'reference_data', fname)
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path)
        m, r = data[:, 0], data[:, 1]
        # Keep Zeng samples bracketing the 0.5-20 M_Earth model range (one below
        # 0.5 and one above 20) so the residual can be interpolated across the
        # whole range; markers outside the plotted x-range are clipped anyway.
        keep = (m >= 0.2) & (m <= 30.0)
        out[cmf] = (m[keep], r[keep])
    return out


# Planet iron weight fractions per core-mass fraction, from the PROTEUS
# iron-fraction helper (proteus.utils.structure_estimate.iron_fractions, mass
# mode). Cached so the figure reproduces without a PROTEUS checkout; the Noack
# & Lasbleis relation below is analytic in x_Fe.
NOACK_X_FE = {0.01: 0.08522005176888357, 0.325: 0.3762863989333297,
              0.5: 0.5379899251357998, 0.999: 0.9990759798502716}


def _noack_lasbleis(cmf, masses):
    """Noack & Lasbleis (2020) Eq. 5 radius [R_earth] for a core-mass fraction.

    R_p [km] = (7030 - 1840 x_Fe) (M/M_earth)^0.282, with x_Fe the planet iron
    weight fraction (cached in ``NOACK_X_FE``).
    """
    x_fe = NOACK_X_FE[cmf]
    r_km = (7030.0 - 1840.0 * x_fe) * masses**0.282
    return r_km / EARTH_RADIUS_KM


def _interp_to(mass_ref, radius_ref, mass_query):
    """Log-log interpolate a reference curve onto query masses (NaN outside range)."""
    lo, hi = mass_ref.min(), mass_ref.max()
    out = np.interp(np.log10(mass_query), np.log10(mass_ref), np.log10(radius_ref))
    out = 10.0**out
    out[(mass_query < lo) | (mass_query > hi)] = np.nan
    return out


def _mass_axis(ax):
    """Log mass axis with labelled ticks at the sampled decade/sub-decade values."""
    ax.set_xscale('log')
    ax.set_xlim(0.45, 22.0)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _inside_label(ax, text, x=0.035, y=0.95, ha='left', va='top'):
    """Bold panel label inside the axes on a white patch (top-left by default)."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'))


def plot_mass_radius(paleos_cold, seager, magrathea, zeng):
    """Two-panel mass-radius comparison: M-R families (a) and residual vs Zeng (b).

    Single-column layout: the two panels stack vertically and share the mass axis.
    """
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 5.4), sharex=True)

    # --- (a) M-R: PALEOS (solid) + MAGRATHEA (dash-dot) + Zeng (circles) ---
    for cmf in CMF_VALUES:
        c = CMF_COLOUR[cmf]
        if cmf in paleos_cold:
            m, r = paleos_cold[cmf]
            ax[0].plot(m, r, '-', lw=2.0, color=c)
        if cmf in magrathea:
            m, r = magrathea[cmf]
            ax[0].plot(m, r, '-.', lw=1.1, color=c)
        if cmf in zeng:
            m, r = zeng[cmf]
            ax[0].plot(m, r, ls='none', marker='o', ms=3, mfc='none', mec=c, mew=0.9)
        # Noack & Lasbleis analytic scaling, thin solid, drawn only over its
        # 0.8-2 M_E calibration range. Force 0.8 and 2.0 into the grid so the
        # segment ends land exactly on the calibration limits.
        ml = np.unique(np.concatenate([np.linspace(0.5, 20.0, 60), [0.8, 2.0]]))
        rl = _noack_lasbleis(cmf, ml)
        cal = (ml >= 0.8) & (ml <= 2.0)
        ax[0].plot(ml[cal], rl[cal], color=c, lw=0.8, alpha=0.8)

    ax[0].set_yscale('log')
    _mass_axis(ax[0])
    ax[0].set_ylim(0.6, 2.55)
    ax[0].set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5])
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax[0].set(ylabel=r'Planet radius [$R_\mathrm{Earth}$]')  # x-axis shared with panel (b) below
    ax[0].grid(True, which='both', alpha=0.25)
    cmf_handles = [plt.Line2D([], [], color=CMF_COLOUR[c], lw=2.5,
                              label=f'{CMF_LABEL[c]} (CMF {c:g})') for c in CMF_VALUES]
    # Single colour legend in the bottom-right; the line styles (PALEOS solid,
    # MAGRATHEA dash-dot, Noack-Lasbleis thin, Zeng circles) are given in the
    # caption and keyed in panel (b).
    leg_cmf = ax[0].legend(handles=cmf_handles, loc='lower right', fontsize=7,
                           frameon=True, facecolor='white', framealpha=0.85,
                           edgecolor='none')
    ax[0].add_artist(leg_cmf)
    # second legend keying the line styles (models), centre-left
    style_handles_a = [
        plt.Line2D([], [], color='0.25', ls='-', lw=2.0, label='Zalmoxis (PALEOS)'),
        plt.Line2D([], [], color='0.25', ls='-.', lw=1.1, label='MAGRATHEA'),
        plt.Line2D([], [], color='0.25', ls='-', lw=0.9, label='Noack & Lasbleis'),
        plt.Line2D([], [], color='0.25', ls='none', marker='o', ms=3, mfc='none',
                   label='Zeng et al. (2016)'),
    ]
    ax[0].legend(handles=style_handles_a, loc='upper left',
                 bbox_to_anchor=(0.03, 0.86), fontsize=6.5,
                 frameon=True, facecolor='white', framealpha=0.85, edgecolor='none')
    ix.set_mono_ticks(ax[0])
    _inside_label(ax[0], '(a) Mass-radius')

    # --- (b) residual vs Zeng, +/-5% band, all models ---
    models = [
        (paleos_cold, 'Zalmoxis (PALEOS)', '-', 2.0, None),
        (seager, 'Zalmoxis (Seager07)', '--', 1.2, None),
        (magrathea, 'MAGRATHEA', '-.', 1.2, 's'),
    ]
    for cmf in ZENG_FILES:
        if cmf not in zeng:
            continue
        c = CMF_COLOUR[cmf]
        zm, zr = zeng[cmf]
        for data, _label, ls, lw, mk in models:
            if cmf not in data:
                continue
            m, r = data[cmf]
            dev = 100.0 * (r - _interp_to(zm, zr, m)) / _interp_to(zm, zr, m)
            ax[1].plot(m, dev, ls=ls, lw=lw, color=c, marker=mk, ms=2.5, mfc='none')

    ax[1].axhspan(-5, 5, color='0.88', zorder=0, lw=0)
    ax[1].axhline(0, color='0.4', lw=0.8)
    _mass_axis(ax[1])
    ax[1].set_ylim(-8, 4)
    ax[1].set_yticks([-8, -6, -4, -2, 0, 2, 4])
    ax[1].set(xlabel=r'Planet mass [$M_\mathrm{Earth}$]', ylabel=r'$\Delta R/R$ vs Zeng 2016 [%]')
    ax[1].grid(True, which='both', alpha=0.25)
    style_handles = [plt.Line2D([], [], color='0.3', ls=ls, lw=lw, marker=mk, ms=2.5,
                                mfc='none', label=lab) for (_d, lab, ls, lw, mk) in models]
    ax[1].legend(handles=style_handles, loc='lower right', fontsize=7, frameon=True,
                 facecolor='white', framealpha=0.85, edgecolor='none')
    ix.set_mono_ticks(ax[1])
    _inside_label(ax[1], '(b) Radius residual')

    fig.tight_layout(pad=0.6)
    out = os.path.join(OUTDIR, 'zalmoxis_literature_mr.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out)


EARTH_MASS_KG = 5.972e24
EARTH_RADIUS_M = 6.371e6
GRAV_CONST = 6.674e-11
PROFILE_MASSES = [1.0, 5.0, 10.0, 20.0]
PROFILE_COLOUR = {1.0: ix.STRATA['cobalt'], 5.0: ix.STRATA['amber'],
                  10.0: ix.STRATA['magma'], 20.0: ix.STRATA['plum']}


def _zalmoxis_profile(mass):
    """Load a cold-PALEOS interior profile (CMF 0.325) as r/R, rho [g/cc], P [GPa], g."""
    path = os.path.join(GRIDS, 'profiles', f'paleos_cold_cmf0.325_M{mass}.csv')
    df = pd.read_csv(path, comment='#')
    df = df[df['density_kg_m3'] > 0]  # trim the zero-density vacuum padding
    r = df['radius_m'].to_numpy()
    return (r / r.max(), df['density_kg_m3'].to_numpy() / 1e3,
            df['pressure_Pa'].to_numpy() / 1e9, df['gravity_m_s2'].to_numpy())


def _magrathea_profile(mass):
    """Load a MAGRATHEA Mode-0 profile as r/R, rho [g/cc], P [GPa], g (from M(r))."""
    path = os.path.join(HERE, 'magrathea', 'profiles',
                        f'magrathea_profile_M{mass}_cmf0.325.txt')
    d = np.genfromtxt(path, skip_header=1, usecols=(1, 2, 3, 4))  # radius, P, M, rho
    r_re, p_gpa, m_me, rho = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    r_m = r_re * EARTH_RADIUS_M
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(r_m > 0, GRAV_CONST * m_me * EARTH_MASS_KG / r_m**2, 0.0)
    return r_re / r_re.max(), rho, p_gpa, g


def plot_profiles():
    """Interior profiles (rho, P, g vs normalised radius) at 1/5/10 M_earth,
    Zalmoxis (solid) against MAGRATHEA (dashed), for the Earth-like CMF."""
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.4))
    for mass in PROFILE_MASSES:
        c = PROFILE_COLOUR[mass]
        zr, zrho, zp, zg = _zalmoxis_profile(mass)
        mr, mrho, mp, mg = _magrathea_profile(mass)
        for j, (zy, my) in enumerate([(zrho, mrho), (zp, mp), (zg, mg)]):
            ax[j].plot(zr, zy, '-', lw=1.8, color=c)
            ax[j].plot(mr, my, '--', lw=1.1, color=c)

    ax[0].set(ylabel=r'Density [g cm$^{-3}$]')
    ax[1].set(ylabel='Pressure [GPa]')
    ax[2].set(ylabel=r'Gravity [m s$^{-2}$]')
    # Density and pressure plotted with high values up (not inverted).
    # Descriptive subtitles in each panel's empty corner: density and pressure
    # fall toward the surface so their labels go top-right; gravity peaks at
    # mid-radius so its label goes bottom-right.
    labels = ['(a) Density', '(b) Pressure', '(c) Gravity']
    label_pos = [dict(x=0.96, ha='right'), dict(x=0.96, ha='right'),
                 dict(x=0.96, y=0.05, ha='right', va='bottom')]
    for j, a in enumerate(ax):
        a.set_xlabel(r'$r / R_\mathrm{p}$')
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.5, 1.0])
        a.set_xticks([0.25, 0.75], minor=True)
        a.grid(True, alpha=0.25)
        ix.set_mono_ticks(a)
        _inside_label(a, labels[j], **label_pos[j])

    # one shared legend in a row above the panels, clear of all data
    mass_handles = [plt.Line2D([], [], color=PROFILE_COLOUR[m], lw=2,
                               label=rf'{m:g} $M_\mathrm{{Earth}}$') for m in PROFILE_MASSES]
    src_handles = [plt.Line2D([], [], color='0.3', lw=1.8, label='Zalmoxis'),
                   plt.Line2D([], [], color='0.3', lw=1.1, ls='--', label='MAGRATHEA')]
    fig.legend(handles=mass_handles + src_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.10), ncol=6, frameon=False, fontsize=7.5,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(pad=0.6)
    out = os.path.join(OUTDIR, 'zalmoxis_literature_profiles.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out)


def main():
    paleos_cold = _load_grid('se_mr_paleos_cold')
    seager = _load_grid('se_mr_seager')
    magrathea = _load_magrathea()
    zeng = _load_zeng()
    plot_mass_radius(paleos_cold, seager, magrathea, zeng)
    plot_profiles()


if __name__ == '__main__':
    main()
