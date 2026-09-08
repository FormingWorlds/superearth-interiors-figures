"""First-principles analytic validation of the Zalmoxis structure solver.

Zalmoxis integrates the three coupled hydrostatic-structure ODEs

    dM/dr = 4 pi r^2 rho(r)
    dg/dr = 4 pi G rho(r) - 2 g(r) / r
    dP/dr = - rho(r) g(r)

outward from the centre, closed by a density law rho(P, T). This script
verifies the numerical machinery against problems with exact closed-form
solutions, independent of any tabulated equation of state. Each test
injects a known density law through the same hook the production solver
uses (``calculate_mixed_density``) and compares the integrated profiles
against the analytic result.

Three figures are produced, each saved as vector PDF:

1. ``spheres.pdf`` - constant-density sphere (top row) and a two-layer
   core + mantle sphere (bottom row), sharing the radius axis. Tests the
   ODE integrator against M ~ r^3, g ~ r, P parabolic, and the layer
   assignment and Gauss's law across a density discontinuity.
2. ``lane_emden.pdf`` - n = 1 polytrope, P = K rho^2, against the exact
   solution rho(r) = rho_c sin(xi) / xi. The top row runs the integrator
   alone (injected density, prescribed central pressure); the bottom row
   runs the full production solver (outer mass-radius search, Picard
   density iteration, Brent central-pressure root-find, and the production
   ``Analytic:`` dispatch via the registered ``polytrope_n1`` material),
   verifying the entire solver chain at its production tolerance.
3. ``conservation_convergence.pdf`` - one four-panel row: pointwise
   Gauss's law and hydrostatic-balance residuals for the constant-density
   sphere (a, b), and the integrator-tolerance convergence for the
   polytrope against the exact uniform sphere (c) with the error's
   independence from the output grid (d).

Notes
-----
By default the figures are replotted from the committed array cache under
``data/`` (numpy + matplotlib only), so they reproduce from the repository
alone. Pass ``--recompute`` to re-run Zalmoxis and refresh the cache; that path
imports the installed ``zalmoxis`` package and must run in the environment where
it is installed. Provenance is recorded in the repository README.

Usage
-----
    python plot_first_principles_validation.py [--outdir DIR] [--recompute]

Figures are written to ``figures/`` next to this script by default; the plotted
arrays live in ``data/``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Zalmoxis is needed only to (re)compute the validation arrays. The committed
# cache under data/ lets the figures reproduce from the repository alone, so
# the import is lazy: a plain numpy + matplotlib environment can replot from
# the cache without a Zalmoxis checkout. Only --recompute requires the package.
G = earth_mass = earth_radius = None
LayerMixture = solve_structure = None
_HAVE_ZALMOXIS = False


def _import_zalmoxis():
    """Bind the Zalmoxis symbols used by the compute path; raise if unavailable."""
    global G, earth_mass, earth_radius, LayerMixture, solve_structure, _HAVE_ZALMOXIS
    if _HAVE_ZALMOXIS:
        return
    from zalmoxis.constants import G as _G, earth_mass as _em, earth_radius as _er
    from zalmoxis.mixing import LayerMixture as _LM
    from zalmoxis.structure_model import solve_structure as _ss
    G, earth_mass, earth_radius = _G, _em, _er
    LayerMixture, solve_structure = _LM, _ss
    _HAVE_ZALMOXIS = True

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# PROTEUS visual identity, vendored under style/ at the repo root.
import sys  # noqa: E402

# Vendored under style/ at the repo root so the figures reproduce from the
# repository alone.
_STYLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

# Mono figure font throughout: equation variables render in true italic with
# roman \mathrm subscripts.
ix.use('white', font='mono')

# Shared colours (strata palette) and style.
C_NUM = ix.STRATA['cobalt']   # numerical solution
C_ANA = ix.STRATA['amber']    # analytic reference
C_CMB = ix.NEUTRALS['fog']    # core-mantle boundary marker

plt.rcParams.update(
    {
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'legend.handlelength': 1.1,
        'legend.handletextpad': 0.4,
        'legend.borderpad': 0.25,
        'legend.labelspacing': 0.2,
        'legend.framealpha': 0.8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 9,
        'lines.linewidth': 2.0,
        'axes.linewidth': 0.9,
        'figure.titlesize': 11.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    }
)

# Single-row figure geometry, sized close to the rendered \textwidth so the
# fonts are not shrunk by \includegraphics scaling.
_ROW4 = (7.6, 2.2)  # four panels in a row
_ROW2 = (7.6, 2.5)  # two panels in a row

# Physics-code outputs are assembled in-repo so the figures reproduce from the
# repository alone (no Zalmoxis checkout needed). Each figure's plotted arrays
# are cached, in the units they are plotted in, under data/<figure>.npz.
HERE_FP = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE_FP, 'data')


def _load_or_compute(name, compute_fn, recompute=False):
    """Return a figure's plotted arrays, from the committed cache or by running
    the Zalmoxis compute path and writing the cache.

    The cache (``data/<name>.npz``) holds every array the figure plots, in the
    units it is plotted in, so the figures reproduce from the repository alone.
    Pass ``recompute=True`` (CLI ``--recompute``) to re-run Zalmoxis and refresh
    the cache; this needs the Zalmoxis package importable.
    """
    path = os.path.join(DATA_DIR, name + '.npz')
    if os.path.exists(path) and not recompute:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    _import_zalmoxis()
    data = compute_fn()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(path, **data)
    logger.info('  wrote cache %s (%d arrays)', os.path.relpath(path, HERE_FP), len(data))
    return data


def _thin_xticks(axes, nbins=4):
    """Reduce x-axis tick density so labels do not crowd narrow panels."""
    for a in np.atleast_1d(axes).ravel():
        a.xaxis.set_major_locator(MaxNLocator(nbins=nbins))


def _more_yticks(axes, nbins=5):
    """Increase y-tick density on linear-scale panels."""
    for a in np.atleast_1d(axes).ravel():
        if a.get_yscale() == 'linear':
            a.yaxis.set_major_locator(MaxNLocator(nbins=nbins))


def _share_column_ylabels(fig, ax, columns):
    """Replace the duplicate per-row y-label of each given column (top and bottom
    rows show the same quantity) with one label centred across both rows, placed
    just left of the column's tick labels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for j in columns:
        text = ax[0, j].get_ylabel()
        ax[0, j].set_ylabel('')
        ax[1, j].set_ylabel('')
        ticks = [t for t in (*ax[0, j].get_yticklabels(), *ax[1, j].get_yticklabels())
                 if t.get_text()]
        left = min(t.get_window_extent(r).x0 for t in ticks)
        x = inv.transform((left, 0))[0] - 0.012
        pt = ax[0, j].get_position()
        pb = ax[1, j].get_position()
        y = 0.5 * (pt.y1 + pb.y0)
        fig.text(x, y, text, rotation='vertical', va='center', ha='center',
                 fontsize=plt.rcParams['axes.labelsize'])


def _panel_label(ax, text, x=0.04, y=0.93, ha='left', alpha=0.9):
    """Place a panel label inside the axes to save the title's vertical space."""
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va='top', fontweight='bold',
        fontsize=plt.rcParams['axes.titlesize'],
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=alpha, edgecolor='none'),
    )


def _log_tick(x, pos):
    """Scientific-notation label for an individual (possibly sub-decade) log tick."""
    if x <= 0:
        return ''
    e = int(math.floor(math.log10(x)))
    m = x / 10.0**e
    if abs(m - 1.0) < 1e-6:
        return rf'$10^{{{e}}}$'
    return rf'${m:g}\times10^{{{e}}}$'


def _axis_extents(axes):
    """Tighten limits to the data and label the extremes, so the full reach of
    each axis (top/bottom on y, ends on x) is readable from the ticks."""
    for ax in np.atleast_1d(axes).ravel():
        if ax.get_yscale() == 'log':
            lo0, hi0 = ax.get_ylim()  # autoscaled positive range (ignores zeros)
            if not (np.isfinite([lo0, hi0]).all() and lo0 > 0 and hi0 > lo0):
                continue
            lo = 10.0 ** math.floor(math.log10(lo0))
            hi = 10.0 ** math.ceil(math.log10(hi0))
            ax.set_ylim(lo, hi)
            ndec = int(round(math.log10(hi / lo)))
            if ndec >= 3:
                step = max(1, round(ndec / 4))
                ticks = [lo * 10.0 ** (step * k) for k in range(ndec // step + 1)]
                if ticks[-1] < hi * (1 - 1e-9):
                    ticks.append(hi)
                ax.set_yticks(ticks)
            else:
                subs = (1.0, 2.0, 5.0) if ndec <= 1 else (1.0, 3.0)
                ticks = []
                dec = lo
                while dec <= hi * (1 + 1e-9):
                    for sub in subs:
                        t = dec * sub
                        if lo * (1 - 1e-9) <= t <= hi * (1 + 1e-9):
                            ticks.append(t)
                    dec *= 10.0
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(FuncFormatter(_log_tick))
        else:
            dy0, dy1 = ax.dataLim.intervaly
            if not (np.isfinite([dy0, dy1]).all() and dy1 > dy0):
                continue
            lo, hi = dy0, dy1
            span = hi - lo
            if -0.05 * span < lo < 0:
                lo = 0.0
            if hi > 0:
                mag = 10.0 ** math.floor(math.log10(hi))
                for nice in (1, 2, 2.5, 5, 10):
                    cand = nice * mag
                    if 0.5 * cand <= hi <= 1.005 * cand:
                        hi = min(hi, cand)
                        break
            ticks = list(MaxNLocator(nbins=5).tick_values(lo, hi))
            ax.set_ylim(ticks[0], ticks[-1])
            ax.set_yticks(ticks)


def _uniform_xlim(axes):
    """Set one tight x-range across all panels so the columns share it and the
    maximum radius is reached and labelled."""
    axes = np.atleast_1d(axes).ravel()
    x0 = min(a.dataLim.intervalx[0] for a in axes if np.isfinite(a.dataLim.intervalx[0]))
    x1 = max(a.dataLim.intervalx[1] for a in axes if np.isfinite(a.dataLim.intervalx[1]))
    for a in axes:
        a.set_xlim(x0, x1)
        a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))


# ----------------------------------------------------------------------------
# Density hooks and solver wrappers
# ----------------------------------------------------------------------------


def _constant_density_mock(rho):
    """Return a density hook that ignores its inputs and returns ``rho``.

    Parameters
    ----------
    rho : float
        Constant mass density [kg m^-3].

    Returns
    -------
    callable
        A drop-in replacement for ``calculate_mixed_density`` returning
        ``None`` at non-physical pressures and ``rho`` otherwise.
    """

    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        return rho

    return _mock


def _two_layer_density_mock(rho_core, rho_mantle):
    """Return a density hook dispatching on layer via the component name."""

    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        if mixture.components[0] == 'mock:core':
            return rho_core
        return rho_mantle

    return _mock


def _polytrope_n1_density_mock(K):
    """Return the n = 1 polytropic density hook ``rho(P) = sqrt(P / K)``."""

    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        return math.sqrt(pressure / K)

    return _mock


def _solve_polytrope_full_chain(M_target, num_layers=600, P_surf=1e2):
    """Run the full production solver on the n = 1 polytrope.

    Drives ``zalmoxis.solver.main``, which runs the outer mass-radius
    search, the Picard density iteration, the Brent central-pressure
    root-find, and the production ``Analytic:`` equation-of-state dispatch,
    using the registered ``polytrope_n1`` verification material.

    Parameters
    ----------
    M_target : float
        Target planet mass [kg]; the solver finds the central pressure and
        radius that reproduce it.
    num_layers : int, optional
        Number of radial grid points.
    P_surf : float, optional
        Target surface pressure [Pa] defining the planet surface.

    Returns
    -------
    dict
        The solver result dictionary (radii, density, mass_enclosed,
        gravity, pressure, p_center, converged, ...).
    """
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.solver import main as solver_main

    config = {
        'planet_mass': M_target,
        'core_mass_fraction': 0.5,
        'mantle_mass_fraction': 0,
        'temperature_mode': 'isothermal',
        'surface_temperature': 300,
        'center_temperature': 2000,
        'temp_profile_file': '',
        'layer_eos_config': {'core': 'Analytic:polytrope_n1', 'mantle': 'Analytic:polytrope_n1'},
        'rock_solidus': 'Stixrude14-solidus',
        'rock_liquidus': 'Stixrude14-liquidus',
        'mushy_zone_factor': 1.0,
        'mushy_zone_factors': {'PALEOS:iron': 1.0, 'PALEOS:MgSiO3': 1.0, 'PALEOS:H2O': 1.0},
        'condensed_rho_min': 322.0,
        'condensed_rho_scale': 50.0,
        'binodal_T_scale': 50.0,
        'num_layers': num_layers,
        'max_iterations_outer': 100,
        'tolerance_outer': 3e-3,
        'max_iterations_inner': 100,
        'tolerance_inner': 1e-4,
        'relative_tolerance': 1e-10,
        'absolute_tolerance': 1e-12,
        'maximum_step': 250000,
        'adaptive_radial_fraction': 0.98,
        'max_center_pressure_guess': 10e12,
        'target_surface_pressure': P_surf,
        'pressure_tolerance': 1e6,
        'max_iterations_pressure': 200,
        'outer_solver': 'newton',
        'data_output_enabled': False,
        'plotting_enabled': False,
        'verbose': False,
        'iteration_profiles_enabled': False,
    }
    input_dir = os.path.join(get_zalmoxis_root(), 'input')
    return solver_main(config, load_material_dictionaries(), None, input_dir)


def _solve(mock_fn, radii, P_center, component='mock:uniform', rtol=1e-10, atol=1e-12):
    """Integrate the structure ODEs with an injected density hook.

    Parameters
    ----------
    mock_fn : callable
        Density hook patched in for ``calculate_mixed_density``.
    radii : numpy.ndarray
        Radial grid [m] from centre to surface.
    P_center : float
        Central pressure [Pa] used as the inner boundary condition.
    component : str, optional
        Single-layer component label passed to the mixture.
    rtol, atol : float, optional
        Relative and absolute integrator tolerances.

    Returns
    -------
    tuple of numpy.ndarray
        Enclosed mass, gravity, and pressure on ``radii``.
    """
    R = radii[-1]
    layer_mixtures = {'core': LayerMixture(components=[component], fractions=[1.0])}
    with patch('zalmoxis.structure_model.calculate_mixed_density', mock_fn):
        mass, gravity, pressure = solve_structure(
            layer_mixtures=layer_mixtures,
            cmb_mass=1e30,
            core_mantle_mass=1e30,
            radii=radii,
            adaptive_radial_fraction=0.98,
            relative_tolerance=rtol,
            absolute_tolerance=atol,
            maximum_step=R / 10,
            material_dictionaries={},
            interpolation_cache={},
            y0=[0, 0, P_center],
            solidus_func=None,
            liquidus_func=None,
        )
    return mass, gravity, pressure


def _solve_two_layer(rho_core, rho_mantle, cmb_mass, M_total, radii, P_center):
    """Integrate a two-layer (core + mantle) constant-density sphere."""
    core_mix = LayerMixture(components=['mock:core'], fractions=[1.0])
    mantle_mix = LayerMixture(components=['mock:mantle'], fractions=[1.0])
    R = radii[-1]
    with patch('zalmoxis.structure_model.calculate_mixed_density', _two_layer_density_mock(rho_core, rho_mantle)):
        mass, gravity, pressure = solve_structure(
            layer_mixtures={'core': core_mix, 'mantle': mantle_mix},
            cmb_mass=cmb_mass,
            core_mantle_mass=M_total,
            radii=radii,
            adaptive_radial_fraction=0.98,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
            maximum_step=R / 10,
            material_dictionaries={},
            interpolation_cache={},
            y0=[0, 0, P_center],
            solidus_func=None,
            liquidus_func=None,
        )
    return mass, gravity, pressure


# ----------------------------------------------------------------------------
# Analytic solutions
# ----------------------------------------------------------------------------


def _analytic_uniform_sphere(rho, P_center, radii):
    """Closed-form profiles for a constant-density self-gravitating sphere."""
    M = (4.0 / 3.0) * math.pi * rho * radii**3
    g = (4.0 / 3.0) * math.pi * G * rho * radii
    P = P_center - (2.0 / 3.0) * math.pi * G * rho**2 * radii**2
    return M, g, P


def _two_layer_central_pressure(rho_core, rho_mantle, R_cmb, R_total, M_cmb):
    """Exact central pressure of a two-layer constant-density sphere."""
    r = np.linspace(0, R_total, 10000)
    rho = np.where(r <= R_cmb, rho_core, rho_mantle)
    M = np.where(
        r <= R_cmb,
        (4.0 / 3.0) * math.pi * rho_core * r**3,
        M_cmb + (4.0 / 3.0) * math.pi * rho_mantle * (r**3 - R_cmb**3),
    )
    g = np.zeros_like(r)
    g[1:] = G * M[1:] / r[1:] ** 2
    return float(np.trapezoid(rho * g, r))


def _analytic_two_layer_mass(rho_core, rho_mantle, R_cmb, M_cmb, radii):
    """Piecewise enclosed-mass profile of a two-layer sphere."""
    return np.where(
        radii <= R_cmb,
        (4.0 / 3.0) * math.pi * rho_core * radii**3,
        M_cmb + (4.0 / 3.0) * math.pi * rho_mantle * (radii**3 - R_cmb**3),
    )


def _lane_emden_n1_constants():
    """Polytropic constant and length scale of the registered ``polytrope_n1``.

    Reads the same K the production equation of state evaluates and returns
    the Lane-Emden length scale alpha = sqrt(K / 2 pi G), so the analytic
    reference is built from the identical constant the solver uses. The n = 1
    polytrope has its surface at xi = r / alpha = pi, fixing the radius
    R = pi alpha independently of the central density.
    """
    from zalmoxis.eos_analytic import _K_POLYTROPE_N1 as K

    alpha = math.sqrt(K / (2.0 * math.pi * G))
    return K, alpha


def _analytic_lane_emden_n1(rho_c, alpha, K, radii):
    """Closed-form n = 1 polytrope profiles rho, M, g, P on ``radii``."""
    xi = radii / alpha
    theta = np.ones_like(xi)
    nz = xi > 0
    theta[nz] = np.sin(xi[nz]) / xi[nz]
    rho = rho_c * theta
    M = 4.0 * math.pi * alpha**3 * rho_c * (np.sin(xi) - xi * np.cos(xi))
    g = np.zeros_like(radii)
    g[nz] = G * M[nz] / radii[nz] ** 2
    P = K * rho**2
    return rho, M, g, P


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _rel_err(num, ana):
    """Relative error |num - ana| / |ana|, masked where ``ana`` is tiny."""
    ana = np.asarray(ana, dtype=float)
    num = np.asarray(num, dtype=float)
    out = np.full_like(ana, np.nan)
    good = np.abs(ana) > 0
    out[good] = np.abs(num[good] - ana[good]) / np.abs(ana[good])
    return out


def compute_spheres():
    """Integrate the constant-density and two-layer spheres (runs Zalmoxis).

    Returns the plotted arrays in Earth units, ready for ``plot_spheres``.
    """
    logger.info('Compute: constant-density and two-layer spheres')
    rho, R, P_c = 5000.0, 6.4e6, 3.6e11
    radii_u = np.linspace(0, R, 300)
    mass_u, grav_u, press_u = _solve(_constant_density_mock(rho), radii_u, P_c)
    M_exu, g_exu, P_exu = _analytic_uniform_sphere(rho, P_c, radii_u)

    rho_c, rho_m, cmf = 13000.0, 4000.0, 0.325
    M_total = earth_mass
    cmb_mass = cmf * M_total
    R_cmb = ((3 * cmb_mass) / (4 * math.pi * rho_c)) ** (1.0 / 3.0)
    M_mantle = M_total - cmb_mass
    R_total = (R_cmb**3 + (3 * M_mantle) / (4 * math.pi * rho_m)) ** (1.0 / 3.0)
    P_ct = _two_layer_central_pressure(rho_c, rho_m, R_cmb, R_total, cmb_mass) * 1.05
    radii_t = np.linspace(0, R_total, 400)
    mass_t, grav_t, press_t = _solve_two_layer(rho_c, rho_m, cmb_mass, M_total, radii_t, P_ct)
    M_ext = _analytic_two_layer_mass(rho_c, rho_m, R_cmb, cmb_mass, radii_t)
    g_gauss = np.zeros_like(radii_t)
    g_gauss[1:] = G * mass_t[1:] / radii_t[1:] ** 2

    return dict(
        xu=radii_u / earth_radius,
        mass_u=mass_u / earth_mass, grav_u=grav_u, press_u=press_u / 1e9,
        Mex_u=M_exu / earth_mass, gex_u=g_exu, Pex_u=P_exu / 1e9,
        err_mass_u=_rel_err(mass_u, M_exu), err_grav_u=_rel_err(grav_u, g_exu),
        err_press_u=_rel_err(press_u, P_exu),
        xt=radii_t / earth_radius,
        mass_t=mass_t / earth_mass, grav_t=grav_t, Mex_t=M_ext / earth_mass,
        g_gauss=g_gauss, rcmb=np.float64(R_cmb / earth_radius),
        err_mass_t=_rel_err(mass_t, M_ext), err_gauss_t=_rel_err(grav_t, g_gauss),
    )


def plot_spheres(d, outdir):
    """Two-layer sphere across a density discontinuity, one row of three panels.

    The columns share the radius axis: enclosed mass, gravity against Gauss's
    law, and the relative errors of both. ``d`` is the array bundle from
    ``compute_spheres`` (or the committed cache).
    """
    logger.info('Figure: two-layer sphere')
    xt = d['xt']
    st = slice(2, None)
    rcmb = float(d['rcmb'])

    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.25))

    # (a) enclosed mass.
    ax[0].plot(xt, d['mass_t'], color=C_NUM, label='Numerical')
    ax[0].plot(xt, d['Mex_t'], '--', color=C_ANA, label='Analytic')
    ax[0].axvline(rcmb, color=C_CMB, ls=':', label='CMB')
    ax[0].set_ylabel(r'Enclosed mass [$M_\mathrm{Earth}$]')
    _panel_label(ax[0], '(a) $M(r)$')
    ax[0].legend(loc='lower right', fontsize=6.4, handlelength=1.4, borderpad=0.25)

    # (b) gravity against Gauss's law.
    ax[1].plot(xt, d['grav_t'], color=C_NUM, label='$g(r)$')
    ax[1].plot(xt[1:], d['g_gauss'][1:], '--', color=C_ANA, label='$G M/r^2$')
    ax[1].axvline(rcmb, color=C_CMB, ls=':')
    ax[1].set_ylabel('Gravity [m s$^{-2}$]')
    _panel_label(ax[1], "(b) Gauss's law")
    ax[1].legend(loc='lower right', fontsize=6.4, handlelength=1.4, borderpad=0.25)

    # (c) relative error of mass and of the Gauss's-law residual.
    ax[2].semilogy(xt[st], d['err_mass_t'][st], color=ix.STRATA['cobalt'], label='mass')
    ax[2].semilogy(xt[st], d['err_gauss_t'][st], color=ix.STRATA['magma'],
                   label="Gauss residual")
    ax[2].axvline(rcmb, color=C_CMB, ls=':')
    ax[2].set_ylabel('Relative error')
    _panel_label(ax[2], '(c) error')
    ax[2].grid(True, which='both', alpha=0.25)
    ax[2].legend(loc='lower right', fontsize=6.0, handlelength=1.4, borderpad=0.25)

    for a in ax:
        a.set_xlabel(r'Radius [$R_\mathrm{Earth}$]')
    _thin_xticks(ax)
    _more_yticks(ax)
    _axis_extents(ax)
    fig.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.1)
    _uniform_xlim(ax)
    for _a in fig.axes:
        ix.set_mono_ticks(_a)
    fig.savefig(os.path.join(outdir, 'zalmoxis_spheres.pdf'))
    plt.close(fig)


def compute_lane_emden():
    """n = 1 polytrope through the integrator and the full solver (runs Zalmoxis).

    Returns the plotted arrays in Earth units, ready for ``plot_lane_emden``.
    """
    logger.info('Compute: Lane-Emden n = 1 polytrope (integrator + full solver)')
    K, alpha = _lane_emden_n1_constants()
    R_exact = math.pi * alpha

    # Integrator: inject rho(P) = sqrt(P/K) at the analytic central pressure.
    rho_ci = 9000.0
    radii_i = np.linspace(0, R_exact, 400)
    mass_i, _, press_i = _solve(
        _polytrope_n1_density_mock(K), radii_i, K * rho_ci**2, component='mock:polytrope'
    )
    rho_exi, M_exi, _, P_exi = _analytic_lane_emden_n1(rho_ci, alpha, K, radii_i)
    rho_numi = np.sqrt(np.clip(press_i, 0.0, None) / K)
    vi = press_i > 0
    si = (radii_i > 0.04 * R_exact) & (radii_i < 0.985 * R_exact) & vi

    # Full solver: register the polytrope as an EoS and converge on target mass.
    res = _solve_polytrope_full_chain(0.5 * earth_mass, num_layers=600, P_surf=1e2)
    if not res.get('converged'):
        raise RuntimeError('polytrope full-chain solve did not converge')
    r = np.asarray(res['radii'])
    rho_f = np.asarray(res['density'])
    mass_f = np.asarray(res['mass_enclosed'])
    press_f = np.asarray(res['pressure'])
    below = np.where(press_f <= 1e2)[0]
    isurf = int(below[0]) if below.size else len(r) - 1
    sl = slice(0, isurf + 1)
    r, rho_f, mass_f, press_f = r[sl], rho_f[sl], mass_f[sl], press_f[sl]
    rho_cf = math.sqrt(res['p_center'] / K)
    rho_exf, M_exf, _, P_exf = _analytic_lane_emden_n1(rho_cf, alpha, K, r)
    R_num = float(np.interp(-1e2, -press_f, r))
    sf = (r > 0.04 * R_exact) & (r < 0.985 * R_exact)

    return dict(
        # Integrator (top row).
        xi=radii_i / earth_radius, vi=vi, si=si,
        rho_numi=rho_numi, rho_exi=rho_exi,
        mass_i=mass_i / earth_mass, Mex_i=M_exi / earth_mass,
        press_i=press_i / 1e9, Pex_i=P_exi / 1e9,
        err_rho_i=_rel_err(rho_numi, rho_exi), err_mass_i=_rel_err(mass_i, M_exi),
        err_press_i=_rel_err(press_i, P_exi),
        # Full solver (bottom row).
        xf=r / earth_radius, sf=sf,
        rho_f=rho_f, rho_exf=rho_exf,
        mass_f=mass_f / earth_mass, Mex_f=M_exf / earth_mass,
        press_f=press_f / 1e9, Pex_f=P_exf / 1e9,
        err_rho_f=_rel_err(rho_f, rho_exf), err_mass_f=_rel_err(mass_f, M_exf),
        err_press_f=_rel_err(press_f, P_exf),
        mass_f_last=np.float64(mass_f[-1] / earth_mass),
        R_ratio=np.float64(R_num / R_exact),
    )


def plot_lane_emden(d, outdir):
    """n = 1 polytrope: the integrator and the full solver overlaid in one row.

    The four columns share the radius axis; the columns are density, mass,
    pressure, and relative error. Each panel overlays the integrator and the
    full production solver against the analytic Lane-Emden reference. ``d`` is
    the array bundle from ``compute_lane_emden`` (or the committed cache).
    """
    logger.info('Figure: Lane-Emden n = 1 polytrope (integrator + full solver)')
    xi, xf = d['xi'], d['xf']
    vi, si, sf = d['vi'], d['si'], d['sf']

    C_INT = C_NUM             # integrator (cobalt)
    C_FUL = ix.STRATA['sage'] if 'sage' in ix.STRATA else '#2e7d32'  # full solver (light blue)

    fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.25))

    # (a) density
    ax[0].plot(xi[vi], d['rho_numi'][vi], color=C_INT, lw=1.6, label='integrator')
    ax[0].plot(xf, d['rho_f'], color=C_FUL, lw=1.2, label='full solver')
    ax[0].plot(xi, d['rho_exi'], '--', color=C_ANA, lw=1.1, label='analytic')
    ax[0].set_ylabel('Density [kg m$^{-3}$]')
    _panel_label(ax[0], r'(a) $\rho(r)$')
    ax[0].legend(loc='lower left', fontsize=6.2, handlelength=1.4, borderpad=0.25,
                 labelspacing=0.25, framealpha=0.85)

    # (b) enclosed mass
    ax[1].plot(xi, d['mass_i'], color=C_INT, lw=1.6)
    ax[1].plot(xf, d['mass_f'], color=C_FUL, lw=1.2)
    ax[1].plot(xi, d['Mex_i'], '--', color=C_ANA, lw=1.1)
    ax[1].set_ylabel(r'Enclosed mass [$M_\mathrm{Earth}$]')
    _panel_label(ax[1], '(b) $M(r)$')

    # (c) pressure
    ax[2].plot(xi[vi], d['press_i'][vi], color=C_INT, lw=1.6)
    ax[2].plot(xf, d['press_f'], color=C_FUL, lw=1.2)
    ax[2].plot(xi, d['Pex_i'], '--', color=C_ANA, lw=1.1)
    ax[2].set_ylabel('Pressure [GPa]')
    _panel_label(ax[2], '(c) $P(r)$', x=0.96, ha='right')

    # (d) relative error: colour by quantity, line style by method.
    cq = {'rho': ix.STRATA['cobalt'], 'mass': ix.STRATA['magma'], 'press': ix.STRATA['amber']}
    ax[3].semilogy(xi[si], d['err_rho_i'][si], '-', color=cq['rho'], lw=0.9)
    ax[3].semilogy(xi[si], d['err_mass_i'][si], '-', color=cq['mass'], lw=0.9)
    ax[3].semilogy(xi[si], d['err_press_i'][si], '-', color=cq['press'], lw=0.9)
    ax[3].semilogy(xf[sf], d['err_rho_f'][sf], '--', color=cq['rho'], lw=0.9)
    ax[3].semilogy(xf[sf], d['err_mass_f'][sf], '--', color=cq['mass'], lw=0.9)
    ax[3].semilogy(xf[sf], d['err_press_f'][sf], '--', color=cq['press'], lw=0.9)
    ax[3].set_ylabel('Relative error')
    _panel_label(ax[3], '(d) error')
    ax[3].grid(True, which='both', alpha=0.25)
    grey = ix.NEUTRALS.get('graphite', '#666')
    handles = [
        Line2D([0], [0], color=cq['rho'], lw=1.6, label='density'),
        Line2D([0], [0], color=cq['mass'], lw=1.6, label='mass'),
        Line2D([0], [0], color=cq['press'], lw=1.6, label='pressure'),
        Line2D([0], [0], color=grey, lw=1.6, ls='-', label='integrator'),
        Line2D([0], [0], color=grey, lw=1.6, ls='--', label='full solver'),
    ]
    # legend in the empty mid-panel band between the full-solver error line
    # above and the integrator error curves below
    ax[3].legend(handles=handles, loc='center', bbox_to_anchor=(0.52, 0.62),
                 fontsize=5.6, handlelength=1.4, borderpad=0.22, labelspacing=0.2,
                 framealpha=0.85, ncol=2, columnspacing=0.8)

    for a in ax:
        a.set_xlabel(r'Radius [$R_\mathrm{Earth}$]')
    _thin_xticks(ax)
    _more_yticks(ax)
    _axis_extents(ax)
    # extra headroom on the error axis so the top-left panel label stays clear
    # of the full-solver error line
    ax[3].set_ylim(top=1e-2)
    fig.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.1)
    _uniform_xlim(ax)
    for _a in fig.axes:
        ix.set_mono_ticks(_a)
    fig.savefig(os.path.join(outdir, 'zalmoxis_lane_emden.pdf'))
    plt.close(fig)
    logger.info(
        '  polytrope: full-solver M=%.4f M_E, R_num/R_exact=%.5f',
        float(d['mass_f_last']),
        float(d['R_ratio']),
    )


def compute_conservation_convergence():
    """Conservation residuals and integrator-tolerance convergence (runs Zalmoxis).

    Returns the plotted residual and convergence arrays, ready for
    ``plot_conservation_and_convergence``.
    """
    logger.info('Compute: conservation and convergence')

    # (a, b) pointwise conservation residuals for the constant-density sphere.
    rho, R, P_c = 5000.0, 6.4e6, 3.6e11
    radii = np.linspace(0, R, 500)
    mass, gravity, pressure = _solve(_constant_density_mock(rho), radii, P_c)
    r_re = radii / earth_radius
    valid = np.where(pressure > 0)[0]
    idx = valid[valid > 2]

    g_gauss = G * mass[idx] / radii[idx] ** 2
    gauss_res = np.abs(gravity[idx] - g_gauss) / g_gauss

    interior = idx[(idx > 1) & (idx < len(radii) - 1)]
    dPdr = (pressure[interior + 1] - pressure[interior - 1]) / (
        radii[interior + 1] - radii[interior - 1]
    )
    P_scale = (2.0 / 3.0) * math.pi * G * rho**2 * R
    hydro_res = np.abs(dPdr + rho * gravity[interior]) / P_scale

    # (c, d) tolerance convergence of the adaptive integrator.
    K, alpha = _lane_emden_n1_constants()
    R_poly, rho_c = math.pi * alpha, 9000.0
    rho_u, R_u, P_c_u = 5000.0, 6.4e6, 3.6e11

    tols = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    poly_err, unif_err = [], []
    for rtol in tols:
        radii_p = np.linspace(0, R_poly, 400)
        poly_err.append(_polytrope_density_error(radii_p, rho_c, K, alpha, rtol))
        radii_u = np.linspace(0, R_u, 400)
        mass_u, _, _ = _solve(_constant_density_mock(rho_u), radii_u, P_c_u, rtol=rtol, atol=rtol * 1e-2)
        M_ex_u, _, _ = _analytic_uniform_sphere(rho_u, P_c_u, radii_u)
        unif_err.append(float(np.nanmax(_rel_err(mass_u, M_ex_u)[2:])))

    grids = [50, 100, 200, 400, 800, 1600]
    polyN_err = [
        _polytrope_density_error(np.linspace(0, R_poly, n), rho_c, K, alpha, 1e-10) for n in grids
    ]

    return dict(
        r_gauss=r_re[idx], gauss_res=gauss_res,
        r_hydro=r_re[interior], hydro_res=hydro_res,
        tols=np.array(tols), poly_err=np.array(poly_err), unif_err=np.array(unif_err),
        grids=np.array(grids), polyN_err=np.array(polyN_err),
    )


def plot_conservation_and_convergence(d, outdir):
    """Conservation residuals and integrator convergence in one four-panel row.

    Panels (a) and (b) are the pointwise Gauss's-law and hydrostatic-balance
    residuals for the constant-density sphere. Panels (c) and (d) are the
    tolerance convergence of the adaptive integrator, with genuine truncation
    error for the n = 1 polytrope against exact integration of the uniform
    sphere, and the independence of the error from the output grid. ``d`` is the
    array bundle from ``compute_conservation_convergence`` (or the cache).
    """
    logger.info('Figure: conservation and convergence')

    fig, ax = plt.subplots(1, 4, figsize=_ROW4)

    # (a, b) pointwise conservation residuals for the constant-density sphere.
    ax[0].semilogy(d['r_gauss'], d['gauss_res'], color=C_NUM)
    ax[0].set(xlabel=r'Radius [$R_\mathrm{Earth}$]', ylabel='$|g - G M/r^2| / (G M/r^2)$')
    _panel_label(ax[0], "(a) Gauss's\nlaw residual", alpha=0.55)
    ax[0].grid(True, which='both', alpha=0.25)

    ax[1].semilogy(d['r_hydro'], d['hydro_res'], color=C_NUM)
    ax[1].set(xlabel=r'Radius [$R_\mathrm{Earth}$]', ylabel=r'$|dP/dr + \rho g| / P_{\rm scale}$')
    _panel_label(ax[1], '(b) hydrostatic\nresidual', alpha=0.55)
    ax[1].grid(True, which='both', alpha=0.25)

    # (c, d) tolerance convergence of the adaptive integrator.
    tols = d['tols']
    grids = d['grids']

    # Curves are identified in the caption by colour and marker (blue circles:
    # polytrope; red squares: sphere); no in-panel legend, to keep the
    # narrow panel uncluttered and the flat red curve visible.
    ax[2].loglog(tols, d['poly_err'], 'o-', color=ix.STRATA['cobalt'])
    ax[2].loglog(tols, d['unif_err'], 's-', color=ix.STRATA['amber'])
    ax[2].set(xlabel='Integrator tolerance', ylabel='Max relative error')
    _panel_label(ax[2], '(c) tolerance\nconvergence')
    ax[2].set_xticks(tols)
    ax[2].xaxis.set_major_formatter(FuncFormatter(_log_tick))
    ax[2].invert_xaxis()
    ax[2].grid(True, which='both', alpha=0.25)

    ax[3].loglog(grids, d['polyN_err'], 'o-', color=ix.STRATA['cobalt'])
    ax[3].set(xlabel='Grid points $N$', ylabel='Max relative error')
    _panel_label(ax[3], '(d) output\ngrid')
    ax[3].set_xlim(grids[0] * 0.85, grids[-1] * 1.18)
    ax[3].set_xticks([100, 200, 500, 1000])
    ax[3].xaxis.set_major_formatter(FuncFormatter(_log_tick))
    ax[3].grid(True, which='both', alpha=0.25)

    # Scientific-notation x labels are wide; rotate them so four or more fit
    # without overlap in the narrow convergence panels.
    for a in (ax[2], ax[3]):
        plt.setp(a.get_xticklabels(), rotation=30, ha='right', fontsize=7.5)

    _more_yticks(ax)
    _axis_extents(ax)
    fig.tight_layout(pad=0.5)
    _uniform_xlim(ax[:2])
    for _a in fig.axes:
        ix.set_mono_ticks(_a)
    fig.savefig(os.path.join(outdir, 'zalmoxis_conservation_convergence.pdf'))
    plt.close(fig)


def _polytrope_density_error(radii, rho_c, K, alpha, rtol):
    """Max relative density error of the integrated n = 1 polytrope."""
    P_c = K * rho_c**2
    _, _, pressure = _solve(
        _polytrope_n1_density_mock(K), radii, P_c, component='mock:polytrope', rtol=rtol, atol=rtol * 1e-2
    )
    rho_ex, _, _, _ = _analytic_lane_emden_n1(rho_c, alpha, K, radii)
    rho_num = np.sqrt(np.clip(pressure, 0.0, None) / K)
    m = (radii > 0.04 * radii[-1]) & (radii < 0.95 * radii[-1]) & (pressure > 0)
    return float(np.nanmax(_rel_err(rho_num, rho_ex)[m]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--outdir',
        default=os.path.join(HERE_FP, 'figures'),
        help='Output directory for the figure PDFs.',
    )
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Re-run Zalmoxis and refresh the data/ cache (needs the Zalmoxis '
             'package). By default the figures are replotted from the committed cache.',
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    plot_spheres(_load_or_compute('spheres', compute_spheres, args.recompute), args.outdir)
    plot_lane_emden(_load_or_compute('lane_emden', compute_lane_emden, args.recompute), args.outdir)
    plot_conservation_and_convergence(
        _load_or_compute('conservation_convergence', compute_conservation_convergence, args.recompute),
        args.outdir,
    )

    logger.info('All figures written to %s', args.outdir)


if __name__ == '__main__':
    main()
