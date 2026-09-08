"""First-principles analytic validation of the Aragog energy solver.

Aragog integrates the entropy-form energy balance

    rho T dS/dt = (1/r^2) d/dr [r^2 F_total] + rho H,
    F_total = F_cond + F_conv + F_grav + F_mix,

on a spherical mantle shell. In its constant-property mode (SPIDER's
``-use_const_properties`` parity path: constant rho, Cp, k, and analytic
T(S) = T_ref exp((S - S_ref)/Cp), with no equation-of-state table) the
entropy equation is exactly equivalent to the classical temperature heat
equation, so the production solver can be checked against closed-form
conduction, energy-conservation, and transient-decay solutions. Every test
here drives the *production* ``EntropySolver`` (the same code path used in
coupled runs) rather than a reimplementation.

Three figures are produced, each saved as vector PDF:

1. ``conduction.pdf`` - steady conduction. Top row: a shell carrying a
   steady conductive flux (prescribed flux at both boundaries), against the
   exact T(r) = A/r + B. Bottom row: uniform internal heating with an
   insulated base, against the exact internally-heated profile. Tests the
   conduction operator, the boundary conditions, and the volumetric source.
2. ``conservation.pdf`` - conservation and invariance laws that hold for
   any valid run: surface-flux energy balance E(t) = E_0 - F A t,
   grey-body radiative closure, and isentropic-state invariance.
3. ``transient.pdf`` - transient conduction in an insulated shell. The
   eigenmodes decay as exp(-t/tau_n) with tau_n = 1/(kappa k_n^2), where the
   wavenumbers k_n solve the Neumann dispersion relation for the spherical
   shell; the fitted timescales are compared to the analytic eigenvalues.
   Tests the time integrator.

Notes
-----
By default the figures are replotted from the committed array caches under
``data/`` (numpy + matplotlib only), so they reproduce from the repository
alone. Pass ``--recompute`` to re-run the Aragog solver and refresh the
cache; that path imports the installed ``aragog`` package and must run in
the environment where it is installed. Provenance is recorded in the
repository README.

Usage
-----
    python plot_aragog_first_principles.py [--outdir DIR] [--recompute]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, MaxNLocator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# PROTEUS visual identity, vendored under style/ at the repo root so the
# figures reproduce from the repository alone.
HERE_FP = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(os.path.dirname(HERE_FP), 'style')
sys.path.insert(0, _STYLE)
import proteus_mpl as ix  # noqa: E402

ix.use('white', font='mono')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'legend.handlelength': 1.1, 'legend.handletextpad': 0.4,
    'legend.borderpad': 0.25, 'legend.labelspacing': 0.2, 'legend.framealpha': 0.85,
    'xtick.labelsize': 8, 'ytick.labelsize': 9, 'lines.linewidth': 2.0,
    'axes.linewidth': 0.9, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_NUM = ix.STRATA['cobalt']   # numerical (production solver)
C_ANA = ix.STRATA['amber']    # analytic reference
C_3 = ix.STRATA['magma']
C_4 = ix.STRATA['plum']

DATA_DIR = os.path.join(HERE_FP, 'data')

# ----------------------------------------------------------------------------
# Constant-property model parameters (shared by every test)
# ----------------------------------------------------------------------------
# A 1000 km mantle shell with constant rho, Cp, k. In const_properties mode
# T(S) = T_REF exp((S - S_REF)/Cp) exactly, so injecting an analytic T(r) as
# the initial entropy reproduces that temperature to machine precision.
R_IN, R_OUT = 5.371e6, 6.371e6       # m (1000 km shell)
RHO, CP, K_COND = 4000.0, 1000.0, 4.0  # kg/m^3, J/kg/K, W/m/K
T_REF, S_REF = 3000.0, 3000.0          # K, J/kg/K
KAPPA = K_COND / (RHO * CP)            # thermal diffusivity, 1e-6 m^2/s
K_HI = 4000.0                          # high conductivity for the transient test: tau_1 ~ 3.2 Myr
D_SHELL = R_OUT - R_IN
SECS_PER_YEAR = 31557600.0


def T_to_S(T):
    """Constant-Cp entropy from temperature, inverse of the const_properties T(S)."""
    return S_REF + CP * np.log(np.asarray(T, dtype=float) / T_REF)


def S_to_T(S):
    """Constant-Cp temperature from entropy (the const_properties T(S) relation)."""
    return T_REF * np.exp((np.asarray(S, dtype=float) - S_REF) / CP)


# ----------------------------------------------------------------------------
# Lazy Aragog import: only the --recompute path needs the package.
# ----------------------------------------------------------------------------
_AR = {}


def _import_aragog():
    """Bind the production Aragog symbols used by the compute path."""
    if _AR:
        return _AR
    from aragog.parser import (
        Parameters, _BoundaryConditionsParameters, _EnergyParameters,
        _InitialConditionParameters, _MeshParameters, _PhaseMixedParameters,
        _PhaseParameters, _SolverParameters,
    )
    from aragog.solver.entropy_solver import EntropySolver
    _AR.update(
        Parameters=Parameters, BC=_BoundaryConditionsParameters, EN=_EnergyParameters,
        IC=_InitialConditionParameters, MESH=_MeshParameters, PM=_PhaseMixedParameters,
        PH=_PhaseParameters, SV=_SolverParameters, EntropySolver=EntropySolver,
    )
    return _AR


def _build_params(*, n_nodes, end_time, outer_bc, outer_val, inner_bc, inner_val,
                  convection=False, alpha=1e-4, log10visc=21.0, tidal_H=0.0,
                  emissivity=1.0, eq_temperature=255.0, atol=1e-8, rtol=1e-8,
                  k_cond=K_COND):
    """Construct production const_properties Parameters for one limiting-case run.

    All physics other than the requested flux is suppressed. ``tidal_H`` is a
    uniform volumetric heating per unit mass [W/kg], injected through the tidal
    heating array (the only per-layer constant source the solver exposes).
    """
    A = _import_aragog()
    bc = A['BC'](outer_boundary_condition=outer_bc, outer_boundary_value=outer_val,
                 inner_boundary_condition=inner_bc, inner_boundary_value=inner_val,
                 emissivity=emissivity, equilibrium_temperature=eq_temperature,
                 core_heat_capacity=880.0, core_bc='quasi_steady')
    en_kw = dict(conduction=True, convection=convection, gravitational_separation=False,
                 mixing=False, radionuclides=False, tidal=(tidal_H != 0.0),
                 solver_method='radau', use_jax_jacobian=False)
    if tidal_H != 0.0:
        en_kw['tidal_array'] = np.full(n_nodes - 1, tidal_H, dtype=float)
    en = A['EN'](**en_kw)
    ic = A['IC'](initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0)
    mesh = A['MESH'](outer_radius=R_OUT, inner_radius=R_IN, number_of_nodes=n_nodes,
                     mixing_length_profile='nearest_boundary', core_density=10500.0, eos_method=1)
    ph = dict(density=RHO, heat_capacity=CP, thermal_conductivity=k_cond, thermal_expansivity=alpha)
    pl = A['PH'](melt_fraction=1.0, viscosity=10.0 ** log10visc, **ph)
    ps = A['PH'](melt_fraction=0.0, viscosity=10.0 ** log10visc, **ph)
    pm = A['PM'](latent_heat_of_fusion=4.0e5, rheological_transition_melt_fraction=0.4,
                 rheological_transition_width=0.15, solidus='solidus.dat', liquidus='liquidus.dat',
                 phase='mixed', phase_transition_width=0.01, grain_size=1e-3, const_properties=True,
                 const_rho=RHO, const_Cp=CP, const_alpha=alpha, const_cond=k_cond,
                 const_log10visc=log10visc, const_T_ref=T_REF, const_S_ref=S_REF)
    sv = A['SV'](start_time=0.0, end_time=end_time, atol=atol, rtol=rtol, tsurf_poststep_change=1e9)
    return A['Parameters'](boundary_conditions=bc, energy=en, initial_condition=ic, mesh=mesh,
                           phase_solid=ps, phase_liquid=pl, phase_mixed=pm, radionuclides=[], solver=sv)


def _run(params, S0):
    """Initialise and solve a production const_properties run from entropy IC ``S0``."""
    A = _import_aragog()
    solver = A['EntropySolver'](params, entropy_eos=None)
    solver.initialize()
    r_stag = np.asarray(solver._r_stag_flat, dtype=float).flatten()
    if np.ndim(S0) == 0:
        solver.set_initial_entropy(float(S0))
    else:
        solver.set_initial_entropy(np.asarray(S0, dtype=float))
    solver.solve()
    status = int(getattr(solver._solution, 'status', 0))
    if status != 0:
        raise RuntimeError(
            f'solve did not reach end_time (solver status {status}); the trajectory '
            'is truncated or the integration failed, so it must not be plotted')
    return solver, r_stag


# ----------------------------------------------------------------------------
# Analytic solutions
# ----------------------------------------------------------------------------


def analytic_conduction(r, T_in, T_out):
    """Steady conduction profile set by boundary temperatures T_in, T_out: T(r) = A/r + B."""
    a, b = R_IN, R_OUT
    A = (T_in - T_out) * a * b / (b - a)
    B = (T_out * b - T_in * a) / (b - a)
    return A / r + B


def analytic_heating(r, H, T_out):
    """Steady internally-heated conduction, insulated base, fixed surface T.

    Solves (k/r^2) d/dr(r^2 dT/dr) + rho H = 0 with dT/dr(R_in) = 0 and
    T(R_out) = T_out. Returns T(r) [K].
    """
    q = RHO * H  # volumetric heating [W/m^3]
    pref = q / (3.0 * K_COND)
    bracket = R_IN**3 * (1.0 / R_OUT - 1.0 / r) + 0.5 * (R_OUT**2 - r**2)
    return T_out + pref * bracket


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------


def _panel_label(ax, text, x=0.04, y=0.93, ha='left', va='top'):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontweight='bold',
            fontsize=plt.rcParams['axes.titlesize'],
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.8, edgecolor='none'))


def _rel_err(num, ana):
    num = np.asarray(num, float)
    ana = np.asarray(ana, float)
    out = np.full_like(ana, np.nan)
    good = np.abs(ana) > 0
    out[good] = np.abs(num[good] - ana[good]) / np.abs(ana[good])
    return out


# Signature baked into each cached array file. It combines the physical constants
# with the source of the compute path, so editing any per-run parameter (boundary
# temperatures, heating rate, node count, tolerances, end times) or any analytic
# helper invalidates a stale cache instead of serving it silently. A mismatch
# rejects the cache and recomputes the figure (which needs the aragog package).
def _cache_signature():
    consts = repr((R_IN, R_OUT, RHO, CP, K_COND, K_HI, T_REF, S_REF))
    srcs = ''.join(inspect.getsource(fn) for fn in (
        _build_params, _run, analytic_conduction, analytic_heating, compute_conduction,
        _neumann_eigenvalues, _neumann_mode, compute_transient,
        _energy_series, _surface_temperature_series, compute_conservation, _init_radii))
    return hashlib.sha256((consts + srcs).encode('utf-8')).hexdigest()


def _load_or_compute(name, compute_fn, recompute=False):
    """Return a figure's plotted arrays, from the committed cache or by running Aragog."""
    sig = _cache_signature()
    path = os.path.join(DATA_DIR, name + '.npz')
    if os.path.exists(path) and not recompute:
        with np.load(path) as f:
            cached = {k: f[k] for k in f.files}
        if str(cached.get('_sig', '')) == sig:
            return {k: v for k, v in cached.items() if k != '_sig'}
        raise RuntimeError(f'cache {name}.npz does not match the compute functions in this '
                           'script; restore the script or rerun with --recompute')
    data = compute_fn()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(path, _sig=np.array(sig), **data)
    logger.info('  wrote cache %s (%d arrays)', os.path.relpath(path, HERE_FP), len(data))
    return data


# ----------------------------------------------------------------------------
# Figure 1: steady conduction (two fixed T) and internal heating
# ----------------------------------------------------------------------------


def compute_conduction():
    """Run the production solver for steady conduction and internal heating.

    Each case injects the exact analytic temperature profile as the initial
    entropy and integrates briefly (the diffusion time is ~1e17 s, so a short
    run probes whether the production operator recognises the analytic state as
    steady). Returns the held profile, the conductive luminosity, and the error.
    """
    logger.info('Compute: steady conduction and internal heating')
    T_in, T_out = 4000.0, 1500.0

    # Top row: the matching conductive flux prescribed at both boundaries
    # (inner kA/R_c^2, outer kA/R_p^2; equal luminosity 4 pi k A, so the
    # energy budget closes) makes the exact T(r) = A/r + B a steady state. The
    # conductive luminosity 4 pi r^2 F is uniform (the flux divergence vanishes).
    A_coeff = (T_in - T_out) * R_IN * R_OUT / (R_OUT - R_IN)
    F_surf_cond = K_COND * A_coeff / R_OUT**2          # analytic surface flux
    F_base_cond = K_COND * A_coeff / R_IN**2           # analytic basal flux
    p = _build_params(n_nodes=100, end_time=1e4, outer_bc=4, outer_val=F_surf_cond,
                      inner_bc=2, inner_val=F_base_cond)
    solver, r = _run(p, T_to_S(analytic_conduction(_init_radii(100), T_in, T_out)))
    out = solver.get_state()
    rb = np.asarray(out.r_basic, float).flatten()
    Tc = np.asarray(out.T_stag, float).flatten()
    Tc_an = analytic_conduction(r, T_in, T_out)
    Fc = np.asarray(out.heat_flux, float).flatten()
    Qc = Fc * 4.0 * math.pi * rb**2          # conductive luminosity (uniform)
    Qc_an = np.full_like(rb, 4.0 * math.pi * K_COND * A_coeff)  # constant

    # Bottom row: uniform internal heating, insulated base, surface flux set to
    # the total heating power. H sized for a ~500 K contrast; the luminosity
    # rises as (r^3 - R_in^3), reaching the total power at the surface.
    H = 1.0e-12  # W/kg
    F_surf_heat = RHO * H * (R_OUT**3 - R_IN**3) / (3.0 * R_OUT**2)
    p = _build_params(n_nodes=100, end_time=1e4, outer_bc=4, outer_val=F_surf_heat,
                      inner_bc=2, inner_val=0.0, tidal_H=H)
    solver, r = _run(p, T_to_S(analytic_heating(_init_radii(100), H, T_out)))
    out = solver.get_state()
    rbh = np.asarray(out.r_basic, float).flatten()
    Th = np.asarray(out.T_stag, float).flatten()
    Th_an = analytic_heating(r, H, T_out)
    Fh = np.asarray(out.heat_flux, float).flatten()
    Qh = Fh * 4.0 * math.pi * rbh**2
    Qh_an = (4.0 * math.pi * RHO * H / 3.0) * (rbh**3 - R_IN**3)

    return dict(
        rc=r / 1e6, Tc=Tc, Tc_an=Tc_an, err_Tc=_rel_err(Tc, Tc_an),
        rbc=rb / 1e6, Qc=Qc, Qc_an=Qc_an,
        rh=r / 1e6, Th=Th, Th_an=Th_an, err_Th=_rel_err(Th, Th_an),
        rbh=rbh / 1e6, Qh=Qh, Qh_an=Qh_an,
    )


def _init_radii(n):
    """Staggered radii the solver will use, to build the analytic IC before solving."""
    a = _import_aragog()
    p = _build_params(n_nodes=n, end_time=1.0, outer_bc=5, outer_val=1500.0, inner_bc=3, inner_val=4000.0)
    s = a['EntropySolver'](p, entropy_eos=None)
    s.initialize()
    return np.asarray(s._r_stag_flat, float).flatten()


def plot_conduction(d, outdir):
    """Steady conduction and internal heating overlaid in one row of three.

    Each panel overlays the no-source conduction case and the internally heated
    case (colour) against their analytic profiles (grey dashed): (a) temperature,
    (b) luminosity, (c) relative error in temperature.
    """
    logger.info('Figure: steady conduction and internal heating')
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.4))

    # Cached radii are in Mm; display in km.
    rc, rbc, rh, rbh = (d['rc'] * 1e3, d['rbc'] * 1e3, d['rh'] * 1e3, d['rbh'] * 1e3)
    C_CON, C_HEAT = C_NUM, C_4          # conduction (cobalt), internal heating (plum)
    C_AN = ix.NEUTRALS['graphite']      # analytic reference (grey dashed)

    # (a) temperature
    ax[0].plot(rc, d['Tc'], '-', color=C_CON, lw=1.6)
    ax[0].plot(rc, d['Tc_an'], '--', color=C_AN, lw=1.0)
    ax[0].plot(rh, d['Th'], '-', color=C_HEAT, lw=1.6)
    ax[0].plot(rh, d['Th_an'], '--', color=C_AN, lw=1.0)
    ax[0].set_ylabel('Temperature [K]')
    _panel_label(ax[0], '(a) $T(r)$')
    handles = [
        Line2D([0], [0], color=C_CON, lw=1.6, label='conduction'),
        Line2D([0], [0], color=C_HEAT, lw=1.6, label='internal heating'),
        Line2D([0], [0], color=C_AN, lw=1.2, ls='--', label='analytic'),
    ]
    ax[0].legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.90),
                 fontsize=6.4, handlelength=1.4, borderpad=0.25, labelspacing=0.22,
                 framealpha=0.85)

    # (b) luminosity, shown on the interior nodes where the returned flux is the
    # physical conductive value (the boundary nodes carry the prescribed BC flux).
    ax[1].plot(rbc[1:-1], d['Qc'][1:-1] / 1e12, '-', color=C_CON, lw=1.6)
    ax[1].plot(rbc[1:-1], d['Qc_an'][1:-1] / 1e12, '--', color=C_AN, lw=1.0)
    ax[1].plot(rbh[1:-1], d['Qh'][1:-1] / 1e12, '-', color=C_HEAT, lw=1.6)
    ax[1].plot(rbh[1:-1], d['Qh_an'][1:-1] / 1e12, '--', color=C_AN, lw=1.0)
    ax[1].set_ylabel(r'$4\pi r^2 F$ [TW]')
    ax[1].set_ylim(0, 1.25 * np.nanmax(d['Qc_an'] / 1e12))
    _panel_label(ax[1], '(b) luminosity')

    # (c) relative error in temperature for both cases.
    ax[2].semilogy(rc, d['err_Tc'], '-', color=C_CON, label='conduction')
    ax[2].semilogy(rh, d['err_Th'], '-', color=C_HEAT, label='internal heating')
    ax[2].set_ylabel('Rel. error in $T$')
    _panel_label(ax[2], '(c) error')
    ax[2].grid(True, which='both', alpha=0.25)
    ax[2].legend(loc='lower left', fontsize=6.4, handlelength=1.4, borderpad=0.25,
                 labelspacing=0.22, framealpha=0.85)

    xlo = min(float(np.min(rc)), float(np.min(rh)))
    xhi = max(float(np.max(rc)), float(np.max(rh)))
    for a in ax:
        a.set_xlim(xlo, xhi)
        a.set_xlabel('Radius [km]')
        a.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ix.set_mono_ticks(a)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.3)
    fig.savefig(os.path.join(outdir, 'aragog_conduction.pdf'))
    plt.close(fig)


STEFAN_BOLTZMANN = 5.670374419e-8


def _neumann_eigenvalues(n_modes):
    """Wavenumbers k_n of the insulated spherical shell: zero-flux at both ends.

    Eigenfunctions T(r) = [A sin(kr) + B cos(kr)]/r with dT/dr = 0 at R_in and
    R_out give the dispersion relation g(a)f(b) - g(b)f(a) = 0, with
    f(r) = kr cos(kr) - sin(kr), g(r) = kr sin(kr) + cos(kr). Returns the first
    ``n_modes`` positive roots (the uniform k=0 mode is conserved, not returned).
    """
    a, b = R_IN, R_OUT

    def fr(r, k):
        return k * r * np.cos(k * r) - np.sin(k * r)

    def gr(r, k):
        return k * r * np.sin(k * r) + np.cos(k * r)

    def disp(k):
        return gr(a, k) * fr(b, k) - gr(b, k) * fr(a, k)

    ks = np.linspace(0.2 * math.pi / D_SHELL, (n_modes + 2) * math.pi / D_SHELL, 40000)
    v = disp(ks)
    roots = []
    for i in np.where(v[:-1] * v[1:] < 0)[0]:
        lo, hi = ks[i], ks[i + 1]
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if disp(lo) * disp(mid) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append(0.5 * (lo + hi))
        if len(roots) >= n_modes:
            break
    return np.array(roots)


def _neumann_mode(r, k):
    """Insulated-shell eigenmode shape for wavenumber ``k``, peak-normalised.

    The zero-flux condition at R_in fixes the amplitude ratio A/B = g(a)/f(a),
    so the shape is [g(a) sin(kr) + f(a) cos(kr)]/r.
    """
    a = R_IN
    fa = k * a * math.cos(k * a) - math.sin(k * a)
    ga = k * a * math.sin(k * a) + math.cos(k * a)
    shape = (ga * np.sin(k * r) + fa * np.cos(k * r)) / r
    return shape / np.max(np.abs(shape))


def compute_transient():
    """Transient conduction in an insulated shell: Neumann eigenmodes decay as
    exp(-t/tau_n), tau_n = 1/(kappa k_n^2)."""
    logger.info('Compute: transient eigenmode decay')
    kappa_hi = K_HI / (RHO * CP)
    T_bg, delta = 3000.0, 50.0
    r = _init_radii(100)
    w = r**2 * np.gradient(r)                  # r^2-weighted projection (modes orthogonal)

    modes = [1, 2, 3]
    kk = _neumann_eigenvalues(len(modes))
    tau_an = 1.0 / (kappa_hi * kk**2) / SECS_PER_YEAR  # yr
    tau_fit = []
    amp1, prof_r, prof_dT, prof_t = None, None, [], []
    for j, n in enumerate(modes):
        phi = _neumann_mode(r, kk[j])
        p = _build_params(n_nodes=100, end_time=2.5 * tau_an[j], outer_bc=4, outer_val=0.0,
                          inner_bc=2, inner_val=0.0, k_cond=K_HI, atol=1e-10, rtol=1e-10)
        s, _ = _run(p, T_to_S(T_bg + delta * phi))
        yS = np.asarray(s._solution.y, float)[:s._n_stag, :]
        tt = np.asarray(s._solution.t, float)                      # yr
        dT = S_to_T(yS) - T_bg
        c = (dT * (phi * w)[:, None]).sum(0) / np.sum(phi**2 * w)   # modal amplitude
        amp = np.abs(c) / max(abs(float(c[0])), 1e-30)
        sel = (amp > 0.05) & (amp < 0.98) & (tt > 0)
        if int(sel.sum()) < 2:
            raise RuntimeError(f'mode {n}: decay-fit window has {int(sel.sum())} points; '
                               'lengthen end_time or relax the amplitude window')
        slope = np.polyfit(tt[sel], np.log(amp[sel]), 1)[0]
        tau_fit.append(-1.0 / slope)
        if j == 0:
            amp1 = (tt / tau_an[j], amp)
            prof_r = r / 1e6
            isnap = [int(f * (len(tt) - 1)) for f in (0.0, 0.25, 0.5, 0.85)]
            prof_dT = [dT[:, k] for k in isnap]
            prof_t = [tt[k] / tau_an[j] for k in isnap]

    return dict(
        modes=np.array(modes), tau_fit=np.array(tau_fit), tau_an=np.array(tau_an),
        amp_t=amp1[0], amp=amp1[1],
        prof_r=prof_r, prof_dT=np.array(prof_dT), prof_t=np.array(prof_t),
    )


def plot_transient(d, outdir):
    """Figure 3: transient eigenmode decay against the shell eigenvalues
    tau_n = 1/(kappa k_n^2), k_n the Neumann roots for the spherical shell."""
    logger.info('Figure: transient eigenmode decay')
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.5))

    tg = np.linspace(0, d['amp_t'].max(), 50)
    ax[0].semilogy(d['amp_t'], d['amp'], color=C_NUM, zorder=2, label='Aragog')
    ax[0].semilogy(tg, np.exp(-tg), '--', color=C_ANA, lw=1.4, zorder=5, label=r'$e^{-t/\tau_1}$')
    ax[0].set(xlabel=r'$t/\tau_1$', ylabel='Mode amplitude $A/A_0$')
    _panel_label(ax[0], '(a) decay', x=0.96, ha='right')
    ax[0].legend(loc='lower left')
    ax[0].grid(True, which='both', alpha=0.25)

    cols = [C_NUM, C_3, C_4, ix.STRATA['gold']]
    for k in range(len(d['prof_t'])):
        ax[1].plot(d['prof_r'] * 1e3, d['prof_dT'][k], color=cols[k],
                   label=rf"$t/\tau_1={d['prof_t'][k]:.2f}$")
    ax[1].axhline(0, color=ix.NEUTRALS['fog'], lw=0.8)
    ax[1].set(xlabel='Radius [km]', ylabel=r'$\delta T(r)$ [K]')
    _panel_label(ax[1], '(b) eigenmode', x=0.96, ha='right')
    ax[1].legend(loc='lower left', fontsize=6.5)

    ax[2].loglog(d['modes'], d['tau_an'], '--', color=C_ANA, label=r'$1/\kappa k_n^2$')
    ax[2].loglog(d['modes'], d['tau_fit'], 'o', color=C_NUM, mfc='none', label='fitted')
    ax[2].set(xlabel='Mode number $n$', ylabel=r'$\tau_n$ [yr]')
    ax[2].set_xticks(d['modes'])
    ax[2].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    _panel_label(ax[2], '(c) eigenvalues', x=0.96, ha='right')
    ax[2].legend(loc='lower left')
    ax[2].grid(True, which='both', alpha=0.25)

    for a in ax.ravel():
        ix.set_mono_ticks(a)
    fig.tight_layout(pad=0.5, w_pad=0.6)
    fig.savefig(os.path.join(outdir, 'aragog_transient.pdf'))
    plt.close(fig)


def _energy_series(solver):
    """Thermal energy E_th(t) = sum rho Cp T vol and the time grid [s] from a solve."""
    out = solver.get_state()
    vol = np.asarray(out.vol, float).flatten()
    n = solver._n_stag
    T = S_to_T(np.asarray(solver._solution.y, float)[:n, :])
    E = (RHO * CP * T * vol[:, None]).sum(axis=0)
    t = np.asarray(solver._solution.t, float) * SECS_PER_YEAR
    r_basic = np.asarray(out.r_basic, float).flatten()
    return t, E, T, 4.0 * math.pi * r_basic[-1] ** 2


def _surface_temperature_series(solver):
    """Surface temperature the grey-body BC radiates from, per saved timestep.

    The solver evaluates the radiative flux from ``top_temperature`` (the
    outermost *basic* node), not the outermost staggered cell, so the closure
    must be checked against that node. Reconstruct it by replaying each saved
    entropy state through ``state.update``.
    """
    y = np.asarray(solver._solution.y, float)
    t = np.asarray(solver._solution.t, float)
    n = solver._n_stag
    Ts = np.empty(y.shape[1])
    for i in range(y.shape[1]):
        solver.state.update(y[:n, i], float(t[i]))
        Ts[i] = float(np.asarray(solver.state.top_temperature).flat[0])
    return Ts


def compute_conservation():
    """Energy-balance and invariance laws that hold for any valid run."""
    logger.info('Compute: conservation and invariance laws')

    # (a) Prescribed-flux energy conservation: E(t) = E0 - F A t.
    F_out = 1.0e3
    p = _build_params(n_nodes=60, end_time=5e4, outer_bc=4, outer_val=F_out,
                      inner_bc=2, inner_val=0.0, convection=True)
    s, _ = _run(p, T_to_S(3000.0))
    t_e, E_e, _, A_surf = _energy_series(s)
    E_line = E_e[0] - F_out * A_surf * t_e

    # (b) Grey-body radiative closure: -dE/dt = eps sigma (T_surf^4 - T_eq^4) A.
    eps, T_eq = 1.0, 255.0
    p = _build_params(n_nodes=60, end_time=3e5, outer_bc=1, outer_val=0.0,
                      inner_bc=2, inner_val=0.0, convection=True,
                      emissivity=eps, eq_temperature=T_eq)
    s, _ = _run(p, T_to_S(4000.0))
    t_g, E_g, T_g, A_g = _energy_series(s)
    loss = -np.gradient(E_g, t_g)                       # numerical energy-loss rate [W]
    T_surf = _surface_temperature_series(s)             # basic-top node, what the BC radiates from
    sb = eps * STEFAN_BOLTZMANN * (T_surf**4 - T_eq**4) * A_g

    # (c) Isentropic invariance: uniform S, insulated both ends, no source.
    p = _build_params(n_nodes=60, end_time=1e6, outer_bc=4, outer_val=0.0,
                      inner_bc=2, inner_val=0.0, convection=True)
    s, _ = _run(p, T_to_S(3000.0))
    n = s._n_stag
    S_t = np.asarray(s._solution.y, float)[:n, :]
    S0 = S_t[:, 0]
    drift = np.max(np.abs(S_t - S0[:, None]), axis=0)
    t_i = np.asarray(s._solution.t, float)

    return dict(
        t_e=t_e / SECS_PER_YEAR / 1e3, E_e=E_e / E_e[0], E_line=E_line / E_e[0],
        F_out=np.float64(F_out), A_surf=np.float64(A_surf), E0=np.float64(E_e[0]),
        t_g=t_g / SECS_PER_YEAR / 1e3, loss_g=loss / 1e12, sb_g=sb / 1e12,
        t_i=t_i / 1e3, drift_i=drift,
    )


def plot_conservation(d, outdir):
    """Figure 2: energy conservation, grey-body closure, isentropic invariance."""
    logger.info('Figure: conservation and invariance laws')
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.5))

    ax[0].plot(d['t_e'], d['E_e'], color=C_NUM, zorder=2, label='Aragog')
    ax[0].plot(d['t_e'], d['E_line'], '--', color=C_ANA, lw=1.4, zorder=5, label=r'$E_0 - F A_\mathrm{s} t$')
    ax[0].set(xlabel='Time [kyr]', ylabel=r'$E_\mathrm{th}/E_0$')
    _panel_label(ax[0], '(a) energy\nbalance', x=0.96, ha='right')
    ax[0].legend(loc='lower left')

    sb = np.abs(d['sb_g'])
    loss = np.abs(d['loss_g'])
    good = (sb > 0) & (loss > 0)
    lim = [0.7 * min(sb[good].min(), loss[good].min()), 1.4 * max(sb[good].max(), loss[good].max())]
    ax[1].loglog(sb[good], loss[good], ls='none', marker='o', ms=2.5, mfc='none',
                 mec=C_NUM, mew=0.8, zorder=2)
    ax[1].plot(lim, lim, color=ix.NEUTRALS['graphite'], lw=1.3, zorder=5)
    ax[1].set(xlabel=r'$\varepsilon\sigma(T_s^4-T_\mathrm{eq}^4)\,A$ [TW]', ylabel=r'$-\,dE/dt$ [TW]')
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    _panel_label(ax[1], '(b) grey-body\nclosure')

    ax[2].semilogy(d['t_i'], np.maximum(d['drift_i'], 1e-15), color=C_NUM)
    ax[2].axhline(1e-8 * S_REF, color=C_ANA, ls='--', lw=1.0, label='solver tol.')
    ax[2].set(xlabel='Time [kyr]', ylabel=r'$\max|S(t)-S_0|$ [J/kg/K]')
    ax[2].set_ylim(1e-16, 1e-3)
    _panel_label(ax[2], '(c) isentropic\ninvariance', y=0.5, va='center')
    # between the tolerance line at the top and the panel label, clear of both
    ax[2].legend(loc='upper right', bbox_to_anchor=(0.98, 0.80))
    ax[2].grid(True, which='both', alpha=0.25)

    # Panels (a) and (c) have a linear time axis; (b) is log-log, so it needs a
    # log locator on x (a linear MaxNLocator leaves the decade ticks unlabelled).
    for a in (ax[0], ax[2]):
        a.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[1].xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax[1].xaxis.set_major_formatter(LogFormatterSciNotation())
    for a in ax.ravel():
        ix.set_mono_ticks(a)
    fig.tight_layout(pad=0.5, w_pad=0.6)
    fig.savefig(os.path.join(outdir, 'aragog_conservation.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--outdir', default=os.path.join(HERE_FP, 'figures'),
                        help='Output directory for the figure PDFs.')
    parser.add_argument('--recompute', action='store_true',
                        help='Re-run Aragog and refresh the data/ cache (needs the aragog package).')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    plot_conduction(_load_or_compute('conduction', compute_conduction, args.recompute), args.outdir)
    plot_conservation(_load_or_compute('conservation', compute_conservation, args.recompute), args.outdir)
    plot_transient(_load_or_compute('transient', compute_transient, args.recompute), args.outdir)

    logger.info('All figures written to %s', args.outdir)


if __name__ == '__main__':
    main()
