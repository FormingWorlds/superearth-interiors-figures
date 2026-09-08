"""PROTEUS Thermocline visual identity as a drop-in matplotlib style.

Exposes ``use``, ``STRATA``, ``NEUTRALS``, ``strata_colors``, ``strata_cmap``
and ``set_mono_ticks`` to the plotting scripts, backed by the PROTEUS design
language:
the magma-red / ocean-blue thermocline palette, Instrument Sans body text with
Spline Sans Mono numerals, and the PROTEUS spines, grid, and colormaps.

Tokens, style sheets, and fonts are vendored in ``proteus_assets/`` so figures
reproduce on any machine without a checkout of the source visual-language
repository. The named ``STRATA`` keys are retained so existing per-series colour
choices keep working; each maps to a PROTEUS brand colour chosen for hue family
and mutual distinctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_hex

_HERE = Path(__file__).resolve().parent
_ASSETS = _HERE / "proteus_assets"
_TOKENS = json.loads((_ASSETS / "proteus_tokens.json").read_text())
_STYLE_LIGHT = _ASSETS / "proteus.mplstyle"
_STYLE_DARK = _ASSETS / "proteus_dark.mplstyle"
_FONT_DIR = _ASSETS / "fonts"

_C = _TOKENS["colors"]
_CYCLE_LIGHT = list(_TOKENS["cycles"]["light"])
_CYCLE_DARK = list(_TOKENS["cycles"]["dark"])
_PHASE_STOPS = list(_TOKENS["colormaps"]["phase"]["stops"])
_SEQ_STOPS = list(_TOKENS["colormaps"]["sequential"]["stops"])

FONT_DISPLAY = _TOKENS["fonts"]["display"]  # Sora
FONT_BODY = _TOKENS["fonts"]["body"]        # Instrument Sans
FONT_MONO = _TOKENS["fonts"]["mono"]        # Spline Sans Mono

# Brand colours, exposed for direct use.
CORE = dict(_C)

# Strata colour names mapped onto PROTEUS brand colours, used by the per-series
# colour choices in the plotting scripts; the mapping keeps the warm/cool sense
# and keeps colours that appear together distinct.
STRATA = {
    "gold": _C["azure"],    # #4FA3D9
    "amber": _C["magma"],   # #E23D28  bright warm accent
    "magma": _C["outgassing"] if "outgassing" in _C else "#A03123",
    "plum": _C["tidal"] if "tidal" in _C else "#593E74",
    "cobalt": _C["ocean"],  # #1B6FA8
    "ink": _C["ink"],       # #10151B
    "sage": _C["ice"],      # #A8D4E8
}
STRATA_ORDER = ["gold", "amber", "magma", "plum", "cobalt", "ink"]

NEUTRALS = {
    "paper": _C["paper"],   # #F2F5F7
    "cream": _C["paper"],
    "bone": "#C8D2DA",
    "mist": _C["mist"],     # #9FB0BE
    "slate": _C["mist"],
    "fog": _C["fog"],       # #7A8894
    "graphite": "#3E4A55",  # slate grey (PROTEUS tick colour)
    "ink": _C["ink"],       # #10151B
}


def register_fonts(font_dir: Path | str | None = None) -> list[str]:
    """Register the bundled PROTEUS fonts with matplotlib's font manager.

    Parameters
    ----------
    font_dir : path-like, optional
        Directory of ``.ttf`` files; defaults to ``proteus_assets/fonts``.

    Returns
    -------
    list of str
        Family names matplotlib now resolves.
    """
    font_dir = Path(font_dir) if font_dir else _FONT_DIR
    names: list[str] = []
    for ttf in sorted(font_dir.glob("*.ttf")):
        fm.fontManager.addfont(str(ttf))
        try:
            names.append(fm.FontProperties(fname=str(ttf)).get_name())
        except Exception:  # noqa: BLE001 - best-effort name lookup
            names.append(ttf.name)
    return names


def _cycle(dark: bool = False) -> list[str]:
    """Return the categorical series cycle for the light or dark identity."""
    return list(_CYCLE_DARK if dark else _CYCLE_LIGHT)


def use(theme: str = "light", register: bool = True, font: str = "mono",
        math: str | None = None) -> None:
    """Apply the PROTEUS matplotlib style.

    Parameters
    ----------
    theme : str, default "light"
        ``"light"`` / ``"paper"`` / ``"white"`` select the light (Paper)
        identity; ``"white"`` additionally forces a pure-white canvas for
        journals that mandate it. ``"dark"`` / ``"void"`` select the dark (Void)
        identity.
    register : bool, default True
        Register the bundled fonts before applying the style.
    font, math : optional
        Accepted and ignored. The PROTEUS style sets Instrument Sans body text with Spline Sans Mono
        numerals regardless; ``set_mono_ticks`` puts tick labels in the mono
        face, matching the design language.
    """
    if register:
        register_fonts()
    dark = theme in ("dark", "void")
    plt.style.use(str(_STYLE_DARK if dark else _STYLE_LIGHT))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=_cycle(dark))
    if theme == "white":
        for key in ("figure.facecolor", "axes.facecolor", "savefig.facecolor"):
            mpl.rcParams[key] = "white"
        mpl.rcParams["legend.facecolor"] = "white"


def strata_colors(n: int, dark: bool = False) -> list[str]:
    """Return ``n`` series colours from the PROTEUS palette.

    Up to the length of the categorical cycle the cycle is returned directly;
    beyond it the continuous phase ramp is sampled so every colour stays
    distinct rather than repeating.

    Parameters
    ----------
    n : int
        Number of colours requested.
    dark : bool, default False
        Use the dark-identity cycle.

    Returns
    -------
    list of str
        ``n`` hex colour strings.
    """
    cycle = _cycle(dark)
    if n <= len(cycle):
        return cycle[:n]
    cmap = LinearSegmentedColormap.from_list("proteus_phase", _PHASE_STOPS)
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def strata_cmap(reverse: bool = False, name: str = "proteus"):
    """Continuous PROTEUS colormap.

    Parameters
    ----------
    reverse : bool, default False
        Reverse the ramp direction.
    name : str, default "proteus"
        Registered colormap name.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The PROTEUS sequential ramp.
    """
    stops = list(reversed(_SEQ_STOPS)) if reverse else _SEQ_STOPS
    return LinearSegmentedColormap.from_list(name, stops)


def phase_cmap(reverse: bool = False, name: str = "proteus_phase"):
    """The full PROTEUS phase ramp (warm through dark to cool)."""
    stops = list(reversed(_PHASE_STOPS)) if reverse else _PHASE_STOPS
    return LinearSegmentedColormap.from_list(name, stops)


def set_mono_ticks(ax) -> None:
    """Set the Spline Sans Mono face on the tick labels of ``ax``."""
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily(FONT_MONO)


def panel_label(ax, text, x=0.04, y=0.95, ha="left", va="top", alpha=0.85,
                fontsize=None) -> None:
    """Bold panel label inside the axes on a translucent white patch.

    The manuscript-wide sublabel style: ``(a) Subtitle`` in bold at the top-left
    corner (or the position given), backed by a rounded white patch so it stays
    readable over gridlines and data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    text : str
        Label text, e.g. ``"(a) Mass-radius"``.
    x, y : float
        Axes-fraction anchor of the label.
    ha, va : str
        Text alignment relative to the anchor.
    alpha : float
        Patch opacity; lower it when data beneath must stay visible.
    fontsize : float, optional
        Override; defaults to the axes title size.
    """
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontweight="bold",
            fontsize=fontsize if fontsize is not None else plt.rcParams["axes.titlesize"],
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=alpha,
                      edgecolor="none"))
