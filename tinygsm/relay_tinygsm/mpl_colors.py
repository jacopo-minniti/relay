"""cics_mpl_colors.py

Matplotlib color utilities built around the CICS theme colors.

Provides:
- Base color definitions (hex + RGB)
- A high-contrast default color cycle (for multi-line plots)
- Registered Matplotlib colormaps:
    * Sequential:  cics_seq
    * Diverging:   cics_div
    * Cyclic:      cics_cyc
    * Qualitative: cics_qual

Typical use:
    import matplotlib.pyplot as plt
    import cics_mpl_colors as cics

    cics.use_cics_style()              # set default prop cycle
    cics.register_colormaps()          # safe to call multiple times

    plt.imshow(Z, cmap='cics_div')

Notes:
- This module only depends on matplotlib.
- All colormaps are registered on-demand (or lazily via use_cics_style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Base theme colors
# -----------------------------------------------------------------------------

CICS_COLORS_HEX: Dict[str, str] = {
    "cics_red": "#990000",
    "cics_light_red": "#FF7070",
    "cics_blue": "#264690",
    "cics_light_blue": "#97AEE4",
    "cics_orange": "#E94A00",
    "cics_yellow": "#EB8F00",
    "cics_gray": "#666666",
}


# -----------------------------------------------------------------------------
# Small color math helpers (pure python; no numpy dependency)
# -----------------------------------------------------------------------------

RGB = Tuple[float, float, float]


def _hex_to_rgb01(hex_color: str) -> RGB:
    """#RRGGBB -> (r,g,b) in [0,1]."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color!r}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def _rgb01_to_hex(rgb: RGB) -> str:
    """(r,g,b) in [0,1] -> #RRGGBB."""
    r, g, b = rgb
    r_i = max(0, min(255, int(round(r * 255.0))))
    g_i = max(0, min(255, int(round(g * 255.0))))
    b_i = max(0, min(255, int(round(b * 255.0))))
    return f"#{r_i:02X}{g_i:02X}{b_i:02X}"


def _blend(a: str, b: str, t: float) -> str:
    """Linear blend between two hex colors (t in [0,1])."""
    t = float(max(0.0, min(1.0, t)))
    ra, ga, ba = _hex_to_rgb01(a)
    rb, gb, bb = _hex_to_rgb01(b)
    r = (1.0 - t) * ra + t * rb
    g = (1.0 - t) * ga + t * gb
    b_ = (1.0 - t) * ba + t * bb
    return _rgb01_to_hex((r, g, b_))


def _lighten(hex_color: str, amount: float) -> str:
    """Lighten by blending towards white."""
    return _blend(hex_color, "#FFFFFF", amount)


def _darken(hex_color: str, amount: float) -> str:
    """Darken by blending towards black."""
    return _blend(hex_color, "#000000", amount)


# -----------------------------------------------------------------------------
# Palette choices
# -----------------------------------------------------------------------------

# A concise, high-contrast ordering for line/marker cycles.
# (We keep the light tones later; they are better as "extra" colors.)
CICS_QUALITATIVE: List[str] = [
    CICS_COLORS_HEX["cics_blue"],
    CICS_COLORS_HEX["cics_orange"],
    CICS_COLORS_HEX["cics_red"],
    CICS_COLORS_HEX["cics_yellow"],
    CICS_COLORS_HEX["cics_gray"],
    CICS_COLORS_HEX["cics_light_blue"],
    CICS_COLORS_HEX["cics_light_red"],
]


def get_base_colors_hex() -> Dict[str, str]:
    """Return a copy of the base color dict (hex strings)."""
    return dict(CICS_COLORS_HEX)


def get_qualitative_palette(n: int = 10) -> List[str]:
    """Return a qualitative palette of length n.

    - For n <= 7, uses the base qualitative list.
    - For n > 7, extends by generating dark/light variations of the
      most distinct anchors (blue, orange, red, yellow, gray).

    The extension is deterministic and keeps reasonable contrast.
    """
    if n <= 0:
        return []
    base = list(CICS_QUALITATIVE)
    if n <= len(base):
        return base[:n]

    anchors = [
        CICS_COLORS_HEX["cics_blue"],
        CICS_COLORS_HEX["cics_orange"],
        CICS_COLORS_HEX["cics_red"],
        CICS_COLORS_HEX["cics_yellow"],
        CICS_COLORS_HEX["cics_gray"],
    ]

    # Add alternating dark/light variants of the anchors.
    # Chosen amounts aim to keep the theme recognizable without washing out.
    variants: List[str] = []
    for amt in (0.18, 0.32, 0.45):
        for c in anchors:
            variants.append(_darken(c, amt))
        for c in anchors:
            variants.append(_lighten(c, amt))

    extended = base + variants
    return extended[:n]


# -----------------------------------------------------------------------------
# Matplotlib integration
# -----------------------------------------------------------------------------


def get_color_cycle(n: int = 10):
    """Return a matplotlib Cycler for the CICS palette."""
    from cycler import cycler

    return cycler(color=get_qualitative_palette(n))


def _build_colormaps():
    """Create colormap objects (unregistered)."""
    import matplotlib.colors as mcolors

    blue = CICS_COLORS_HEX["cics_blue"]
    lblue = CICS_COLORS_HEX["cics_light_blue"]
    red = CICS_COLORS_HEX["cics_red"]
    lred = CICS_COLORS_HEX["cics_light_red"]
    orange = CICS_COLORS_HEX["cics_orange"]
    yellow = CICS_COLORS_HEX["cics_yellow"]
    gray = CICS_COLORS_HEX["cics_gray"]

    # Sequential (cool): very light blue -> light blue -> blue -> slightly darker blue
    seq_list = [
        _lighten(lblue, 0.55),
        lblue,
        blue,
        _darken(blue, 0.25),
    ]
    cics_seq = mcolors.LinearSegmentedColormap.from_list(
        "cics_seq", seq_list, N=256
    )

    # Diverging: blue -> light blue -> near-neutral -> light red -> red
    # Use a lightened gray as a gentle neutral center.
    center = _lighten(gray, 0.75)
    div_list = [
        _darken(blue, 0.15),
        blue,
        lblue,
        center,
        lred,
        red,
        _darken(red, 0.10),
    ]
    cics_div = mcolors.LinearSegmentedColormap.from_list(
        "cics_div", div_list, N=256
    )

    # Cyclic: make a hue-like loop using the available anchors.
    # Ensure the first and last colors match (cyclic continuity).
    cyc_list = [
        blue,
        lblue,
        _blend(lblue, yellow, 0.45),
        yellow,
        orange,
        _blend(orange, red, 0.55),
        red,
        lred,
        _blend(lred, blue, 0.35),
        blue,
    ]
    cics_cyc = mcolors.LinearSegmentedColormap.from_list(
        "cics_cyc", cyc_list, N=256
    )

    # Qualitative ListedColormap (for categorical images)
    cics_qual = mcolors.ListedColormap(CICS_QUALITATIVE, name="cics_qual")

    return {
        "cics_seq": cics_seq,
        "cics_div": cics_div,
        "cics_cyc": cics_cyc,
        "cics_qual": cics_qual,
    }


def register_colormaps(override: bool = False) -> None:
    """Register all CICS colormaps in Matplotlib.

    Args:
        override: If True, re-register even if a cmap with the name exists.
                  (This uses matplotlib's 'force' when available.)
    """
    import matplotlib as mpl

    cmaps = _build_colormaps()

    # Matplotlib 3.7+ supports 'force' kwarg; earlier versions error.
    for name, cmap in cmaps.items():
        if not override:
            try:
                mpl.colormaps[name]
                continue
            except KeyError:
                pass

        try:
            mpl.colormaps.register(cmap, name=name, force=override)
        except TypeError:
            # Older matplotlib: no 'force' kwarg
            try:
                mpl.cm.register_cmap(name=name, cmap=cmap)
            except ValueError:
                # Exists and can't override on old versions
                if override:
                    raise


def use_cics_style(
    *,
    n_cycle: int = 10,
    set_as_default: bool = True,
    register_cmaps: bool = True,
) -> None:
    """Convenience function to apply the CICS theme to Matplotlib.

    This sets the default property cycle for axes (line colors). Optionally
    registers colormaps.

    Args:
        n_cycle: number of colors to include in the default cycle.
        set_as_default: if True, sets mpl.rcParams['axes.prop_cycle'].
        register_cmaps: if True, registers the CICS colormaps.
    """
    import matplotlib as mpl

    if register_cmaps:
        register_colormaps(override=False)

    if set_as_default:
        mpl.rcParams["axes.prop_cycle"] = get_color_cycle(n_cycle)


# -----------------------------------------------------------------------------
# Public names
# -----------------------------------------------------------------------------

__all__ = [
    "CICS_COLORS_HEX",
    "CICS_QUALITATIVE",
    "get_base_colors_hex",
    "get_qualitative_palette",
    "get_color_cycle",
    "register_colormaps",
    "use_cics_style",
]

"""
# Usage: Load everything
import matplotlib.pyplot as plt
import cics_mpl_colors as cics

cics.use_cics_style(n_cycle=12)     # sets default line color cycle + registers cmaps

plt.plot(x, y1)
plt.plot(x, y2)

plt.imshow(Z, cmap="cics_div")
plt.colorbar()

# Usage: colormaps only
import cics_mpl_colors as cics
cics.register_colormaps()
"""