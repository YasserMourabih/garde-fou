"""Chart styling: one theme, two surfaces.

Every figure is rendered twice — once for a light page, once for a dark one —
and the README serves the right file with `<picture>`. Dark mode is a *selected*
set of steps for the dark surface, not an inverted light palette.

Only documented-validated colour configurations are used: slots 1-3 where a
reader compares every pair (dot plots, categorical fills), and the fixed slot
order for line charts, where the validated adjacent-pair guarantee applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


@dataclass(frozen=True)
class Theme:
    mode: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series: tuple[str, ...]
    muted: str


LIGHT = Theme(
    mode="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e6e5e1",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
    muted="#b9b8b2",
)

DARK = Theme(
    mode="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#333331",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"),
    muted="#5a5a56",
)

THEMES = (LIGHT, DARK)


def apply(theme: Theme) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.text_primary,
            "axes.labelcolor": theme.text_secondary,
            "axes.edgecolor": theme.grid,
            "xtick.color": theme.text_secondary,
            "ytick.color": theme.text_secondary,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titlecolor": theme.text_primary,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def rounded_barh(ax, y, width, *, height, color, theme: Theme, radius_px: float = 4.0):
    """Horizontal bar with a 4px rounded data-end and a square baseline.

    Matplotlib has no rounded-bar primitive, so we draw a fully rounded box and
    then square off the baseline end with a small patch in the same colour.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig = ax.figure
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    px_per_unit = max(bbox.width, 1.0) / max(x1 - x0, 1e-9)
    radius = radius_px / px_per_unit
    radius = min(radius, max(width, 1e-9) / 2, 0.45 * (x1 - x0))

    if width <= radius * 2:  # too short to round without distorting
        patch = Rectangle((0, y - height / 2), width, height, color=color, linewidth=0)
        ax.add_patch(patch)
        return patch

    box = FancyBboxPatch(
        (radius, y - height / 2 + radius),
        width - 2 * radius,
        height - 2 * radius,
        boxstyle=f"round,pad={radius},rounding_size={radius}",
        linewidth=0,
        facecolor=color,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)
    # Square the baseline end.
    ax.add_patch(
        Rectangle((0, y - height / 2), radius, height, color=color, linewidth=0)
    )
    _ = fig
    return box
