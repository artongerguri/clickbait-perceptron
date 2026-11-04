"""Shared matplotlib styling so every figure reads as one visual system.

Colors come from a validated, colorblind-safe palette (worst adjacent CVD ΔE
well above the safe threshold). We render static, print-ready figures in light
mode (the mode an IEEE paper is read in), with thin marks, a recessive grid and
restrained ink, following data-visualization best practice.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# -- palette (light mode) --------------------------------------------------- #
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots (fixed order, never cycled arbitrarily).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

CATEGORICAL = [BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, GREEN, RED]

# Diverging poles for signed quantities (e.g. feature weights):
#   positive  -> clickbait  -> warm red
#   negative  -> legitimate -> cool blue
DIVERGING_POS = RED
DIVERGING_NEG = BLUE

# Sequential blue ramp (for heatmaps such as the confusion matrix).
SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]


def apply_style() -> None:
    """Install the shared rcParams. Call once before plotting."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "figure.dpi": 120,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK_PRIMARY,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sequential_cmap():
    """Return a matplotlib colormap from the sequential blue ramp."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


def save(fig, path: str) -> None:
    """Tight, consistent figure export."""
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
