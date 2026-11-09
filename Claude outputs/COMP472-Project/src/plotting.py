"""Consistent, readable matplotlib styling for every figure in the project."""
import matplotlib
import matplotlib.pyplot as plt

from .common import FIG_DIR

# Fixed categorical colour order (colour-blind-checked). Series always get
# colours in this order, so e.g. "k-NN" is the same colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"k-NN": PALETTE[0], "Decision Tree": PALETTE[1],
                "Random Forest": PALETTE[2], "Neural Network": PALETTE[6]}
SEQ_CMAP = "Blues"   # one hue, light -> dark, for magnitudes (confusion matrices)


def setup():
    matplotlib.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a",
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path


setup()
