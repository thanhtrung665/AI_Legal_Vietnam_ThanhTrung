"""One visual style for every figure in the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {"primary": "#1f5fa8", "accent": "#d1495b", "neutral": "#9aa5b1", "good": "#2a9d8f"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    mpl.rcParams.update({
        "figure.figsize": (10, 5.5), "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlelocation": "left", "font.family": "DejaVu Sans",
    })


def save(fig: plt.Figure, name: str, folder: str = "reports/figures") -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(Path(folder) / f"{name}.{ext}")
