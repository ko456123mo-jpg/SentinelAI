"""
SentinelAI - Shared visualization helpers
=========================================
One place for figure styling / saving so every stage produces consistent,
good-looking charts saved into results/figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

from src import config           # noqa: E402

# ---- global style -----------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.autolayout": True,
})

PALETTE = ["#2e86ab", "#e63946", "#06a77d", "#f4a261", "#8d5a97",
           "#b08968", "#00b4d8"]


def save_fig(fig, name: str) -> str:
    """Save a matplotlib figure into results/figures/ and close it."""
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    [figure] {os.path.relpath(path, config.ROOT)}")
    return path


def confusion_matrix_heatmap(cm, classes, title, normalize=False):
    """Return a styled confusion-matrix figure (uses seaborn if available)."""
    try:
        import seaborn as sns
        sns.set_style("white")
    except Exception:  # pragma: no cover
        pass

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if val > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def grouped_bar(metrics_dict, metric_keys, title, ylabel,
                fname=None, figsize=(10, 4.6)):
    """
    grouped_bar({"LogReg": {"acc":.9,"f1":.8}, ...}, ["acc","f1"], ...)
    -> grouped bar figure comparing models.
    """
    names = list(metrics_dict.keys())
    x = np.arange(len(names))
    width = 0.8 / len(metric_keys)

    fig, ax = plt.subplots(figsize=figsize)
    for i, key in enumerate(metric_keys):
        vals = [metrics_dict[n].get(key, 0) for n in names]
        bars = ax.bar(x + (i - len(metric_keys) / 2 + 0.5) * width, vals,
                      width, label=key, color=PALETTE[i % len(PALETTE)])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", ncols=len(metric_keys), fontsize=8)
    return fig


def line_chart(x, ys, labels, title, xlabel, ylabel, log_x=False):
    """Simple multi-line chart used for learning / reward curves."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for y, lab, c in zip(ys, labels, PALETTE):
        ax.plot(x, y, label=lab, color=c, linewidth=1.8)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig
