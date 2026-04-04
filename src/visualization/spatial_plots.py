"""
Spatial visualisation utilities for segmentation benchmarking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Transcript scatter
# ---------------------------------------------------------------------------

def plot_transcript_density(
    transcripts: pd.DataFrame,
    x_col: str = "x_location",
    y_col: str = "y_location",
    gene_col: str = "feature_name",
    genes: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Transcript Density",
    alpha: float = 0.3,
    s: float = 0.5,
) -> plt.Axes:
    """Scatter plot of transcript positions, optionally coloured by gene."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if genes is not None:
        subset = transcripts[transcripts[gene_col].isin(genes)]
        for gene in genes:
            g = subset[subset[gene_col] == gene]
            ax.scatter(g[x_col], g[y_col], s=s, alpha=alpha, label=gene)
        ax.legend(markerscale=6, fontsize=8)
    else:
        ax.scatter(transcripts[x_col], transcripts[y_col], s=s, alpha=alpha, color="steelblue")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# Cell boundary overlay
# ---------------------------------------------------------------------------

def plot_cell_boundaries(
    polys: List[Polygon],
    image: Optional[np.ndarray] = None,
    color: str = "cyan",
    linewidth: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: str = "Cell Boundaries",
) -> plt.Axes:
    """Overlay cell boundary polygons on an optional background image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if image is not None:
        ax.imshow(image, cmap="gray", origin="lower")

    for poly in polys:
        if poly.geom_type == "Polygon":
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth)

    ax.set_title(title)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Side-by-side method comparison
# ---------------------------------------------------------------------------

def compare_segmentations(
    method_polys: Dict[str, List[Polygon]],
    image: Optional[np.ndarray] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (18, 6),
) -> plt.Figure:
    """
    Side-by-side panel showing cell boundaries from each method on the same FOV.

    Parameters
    ----------
    method_polys : {'Cellpose': [...], 'Baysor': [...], 'Xenium': [...]}
    image        : background DAPI image array (shared across panels)
    colors       : per-method boundary colour
    """
    default_colors = {"Xenium": "yellow", "Cellpose": "cyan", "Baysor": "lime"}
    if colors is not None:
        default_colors.update(colors)

    methods = list(method_polys.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=figsize)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        plot_cell_boundaries(
            method_polys[method],
            image=image,
            color=default_colors.get(method, "white"),
            ax=ax,
            title=f"{method}\n({len(method_polys[method])} cells)",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------

def plot_transcripts_per_cell(
    method_counts: Dict[str, pd.Series],
    ax: Optional[plt.Axes] = None,
    bins: int = 60,
    log_scale: bool = True,
) -> plt.Axes:
    """Overlapping histograms of transcripts-per-cell for each method."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    colors = ["steelblue", "tomato", "seagreen"]
    for (method, counts), color in zip(method_counts.items(), colors):
        ax.hist(counts, bins=bins, alpha=0.6, label=method, color=color, density=True)

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel("Transcripts per cell")
    ax.set_ylabel("Density")
    ax.set_title("Transcripts per Cell Distribution")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# IoU heatmap / bar chart
# ---------------------------------------------------------------------------

def plot_iou_comparison(
    method_iou: Dict[str, List[float]],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of mean IoU ± std per method vs. DAPI reference."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    methods = list(method_iou.keys())
    means = [np.mean(v) for v in method_iou.values()]
    stds = [np.std(v) for v in method_iou.values()]
    colors = ["steelblue", "tomato", "seagreen"]

    bars = ax.bar(methods, means, yerr=stds, color=colors[: len(methods)], capsize=5, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean IoU vs. DAPI reference")
    ax.set_title("Segmentation Accuracy (IoU)")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.2f}", ha="center", fontsize=10)

    return ax
