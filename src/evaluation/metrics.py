"""
Segmentation evaluation metrics.

Computes overlap-based accuracy (IoU, Dice) against DAPI nucleus reference boundaries,
plus transcript-level and robustness summary statistics.
"""

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Polygon overlap metrics
# ---------------------------------------------------------------------------

def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Intersection-over-Union between two Shapely polygons."""
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return intersection / union if union > 0 else 0.0


def polygon_dice(poly_a: Polygon, poly_b: Polygon) -> float:
    """Dice coefficient between two Shapely polygons."""
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    intersection = poly_a.intersection(poly_b).area
    return (2 * intersection) / (poly_a.area + poly_b.area) if (poly_a.area + poly_b.area) > 0 else 0.0


def match_cells_to_reference(
    pred_polys: List[Polygon],
    ref_polys: List[Polygon],
    iou_threshold: float = 0.3,
) -> Dict:
    """
    Greedy matching of predicted cell polygons to reference (DAPI nucleus) polygons.

    Returns a dict with:
      - matched_iou: list of IoU scores for matched pairs
      - matched_dice: list of Dice scores for matched pairs
      - matched_cell_rate: fraction of reference nuclei that have a match
      - mean_iou, mean_dice
    """
    matched_iou = []
    matched_dice = []
    matched_ref = set()

    for pred in pred_polys:
        best_iou = 0.0
        best_ref_idx = -1
        for i, ref in enumerate(ref_polys):
            if i in matched_ref:
                continue
            iou = polygon_iou(pred, ref)
            if iou > best_iou:
                best_iou = iou
                best_ref_idx = i
        if best_iou >= iou_threshold and best_ref_idx >= 0:
            matched_iou.append(best_iou)
            matched_dice.append(polygon_dice(pred, ref_polys[best_ref_idx]))
            matched_ref.add(best_ref_idx)

    return {
        "matched_iou": matched_iou,
        "matched_dice": matched_dice,
        "matched_cell_rate": len(matched_ref) / len(ref_polys) if ref_polys else 0.0,
        "mean_iou": np.mean(matched_iou) if matched_iou else 0.0,
        "mean_dice": np.mean(matched_dice) if matched_dice else 0.0,
    }


# ---------------------------------------------------------------------------
# Transcript-level metrics
# ---------------------------------------------------------------------------

def transcript_assignment_stats(transcripts_df: pd.DataFrame, cell_col: str = "cell_id") -> Dict:
    """
    Summarise how transcripts are distributed across cells.

    Parameters
    ----------
    transcripts_df : DataFrame with at least `cell_col` column.
                     Unassigned transcripts should have cell_col == -1 or NaN.
    cell_col       : column name containing the cell assignment.
    """
    assigned = transcripts_df[transcripts_df[cell_col].notna() & (transcripts_df[cell_col] != -1)]
    unassigned_frac = 1 - len(assigned) / len(transcripts_df)

    per_cell = assigned.groupby(cell_col).size()
    return {
        "n_cells": per_cell.shape[0],
        "total_transcripts": len(transcripts_df),
        "assigned_transcripts": len(assigned),
        "unassigned_fraction": round(unassigned_frac, 4),
        "mean_transcripts_per_cell": round(per_cell.mean(), 2),
        "median_transcripts_per_cell": round(per_cell.median(), 2),
        "std_transcripts_per_cell": round(per_cell.std(), 2),
        "min_transcripts_per_cell": int(per_cell.min()),
        "max_transcripts_per_cell": int(per_cell.max()),
    }


# ---------------------------------------------------------------------------
# Robustness / stability metrics
# ---------------------------------------------------------------------------

def parameter_sensitivity(results: List[Dict], param_name: str, metric: str = "n_cells") -> pd.DataFrame:
    """
    Summarise how a metric varies across a parameter sweep.

    Parameters
    ----------
    results    : list of dicts, each with keys including `param_name` and `metric`.
    param_name : name of the swept parameter (e.g. 'diameter', 'prior_cell_radius').
    metric     : metric to analyse stability for.

    Returns a DataFrame and prints coefficient of variation (CV).
    """
    df = pd.DataFrame(results)[[param_name, metric]].sort_values(param_name)
    cv = df[metric].std() / df[metric].mean() if df[metric].mean() != 0 else float("nan")
    df["cv"] = round(cv, 4)
    return df


# ---------------------------------------------------------------------------
# Summary table across methods
# ---------------------------------------------------------------------------

def build_summary_table(method_stats: Dict[str, Dict]) -> pd.DataFrame:
    """
    Combine per-method stats dicts into a single comparison DataFrame.

    Parameters
    ----------
    method_stats : {'Cellpose': {...}, 'Baysor': {...}, 'Xenium': {...}}
    """
    rows = []
    for method, stats in method_stats.items():
        row = {"Method": method}
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")
