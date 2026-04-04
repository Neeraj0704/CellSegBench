"""
Notebook 01 — Preprocessing & Data Exploration
Uses PyArrow predicate pushdown to filter before loading into pandas.
"""
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

DATA_DIR = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── 1. Load & filter with PyArrow (pushdown — never loads full file) ────────
print("Reading transcripts with PyArrow filter pushdown...")
pf = pq.ParquetFile(DATA_DIR / "transcripts.parquet")
print(f"Total rows in file: {pf.metadata.num_rows:,}")

# Filter: is_gene=True AND qv>=20; load only needed columns
COLS = ["transcript_id", "cell_id", "feature_name",
        "x_location", "y_location", "z_location", "qv"]

import pyarrow as pa
table = pq.read_table(
    DATA_DIR / "transcripts.parquet",
    columns=COLS + ["is_gene"],
    filters=[("is_gene", "=", True), ("qv", ">=", 20.0)],
)
transcripts_filtered = table.to_pandas()
del table
transcripts_filtered.drop(columns=["is_gene"], inplace=True)

# Downcast to save RAM
for col in ["x_location", "y_location", "z_location", "qv"]:
    transcripts_filtered[col] = transcripts_filtered[col].astype("float32")
transcripts_filtered["feature_name"] = transcripts_filtered["feature_name"].astype("category")
transcripts_filtered["cell_id"] = transcripts_filtered["cell_id"].astype("category")

n_filt = len(transcripts_filtered)
mem_gb = transcripts_filtered.memory_usage(deep=True).sum() / 1e9
print(f"Filtered transcripts: {n_filt:,} ({mem_gb:.2f} GB in memory)")

# ── 2. Q-score plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(transcripts_filtered["qv"], bins=50, color="steelblue", edgecolor="none")
ax.axvline(20, color="red", linestyle="--", label="Q20 threshold")
ax.set_xlabel("Q-score"); ax.set_ylabel("Count")
ax.set_title(f"Q-score Distribution (Q≥20 filter applied, n={n_filt:,})")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "qscore_filtering.png", dpi=150)
plt.close()
print("Saved: qscore_filtering.png")

# ── 3. Gene panel overview ─────────────────────────────────────────────────
gene_counts = transcripts_filtered["feature_name"].value_counts()
print(f"\nGenes in panel: {len(gene_counts)}")
print("Top 15:")
print(gene_counts.head(15).to_string())

fig, ax = plt.subplots(figsize=(12, 4))
gene_counts.head(30).plot(kind="bar", ax=ax, color="steelblue", edgecolor="none")
ax.set_xlabel("Gene"); ax.set_ylabel("Transcript count")
ax.set_title("Top 30 Most Detected Genes (Q ≥ 20)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "top_genes.png", dpi=150)
plt.close()
print("Saved: top_genes.png")

# ── 4. Spatial distribution ────────────────────────────────────────────────
print("\nSpatial distribution plot (150k sample)...")
sample = transcripts_filtered.sample(n=min(150_000, n_filt), random_state=42)
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(sample["x_location"], sample["y_location"], s=0.1, alpha=0.15, color="steelblue")
ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
ax.set_title("Spatial Distribution of Transcripts (150k sample)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "transcript_spatial_distribution.png", dpi=150)
plt.close()
del sample
print("Saved: transcript_spatial_distribution.png")

# ── 5. Marker genes ────────────────────────────────────────────────────────
candidates = ["EPCAM", "CD68", "CD3E", "KRT17", "KRT8", "VIM", "ACTA2", "PTPRC"]
marker_genes = [g for g in candidates if g in gene_counts.index][:4]
print(f"\nMarker genes: {marker_genes}")

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()
colors = ["tomato", "steelblue", "seagreen", "darkorange"]
for ax, gene, color in zip(axes, marker_genes, colors):
    sub = transcripts_filtered[transcripts_filtered["feature_name"] == gene]
    ax.scatter(sub["x_location"], sub["y_location"], s=0.3, alpha=0.4, color=color)
    ax.set_title(f"{gene} ({len(sub):,} transcripts)", fontsize=13)
    ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Marker Gene Spatial Patterns", fontsize=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "marker_genes_spatial.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: marker_genes_spatial.png")

# ── 6. 10x reference ───────────────────────────────────────────────────────
print("\nLoading 10x reference data...")
cells = pd.read_parquet(DATA_DIR / "cells.parquet")
n_cells = len(cells)
print(f"10x cells: {n_cells:,}")
print(f"Segmentation method: {cells['segmentation_method'].iloc[0]}")
print("\nCell area (µm²):")
print(cells["cell_area"].describe().round(2).to_string())

assigned_10x = transcripts_filtered[transcripts_filtered["cell_id"] != "UNASSIGNED"]
n_assigned = len(assigned_10x)
print(f"\n10x assigned:   {n_assigned:,} ({100*n_assigned/n_filt:.1f}%)")
print(f"10x unassigned: {n_filt-n_assigned:,} ({100*(n_filt-n_assigned)/n_filt:.1f}%)")
per_cell = assigned_10x.groupby("cell_id", observed=True).size()
print(f"Transcripts/cell — mean: {per_cell.mean():.1f}, median: {per_cell.median():.1f}")

# ── 7. Cell area distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(cells["cell_area"], bins=60, color="steelblue", edgecolor="none")
ax.set_xlabel("Cell area (µm²)"); ax.set_ylabel("Count")
ax.set_title(f"10x Cell Area Distribution (n={n_cells:,})")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cell_area_distribution_10x.png", dpi=150)
plt.close()
print("Saved: cell_area_distribution_10x.png")

# ── 8. Save filtered transcripts (before DAPI to avoid OOM) ───────────────
print("\nSaving filtered transcripts...")
transcripts_filtered.to_parquet(DATA_DIR / "transcripts_filtered.parquet", index=False)
print(f"Saved: transcripts_filtered.parquet ({n_filt:,} rows)")
del transcripts_filtered  # free RAM before loading DAPI

# ── 9. DAPI image (load only a crop via memmap) ────────────────────────────
print("\nLoading DAPI image (cropped thumbnail for visualisation)...")
import json
dapi_path = DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif"
with tifffile.TiffFile(dapi_path) as tif:
    # Read metadata without loading data
    page = tif.pages[0]
    h, w = page.shape
    dtype = page.dtype
    print(f"DAPI full shape: ({h}, {w}), dtype: {dtype}")
    # Load only a 2048x2048 crop from the centre for visualisation
    cy, cx = h // 2, w // 2
    r0, r1 = max(0, cy - 1024), min(h, cy + 1024)
    c0, c1 = max(0, cx - 1024), min(w, cx + 1024)
    dapi_crop = page.asarray()[r0:r1, c0:c1]

with open(DATA_DIR / "dapi_meta.json", "w") as f:
    json.dump({"shape": [h, w], "dtype": str(dtype),
               "pixel_size_um": 0.2125,
               "crop": [r0, r1, c0, c1]}, f)

dapi_norm = (dapi_crop.astype("float32") - dapi_crop.min()) / (dapi_crop.max() - dapi_crop.min())
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(dapi_norm, cmap="gray", origin="lower")
ax.set_title(f"DAPI Nuclear Staining (2048×2048 centre crop)")
ax.axis("off")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "dapi_image.png", dpi=150)
plt.close()
del dapi_crop, dapi_norm
print("Saved: dapi_image.png")

print("\n=== NOTEBOOK 01 COMPLETE ===")
print(f"  Filtered transcripts:  {n_filt:,}")
print(f"  Unique genes:          {len(gene_counts)}")
print(f"  10x cells:             {n_cells:,}")
print(f"  10x assignment rate:   {100*n_assigned/n_filt:.1f}%")
print(f"  Results → {RESULTS_DIR.resolve()}")
