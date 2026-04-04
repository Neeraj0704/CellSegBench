"""
Notebook 02 — Cellpose Segmentation
Runs Cellpose on a representative tile of the DAPI image, extracts polygons,
assigns transcripts, and sweeps over diameter parameter.
"""
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
import json
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon
from cellpose import models

DATA_DIR = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)
PIXEL_SIZE = 0.2125  # µm per pixel

# ── 1. Load DAPI tile (ROI crop — manageable size for Cellpose) ─────────────
print("Loading DAPI crop for Cellpose...")
with open(DATA_DIR / "dapi_meta.json") as f:
    meta = json.load(f)

dapi_path = DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif"
with tifffile.TiffFile(dapi_path) as tif:
    page = tif.pages[0]
    H, W = page.shape
    # Use a 512 x 512 tile from the centre
    cy, cx = H // 2, W // 2
    r0 = max(0, cy - 256); r1 = min(H, cy + 256)
    c0 = max(0, cx - 256); c1 = min(W, cx + 256)
    dapi_tile = page.asarray()[r0:r1, c0:c1]

print(f"DAPI tile shape: {dapi_tile.shape}  ({dapi_tile.shape[0]*PIXEL_SIZE:.0f} x {dapi_tile.shape[1]*PIXEL_SIZE:.0f} µm)")
# Save tile metadata for later notebooks
tile_meta = {"r0": r0, "r1": r1, "c0": c0, "c1": c1,
             "x0_um": c0 * PIXEL_SIZE, "y0_um": r0 * PIXEL_SIZE,
             "x1_um": c1 * PIXEL_SIZE, "y1_um": r1 * PIXEL_SIZE}
with open(DATA_DIR / "tile_meta.json", "w") as f:
    json.dump(tile_meta, f)

# ── 2. Run Cellpose ────────────────────────────────────────────────────────
print("\nRunning Cellpose (nuclei model)...")
model = models.CellposeModel(gpu=False, model_type="nuclei")

DEFAULT_DIAMETER = 30  # pixels ≈ 6.4 µm nucleus diameter

masks, flows, styles = model.eval(
    dapi_tile,
    diameter=DEFAULT_DIAMETER,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
)
n_cells = int(masks.max())
print(f"Cellpose detected: {n_cells:,} nuclei (diameter={DEFAULT_DIAMETER}px)")

# ── 3. Visualise segmentation ──────────────────────────────────────────────
dapi_norm = (dapi_tile.astype("float32") - dapi_tile.min()) / (dapi_tile.max() - dapi_tile.min())

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(dapi_norm, cmap="gray", origin="lower")
axes[0].set_title("DAPI Input (4096×4096 tile)"); axes[0].axis("off")

axes[1].imshow(dapi_norm, cmap="gray", origin="lower")
axes[1].imshow(masks > 0, cmap="Reds", alpha=0.4, origin="lower")
axes[1].set_title(f"Cellpose Segmentation ({n_cells:,} nuclei)"); axes[1].axis("off")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "cellpose_segmentation_map.png", dpi=150)
plt.close()
print("Saved: cellpose_segmentation_map.png")
del dapi_norm

# ── 4. Extract polygon boundaries (pixel → µm) ─────────────────────────────
print("\nExtracting polygon boundaries...")

def masks_to_polygons_um(mask_array, pixel_size, x0_um=0, y0_um=0):
    polys = []
    for region in measure.regionprops(mask_array):
        contours = measure.find_contours(mask_array == region.label, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        # (col, row) in pixels → (x, y) in µm, offset to global coords
        xy_um = np.column_stack([
            contour[:, 1] * pixel_size + x0_um,
            contour[:, 0] * pixel_size + y0_um,
        ])
        poly = Polygon(xy_um)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
    return polys

cellpose_polys = masks_to_polygons_um(
    masks, PIXEL_SIZE,
    x0_um=tile_meta["x0_um"], y0_um=tile_meta["y0_um"]
)
print(f"Valid polygons: {len(cellpose_polys):,}")

areas = [p.area for p in cellpose_polys]
print(f"Cell area (µm²) — mean: {np.mean(areas):.1f}, median: {np.median(areas):.1f}, std: {np.std(areas):.1f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(areas, bins=60, color="steelblue", edgecolor="none")
ax.set_xlabel("Cell area (µm²)"); ax.set_ylabel("Count")
ax.set_title(f"Cellpose — Cell Area Distribution (n={len(areas):,})")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cellpose_area_distribution.png", dpi=150)
plt.close()
print("Saved: cellpose_area_distribution.png")

# ── 5. Assign transcripts to Cellpose cells ────────────────────────────────
print("\nAssigning transcripts to Cellpose cells...")
import pyarrow.parquet as pq

# Load only transcripts within tile bounds
x0, x1 = tile_meta["x0_um"], tile_meta["x1_um"]
y0, y1 = tile_meta["y0_um"], tile_meta["y1_um"]

t = pq.read_table(
    DATA_DIR / "transcripts_filtered.parquet",
    columns=["transcript_id", "cell_id", "feature_name", "x_location", "y_location", "qv"],
    filters=[
        ("x_location", ">=", x0), ("x_location", "<", x1),
        ("y_location", ">=", y0), ("y_location", "<", y1),
    ]
).to_pandas()

print(f"Transcripts in tile: {len(t):,}")

# Map µm → pixel → mask lookup
t["col_px"] = ((t["x_location"] - x0) / PIXEL_SIZE).astype(int).clip(0, masks.shape[1]-1)
t["row_px"] = ((t["y_location"] - y0) / PIXEL_SIZE).astype(int).clip(0, masks.shape[0]-1)
t["cellpose_cell_id"] = masks[t["row_px"].values, t["col_px"].values]

assigned = t[t["cellpose_cell_id"] > 0]
per_cell = assigned.groupby("cellpose_cell_id").size()

print(f"Assigned:   {len(assigned):,} ({100*len(assigned)/len(t):.1f}%)")
print(f"Unassigned: {len(t)-len(assigned):,} ({100*(len(t)-len(assigned))/len(t):.1f}%)")
print(f"Cells with transcripts: {per_cell.shape[0]:,}")
print(f"Transcripts/cell — mean: {per_cell.mean():.1f}, median: {per_cell.median():.1f}")

t.to_parquet(DATA_DIR / "transcripts_cellpose.parquet", index=False)
print("Saved: transcripts_cellpose.parquet")
del t, assigned

# ── 6. Parameter sweep — diameter ─────────────────────────────────────────
print("\nParameter sweep over Cellpose diameter...")
diameters = [20, 30, 40]
sweep = []
for diam in diameters:
    m, _, _ = model.eval(dapi_tile, diameter=diam)
    n = int(m.max())
    sweep.append({"diameter_px": diam, "diameter_um": round(diam * PIXEL_SIZE, 2), "n_cells": n})
    print(f"  diameter={diam}px ({diam*PIXEL_SIZE:.1f}µm) → {n:,} cells")
    del m

sweep_df = pd.DataFrame(sweep)
cv = sweep_df["n_cells"].std() / sweep_df["n_cells"].mean()
print(f"CV (cell count across sweep): {cv:.3f}")
sweep_df["cv"] = round(cv, 4)
sweep_df.to_csv(RESULTS_DIR / "cellpose_diameter_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep_df["diameter_px"], sweep_df["n_cells"], marker="o", color="steelblue")
ax.axvline(DEFAULT_DIAMETER, color="red", linestyle="--", label=f"Default ({DEFAULT_DIAMETER}px)")
ax.set_xlabel("Diameter (px)"); ax.set_ylabel("Cells detected")
ax.set_title("Cellpose — Parameter Sensitivity (Diameter)")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cellpose_diameter_sweep.png", dpi=150)
plt.close()
print("Saved: cellpose_diameter_sweep.png")

# ── 7. Save masks ──────────────────────────────────────────────────────────
np.save(DATA_DIR / "cellpose_masks.npy", masks)
print("Saved: cellpose_masks.npy")

print("\n=== NOTEBOOK 02 COMPLETE ===")
print(f"  Cellpose cells (tile):     {n_cells:,}")
print(f"  Valid polygons:            {len(cellpose_polys):,}")
print(f"  Mean cell area:            {np.mean(areas):.1f} µm²")
print(f"  CV across diameter sweep:  {cv:.3f}")
