"""
Full CellSegBench Pipeline — runs in ~1 hour on CPU
Covers: Cellpose segmentation, evaluation vs 10x reference, downstream analysis
Baysor skipped (requires Julia — handled separately)
"""
import sys, json, time
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)
PIXEL_SIZE = 0.2125  # µm/px

t_start = time.time()
def elapsed(): return f"[{(time.time()-t_start)/60:.1f} min]"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load DAPI tile (512×512 px = ~109×109 µm)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 1: Load DAPI tile  {elapsed()}")
print(f"{'='*60}")

dapi_path = DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif"
with tifffile.TiffFile(dapi_path) as tif:
    page = tif.pages[0]
    H, W = page.shape
    cy, cx = H // 2, W // 2
    r0, r1 = cy - 256, cy + 256
    c0, c1 = cx - 256, cx + 256
    dapi_tile = page.asarray()[r0:r1, c0:c1]

tile_meta = dict(r0=r0, r1=r1, c0=c0, c1=c1,
                 x0_um=c0*PIXEL_SIZE, y0_um=r0*PIXEL_SIZE,
                 x1_um=c1*PIXEL_SIZE, y1_um=r1*PIXEL_SIZE)
with open(DATA_DIR / "tile_meta.json", "w") as f:
    json.dump(tile_meta, f)

print(f"Tile: {dapi_tile.shape}  ({dapi_tile.shape[1]*PIXEL_SIZE:.0f}×{dapi_tile.shape[0]*PIXEL_SIZE:.0f} µm)")
print(f"Tile bounds: x=[{tile_meta['x0_um']:.0f}, {tile_meta['x1_um']:.0f}] µm, "
      f"y=[{tile_meta['y0_um']:.0f}, {tile_meta['y1_um']:.0f}] µm  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run Cellpose
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 2: Cellpose segmentation  {elapsed()}")
print(f"{'='*60}")

from cellpose import models as cp_models
model = cp_models.CellposeModel(gpu=False, model_type="nuclei")
print(f"Model loaded  {elapsed()}")

DEFAULT_DIAM = 30
masks, flows, styles = model.eval(dapi_tile, diameter=DEFAULT_DIAM,
                                   flow_threshold=0.4, cellprob_threshold=0.0)
n_cp = int(masks.max())
print(f"Cellpose: {n_cp} nuclei detected (diameter={DEFAULT_DIAM}px)  {elapsed()}")

# Save segmentation map
dapi_norm = (dapi_tile.astype("float32") - dapi_tile.min()) / (dapi_tile.max() - dapi_tile.min())
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(dapi_norm, cmap="gray", origin="lower")
axes[0].set_title("DAPI Input"); axes[0].axis("off")
axes[1].imshow(dapi_norm, cmap="gray", origin="lower")
axes[1].imshow(masks > 0, cmap="Reds", alpha=0.45, origin="lower")
axes[1].set_title(f"Cellpose ({n_cp} nuclei)"); axes[1].axis("off")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cellpose_segmentation_map.png", dpi=150)
plt.close()
print(f"Saved: cellpose_segmentation_map.png  {elapsed()}")
np.save(DATA_DIR / "cellpose_masks.npy", masks)

# ── Extract polygons (pixel → µm) ──────────────────────────────────────────
cp_polys = []
x0_um, y0_um = tile_meta["x0_um"], tile_meta["y0_um"]
for region in measure.regionprops(masks):
    contours = measure.find_contours(masks == region.label, 0.5)
    if not contours: continue
    c = max(contours, key=len)
    xy = np.column_stack([c[:,1]*PIXEL_SIZE + x0_um, c[:,0]*PIXEL_SIZE + y0_um])
    p = Polygon(xy)
    if p.is_valid and p.area > 0:
        cp_polys.append(p)
print(f"Polygons extracted: {len(cp_polys)}  {elapsed()}")

cp_areas = [p.area for p in cp_polys]
print(f"Cell area (µm²): mean={np.mean(cp_areas):.1f}, median={np.median(cp_areas):.1f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(cp_areas, bins=40, color="steelblue", edgecolor="none")
ax.set_xlabel("Cell area (µm²)"); ax.set_ylabel("Count")
ax.set_title(f"Cellpose — Cell Area Distribution (n={len(cp_areas)})")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "cellpose_area_distribution.png", dpi=150); plt.close()

# ── Assign transcripts to Cellpose cells ───────────────────────────────────
t = pq.read_table(DATA_DIR / "transcripts_filtered.parquet",
                  columns=["transcript_id","cell_id","feature_name","x_location","y_location"],
                  filters=[("x_location",">=",tile_meta["x0_um"]),
                            ("x_location","<", tile_meta["x1_um"]),
                            ("y_location",">=",tile_meta["y0_um"]),
                            ("y_location","<", tile_meta["y1_um"])]).to_pandas()

t["col_px"] = ((t["x_location"]-x0_um)/PIXEL_SIZE).astype(int).clip(0, masks.shape[1]-1)
t["row_px"] = ((t["y_location"]-y0_um)/PIXEL_SIZE).astype(int).clip(0, masks.shape[0]-1)
t["cellpose_cell_id"] = masks[t["row_px"].values, t["col_px"].values]

cp_assigned = t[t["cellpose_cell_id"] > 0]
cp_tpc = cp_assigned.groupby("cellpose_cell_id").size()
print(f"Transcript assignment: {len(cp_assigned):,}/{len(t):,} assigned ({100*len(cp_assigned)/len(t):.1f}%)")
print(f"Transcripts/cell: mean={cp_tpc.mean():.1f}, median={cp_tpc.median():.1f}  {elapsed()}")
t.to_parquet(DATA_DIR / "transcripts_cellpose.parquet", index=False)
del t, masks, flows, styles, dapi_tile, dapi_norm

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Evaluation vs 10x reference
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 3: Evaluation vs 10x reference  {elapsed()}")
print(f"{'='*60}")

# Load 10x nucleus boundaries clipped to tile
nb = pd.read_parquet(DATA_DIR / "nucleus_boundaries.parquet")
cells = pd.read_parquet(DATA_DIR / "cells.parquet")

# Clip to tile region
cells_in_tile = cells[
    (cells["x_centroid"] >= tile_meta["x0_um"]) &
    (cells["x_centroid"] <  tile_meta["x1_um"]) &
    (cells["y_centroid"] >= tile_meta["y0_um"]) &
    (cells["y_centroid"] <  tile_meta["y1_um"])
]
tile_cell_ids = set(cells_in_tile["cell_id"])
nb_tile = nb[nb["cell_id"].isin(tile_cell_ids)]
print(f"10x cells in tile: {len(cells_in_tile)}  (nucleus boundary vertices: {len(nb_tile):,})")

# Build reference polygons from 10x nucleus boundaries
ref_polys = []
for cid, grp in nb_tile.groupby("cell_id"):
    coords = list(zip(grp["vertex_x"], grp["vertex_y"]))
    if len(coords) >= 3:
        p = Polygon(coords)
        if p.is_valid and p.area > 0:
            ref_polys.append(p)
print(f"Reference polygons built: {len(ref_polys)}  {elapsed()}")

# IoU matching (greedy)
def match_iou(pred_polys, ref_polys, thresh=0.3):
    from shapely.strtree import STRtree
    tree = STRtree(ref_polys)
    matched_iou, matched_dice, matched_ref = [], [], set()
    for pred in pred_polys:
        candidates = tree.query(pred)
        best_iou, best_idx = 0, -1
        for idx in candidates:
            if idx in matched_ref: continue
            ref = ref_polys[idx]
            inter = pred.intersection(ref).area
            union = pred.union(ref).area
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= thresh and best_idx >= 0:
            matched_iou.append(best_iou)
            ref = ref_polys[best_idx]
            dice = 2*pred.intersection(ref).area / (pred.area+ref.area)
            matched_dice.append(dice)
            matched_ref.add(best_idx)
    return matched_iou, matched_dice, len(matched_ref)/len(ref_polys) if ref_polys else 0

cp_iou, cp_dice, cp_recall = match_iou(cp_polys, ref_polys)
print(f"\n--- Cellpose vs 10x reference ---")
print(f"  Matched cells:      {len(cp_iou)} / {len(ref_polys)} reference nuclei ({cp_recall*100:.1f}% recall)")
print(f"  Mean IoU:           {np.mean(cp_iou):.3f}")
print(f"  Mean Dice:          {np.mean(cp_dice):.3f}")
print(f"  {elapsed()}")

# Summary stats table
cells_in_tile_data = pd.read_parquet(DATA_DIR / "cells.parquet")
xenium_tpc = pd.read_parquet(DATA_DIR / "transcripts_filtered.parquet",
                              columns=["cell_id","x_location","y_location"]).pipe(
    lambda df: df[(df["x_location"] >= tile_meta["x0_um"]) &
                  (df["x_location"] <  tile_meta["x1_um"]) &
                  (df["y_location"] >= tile_meta["y0_um"]) &
                  (df["y_location"] <  tile_meta["y1_um"])]
)
xen_assigned = xenium_tpc[xenium_tpc["cell_id"] != "UNASSIGNED"]
xen_tpc = xen_assigned.groupby("cell_id").size()

summary = pd.DataFrame({
    "Method": ["Xenium (10x)", "Cellpose"],
    "Cells detected": [len(cells_in_tile), n_cp],
    "Transcripts assigned": [len(xen_assigned), len(cp_assigned)],
    "Unassigned %": [
        round(100*(len(xenium_tpc)-len(xen_assigned))/len(xenium_tpc),1),
        round(100*(len(xenium_tpc)-len(cp_assigned))/len(xenium_tpc),1),
    ],
    "Mean TPC": [round(xen_tpc.mean(),1), round(cp_tpc.mean(),1)],
    "Median TPC": [round(xen_tpc.median(),1), round(cp_tpc.median(),1)],
    "Mean IoU vs ref": ["—", round(np.mean(cp_iou),3)],
    "Mean Dice vs ref": ["—", round(np.mean(cp_dice),3)],
    "Recall vs ref": ["—", round(cp_recall,3)],
})
summary.to_csv(RESULTS_DIR / "method_summary_stats.csv", index=False)
print(f"\n{summary.to_string(index=False)}")

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics = ["Cells detected", "Mean TPC", "Unassigned %"]
colors  = [["steelblue","tomato"], ["steelblue","tomato"], ["steelblue","tomato"]]
for ax, metric, col in zip(axes, metrics, colors):
    vals = summary[metric].tolist()
    ax.bar(summary["Method"], [float(v) for v in vals], color=col)
    ax.set_title(metric); ax.set_ylabel(metric)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "method_comparison_bar.png", dpi=150); plt.close()

# IoU distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(cp_iou, bins=30, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(np.mean(cp_iou), color="red", linestyle="--", label=f"Mean={np.mean(cp_iou):.3f}")
ax.set_xlabel("IoU vs 10x nucleus reference"); ax.set_ylabel("Count")
ax.set_title("Cellpose — IoU Distribution"); ax.legend()
plt.tight_layout(); plt.savefig(RESULTS_DIR / "cellpose_iou_distribution.png", dpi=150); plt.close()
print(f"Saved evaluation figures  {elapsed()}")

# Side-by-side boundary overlay
dapi_path2 = DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif"
with tifffile.TiffFile(dapi_path2) as tif:
    bg = tif.pages[0].asarray()[r0:r1, c0:c1]
bg_norm = (bg.astype("float32") - bg.min()) / (bg.max() - bg.min())

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, polys, color, title in zip(
    axes,
    [ref_polys, cp_polys],
    ["yellow", "cyan"],
    [f"10x Reference ({len(ref_polys)} nuclei)", f"Cellpose ({len(cp_polys)} cells)"]
):
    ax.imshow(bg_norm, cmap="gray", origin="lower")
    for poly in polys:
        xs = (np.array(poly.exterior.xy[0]) - x0_um) / PIXEL_SIZE
        ys = (np.array(poly.exterior.xy[1]) - y0_um) / PIXEL_SIZE
        ax.plot(xs, ys, color=color, linewidth=0.6)
    ax.set_title(title); ax.axis("off")
plt.suptitle("Segmentation Comparison — 109×109 µm tile", fontsize=13)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "side_by_side_segmentation.png", dpi=150); plt.close()
del bg, bg_norm
print(f"Saved: side_by_side_segmentation.png  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Downstream biological analysis (using full 10x matrix)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 4: Downstream analysis  {elapsed()}")
print(f"{'='*60}")

import scanpy as sc
import anndata as ad
sc.settings.verbosity = 0

# Load 10x cell feature matrix (full tissue — already segmented)
print("Loading 10x cell feature matrix...")
adata = sc.read_10x_h5(DATA_DIR / "cell_feature_matrix.h5")
cells_df = pd.read_parquet(DATA_DIR / "cells.parquet").set_index("cell_id")
adata.obs = adata.obs.join(cells_df[["x_centroid","y_centroid","cell_area","nucleus_area"]], how="left")
adata.obsm["spatial"] = adata.obs[["x_centroid","y_centroid"]].values
print(f"AnnData: {adata.n_obs:,} cells × {adata.n_vars} genes  {elapsed()}")

# QC filter
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=10)
print(f"After QC: {adata.n_obs:,} cells × {adata.n_vars} genes")

# Normalise + log
sc.pp.normalize_total(adata, target_sum=100)
sc.pp.log1p(adata)

# HVGs → PCA → neighbours → UMAP → Leiden
# Only 280 genes — use all of them, skip HVG selection
adata.var["highly_variable"] = True
sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.4, key_added="leiden")
print(f"Clustering done: {adata.obs['leiden'].nunique()} Leiden clusters  {elapsed()}")

# UMAP plot
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata, color="leiden", ax=ax, show=False,
           title=f"10x Xenium — UMAP ({adata.n_obs:,} cells, {adata.obs['leiden'].nunique()} clusters)",
           legend_loc="on data", legend_fontsize=9)
plt.tight_layout(); plt.savefig(RESULTS_DIR / "umap_leiden_10x.png", dpi=150); plt.close()
print(f"Saved: umap_leiden_10x.png  {elapsed()}")

# Marker gene dot plot
candidates = ["EPCAM","KRT17","CD68","CD3E","VIM","ACTA2","PECAM1","MKI67"]
markers = [g for g in candidates if g in adata.var_names]
if markers:
    fig, ax = plt.subplots(figsize=(max(8, len(markers)*1.2), 5))
    sc.pl.dotplot(adata, var_names=markers, groupby="leiden",
                  ax=ax, show=False, standard_scale="var",
                  title="Marker Gene Expression by Cluster")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "marker_dotplot_10x.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: marker_dotplot_10x.png  {elapsed()}")

# Cluster proportions
fig, ax = plt.subplots(figsize=(8, 4))
props = adata.obs["leiden"].value_counts(normalize=True).sort_index()
ax.bar(props.index.astype(str), props.values,
       color=plt.cm.tab20.colors[:len(props)])
ax.set_xlabel("Leiden cluster"); ax.set_ylabel("Proportion of cells")
ax.set_title("Cell Type Cluster Proportions (10x segmentation)")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "cluster_proportions_10x.png", dpi=150); plt.close()
print(f"Saved: cluster_proportions_10x.png  {elapsed()}")

# Spatial plot
fig, ax = plt.subplots(figsize=(9, 9))
palette = plt.cm.tab20.colors
for i, cluster in enumerate(sorted(adata.obs["leiden"].unique())):
    mask = adata.obs["leiden"] == cluster
    coords = adata.obsm["spatial"][mask.values]
    ax.scatter(coords[:,0], coords[:,1], s=0.5, alpha=0.6,
               color=palette[int(cluster) % 20], label=cluster)
ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
ax.set_title("Spatial Distribution of Leiden Clusters (10x)")
ax.set_aspect("equal")
ax.legend(markerscale=8, fontsize=7, ncol=2, loc="upper right")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "spatial_clusters_10x.png", dpi=150); plt.close()
print(f"Saved: spatial_clusters_10x.png  {elapsed()}")

# Save processed AnnData
adata.write_h5ad(DATA_DIR / "adata_10x_processed.h5ad")
print(f"Saved: adata_10x_processed.h5ad  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
total = (time.time()-t_start)/60
print(f"\n{'='*60}")
print(f"PIPELINE COMPLETE in {total:.1f} minutes")
print(f"{'='*60}")
print(f"\nResults saved to: {RESULTS_DIR.resolve()}")
for f in sorted(RESULTS_DIR.iterdir()):
    if f.suffix in [".png",".csv"]:
        print(f"  {f.name}")

print(f"\nKey results:")
print(f"  10x cells (full tissue):    {adata.n_obs:,}")
print(f"  Leiden clusters:            {adata.obs['leiden'].nunique()}")
print(f"  Cellpose cells (tile):      {n_cp}")
print(f"  Cellpose mean IoU vs ref:   {np.mean(cp_iou):.3f}")
print(f"  Cellpose recall vs ref:     {cp_recall*100:.1f}%")
