"""
Remaining pipeline:
  1. Transcript-based GMM segmentation (Baysor-style spatial baseline)
  2. Cellpose parameter sweep
  3. 3-way evaluation comparison
  4. Per-method downstream comparison
"""
import sys, json, time, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import scanpy as sc
from scipy.sparse import csr_matrix
import anndata as ad

DATA_DIR    = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)
PIXEL_SIZE  = 0.2125

t0 = time.time()
def elapsed(): return f"[{(time.time()-t0)/60:.1f} min]"

# ─── Load tile meta & tile transcripts ────────────────────────────────────
with open(DATA_DIR / "tile_meta.json") as f:
    tile = json.load(f)
x0, x1 = tile["x0_um"], tile["x1_um"]
y0, y1 = tile["y0_um"], tile["y1_um"]

t = pq.read_table(
    DATA_DIR / "transcripts_filtered.parquet",
    columns=["transcript_id","cell_id","feature_name","x_location","y_location"],
    filters=[("x_location",">=",x0),("x_location","<",x1),
             ("y_location",">=",y0),("y_location","<",y1)]
).to_pandas()
print(f"Tile transcripts: {len(t):,}  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Transcript-based GMM segmentation (Baysor-style baseline)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 1: Transcript-based GMM segmentation  {elapsed()}")
print(f"{'='*60}")

# Use 10x cell count as prior for n_components
n_ref = 97  # from step 3 earlier
coords = t[["x_location","y_location"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

gmm = GaussianMixture(n_components=n_ref, covariance_type="full",
                       max_iter=100, random_state=42, n_init=1)
gmm.fit(coords_scaled)
labels = gmm.predict(coords_scaled)  # 0-based cell labels
t["gmm_cell_id"] = labels + 1        # 1-based

n_gmm = t["gmm_cell_id"].nunique()
gmm_assigned = t[t["gmm_cell_id"] > 0]
gmm_tpc = gmm_assigned.groupby("gmm_cell_id").size()
print(f"GMM cells: {n_gmm}  {elapsed()}")
print(f"Transcripts/cell: mean={gmm_tpc.mean():.1f}, median={gmm_tpc.median():.1f}")

# Build convex-hull polygons per GMM cell
gmm_polys = []
for cid, grp in t.groupby("gmm_cell_id"):
    pts = grp[["x_location","y_location"]].values
    if len(pts) < 4:
        continue
    try:
        hull = ConvexHull(pts)
        poly = Polygon(pts[hull.vertices])
        if poly.is_valid and poly.area > 0:
            gmm_polys.append(poly)
    except Exception:
        pass
print(f"GMM polygons: {len(gmm_polys)}  {elapsed()}")

t.to_parquet(DATA_DIR / "transcripts_gmm.parquet", index=False)

# ── GMM spatial map ────────────────────────────────────────────────────────
import tifffile
with tifffile.TiffFile(DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif") as tif:
    r0i, r1i = int(y0/PIXEL_SIZE), int(y1/PIXEL_SIZE)
    c0i, c1i = int(x0/PIXEL_SIZE), int(x1/PIXEL_SIZE)
    bg = tif.pages[0].asarray()[r0i:r1i, c0i:c1i]
bg_norm = (bg.astype("float32") - bg.min()) / (bg.max() - bg.min())

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(bg_norm, cmap="gray", origin="lower")
colors = plt.cm.tab20.colors
for i, poly in enumerate(gmm_polys):
    xs = (np.array(poly.exterior.xy[0]) - x0) / PIXEL_SIZE
    ys = (np.array(poly.exterior.xy[1]) - y0) / PIXEL_SIZE
    ax.plot(xs, ys, color=colors[i % 20], linewidth=0.7, alpha=0.8)
ax.set_title(f"GMM Transcript Segmentation ({len(gmm_polys)} cells)"); ax.axis("off")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gmm_segmentation_map.png", dpi=150); plt.close()
print(f"Saved: gmm_segmentation_map.png  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Cellpose parameter sweep (diameters 20, 30, 40)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 2: Cellpose parameter sweep  {elapsed()}")
print(f"{'='*60}")

from cellpose import models as cp_models
model = cp_models.CellposeModel(gpu=False, model_type="nuclei")

with tifffile.TiffFile(DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif") as tif:
    dapi_tile = tif.pages[0].asarray()[r0i:r1i, c0i:c1i]

sweep_rows = []
diameters = [20, 30, 40]
for diam in diameters:
    m, _, _ = model.eval(dapi_tile, diameter=diam,
                          flow_threshold=0.4, cellprob_threshold=0.0)
    n = int(m.max())
    sweep_rows.append({"diameter_px": diam,
                        "diameter_um": round(diam * PIXEL_SIZE, 2),
                        "n_cells": n})
    print(f"  diameter={diam}px ({diam*PIXEL_SIZE:.1f}µm) → {n} cells  {elapsed()}")

sweep_df = pd.DataFrame(sweep_rows)
cv = sweep_df["n_cells"].std() / sweep_df["n_cells"].mean()
sweep_df["cv"] = round(cv, 4)
sweep_df.to_csv(RESULTS_DIR / "cellpose_diameter_sweep.csv", index=False)
print(f"CV (cell count stability): {cv:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep_df["diameter_px"], sweep_df["n_cells"], marker="o", color="steelblue", linewidth=2)
ax.axvline(30, color="red", linestyle="--", label="Default (30px)")
for _, row in sweep_df.iterrows():
    ax.annotate(f"{int(row['n_cells'])}", (row["diameter_px"], row["n_cells"]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_xlabel("Diameter (px)"); ax.set_ylabel("Cells detected")
ax.set_title("Cellpose — Parameter Sensitivity (Diameter)")
ax.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR / "cellpose_diameter_sweep.png", dpi=150); plt.close()
print(f"Saved: cellpose_diameter_sweep.png  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — 3-way evaluation comparison
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 3: 3-way evaluation  {elapsed()}")
print(f"{'='*60}")

# Load Cellpose results & reference polygons
masks = np.load(DATA_DIR / "cellpose_masks.npy")

cp_polys = []
for region in measure.regionprops(masks):
    contours = measure.find_contours(masks == region.label, 0.5)
    if not contours: continue
    c = max(contours, key=len)
    xy = np.column_stack([c[:,1]*PIXEL_SIZE + x0, c[:,0]*PIXEL_SIZE + y0])
    p = Polygon(xy)
    if p.is_valid and p.area > 0:
        cp_polys.append(p)

cells   = pd.read_parquet(DATA_DIR / "cells.parquet")
nb      = pd.read_parquet(DATA_DIR / "nucleus_boundaries.parquet")
tile_cells = cells[(cells["x_centroid"]>=x0)&(cells["x_centroid"]<x1)&
                    (cells["y_centroid"]>=y0)&(cells["y_centroid"]<y1)]
nb_tile = nb[nb["cell_id"].isin(set(tile_cells["cell_id"]))]

ref_polys = []
for cid, grp in nb_tile.groupby("cell_id"):
    coords_poly = list(zip(grp["vertex_x"], grp["vertex_y"]))
    if len(coords_poly) >= 3:
        p = Polygon(coords_poly)
        if p.is_valid and p.area > 0:
            ref_polys.append(p)

def match_iou(pred_polys, ref_polys, thresh=0.3):
    from shapely.strtree import STRtree
    tree = STRtree(ref_polys)
    iou_scores, dice_scores, matched_ref = [], [], set()
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
            iou_scores.append(best_iou)
            ref = ref_polys[best_idx]
            dice_scores.append(2*pred.intersection(ref).area/(pred.area+ref.area))
            matched_ref.add(best_idx)
    recall = len(matched_ref)/len(ref_polys) if ref_polys else 0
    return iou_scores, dice_scores, recall

cp_iou,  cp_dice,  cp_recall  = match_iou(cp_polys,  ref_polys)
gmm_iou, gmm_dice, gmm_recall = match_iou(gmm_polys, ref_polys)

# Transcript stats per method
cp_t  = pd.read_parquet(DATA_DIR / "transcripts_cellpose.parquet")
gmm_t = pd.read_parquet(DATA_DIR / "transcripts_gmm.parquet")

xen_assigned = t[t["cell_id"] != "UNASSIGNED"]
cp_assigned  = cp_t[cp_t["cellpose_cell_id"] > 0]
gmm_assigned_df = gmm_t[gmm_t["gmm_cell_id"] > 0]

xen_tpc = xen_assigned.groupby("cell_id").size()
cp_tpc  = cp_assigned.groupby("cellpose_cell_id").size()
gmm_tpc2 = gmm_assigned_df.groupby("gmm_cell_id").size()

summary = pd.DataFrame([
    {"Method": "Xenium (10x)",
     "Cells": len(tile_cells),
     "Assigned %": round(100*len(xen_assigned)/len(t), 1),
     "Mean TPC": round(xen_tpc.mean(), 1),
     "Median TPC": round(xen_tpc.median(), 1),
     "Mean IoU": "—", "Mean Dice": "—", "Recall": "—"},
    {"Method": "Cellpose",
     "Cells": len(cp_polys),
     "Assigned %": round(100*len(cp_assigned)/len(t), 1),
     "Mean TPC": round(cp_tpc.mean(), 1),
     "Median TPC": round(cp_tpc.median(), 1),
     "Mean IoU": round(np.mean(cp_iou), 3),
     "Mean Dice": round(np.mean(cp_dice), 3),
     "Recall": round(cp_recall, 3)},
    {"Method": "GMM (transcript)",
     "Cells": len(gmm_polys),
     "Assigned %": 100.0,
     "Mean TPC": round(gmm_tpc2.mean(), 1),
     "Median TPC": round(gmm_tpc2.median(), 1),
     "Mean IoU": round(np.mean(gmm_iou), 3),
     "Mean Dice": round(np.mean(gmm_dice), 3),
     "Recall": round(gmm_recall, 3)},
])
summary.to_csv(RESULTS_DIR / "method_summary_stats.csv", index=False)
print(f"\n{summary.to_string(index=False)}")

# IoU comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods_with_iou = ["Cellpose", "GMM (transcript)"]
iou_vals  = [np.mean(cp_iou), np.mean(gmm_iou)]
dice_vals = [np.mean(cp_dice), np.mean(gmm_dice)]
rec_vals  = [cp_recall, gmm_recall]

for ax, vals, ylabel in zip(axes,
    [iou_vals, dice_vals, rec_vals],
    ["Mean IoU", "Mean Dice", "Recall (vs 10x ref)"]):
    bars = ax.bar(methods_with_iou, vals,
                  color=["steelblue","seagreen"], alpha=0.85)
    ax.set_ylim(0, 1); ax.set_ylabel(ylabel); ax.set_title(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
plt.suptitle("Segmentation Method Comparison vs 10x Reference", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "three_way_iou_comparison.png", dpi=150); plt.close()
print(f"Saved: three_way_iou_comparison.png  {elapsed()}")

# Side-by-side 3-panel
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
panels = [
    (ref_polys,  "yellow", f"Xenium 10x Reference ({len(ref_polys)} nuclei)"),
    (cp_polys,   "cyan",   f"Cellpose ({len(cp_polys)} cells, IoU={np.mean(cp_iou):.2f})"),
    (gmm_polys,  "lime",   f"GMM Transcript ({len(gmm_polys)} cells, IoU={np.mean(gmm_iou):.2f})"),
]
for ax, (polys, color, title) in zip(axes, panels):
    ax.imshow(bg_norm, cmap="gray", origin="lower")
    for poly in polys:
        xs = (np.array(poly.exterior.xy[0]) - x0) / PIXEL_SIZE
        ys = (np.array(poly.exterior.xy[1]) - y0) / PIXEL_SIZE
        ax.plot(xs, ys, color=color, linewidth=0.7)
    ax.set_title(title, fontsize=11); ax.axis("off")
plt.suptitle("Segmentation Comparison — 109×109 µm tile", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "side_by_side_3way.png", dpi=150); plt.close()
print(f"Saved: side_by_side_3way.png  {elapsed()}")

# Transcripts-per-cell distribution
fig, ax = plt.subplots(figsize=(8, 5))
for tpc, label, color in [
    (xen_tpc, "Xenium 10x", "gold"),
    (cp_tpc,  "Cellpose",   "steelblue"),
    (gmm_tpc2,"GMM",        "seagreen"),
]:
    ax.hist(tpc, bins=30, alpha=0.6, label=label, color=color, density=True)
ax.set_xlabel("Transcripts per cell"); ax.set_ylabel("Density")
ax.set_title("Transcripts per Cell — All Methods")
ax.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR / "transcripts_per_cell_comparison.png", dpi=150); plt.close()
print(f"Saved: transcripts_per_cell_comparison.png  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Per-method downstream comparison
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 4: Per-method downstream (tile-level)  {elapsed()}")
print(f"{'='*60}")

sc.settings.verbosity = 0

def build_adata(transcripts_df, cell_col, gene_col="feature_name"):
    assigned = transcripts_df[
        transcripts_df[cell_col].notna() &
        (transcripts_df[cell_col] != 0) &
        (transcripts_df[cell_col] != "UNASSIGNED")
    ].copy()
    assigned[cell_col] = assigned[cell_col].astype(str)
    counts = assigned.groupby([cell_col, gene_col]).size().unstack(fill_value=0)
    if counts.shape[0] < 10:
        return None
    adata = ad.AnnData(X=csr_matrix(counts.values))
    adata.obs_names = counts.index.astype(str)
    adata.var_names = counts.columns
    centroids = assigned.groupby(cell_col)[["x_location","y_location"]].mean()
    adata.obs = adata.obs.join(centroids.rename(columns={"x_location":"x","y_location":"y"}))
    adata.obsm["spatial"] = adata.obs[["x","y"]].values
    return adata

methods_data = {
    "Xenium_10x": (t,     "cell_id"),
    "Cellpose":   (cp_t,  "cellpose_cell_id"),
    "GMM":        (gmm_t, "gmm_cell_id"),
}

adatas = {}
for name, (df, col) in methods_data.items():
    a = build_adata(df, col)
    if a is None:
        print(f"  {name}: too few cells, skipping")
        continue
    sc.pp.filter_cells(a, min_counts=5)
    sc.pp.filter_genes(a, min_cells=2)
    if a.n_obs < 5:
        print(f"  {name}: {a.n_obs} cells after QC, skipping")
        continue
    sc.pp.normalize_total(a, target_sum=100)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=min(20, a.n_obs-1, a.n_vars-1))
    sc.pp.neighbors(a, n_neighbors=min(10, a.n_obs-1), n_pcs=min(10, a.n_obs-1))
    sc.tl.umap(a)
    sc.tl.leiden(a, resolution=0.5, key_added="leiden")
    adatas[name] = a
    print(f"  {name}: {a.n_obs} cells, {a.obs['leiden'].nunique()} clusters  {elapsed()}")

if adatas:
    ncols = len(adatas)
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6))
    if ncols == 1: axes = [axes]
    for ax, (name, a) in zip(axes, adatas.items()):
        sc.pl.umap(a, color="leiden", ax=ax, show=False,
                   title=f"{name}\n({a.n_obs} cells, {a.obs['leiden'].nunique()} clusters)",
                   legend_loc="on data", legend_fontsize=9)
    plt.suptitle("UMAP Comparison Across Segmentation Methods (tile)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_comparison_methods.png", dpi=150); plt.close()
    print(f"Saved: umap_comparison_methods.png  {elapsed()}")

    # Spatial cluster maps per method
    fig, axes = plt.subplots(1, ncols, figsize=(8*ncols, 7))
    if ncols == 1: axes = [axes]
    palette = plt.cm.tab10.colors
    for ax, (name, a) in zip(axes, adatas.items()):
        for cluster in sorted(a.obs["leiden"].unique(), key=int):
            mask = a.obs["leiden"] == cluster
            coords = a.obsm["spatial"][mask.values]
            ax.scatter(coords[:,0], coords[:,1], s=30, alpha=0.8,
                       color=palette[int(cluster) % 10], label=f"C{cluster}")
        ax.set_title(f"{name} — Spatial Clusters"); ax.set_aspect("equal")
        ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
        ax.legend(markerscale=2, fontsize=8)
    plt.suptitle("Spatial Cluster Comparison (tile)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "spatial_comparison_methods.png", dpi=150); plt.close()
    print(f"Saved: spatial_comparison_methods.png  {elapsed()}")

# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
total = (time.time()-t0)/60
print(f"\n{'='*60}")
print(f"ALL REMAINING STEPS COMPLETE in {total:.1f} minutes")
print(f"{'='*60}")
print(f"\nFinal figures & results:")
for f in sorted(RESULTS_DIR.iterdir()):
    if f.suffix in [".png", ".csv"]:
        print(f"  {f.name}")
print(f"\nKey results:")
print(f"  Cellpose  — IoU: {np.mean(cp_iou):.3f}, Dice: {np.mean(cp_dice):.3f}, Recall: {cp_recall*100:.1f}%")
print(f"  GMM       — IoU: {np.mean(gmm_iou):.3f}, Dice: {np.mean(gmm_dice):.3f}, Recall: {gmm_recall*100:.1f}%")
print(f"  Cellpose sweep CV: {cv:.3f}")
