"""
Process Baysor output, evaluate against 10x reference,
update summary stats, and regenerate comparison figures.
"""
import sys, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

DATA_DIR    = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
BAYSOR_DIR  = DATA_DIR / "baysor_output"
RESULTS_DIR = Path("../results")
PIXEL_SIZE  = 0.2125

with open(DATA_DIR / "tile_meta.json") as f:
    tile = json.load(f)
x0, x1, y0, y1 = tile["x0_um"], tile["x1_um"], tile["y0_um"], tile["y1_um"]

# ── 1. Load Baysor segmentation ────────────────────────────────────────────
print("Loading Baysor output...")
seg = pd.read_csv(BAYSOR_DIR / "segmentation.csv")
stats = pd.read_csv(BAYSOR_DIR / "segmentation_cell_stats.csv")

assigned = seg[~seg["is_noise"] & seg["cell"].notna() & (seg["cell"] != "")]
n_cells  = assigned["cell"].nunique()
n_noise  = seg["is_noise"].sum()
tpc      = assigned.groupby("cell").size()

print(f"Baysor cells:          {n_cells}")
print(f"Noise transcripts:     {n_noise} ({100*n_noise/len(seg):.1f}%)")
print(f"Assigned transcripts:  {len(assigned)} ({100*len(assigned)/len(seg):.1f}%)")
print(f"Transcripts/cell:      mean={tpc.mean():.1f}, median={tpc.median():.1f}")

# ── 2. Build Baysor polygons from GeoJSON ──────────────────────────────────
print("\nLoading Baysor polygons...")
with open(BAYSOR_DIR / "segmentation_polygons_2d.json") as f:
    geojson = json.load(f)

baysor_polys = []
for feature in geojson.get("features", []):
    geom = feature.get("geometry", {})
    if geom.get("type") == "Polygon":
        coords = geom["coordinates"][0]
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            baysor_polys.append(poly)
print(f"Baysor polygons loaded: {len(baysor_polys)}")

# ── 3. Load reference & Cellpose polygons ─────────────────────────────────
nb    = pd.read_parquet(DATA_DIR / "nucleus_boundaries.parquet")
cells = pd.read_parquet(DATA_DIR / "cells.parquet")
tile_cells = cells[(cells["x_centroid"]>=x0)&(cells["x_centroid"]<x1)&
                    (cells["y_centroid"]>=y0)&(cells["y_centroid"]<y1)]
nb_tile = nb[nb["cell_id"].isin(set(tile_cells["cell_id"]))]

ref_polys = []
for cid, grp in nb_tile.groupby("cell_id"):
    coords = list(zip(grp["vertex_x"], grp["vertex_y"]))
    if len(coords) >= 3:
        p = Polygon(coords)
        if p.is_valid and p.area > 0:
            ref_polys.append(p)

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

# ── 4. IoU matching ────────────────────────────────────────────────────────
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

cp_iou,  cp_dice,  cp_recall  = match_iou(cp_polys,   ref_polys)
by_iou,  by_dice,  by_recall  = match_iou(baysor_polys, ref_polys)

print(f"\nCellpose  — IoU: {np.mean(cp_iou):.3f}, Dice: {np.mean(cp_dice):.3f}, Recall: {cp_recall*100:.1f}%")
print(f"Baysor    — IoU: {np.mean(by_iou):.3f}, Dice: {np.mean(by_dice):.3f}, Recall: {by_recall*100:.1f}%")

# ── 5. Transcript stats for all methods ───────────────────────────────────
tile_t = pq.read_table(DATA_DIR / "transcripts_filtered.parquet",
    columns=["cell_id","x_location","y_location","feature_name"],
    filters=[("x_location",">=",x0),("x_location","<",x1),
             ("y_location",">=",y0),("y_location","<",y1)]).to_pandas()
cp_t  = pd.read_parquet(DATA_DIR / "transcripts_cellpose.parquet")

xen_assigned = tile_t[tile_t["cell_id"] != "UNASSIGNED"]
cp_assigned  = cp_t[cp_t["cellpose_cell_id"] > 0]
xen_tpc = xen_assigned.groupby("cell_id").size()
cp_tpc  = cp_assigned.groupby("cellpose_cell_id").size()

# ── 6. Summary table ──────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"Method":"Xenium (10x)", "Cells":len(tile_cells),
     "Assigned %": round(100*len(xen_assigned)/len(tile_t),1),
     "Mean TPC": round(xen_tpc.mean(),1), "Median TPC": round(xen_tpc.median(),1),
     "Mean IoU":"—","Mean Dice":"—","Recall":"—"},
    {"Method":"Cellpose", "Cells":len(cp_polys),
     "Assigned %": round(100*len(cp_assigned)/len(tile_t),1),
     "Mean TPC": round(cp_tpc.mean(),1), "Median TPC": round(cp_tpc.median(),1),
     "Mean IoU": round(np.mean(cp_iou),3), "Mean Dice": round(np.mean(cp_dice),3),
     "Recall": f"{cp_recall*100:.1f}%"},
    {"Method":"Baysor", "Cells":n_cells,
     "Assigned %": round(100*len(assigned)/len(seg),1),
     "Mean TPC": round(tpc.mean(),1), "Median TPC": round(tpc.median(),1),
     "Mean IoU": round(np.mean(by_iou),3), "Mean Dice": round(np.mean(by_dice),3),
     "Recall": f"{by_recall*100:.1f}%"},
])
summary.to_csv(RESULTS_DIR / "method_summary_stats.csv", index=False)
print(f"\n{summary.to_string(index=False)}")

# ── 7. IoU comparison bar chart (3-way) ───────────────────────────────────
import tifffile
with tifffile.TiffFile(DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif") as tif:
    r0i,r1i = int(y0/PIXEL_SIZE), int(y1/PIXEL_SIZE)
    c0i,c1i = int(x0/PIXEL_SIZE), int(x1/PIXEL_SIZE)
    bg = tif.pages[0].asarray()[r0i:r1i, c0i:c1i]
bg_norm = (bg.astype("float32")-bg.min())/(bg.max()-bg.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods = ["Cellpose", "Baysor"]
for ax, (vals, ylabel) in zip(axes, [
    ([np.mean(cp_iou), np.mean(by_iou)], "Mean IoU"),
    ([np.mean(cp_dice), np.mean(by_dice)], "Mean Dice"),
    ([cp_recall, by_recall], "Recall"),
]):
    bars = ax.bar(methods, vals, color=["steelblue","tomato"], alpha=0.85)
    ax.set_ylim(0,1); ax.set_ylabel(ylabel); ax.set_title(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
plt.suptitle("Cellpose vs Baysor vs 10x Reference", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "three_way_iou_comparison.png", dpi=150); plt.close()
print("Saved: three_way_iou_comparison.png")

# ── 8. Side-by-side 3-panel segmentation map ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
panels = [
    (ref_polys,    "yellow", f"Xenium 10x Reference\n({len(ref_polys)} nuclei)"),
    (cp_polys,     "cyan",   f"Cellpose\n({len(cp_polys)} cells, IoU={np.mean(cp_iou):.3f})"),
    (baysor_polys, "lime",   f"Baysor\n({len(baysor_polys)} cells, IoU={np.mean(by_iou):.3f})"),
]
for ax, (polys, color, title) in zip(axes, panels):
    ax.imshow(bg_norm, cmap="gray", origin="lower")
    for poly in polys:
        xs = (np.array(poly.exterior.xy[0]) - x0) / PIXEL_SIZE
        ys = (np.array(poly.exterior.xy[1]) - y0) / PIXEL_SIZE
        ax.plot(xs, ys, color=color, linewidth=0.8)
    ax.set_title(title, fontsize=11); ax.axis("off")
plt.suptitle("Segmentation Comparison — 109×109 µm tile", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "side_by_side_3way.png", dpi=150); plt.close()
print("Saved: side_by_side_3way.png")

# ── 9. Transcripts per cell distribution ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for tpc_data, label, color in [
    (xen_tpc, "Xenium 10x", "gold"),
    (cp_tpc,  "Cellpose",   "steelblue"),
    (tpc,     "Baysor",     "tomato"),
]:
    ax.hist(tpc_data, bins=30, alpha=0.6, label=label, color=color, density=True)
ax.set_xlabel("Transcripts per cell"); ax.set_ylabel("Density")
ax.set_title("Transcripts per Cell — All Methods")
ax.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR / "transcripts_per_cell_comparison.png", dpi=150); plt.close()
print("Saved: transcripts_per_cell_comparison.png")

# ── 10. Baysor downstream ─────────────────────────────────────────────────
print("\nRunning Baysor downstream analysis...")
sc.settings.verbosity = 0
bay_assigned = assigned.rename(columns={"cell":"cell_id","gene":"feature_name",
                                         "x":"x_location","y":"y_location"})
counts = bay_assigned.groupby(["cell_id","feature_name"]).size().unstack(fill_value=0)
if counts.shape[0] >= 5:
    adata = ad.AnnData(X=csr_matrix(counts.values))
    adata.obs_names = counts.index.astype(str)
    adata.var_names = counts.columns
    centroids = bay_assigned.groupby("cell_id")[["x_location","y_location"]].mean()
    adata.obs = adata.obs.join(centroids)
    adata.obsm["spatial"] = adata.obs[["x_location","y_location"]].values
    sc.pp.filter_cells(adata, min_counts=5)
    sc.pp.normalize_total(adata, target_sum=100)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(10, adata.n_obs-1, adata.n_vars-1))
    sc.pp.neighbors(adata, n_neighbors=min(5, adata.n_obs-1))
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
    print(f"Baysor: {adata.n_obs} cells, {adata.obs['leiden'].nunique()} clusters")

    fig, ax = plt.subplots(figsize=(6,5))
    sc.pl.umap(adata, color="leiden", ax=ax, show=False,
               title=f"Baysor UMAP ({adata.n_obs} cells)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_baysor.png", dpi=150); plt.close()
    print("Saved: umap_baysor.png")

print("\n=== BAYSOR EVALUATION COMPLETE ===")
print(f"Baysor  — cells: {n_cells}, IoU: {np.mean(by_iou):.3f}, Dice: {np.mean(by_dice):.3f}, Recall: {by_recall*100:.1f}%")
print(f"Cellpose — cells: {len(cp_polys)}, IoU: {np.mean(cp_iou):.3f}, Dice: {np.mean(cp_dice):.3f}, Recall: {cp_recall*100:.1f}%")
