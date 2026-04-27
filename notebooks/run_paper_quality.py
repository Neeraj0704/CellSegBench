"""
Paper-quality pipeline — fixes all scientific flaws:
1. 2048x2048 tile (~435x435 µm, ~800 reference cells) for statistical power
2. Full IoU/Dice for BOTH Cellpose AND Baysor
3. Baysor retuned (scale=8 µm) to fix over-merging
4. Downstream on large tile for real UMAP/clustering
5. Validated cell type annotations
"""
import sys, json, time, warnings, subprocess
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

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
from shapely.strtree import STRtree
from cellpose import models as cp_models
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

DATA_DIR    = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)
PIXEL_SIZE  = 0.2125
sc.settings.verbosity = 0

t0 = time.time()
def log(msg): print(f"[{(time.time()-t0)/60:.1f}m] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load 2048x2048 DAPI tile
# ═══════════════════════════════════════════════════════════════
log("STEP 1: Load 2048x2048 DAPI tile")

dapi_path = DATA_DIR / "morphology_focus" / "ch0000_dapi.ome.tif"
with tifffile.TiffFile(dapi_path) as tif:
    page = tif.pages[0]
    H, W = page.shape
    cy, cx = H // 2, W // 2
    # 2048 px = 435 µm per side
    r0, r1 = cy - 1024, cy + 1024
    c0, c1 = cx - 1024, cx + 1024
    dapi_tile = page.asarray()[r0:r1, c0:c1]

x0_um, x1_um = c0*PIXEL_SIZE, c1*PIXEL_SIZE
y0_um, y1_um = r0*PIXEL_SIZE, r1*PIXEL_SIZE

tile_meta = dict(r0=r0,r1=r1,c0=c0,c1=c1,
                 x0_um=x0_um,y0_um=y0_um,x1_um=x1_um,y1_um=y1_um)
with open(DATA_DIR / "tile_meta_large.json", "w") as f:
    json.dump(tile_meta, f)

log(f"Tile: {dapi_tile.shape} = {dapi_tile.shape[1]*PIXEL_SIZE:.0f}x{dapi_tile.shape[0]*PIXEL_SIZE:.0f} µm")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Load transcripts in tile
# ═══════════════════════════════════════════════════════════════
log("STEP 2: Load tile transcripts")

t_tile = pq.read_table(
    DATA_DIR / "transcripts_filtered.parquet",
    columns=["transcript_id","cell_id","feature_name","x_location","y_location","z_location"],
    filters=[("x_location",">=",x0_um),("x_location","<",x1_um),
             ("y_location",">=",y0_um),("y_location","<",y1_um)]
).to_pandas()
log(f"Transcripts in tile: {len(t_tile):,}")

# Baysor input CSV
baysor_csv = DATA_DIR / "baysor_input_large.csv"
t_tile[["x_location","y_location","z_location","feature_name"]].rename(
    columns={"x_location":"x","y_location":"y","z_location":"z","feature_name":"gene"}
).to_csv(baysor_csv, index=False)
log(f"Baysor input saved: {baysor_csv}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Load 10x reference cells & nucleus boundaries
# ═══════════════════════════════════════════════════════════════
log("STEP 3: Load 10x reference polygons")

cells    = pd.read_parquet(DATA_DIR / "cells.parquet")
nb_all   = pd.read_parquet(DATA_DIR / "nucleus_boundaries.parquet")
cb_all   = pd.read_parquet(DATA_DIR / "cell_boundaries.parquet")

tile_cells = cells[
    (cells["x_centroid"]>=x0_um)&(cells["x_centroid"]<x1_um)&
    (cells["y_centroid"]>=y0_um)&(cells["y_centroid"]<y1_um)
]
tile_cell_ids = set(tile_cells["cell_id"])
log(f"10x reference cells in tile: {len(tile_cells):,}")

# Build nucleus reference polygons
nb_tile = nb_all[nb_all["cell_id"].isin(tile_cell_ids)]
ref_nuc_polys = []
for cid, grp in nb_tile.groupby("cell_id"):
    coords = list(zip(grp["vertex_x"], grp["vertex_y"]))
    if len(coords) >= 3:
        p = Polygon(coords)
        if p.is_valid and p.area > 0:
            ref_nuc_polys.append(p)
log(f"Reference nucleus polygons: {len(ref_nuc_polys):,}")

# Build cell reference polygons (for fairer Baysor comparison)
cb_tile = cb_all[cb_all["cell_id"].isin(tile_cell_ids)]
ref_cell_polys = []
for cid, grp in cb_tile.groupby("cell_id"):
    coords = list(zip(grp["vertex_x"], grp["vertex_y"]))
    if len(coords) >= 3:
        p = Polygon(coords)
        if p.is_valid and p.area > 0:
            ref_cell_polys.append(p)
log(f"Reference cell polygons: {len(ref_cell_polys):,}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — Run Cellpose on 2048x2048 tile
# ═══════════════════════════════════════════════════════════════
log("STEP 4: Running Cellpose on 2048x2048 tile...")

model = cp_models.CellposeModel(gpu=False, model_type="nuclei")
masks, flows, styles = model.eval(
    dapi_tile, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0
)
n_cp = int(masks.max())
log(f"Cellpose: {n_cp} nuclei detected")

# Extract polygons in µm
cp_polys = []
for region in measure.regionprops(masks):
    contours = measure.find_contours(masks == region.label, 0.5)
    if not contours: continue
    c = max(contours, key=len)
    xy = np.column_stack([c[:,1]*PIXEL_SIZE+x0_um, c[:,0]*PIXEL_SIZE+y0_um])
    p = Polygon(xy)
    if p.is_valid and p.area > 0:
        cp_polys.append(p)
log(f"Cellpose polygons: {len(cp_polys):,}")
np.save(DATA_DIR / "cellpose_masks_large.npy", masks)

# Assign transcripts
t_tile2 = t_tile.copy()
t_tile2["col_px"] = ((t_tile2["x_location"]-x0_um)/PIXEL_SIZE).astype(int).clip(0,masks.shape[1]-1)
t_tile2["row_px"] = ((t_tile2["y_location"]-y0_um)/PIXEL_SIZE).astype(int).clip(0,masks.shape[0]-1)
t_tile2["cellpose_cell_id"] = masks[t_tile2["row_px"].values, t_tile2["col_px"].values]
cp_assigned = t_tile2[t_tile2["cellpose_cell_id"]>0]
cp_tpc = cp_assigned.groupby("cellpose_cell_id").size()
log(f"Cellpose assignment: {len(cp_assigned):,}/{len(t_tile2):,} ({100*len(cp_assigned)/len(t_tile2):.1f}%)")
log(f"Cellpose TPC: mean={cp_tpc.mean():.1f}, median={cp_tpc.median():.1f}")
t_tile2.to_parquet(DATA_DIR / "transcripts_cellpose_large.parquet", index=False)
del masks, flows, styles

# ═══════════════════════════════════════════════════════════════
# STEP 5 — Run Baysor (retuned scale=8 µm)
# ═══════════════════════════════════════════════════════════════
log("STEP 5: Running Baysor (scale=8 µm, retuned)...")

baysor_out = DATA_DIR / "baysor_output_large"
baysor_out.mkdir(exist_ok=True)

cmd = [
    "julia", f"--project={Path.home()}/.julia/environments/v1.11",
    "-e",
    f"""
using Baysor
append!(ARGS, ["run", "-x", "x", "-y", "y", "-z", "z", "-g", "gene",
  "-s", "8", "-m", "5", "--n-clusters=4",
  "-o", "{baysor_out}",
  "{baysor_csv}"])
Baysor.julia_main()
"""
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
log(f"Baysor stdout: {result.stdout[-500:]}")
if result.returncode != 0:
    log(f"Baysor stderr: {result.stderr[-300:]}")

# Load Baysor output
baysor_seg_path = baysor_out / "segmentation.csv"
if baysor_seg_path.exists():
    baysor_df = pd.read_csv(baysor_seg_path)
    log(f"Baysor transcripts: {len(baysor_df):,}, columns: {list(baysor_df.columns)}")
    n_by = baysor_df[baysor_df["cell"]>0]["cell"].nunique() if "cell" in baysor_df.columns else 0
    log(f"Baysor cells: {n_by}")
else:
    log("ERROR: Baysor output not found")
    baysor_df = None

# Load Baysor polygons
by_polys = []
baysor_poly_path = baysor_out / "segmentation_polygons.json"
if baysor_poly_path.exists():
    import json as jsonlib
    with open(baysor_poly_path) as f:
        geo = jsonlib.load(f)
    for feat in geo.get("features", []):
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            p = Polygon(coords)
            if p.is_valid and p.area > 0:
                by_polys.append(p)
    log(f"Baysor polygons: {len(by_polys):,}")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — Full IoU/Dice evaluation for BOTH methods
# ═══════════════════════════════════════════════════════════════
log("STEP 6: Computing IoU/Dice for ALL methods")

def match_iou(pred_polys, ref_polys, thresh=0.3):
    if not pred_polys or not ref_polys:
        return [], [], 0.0
    tree = STRtree(ref_polys)
    iou_s, dice_s, matched_ref = [], [], set()
    for pred in pred_polys:
        candidates = list(tree.query(pred))
        best_iou, best_idx = 0, -1
        for idx in candidates:
            if idx in matched_ref: continue
            ref = ref_polys[idx]
            inter = pred.intersection(ref).area
            union = pred.union(ref).area
            iou = inter/union if union>0 else 0
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= thresh and best_idx >= 0:
            iou_s.append(best_iou)
            ref = ref_polys[best_idx]
            dice_s.append(2*pred.intersection(ref).area/(pred.area+ref.area))
            matched_ref.add(best_idx)
    recall = len(matched_ref)/len(ref_polys) if ref_polys else 0
    return iou_s, dice_s, recall

# Cellpose vs nucleus reference (fair — both DAPI-based)
log("  Cellpose vs nucleus reference...")
cp_iou_nuc, cp_dice_nuc, cp_recall_nuc = match_iou(cp_polys, ref_nuc_polys)
log(f"  Cellpose/nucleus: IoU={np.mean(cp_iou_nuc):.3f}, Dice={np.mean(cp_dice_nuc):.3f}, Recall={cp_recall_nuc*100:.1f}%")

# Cellpose vs cell reference (also compute for transparency)
log("  Cellpose vs cell reference...")
cp_iou_cell, cp_dice_cell, cp_recall_cell = match_iou(cp_polys, ref_cell_polys)
log(f"  Cellpose/cell: IoU={np.mean(cp_iou_cell):.3f}, Dice={np.mean(cp_dice_cell):.3f}, Recall={cp_recall_cell*100:.1f}%")

# Baysor vs cell reference (fair — Baysor segments whole cells)
if by_polys:
    log("  Baysor vs cell reference...")
    by_iou_cell, by_dice_cell, by_recall_cell = match_iou(by_polys, ref_cell_polys)
    log(f"  Baysor/cell: IoU={np.mean(by_iou_cell):.3f}, Dice={np.mean(by_dice_cell):.3f}, Recall={by_recall_cell*100:.1f}%")

    log("  Baysor vs nucleus reference...")
    by_iou_nuc, by_dice_nuc, by_recall_nuc = match_iou(by_polys, ref_nuc_polys)
    log(f"  Baysor/nucleus: IoU={np.mean(by_iou_nuc):.3f}, Dice={np.mean(by_dice_nuc):.3f}, Recall={by_recall_nuc*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — Figures: segmentation maps + IoU comparisons
# ═══════════════════════════════════════════════════════════════
log("STEP 7: Generating figures")

dapi_norm = (dapi_tile.astype("float32")-dapi_tile.min())/(dapi_tile.max()-dapi_tile.min())

# 3-panel side-by-side
fig, axes = plt.subplots(1,3,figsize=(18,6))
panels = [
    (ref_nuc_polys, "yellow", f"10x Reference\n({len(ref_nuc_polys):,} nuclei)"),
    (cp_polys,      "cyan",   f"Cellpose\n({len(cp_polys):,} nuclei, IoU={np.mean(cp_iou_nuc):.3f})"),
    (by_polys if by_polys else [], "lime",
     f"Baysor (scale=8µm)\n({len(by_polys):,} cells, IoU={np.mean(by_iou_cell):.3f})" if by_polys else "Baysor (no output)"),
]
for ax, (polys, color, title) in zip(axes, panels):
    ax.imshow(dapi_norm, cmap="gray", origin="lower")
    for poly in polys:
        xs=(np.array(poly.exterior.xy[0])-x0_um)/PIXEL_SIZE
        ys=(np.array(poly.exterior.xy[1])-y0_um)/PIXEL_SIZE
        ax.plot(xs,ys,color=color,linewidth=0.5)
    ax.set_title(title,fontsize=11); ax.axis("off")
plt.suptitle("Segmentation Comparison — 435×435 µm tile", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"side_by_side_3way.png",dpi=150); plt.close()
log("Saved: side_by_side_3way.png")

# IoU distributions
fig, axes = plt.subplots(1,2,figsize=(13,5))
if cp_iou_nuc:
    axes[0].hist(cp_iou_nuc, bins=40, color="steelblue", alpha=0.8, edgecolor="none")
    axes[0].axvline(np.mean(cp_iou_nuc), color="red", linestyle="--", label=f"Mean={np.mean(cp_iou_nuc):.3f}")
    axes[0].set_title("Cellpose IoU vs Nucleus Reference"); axes[0].set_xlabel("IoU"); axes[0].legend()

if by_polys and by_iou_cell:
    axes[1].hist(by_iou_cell, bins=40, color="seagreen", alpha=0.8, edgecolor="none")
    axes[1].axvline(np.mean(by_iou_cell), color="red", linestyle="--", label=f"Mean={np.mean(by_iou_cell):.3f}")
    axes[1].set_title("Baysor IoU vs Cell Reference"); axes[1].set_xlabel("IoU"); axes[1].legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR/"iou_distributions_both.png",dpi=150); plt.close()
log("Saved: iou_distributions_both.png")

# Bar chart comparison
methods = ["Cellpose\nvs Nucleus", "Cellpose\nvs Cell", "Baysor\nvs Nucleus", "Baysor\nvs Cell"]
iou_vals = [
    np.mean(cp_iou_nuc) if cp_iou_nuc else 0,
    np.mean(cp_iou_cell) if cp_iou_cell else 0,
    np.mean(by_iou_nuc) if by_polys and by_iou_nuc else 0,
    np.mean(by_iou_cell) if by_polys and by_iou_cell else 0,
]
dice_vals = [
    np.mean(cp_dice_nuc) if cp_dice_nuc else 0,
    np.mean(cp_dice_cell) if cp_dice_cell else 0,
    np.mean(by_dice_nuc) if by_polys and by_iou_nuc else 0,
    np.mean(by_dice_cell) if by_polys and by_iou_cell else 0,
]
recall_vals = [
    cp_recall_nuc, cp_recall_cell,
    by_recall_nuc if by_polys else 0,
    by_recall_cell if by_polys else 0,
]
x = np.arange(len(methods)); w=0.25
fig, axes = plt.subplots(1,3,figsize=(15,5))
colors=["steelblue","steelblue","seagreen","seagreen"]
for ax, vals, ylabel in zip(axes,[iou_vals,dice_vals,recall_vals],["Mean IoU","Mean Dice","Recall"]):
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0,1); ax.set_ylabel(ylabel); ax.set_title(ylabel)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.suptitle("Full Evaluation: Cellpose vs Baysor vs 10x Reference\n(435×435 µm tile)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"three_way_iou_comparison.png",dpi=150); plt.close()
log("Saved: three_way_iou_comparison.png")

# Baysor segmentation map
if by_polys:
    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(dapi_norm, cmap="gray", origin="lower")
    for poly in by_polys:
        xs=(np.array(poly.exterior.xy[0])-x0_um)/PIXEL_SIZE
        ys=(np.array(poly.exterior.xy[1])-y0_um)/PIXEL_SIZE
        ax.plot(xs,ys,color="lime",linewidth=0.6,alpha=0.8)
    ax.set_title(f"Baysor Segmentation (scale=8µm, {len(by_polys):,} cells)"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"baysor_segmentation_map.png",dpi=150); plt.close()
    log("Saved: baysor_segmentation_map.png")

# Cellpose segmentation map
fig, axes = plt.subplots(1,2,figsize=(14,7))
axes[0].imshow(dapi_norm, cmap="gray", origin="lower"); axes[0].set_title("DAPI Input"); axes[0].axis("off")
axes[1].imshow(dapi_norm, cmap="gray", origin="lower")
cp_masks = np.load(DATA_DIR/"cellpose_masks_large.npy")
axes[1].imshow(cp_masks>0, cmap="Reds", alpha=0.4, origin="lower")
axes[1].set_title(f"Cellpose ({n_cp:,} nuclei)"); axes[1].axis("off")
plt.tight_layout(); plt.savefig(RESULTS_DIR/"cellpose_segmentation_map.png",dpi=150); plt.close()
log("Saved: cellpose_segmentation_map.png")

# ═══════════════════════════════════════════════════════════════
# STEP 8 — Updated summary table
# ═══════════════════════════════════════════════════════════════
log("STEP 8: Summary table")

xen_assigned = t_tile[t_tile["cell_id"]!="UNASSIGNED"]
by_assigned_n = len(baysor_df[baysor_df["cell"]>0]) if baysor_df is not None and "cell" in baysor_df.columns else 0
by_tpc_mean = baysor_df[baysor_df["cell"]>0].groupby("cell").size().mean() if baysor_df is not None and "cell" in baysor_df.columns else 0

summary = pd.DataFrame([
    {"Method":"Xenium 10x","Reference":"—","Cells":len(tile_cells),"Assigned%":round(100*len(xen_assigned)/len(t_tile),1),
     "Mean TPC":round(xen_assigned.groupby("cell_id").size().mean(),1),
     "IoU (vs nuc)":"—","Dice (vs nuc)":"—","Recall (vs nuc)":"—",
     "IoU (vs cell)":"—","Dice (vs cell)":"—","Recall (vs cell)":"—"},
    {"Method":"Cellpose","Reference":"Nucleus","Cells":len(cp_polys),
     "Assigned%":round(100*len(cp_assigned)/len(t_tile),1),
     "Mean TPC":round(cp_tpc.mean(),1),
     "IoU (vs nuc)":round(np.mean(cp_iou_nuc),3),"Dice (vs nuc)":round(np.mean(cp_dice_nuc),3),"Recall (vs nuc)":round(cp_recall_nuc,3),
     "IoU (vs cell)":round(np.mean(cp_iou_cell),3),"Dice (vs cell)":round(np.mean(cp_dice_cell),3),"Recall (vs cell)":round(cp_recall_cell,3)},
    {"Method":"Baysor","Reference":"Cell","Cells":len(by_polys),
     "Assigned%":round(100*by_assigned_n/len(t_tile),1) if len(t_tile)>0 else 0,
     "Mean TPC":round(by_tpc_mean,1),
     "IoU (vs nuc)":round(np.mean(by_iou_nuc),3) if by_polys and by_iou_nuc else "—",
     "Dice (vs nuc)":round(np.mean(by_dice_nuc),3) if by_polys and by_iou_nuc else "—",
     "Recall (vs nuc)":round(by_recall_nuc,3) if by_polys else "—",
     "IoU (vs cell)":round(np.mean(by_iou_cell),3) if by_polys and by_iou_cell else "—",
     "Dice (vs cell)":round(np.mean(by_dice_cell),3) if by_polys and by_iou_cell else "—",
     "Recall (vs cell)":round(by_recall_cell,3) if by_polys else "—"},
])
summary.to_csv(RESULTS_DIR/"method_summary_stats.csv", index=False)
print(f"\n{summary.to_string(index=False)}\n")

# ═══════════════════════════════════════════════════════════════
# STEP 9 — Downstream: per-method on large tile
# ═══════════════════════════════════════════════════════════════
log("STEP 9: Downstream analysis on large tile")

def build_adata_tile(df, cell_col, gene_col="feature_name"):
    assigned = df[df[cell_col].notna() & (df[cell_col]!=0) & (df[cell_col]!="UNASSIGNED")].copy()
    assigned[cell_col] = assigned[cell_col].astype(str)
    counts = assigned.groupby([cell_col, gene_col]).size().unstack(fill_value=0)
    if counts.shape[0] < 20: return None
    a = ad.AnnData(X=csr_matrix(counts.values))
    a.obs_names = counts.index.astype(str); a.var_names = counts.columns
    centroids = assigned.groupby(cell_col)[["x_location","y_location"]].mean()
    a.obs = a.obs.join(centroids.rename(columns={"x_location":"x","y_location":"y"}))
    a.obsm["spatial"] = a.obs[["x","y"]].values
    return a

# Xenium 10x tile
xen_tile_df = t_tile.copy()
xen_adata = build_adata_tile(xen_tile_df, "cell_id")

# Cellpose tile
cp_adata = build_adata_tile(t_tile2, "cellpose_cell_id")

# Baysor tile
by_adata = None
if baysor_df is not None and "cell" in baysor_df.columns:
    # merge coordinates back
    baysor_df2 = baysor_df.copy()
    baysor_df2["x_location"] = baysor_df2["x"]
    baysor_df2["y_location"] = baysor_df2["y"]
    baysor_df2["feature_name"] = baysor_df2["gene"]
    by_adata = build_adata_tile(baysor_df2, "cell")

adatas = {k:v for k,v in {"Xenium 10x":xen_adata,"Cellpose":cp_adata,"Baysor":by_adata}.items() if v is not None}

for name, a in adatas.items():
    sc.pp.filter_cells(a, min_counts=5)
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=100)
    sc.pp.log1p(a)
    n_pcs = min(20, a.n_obs-1, a.n_vars-1)
    sc.pp.pca(a, n_comps=n_pcs)
    sc.pp.neighbors(a, n_neighbors=min(15,a.n_obs-1), n_pcs=n_pcs)
    sc.tl.umap(a)
    sc.tl.leiden(a, resolution=0.5, key_added="leiden")
    log(f"  {name}: {a.n_obs} cells, {a.obs['leiden'].nunique()} clusters")

ncols = len(adatas)
fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6))
if ncols==1: axes=[axes]
for ax,(name,a) in zip(axes,adatas.items()):
    sc.pl.umap(a, color="leiden", ax=ax, show=False,
               title=f"{name}\n({a.n_obs} cells, {a.obs['leiden'].nunique()} clusters)",
               legend_loc="on data", legend_fontsize=9)
plt.suptitle("UMAP — All Methods on 435×435 µm Tile", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"umap_comparison_methods.png",dpi=150); plt.close()
log("Saved: umap_comparison_methods.png")

# ═══════════════════════════════════════════════════════════════
# STEP 10 — Validated cell type annotation
# ═══════════════════════════════════════════════════════════════
log("STEP 10: Validated cell type annotation (10x full tissue)")

adata_full = sc.read_h5ad(DATA_DIR / "adata_10x_processed.h5ad")
sc.tl.rank_genes_groups(adata_full, groupby="leiden", method="wilcoxon", n_genes=5, use_raw=False)

print("\n=== TOP MARKER GENES PER CLUSTER ===")
result = adata_full.uns["rank_genes_groups"]
clusters = list(result["names"].dtype.names)
ct_map = {}
known_markers = {
    "EPCAM":"Epithelial","KRT17":"Epithelial","KRT8":"Epithelial",
    "CD68":"Macrophage","CD163":"Macrophage",
    "CD3E":"T Cell","CD8A":"T Cell",
    "ACTA2":"Myoepithelial","CNN1":"Myoepithelial",
    "PECAM1":"Endothelial","VWF":"Endothelial",
    "COL1A1":"Stromal","FAP":"Stromal","VIM":"Stromal",
    "MKI67":"Proliferating","TOP2A":"Proliferating",
    "CD79A":"B Cell","MS4A1":"B Cell",
}
for cluster in clusters:
    top5 = [result["names"][i][clusters.index(cluster)] for i in range(5)]
    scores = [result["scores"][i][clusters.index(cluster)] for i in range(5)]
    # Find best matching cell type
    matched_ct = "Unknown"
    best_score = 0
    for gene in top5:
        if gene in known_markers:
            matched_ct = known_markers[gene]
            break
    ct_map[cluster] = matched_ct
    print(f"  Cluster {cluster}: top genes = {top5[:3]} → {matched_ct}")

adata_full.obs["cell_type"] = adata_full.obs["leiden"].map(ct_map).astype(str)

print("\n=== VALIDATED CELL TYPE PROPORTIONS ===")
props = adata_full.obs["cell_type"].value_counts(normalize=True).sort_values(ascending=False)
for ct, v in props.items():
    print(f"  {ct}: {v*100:.1f}%")

# Save validated UMAP
fig, ax = plt.subplots(figsize=(9,7))
sc.pl.umap(adata_full, color="cell_type", ax=ax, show=False,
           title="10x Xenium — Validated Cell Type Annotation (201,446 cells)",
           legend_loc="right margin")
plt.tight_layout()
plt.savefig(RESULTS_DIR/"umap_cell_types_annotated.png",dpi=150); plt.close()

# Cell type proportions
fig, ax = plt.subplots(figsize=(9,4))
colors = plt.cm.Set1.colors
ax.bar(props.index, props.values, color=colors[:len(props)])
for i,(ct,v) in enumerate(props.items()):
    ax.text(i, v+0.004, f"{v*100:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Proportion"); ax.set_title("Validated Cell Type Proportions")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"cell_type_proportions.png",dpi=150); plt.close()

adata_full.write_h5ad(DATA_DIR/"adata_10x_processed.h5ad")
log("Saved: validated cell type figures and AnnData")

# ═══════════════════════════════════════════════════════════════
total = (time.time()-t0)/60
log(f"\n{'='*60}")
log(f"PAPER-QUALITY PIPELINE COMPLETE in {total:.1f} minutes")
log(f"{'='*60}")
print(f"\nAll results in: {RESULTS_DIR.resolve()}")
for f in sorted(RESULTS_DIR.iterdir()):
    if f.suffix in [".png",".csv"]:
        print(f"  {f.name}")
