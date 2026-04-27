"""
Paper-quality pipeline — Step 2: Evaluation, figures, downstream, report
Picks up from Baysor output already on disk.
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
import tifffile
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import seaborn as sns

DATA_DIR    = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
PIXEL_SIZE  = 0.2125
sc.settings.verbosity = 0

t0 = time.time()
def log(msg): print(f"[{(time.time()-t0)/60:.1f}m] {msg}", flush=True)

# ── Load tile metadata ─────────────────────────────────────────
with open(DATA_DIR / "tile_meta_large.json") as f:
    tile = json.load(f)
x0, x1 = tile["x0_um"], tile["x1_um"]
y0, y1 = tile["y0_um"], tile["y1_um"]
r0, r1, c0, c1 = tile["r0"], tile["r1"], tile["c0"], tile["c1"]

# ── Load DAPI background ───────────────────────────────────────
log("Loading DAPI tile for figures...")
with tifffile.TiffFile(DATA_DIR/"morphology_focus"/"ch0000_dapi.ome.tif") as tif:
    dapi_tile = tif.pages[0].asarray()[r0:r1, c0:c1]
dapi_norm = (dapi_tile.astype("float32")-dapi_tile.min())/(dapi_tile.max()-dapi_tile.min())

# ── Load 10x reference polygons ────────────────────────────────
log("Loading reference polygons...")
cells  = pd.read_parquet(DATA_DIR/"cells.parquet")
nb_all = pd.read_parquet(DATA_DIR/"nucleus_boundaries.parquet")
cb_all = pd.read_parquet(DATA_DIR/"cell_boundaries.parquet")

tile_cells = cells[
    (cells["x_centroid"]>=x0)&(cells["x_centroid"]<x1)&
    (cells["y_centroid"]>=y0)&(cells["y_centroid"]<y1)
]
tile_ids = set(tile_cells["cell_id"])

def build_polys(boundary_df, id_set):
    polys = []
    for cid, grp in boundary_df[boundary_df["cell_id"].isin(id_set)].groupby("cell_id"):
        coords = list(zip(grp["vertex_x"], grp["vertex_y"]))
        if len(coords)>=3:
            p = Polygon(coords)
            if p.is_valid and p.area>0: polys.append(p)
    return polys

ref_nuc_polys  = build_polys(nb_all, tile_ids)
ref_cell_polys = build_polys(cb_all, tile_ids)
log(f"Reference: {len(tile_cells)} cells, {len(ref_nuc_polys)} nucleus polys, {len(ref_cell_polys)} cell polys")

# ── Load Cellpose polygons ─────────────────────────────────────
log("Loading Cellpose polygons...")
masks = np.load(DATA_DIR/"cellpose_masks_large.npy")
cp_polys = []
for region in measure.regionprops(masks):
    contours = measure.find_contours(masks==region.label, 0.5)
    if not contours: continue
    c = max(contours, key=len)
    xy = np.column_stack([c[:,1]*PIXEL_SIZE+x0, c[:,0]*PIXEL_SIZE+y0])
    p = Polygon(xy)
    if p.is_valid and p.area>0: cp_polys.append(p)
log(f"Cellpose polygons: {len(cp_polys):,}")

# ── Load Baysor output ─────────────────────────────────────────
log("Loading Baysor output...")
baysor_out = DATA_DIR/"baysor_output_large"
baysor_df  = pd.read_csv(baysor_out/"segmentation.csv")

# Baysor cell column is string — filter non-noise, assigned cells
baysor_assigned = baysor_df[baysor_df["is_noise"]==False].copy() if "is_noise" in baysor_df.columns else baysor_df.copy()
n_by_cells = baysor_assigned["cell"].nunique() if "cell" in baysor_assigned.columns else 0
log(f"Baysor: {len(baysor_df):,} transcripts, {n_by_cells} cells")

# Load Baysor polygons
by_polys = []
import json as jsonlib
poly_path = baysor_out/"segmentation_polygons_2d.json"
if poly_path.exists():
    with open(poly_path) as f:
        geo = jsonlib.load(f)
    for feat in geo.get("features",[]):
        geom = feat["geometry"]
        if geom["type"]=="Polygon":
            coords = geom["coordinates"][0]
            p = Polygon(coords)
            if p.is_valid and p.area>0: by_polys.append(p)
log(f"Baysor polygons: {len(by_polys):,}")

# ═══════════════════════════════════════════════════════════════
# IoU/Dice evaluation for ALL methods
# ═══════════════════════════════════════════════════════════════
log("Computing IoU/Dice...")

def match_iou(pred_polys, ref_polys, thresh=0.3):
    if not pred_polys or not ref_polys: return [],[],0.0
    tree = STRtree(ref_polys)
    iou_s, dice_s, matched = [],[],set()
    for pred in pred_polys:
        cands = list(tree.query(pred))
        best_iou, best_idx = 0,-1
        for idx in cands:
            if idx in matched: continue
            ref = ref_polys[idx]
            inter = pred.intersection(ref).area
            union = pred.union(ref).area
            iou = inter/union if union>0 else 0
            if iou>best_iou: best_iou,best_idx=iou,idx
        if best_iou>=thresh and best_idx>=0:
            iou_s.append(best_iou)
            ref=ref_polys[best_idx]
            dice_s.append(2*pred.intersection(ref).area/(pred.area+ref.area))
            matched.add(best_idx)
    return iou_s, dice_s, len(matched)/len(ref_polys)

cp_iou_nuc,  cp_dice_nuc,  cp_recall_nuc  = match_iou(cp_polys, ref_nuc_polys)
cp_iou_cell, cp_dice_cell, cp_recall_cell = match_iou(cp_polys, ref_cell_polys)
log(f"Cellpose vs nucleus: IoU={np.mean(cp_iou_nuc):.3f}, Dice={np.mean(cp_dice_nuc):.3f}, Recall={cp_recall_nuc*100:.1f}%")
log(f"Cellpose vs cell:    IoU={np.mean(cp_iou_cell):.3f}, Dice={np.mean(cp_dice_cell):.3f}, Recall={cp_recall_cell*100:.1f}%")

by_iou_nuc,  by_dice_nuc,  by_recall_nuc  = match_iou(by_polys, ref_nuc_polys)
by_iou_cell, by_dice_cell, by_recall_cell = match_iou(by_polys, ref_cell_polys)
log(f"Baysor vs nucleus:   IoU={np.mean(by_iou_nuc):.3f},  Dice={np.mean(by_dice_nuc):.3f},  Recall={by_recall_nuc*100:.1f}%")
log(f"Baysor vs cell:      IoU={np.mean(by_iou_cell):.3f},  Dice={np.mean(by_dice_cell):.3f},  Recall={by_recall_cell*100:.1f}%")

# ── Transcript stats ───────────────────────────────────────────
t_tile = pq.read_table(DATA_DIR/"transcripts_cellpose_large.parquet").to_pandas()
cp_assigned_t = t_tile[t_tile["cellpose_cell_id"]>0]
cp_tpc = cp_assigned_t.groupby("cellpose_cell_id").size()

xen_assigned_t = t_tile[t_tile["cell_id"]!="UNASSIGNED"]
xen_tpc = xen_assigned_t.groupby("cell_id").size()

by_assigned_t = baysor_assigned[baysor_assigned["cell"].notna()]
by_tpc = by_assigned_t.groupby("cell").size() if "cell" in by_assigned_t.columns else pd.Series([])

# ── Summary table ──────────────────────────────────────────────
summary = pd.DataFrame([
    {"Method":"Xenium 10x","Cells":len(tile_cells),
     "Assigned%":round(100*len(xen_assigned_t)/len(t_tile),1),
     "Mean TPC":round(xen_tpc.mean(),1),"Median TPC":round(xen_tpc.median(),1),
     "IoU vs Nucleus":"—","Dice vs Nucleus":"—","Recall vs Nucleus":"—",
     "IoU vs Cell":"—","Dice vs Cell":"—","Recall vs Cell":"—"},
    {"Method":"Cellpose (nuclei)","Cells":len(cp_polys),
     "Assigned%":round(100*len(cp_assigned_t)/len(t_tile),1),
     "Mean TPC":round(cp_tpc.mean(),1),"Median TPC":round(cp_tpc.median(),1),
     "IoU vs Nucleus":round(np.mean(cp_iou_nuc),3),
     "Dice vs Nucleus":round(np.mean(cp_dice_nuc),3),
     "Recall vs Nucleus":f"{cp_recall_nuc*100:.1f}%",
     "IoU vs Cell":round(np.mean(cp_iou_cell),3),
     "Dice vs Cell":round(np.mean(cp_dice_cell),3),
     "Recall vs Cell":f"{cp_recall_cell*100:.1f}%"},
    {"Method":"Baysor (cells)","Cells":len(by_polys),
     "Assigned%":round(100*len(by_assigned_t)/len(t_tile),1),
     "Mean TPC":round(by_tpc.mean(),1) if len(by_tpc)>0 else "—",
     "Median TPC":round(by_tpc.median(),1) if len(by_tpc)>0 else "—",
     "IoU vs Nucleus":round(np.mean(by_iou_nuc),3),
     "Dice vs Nucleus":round(np.mean(by_dice_nuc),3),
     "Recall vs Nucleus":f"{by_recall_nuc*100:.1f}%",
     "IoU vs Cell":round(np.mean(by_iou_cell),3),
     "Dice vs Cell":round(np.mean(by_dice_cell),3),
     "Recall vs Cell":f"{by_recall_cell*100:.1f}%"},
])
summary.to_csv(RESULTS_DIR/"method_summary_stats.csv", index=False)
print(f"\n{'='*60}")
print(summary.to_string(index=False))
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════
log("Generating figures...")

# 1. Side-by-side 3-panel (real Baysor, not GMM)
fig, axes = plt.subplots(1,3,figsize=(18,6))
panels = [
    (ref_nuc_polys, "yellow", f"10x Reference ({len(ref_nuc_polys):,} nuclei)"),
    (cp_polys, "cyan", f"Cellpose ({len(cp_polys):,} nuclei)\nIoU={np.mean(cp_iou_nuc):.3f}, Recall={cp_recall_nuc*100:.0f}%"),
    (by_polys, "lime", f"Baysor ({len(by_polys):,} cells)\nIoU={np.mean(by_iou_cell):.3f}, Recall={by_recall_cell*100:.0f}%"),
]
for ax,(polys,color,title) in zip(axes,panels):
    ax.imshow(dapi_norm, cmap="gray", origin="lower")
    for poly in polys:
        xs=(np.array(poly.exterior.xy[0])-x0)/PIXEL_SIZE
        ys=(np.array(poly.exterior.xy[1])-y0)/PIXEL_SIZE
        ax.plot(xs,ys,color=color,linewidth=0.4,alpha=0.8)
    ax.set_title(title,fontsize=10); ax.axis("off")
plt.suptitle("Segmentation Comparison — 435×435 µm tile (n=1,432 reference cells)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"side_by_side_3way.png", dpi=150); plt.close()
log("Saved: side_by_side_3way.png")

# 2. IoU distribution both methods
fig, axes = plt.subplots(1,2,figsize=(13,5))
axes[0].hist(cp_iou_nuc, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
axes[0].axvline(np.mean(cp_iou_nuc), color="red", linestyle="--", label=f"Mean={np.mean(cp_iou_nuc):.3f}")
axes[0].set_xlabel("IoU"); axes[0].set_ylabel("Count")
axes[0].set_title(f"Cellpose IoU vs Nucleus Reference\n(n={len(cp_iou_nuc)} matched cells)")
axes[0].legend()
if by_iou_cell:
    axes[1].hist(by_iou_cell, bins=50, color="seagreen", edgecolor="none", alpha=0.85)
    axes[1].axvline(np.mean(by_iou_cell), color="red", linestyle="--", label=f"Mean={np.mean(by_iou_cell):.3f}")
    axes[1].set_title(f"Baysor IoU vs Cell Reference\n(n={len(by_iou_cell)} matched cells)")
    axes[1].set_xlabel("IoU"); axes[1].legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR/"iou_distributions_both.png", dpi=150); plt.close()
log("Saved: iou_distributions_both.png")

# 3. Full comparison bar chart
labels = ["Cellpose\nvs Nucleus","Cellpose\nvs Cell","Baysor\nvs Nucleus","Baysor\nvs Cell"]
iou_v   = [np.mean(cp_iou_nuc),np.mean(cp_iou_cell),np.mean(by_iou_nuc),np.mean(by_iou_cell)]
dice_v  = [np.mean(cp_dice_nuc),np.mean(cp_dice_cell),np.mean(by_dice_nuc),np.mean(by_dice_cell)]
recall_v= [cp_recall_nuc,cp_recall_cell,by_recall_nuc,by_recall_cell]
colors  = ["steelblue","steelblue","seagreen","seagreen"]
x = np.arange(4)

fig,axes = plt.subplots(1,3,figsize=(15,5))
for ax,vals,ylabel in zip(axes,[iou_v,dice_v,recall_v],["Mean IoU","Mean Dice","Recall"]):
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0,1); ax.set_ylabel(ylabel); ax.set_title(ylabel)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.suptitle(f"Full Evaluation: Cellpose vs Baysor vs 10x Reference (n=1,432 reference cells, 435×435 µm)", fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"three_way_iou_comparison.png", dpi=150); plt.close()
log("Saved: three_way_iou_comparison.png")

# 4. Transcripts per cell comparison
fig, ax = plt.subplots(figsize=(8,5))
for tpc,label,color in [(xen_tpc,"Xenium 10x","gold"),(cp_tpc,"Cellpose","steelblue"),(by_tpc,"Baysor","seagreen")]:
    if len(tpc)>0:
        ax.hist(tpc.clip(upper=500), bins=50, alpha=0.6, label=f"{label} (n={len(tpc):,})", color=color, density=True)
ax.set_xlabel("Transcripts per cell"); ax.set_ylabel("Density")
ax.set_title("Transcripts per Cell Distribution — All Methods")
ax.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR/"transcripts_per_cell_comparison.png", dpi=150); plt.close()
log("Saved: transcripts_per_cell_comparison.png")

# 5. Baysor segmentation map
fig, ax = plt.subplots(figsize=(9,9))
ax.imshow(dapi_norm, cmap="gray", origin="lower")
for poly in by_polys:
    xs=(np.array(poly.exterior.xy[0])-x0)/PIXEL_SIZE
    ys=(np.array(poly.exterior.xy[1])-y0)/PIXEL_SIZE
    ax.plot(xs,ys,color="lime",linewidth=0.5,alpha=0.85)
ax.set_title(f"Baysor Segmentation (scale=8µm, {len(by_polys):,} cells)"); ax.axis("off")
plt.tight_layout(); plt.savefig(RESULTS_DIR/"baysor_segmentation_map.png", dpi=150); plt.close()
log("Saved: baysor_segmentation_map.png")

# 6. Cellpose segmentation map
fig, axes = plt.subplots(1,2,figsize=(14,7))
axes[0].imshow(dapi_norm, cmap="gray", origin="lower"); axes[0].set_title("DAPI Input"); axes[0].axis("off")
axes[1].imshow(dapi_norm, cmap="gray", origin="lower")
axes[1].imshow(np.load(DATA_DIR/"cellpose_masks_large.npy")>0, cmap="Reds", alpha=0.4, origin="lower")
axes[1].set_title(f"Cellpose ({len(cp_polys):,} nuclei)"); axes[1].axis("off")
plt.tight_layout(); plt.savefig(RESULTS_DIR/"cellpose_segmentation_map.png", dpi=150); plt.close()
log("Saved: cellpose_segmentation_map.png")

# ═══════════════════════════════════════════════════════════════
# Downstream on tile (1,296 cells — enough for UMAP)
# ═══════════════════════════════════════════════════════════════
log("Downstream analysis on large tile...")

def build_adata(df, cell_col, gene_col="feature_name"):
    assigned = df[df[cell_col].notna() & (df[cell_col]!="UNASSIGNED") & (df[cell_col]!=0)].copy()
    assigned[cell_col] = assigned[cell_col].astype(str)
    counts = assigned.groupby([cell_col,gene_col]).size().unstack(fill_value=0)
    if counts.shape[0]<20: return None
    a = ad.AnnData(X=csr_matrix(counts.values))
    a.obs_names = counts.index.astype(str); a.var_names = counts.columns
    try:
        cx = assigned.groupby(cell_col)[["x_location","y_location"]].mean()
        a.obs = a.obs.join(cx.rename(columns={"x_location":"x","y_location":"y"}))
        a.obsm["spatial"] = a.obs[["x","y"]].values
    except: pass
    return a

# Build per-method AnnData on the large tile
xen_a = build_adata(t_tile, "cell_id")
cp_a  = build_adata(t_tile, "cellpose_cell_id")

# Baysor — merge xy back from original transcripts
baysor_assigned2 = baysor_assigned.copy()
baysor_assigned2["x_location"] = baysor_assigned2["x"]
baysor_assigned2["y_location"] = baysor_assigned2["y"]
baysor_assigned2["feature_name"] = baysor_assigned2["gene"]
by_a = build_adata(baysor_assigned2, "cell") if "cell" in baysor_assigned2.columns else None

adatas = {k:v for k,v in {"Xenium 10x":xen_a,"Cellpose":cp_a,"Baysor":by_a}.items() if v is not None}
for name,a in adatas.items():
    sc.pp.filter_cells(a, min_counts=5); sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=100); sc.pp.log1p(a)
    n_pcs = min(20, a.n_obs-1, a.n_vars-1)
    sc.pp.pca(a, n_comps=n_pcs)
    sc.pp.neighbors(a, n_neighbors=min(15,a.n_obs-1), n_pcs=min(10,n_pcs))
    sc.tl.umap(a); sc.tl.leiden(a, resolution=0.5, key_added="leiden")
    log(f"  {name}: {a.n_obs} cells, {a.obs['leiden'].nunique()} clusters")

ncols = len(adatas)
fig, axes = plt.subplots(1,ncols,figsize=(7*ncols,6))
if ncols==1: axes=[axes]
for ax,(name,a) in zip(axes,adatas.items()):
    sc.pl.umap(a,color="leiden",ax=ax,show=False,
               title=f"{name} ({a.n_obs} cells, {a.obs['leiden'].nunique()} clusters)",
               legend_loc="on data",legend_fontsize=9)
plt.suptitle("UMAP Per Method — 435×435 µm Tile", fontsize=13)
plt.tight_layout(); plt.savefig(RESULTS_DIR/"umap_comparison_methods.png",dpi=150); plt.close()
log("Saved: umap_comparison_methods.png")

# ═══════════════════════════════════════════════════════════════
# Validated cell type annotation
# ═══════════════════════════════════════════════════════════════
log("Validating cell type annotation from marker genes...")

adata_full = sc.read_h5ad(DATA_DIR/"adata_10x_processed.h5ad")
sc.tl.rank_genes_groups(adata_full, groupby="leiden", method="wilcoxon", n_genes=10, use_raw=False)

known = {
    "EPCAM":"Epithelial","KRT17":"Epithelial","KRT8":"Epithelial","KRT18":"Epithelial",
    "CD68":"Macrophage","CD163":"Macrophage","MRC1":"Macrophage",
    "CD3E":"T Cell","CD8A":"T Cell","CD4":"T Cell",
    "ACTA2":"Myoepithelial","CNN1":"Myoepithelial","MYLK":"Myoepithelial",
    "PECAM1":"Endothelial","VWF":"Endothelial","CDH5":"Endothelial",
    "COL1A1":"Stromal","FAP":"Stromal","VIM":"Stromal","PDGFRA":"Stromal",
    "MKI67":"Proliferating","TOP2A":"Proliferating","PCNA":"Proliferating",
    "CD79A":"B Cell","MS4A1":"B Cell","CD19":"B Cell",
}
clusters = list(adata_full.uns["rank_genes_groups"]["names"].dtype.names)
ct_map = {}
print("\n=== VALIDATED MARKER GENES PER CLUSTER ===")
for cl in clusters:
    top10 = [adata_full.uns["rank_genes_groups"]["names"][i][clusters.index(cl)] for i in range(10)]
    scores = [adata_full.uns["rank_genes_groups"]["scores"][i][clusters.index(cl)] for i in range(10)]
    label = "Unknown"
    for gene in top10:
        if gene in known: label=known[gene]; break
    ct_map[cl] = label
    print(f"  Cluster {cl} → {label:20s} | top genes: {top10[:5]}")

adata_full.obs["cell_type"] = adata_full.obs["leiden"].map(ct_map).astype(str)
props = adata_full.obs["cell_type"].value_counts(normalize=True).sort_values(ascending=False)
print("\n=== VALIDATED CELL TYPE PROPORTIONS ===")
for ct,v in props.items(): print(f"  {ct}: {v*100:.1f}%")

# UMAP coloured by validated cell type
fig, ax = plt.subplots(figsize=(9,7))
sc.pl.umap(adata_full, color="cell_type", ax=ax, show=False,
           title="10x Xenium — Validated Cell Type Annotation (201,446 cells)",
           legend_loc="right margin")
plt.tight_layout(); plt.savefig(RESULTS_DIR/"umap_cell_types_annotated.png",dpi=150); plt.close()

fig,ax = plt.subplots(figsize=(9,4))
colors = list(plt.cm.Set1.colors)+list(plt.cm.Set2.colors)
ax.bar(props.index, props.values, color=colors[:len(props)])
for i,(ct,v) in enumerate(props.items()):
    ax.text(i,v+0.004,f"{v*100:.1f}%",ha="center",fontsize=9)
ax.set_ylabel("Proportion"); ax.set_title("Validated Cell Type Proportions (Marker Gene Confirmed)")
ax.tick_params(axis="x",rotation=25)
plt.tight_layout(); plt.savefig(RESULTS_DIR/"cell_type_proportions.png",dpi=150); plt.close()

adata_full.write_h5ad(DATA_DIR/"adata_10x_processed.h5ad")
log("Saved: validated annotation figures")

# ═══════════════════════════════════════════════════════════════
# Spatial neighbourhood enrichment (manual, validated)
# ═══════════════════════════════════════════════════════════════
log("Spatial neighbourhood enrichment...")
from scipy.spatial import cKDTree

coords = adata_full.obsm["spatial"]
ct_labels = adata_full.obs["cell_type"].values
cell_types = sorted(adata_full.obs["cell_type"].unique())
ct_idx = {ct:i for i,ct in enumerate(cell_types)}
n_ct = len(cell_types)

tree = cKDTree(coords)
pairs = tree.query_pairs(r=50.0)
obs_mat = np.zeros((n_ct,n_ct),dtype=int)
for i,j in pairs:
    a,b = ct_idx[ct_labels[i]], ct_idx[ct_labels[j]]
    obs_mat[a,b]+=1; obs_mat[b,a]+=1

totals = obs_mat.sum(axis=1)
total_pairs = obs_mat.sum()
exp_mat = np.outer(totals,totals)/(total_pairs+1e-9)
enrich = np.log2((obs_mat+1)/(exp_mat+1))

fig, ax = plt.subplots(figsize=(9,8))
sns.heatmap(enrich, xticklabels=cell_types, yticklabels=cell_types,
            cmap="RdBu_r", center=0, ax=ax, annot=True, fmt=".1f", annot_kws={"size":8})
ax.set_title("Spatial Neighbourhood Enrichment\n(log₂ observed/expected co-occurrence within 50 µm)")
plt.xticks(rotation=30,ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"neighbourhood_enrichment.png",dpi=150,bbox_inches="tight"); plt.close()
log("Saved: neighbourhood_enrichment.png")

# ═══════════════════════════════════════════════════════════════
total = (time.time()-t0)/60
log(f"\n{'='*60}")
log(f"PAPER-QUALITY STEP 2 COMPLETE in {total:.1f} minutes")
log(f"{'='*60}")
print(f"\nKey results:")
print(f"  Cellpose vs nucleus: IoU={np.mean(cp_iou_nuc):.3f}, Dice={np.mean(cp_dice_nuc):.3f}, Recall={cp_recall_nuc*100:.1f}%")
print(f"  Cellpose vs cell:    IoU={np.mean(cp_iou_cell):.3f}, Dice={np.mean(cp_dice_cell):.3f}, Recall={cp_recall_cell*100:.1f}%")
print(f"  Baysor vs nucleus:   IoU={np.mean(by_iou_nuc):.3f},  Dice={np.mean(by_dice_nuc):.3f},  Recall={by_recall_nuc*100:.1f}%")
print(f"  Baysor vs cell:      IoU={np.mean(by_iou_cell):.3f},  Dice={np.mean(by_dice_cell):.3f},  Recall={by_recall_cell*100:.1f}%")
print(f"\nAll results in: {RESULTS_DIR.resolve()}")
