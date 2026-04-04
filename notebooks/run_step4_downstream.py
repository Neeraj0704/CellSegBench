"""
Step 4 only — Downstream biological analysis
Picks up after Cellpose + evaluation are already done.
"""
import sys, time, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from pathlib import Path

DATA_DIR   = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
t0 = time.time()
def elapsed(): return f"[{(time.time()-t0)/60:.1f} min]"

sc.settings.verbosity = 0

print(f"STEP 4: Downstream analysis  {elapsed()}")

# ── Load 10x cell feature matrix ───────────────────────────────────────────
print("Loading 10x AnnData...")
adata = sc.read_10x_h5(DATA_DIR / "cell_feature_matrix.h5")
cells_df = pd.read_parquet(DATA_DIR / "cells.parquet").set_index("cell_id")
adata.obs = adata.obs.join(
    cells_df[["x_centroid", "y_centroid", "cell_area", "nucleus_area"]], how="left"
)
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars} genes  {elapsed()}")

# ── QC filter ──────────────────────────────────────────────────────────────
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=10)
print(f"After QC: {adata.n_obs:,} cells × {adata.n_vars} genes  {elapsed()}")

# ── Normalise + log ────────────────────────────────────────────────────────
sc.pp.normalize_total(adata, target_sum=100)
sc.pp.log1p(adata)

# ── PCA → neighbours → UMAP → Leiden ──────────────────────────────────────
# 280 genes — use all, skip HVG selection
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.4, key_added="leiden")
n_clusters = adata.obs["leiden"].nunique()
print(f"Clustering: {n_clusters} Leiden clusters  {elapsed()}")

# ── UMAP ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata, color="leiden", ax=ax, show=False,
           title=f"10x Xenium UMAP — {adata.n_obs:,} cells, {n_clusters} clusters",
           legend_loc="on data", legend_fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "umap_leiden_10x.png", dpi=150)
plt.close()
print(f"Saved: umap_leiden_10x.png  {elapsed()}")

# ── Marker gene dot plot ───────────────────────────────────────────────────
candidates = ["EPCAM", "KRT17", "KRT8", "CD68", "CD3E", "VIM", "ACTA2", "PECAM1", "MKI67", "CD79A"]
markers = [g for g in candidates if g in adata.var_names]
print(f"Marker genes available: {markers}")

if markers:
    fig, ax = plt.subplots(figsize=(max(8, len(markers)*1.1), 5))
    sc.pl.dotplot(adata, var_names=markers, groupby="leiden",
                  ax=ax, show=False, standard_scale="var",
                  title="Marker Gene Expression by Cluster")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "marker_dotplot_10x.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: marker_dotplot_10x.png  {elapsed()}")

# ── Cluster proportions ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
props = adata.obs["leiden"].value_counts(normalize=True).sort_index()
colors = [plt.cm.tab20.colors[int(i) % 20] for i in props.index]
ax.bar(props.index.astype(str), props.values, color=colors)
ax.set_xlabel("Leiden cluster")
ax.set_ylabel("Proportion of cells")
ax.set_title(f"Cell Cluster Proportions — 10x Xenium ({adata.n_obs:,} cells)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cluster_proportions_10x.png", dpi=150)
plt.close()
print(f"Saved: cluster_proportions_10x.png  {elapsed()}")

# ── Spatial cluster map (subsample for speed) ──────────────────────────────
print("Plotting spatial clusters...")
sub_idx = np.random.choice(adata.n_obs, size=min(50_000, adata.n_obs), replace=False)
sub = adata[sub_idx]

fig, ax = plt.subplots(figsize=(10, 10))
palette = plt.cm.tab20.colors
for cluster in sorted(sub.obs["leiden"].unique(), key=int):
    mask = sub.obs["leiden"] == cluster
    coords = sub.obsm["spatial"][mask.values]
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.5,
               color=palette[int(cluster) % 20], label=f"C{cluster}")
ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
ax.set_title(f"Spatial Cluster Map — 10x Xenium (50k cell sample)")
ax.set_aspect("equal")
ax.legend(markerscale=8, fontsize=8, ncol=2, loc="upper right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "spatial_clusters_10x.png", dpi=150)
plt.close()
print(f"Saved: spatial_clusters_10x.png  {elapsed()}")

# ── Save AnnData ───────────────────────────────────────────────────────────
adata.write_h5ad(DATA_DIR / "adata_10x_processed.h5ad")
print(f"Saved: adata_10x_processed.h5ad  {elapsed()}")

# ── Summary ────────────────────────────────────────────────────────────────
total = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"STEP 4 COMPLETE in {total:.1f} minutes")
print(f"{'='*60}")
print(f"  Cells analysed:    {adata.n_obs:,}")
print(f"  Genes:             {adata.n_vars}")
print(f"  Leiden clusters:   {n_clusters}")
print(f"\nAll results in: {RESULTS_DIR.resolve()}")
for f in sorted(RESULTS_DIR.iterdir()):
    if f.suffix in [".png", ".csv"]:
        print(f"  {f.name}")
