"""
Final steps: cell type annotation + neighbourhood enrichment
"""
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from pathlib import Path

DATA_DIR    = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
sc.settings.verbosity = 0

# ── Load processed AnnData ────────────────────────────────────────────────
print("Loading processed AnnData...")
adata = sc.read_h5ad(DATA_DIR / "adata_10x_processed.h5ad")
print(f"Cells: {adata.n_obs:,}, Clusters: {adata.obs['leiden'].nunique()}")

# ── Cell type annotation from marker genes ────────────────────────────────
print("Annotating cell types...")
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", n_genes=10)

# Map clusters to cell types based on known markers
# EPCAM/KRT17/KRT8 = Epithelial, CD68 = Macrophage, CD3E = T cell,
# ACTA2 = Myoepithelial, PECAM1 = Endothelial, MKI67 = Proliferating, VIM = Stromal
cluster_labels = {
    "0": "Epithelial",
    "1": "Stromal/Fibroblast",
    "2": "Macrophage",
    "3": "Epithelial (basal)",
    "4": "T Cell",
    "5": "Endothelial",
    "6": "Myoepithelial",
}
adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels).astype(str)

# UMAP coloured by cell type
fig, ax = plt.subplots(figsize=(9, 7))
sc.pl.umap(adata, color="cell_type", ax=ax, show=False,
           title="10x Xenium — Annotated Cell Types (201,446 cells)",
           legend_loc="right margin", legend_fontsize=9,
           palette=["#e41a1c","#ff7f00","#4daf4a","#984ea3",
                    "#377eb8","#a65628","#f781bf"])
plt.tight_layout()
plt.savefig(RESULTS_DIR / "umap_cell_types_annotated.png", dpi=150)
plt.close()
print("Saved: umap_cell_types_annotated.png")

# Cell type proportion bar chart
fig, ax = plt.subplots(figsize=(9, 4))
props = adata.obs["cell_type"].value_counts(normalize=True).sort_values(ascending=False)
colors = ["#e41a1c","#ff7f00","#4daf4a","#984ea3","#377eb8","#a65628","#f781bf"]
ax.bar(props.index, props.values, color=colors[:len(props)])
ax.set_xlabel("Cell Type"); ax.set_ylabel("Proportion")
ax.set_title("Cell Type Proportions — 10x Xenium (201,446 cells)")
ax.tick_params(axis="x", rotation=20)
for i, (ct, v) in enumerate(props.items()):
    ax.text(i, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cell_type_proportions.png", dpi=150)
plt.close()
print("Saved: cell_type_proportions.png")
print(f"\nCell type proportions:")
for ct, v in props.items():
    print(f"  {ct}: {v*100:.1f}%")

# Spatial plot coloured by cell type
sub_idx = np.random.choice(adata.n_obs, size=min(60_000, adata.n_obs), replace=False)
sub = adata[sub_idx]
fig, ax = plt.subplots(figsize=(10, 10))
ct_colors = dict(zip(props.index, colors[:len(props)]))
for ct in sub.obs["cell_type"].unique():
    mask = sub.obs["cell_type"] == ct
    coords = sub.obsm["spatial"][mask.values]
    ax.scatter(coords[:,0], coords[:,1], s=0.5, alpha=0.5,
               color=ct_colors.get(ct, "grey"), label=ct)
ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
ax.set_title("Spatial Cell Type Map — 10x Xenium (60k sample)")
ax.set_aspect("equal")
ax.legend(markerscale=8, fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "spatial_cell_types.png", dpi=150)
plt.close()
print("Saved: spatial_cell_types.png")

# ── Spatial co-occurrence (manual, no multiprocessing) ───────────────────
print("\nComputing spatial co-occurrence...")
from scipy.spatial import cKDTree
import seaborn as sns

coords = adata.obsm["spatial"]
ct_labels = adata.obs["cell_type"].values
cell_types = sorted(adata.obs["cell_type"].unique())
n_ct = len(cell_types)
ct_idx = {ct: i for i, ct in enumerate(cell_types)}

# Build KD-tree and count neighbours within 50 µm
tree = cKDTree(coords)
pairs = tree.query_pairs(r=50.0)

# Count observed co-occurrences
obs_matrix = np.zeros((n_ct, n_ct), dtype=int)
for i, j in pairs:
    a, b = ct_idx[ct_labels[i]], ct_idx[ct_labels[j]]
    obs_matrix[a, b] += 1
    obs_matrix[b, a] += 1

# Expected = product of marginals / total
totals = obs_matrix.sum(axis=1)
total_pairs = obs_matrix.sum()
exp_matrix = np.outer(totals, totals) / (total_pairs + 1e-9)

# Enrichment = log2(obs/exp)
enrich = np.log2((obs_matrix + 1) / (exp_matrix + 1))

fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(enrich, xticklabels=cell_types, yticklabels=cell_types,
            cmap="RdBu_r", center=0, ax=ax,
            annot=True, fmt=".1f", annot_kws={"size": 8})
ax.set_title("Spatial Neighbourhood Enrichment\n(log₂ observed/expected within 50 µm)")
plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "neighbourhood_enrichment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: neighbourhood_enrichment.png")

print("\n=== FINAL STEPS COMPLETE ===")
print("Cell types annotated, neighbourhood enrichment computed.")
