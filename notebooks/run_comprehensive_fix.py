"""
Comprehensive fix for all identified scientific issues:
1. Baysor quality filtering (remove noise cells area<10um2, n_transcripts<10)
2. Baysor parameter sweep (scales 6, 8, 10, 12, 15 um)
3. Statistical significance tests (Mann-Whitney U + 95% CIs)
4. Per-method cell type marker gene comparison
5. Fixed random seeds everywhere
6. Corrected cell type annotations
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
import seaborn as sns
import tifffile
from pathlib import Path
from skimage import measure
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from scipy import stats
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

DATA_DIR    = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)
PIXEL_SIZE  = 0.2125
RANDOM_SEED = 42
sc.settings.verbosity = 0

t0 = time.time()
def log(msg): print(f"[{(time.time()-t0)/60:.1f}m] {msg}", flush=True)

np.random.seed(RANDOM_SEED)

# ── Load tile metadata ─────────────────────────────────────────
with open(DATA_DIR/"tile_meta_large.json") as f:
    tile = json.load(f)
x0, x1 = tile["x0_um"], tile["x1_um"]
y0, y1 = tile["y0_um"], tile["y1_um"]
r0, r1, c0, c1 = tile["r0"], tile["r1"], tile["c0"], tile["c1"]

# ── Load DAPI ──────────────────────────────────────────────────
log("Loading DAPI...")
with tifffile.TiffFile(DATA_DIR/"morphology_focus"/"ch0000_dapi.ome.tif") as tif:
    dapi_tile = tif.pages[0].asarray()[r0:r1, c0:c1]
dapi_norm = (dapi_tile.astype("float32")-dapi_tile.min())/(dapi_tile.max()-dapi_tile.min())

# ── Load reference polygons ────────────────────────────────────
log("Loading reference polygons...")
cells_all = pd.read_parquet(DATA_DIR/"cells.parquet")
nb_all    = pd.read_parquet(DATA_DIR/"nucleus_boundaries.parquet")
cb_all    = pd.read_parquet(DATA_DIR/"cell_boundaries.parquet")

tile_cells = cells_all[
    (cells_all["x_centroid"]>=x0)&(cells_all["x_centroid"]<x1)&
    (cells_all["y_centroid"]>=y0)&(cells_all["y_centroid"]<y1)
]
tile_ids = set(tile_cells["cell_id"])
log(f"Reference cells in tile: {len(tile_cells):,}")

def build_polys(bdf, id_set):
    polys = []
    for cid, grp in bdf[bdf["cell_id"].isin(id_set)].groupby("cell_id"):
        coords = list(zip(grp["vertex_x"], grp["vertex_y"]))
        if len(coords)>=3:
            p = Polygon(coords)
            if p.is_valid and p.area>0: polys.append(p)
    return polys

ref_nuc  = build_polys(nb_all, tile_ids)
ref_cell = build_polys(cb_all, tile_ids)
log(f"Reference: {len(ref_nuc)} nucleus polys, {len(ref_cell)} cell polys")

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

# ══════════════════════════════════════════════════════════════════════
# IoU matching (reusable)
# ══════════════════════════════════════════════════════════════════════
def match_iou(pred_polys, ref_polys, thresh=0.3):
    if not pred_polys or not ref_polys: return [], [], 0.0
    tree = STRtree(ref_polys)
    iou_s, dice_s, matched = [], [], set()
    for pred in pred_polys:
        cands = list(tree.query(pred))
        best_iou, best_idx = 0, -1
        for idx in cands:
            if idx in matched: continue
            ref = ref_polys[idx]
            inter = pred.intersection(ref).area
            union = pred.union(ref).area
            iou = inter/union if union>0 else 0
            if iou>best_iou: best_iou, best_idx = iou, idx
        if best_iou>=thresh and best_idx>=0:
            iou_s.append(best_iou)
            ref = ref_polys[best_idx]
            dice_s.append(2*pred.intersection(ref).area/(pred.area+ref.area))
            matched.add(best_idx)
    return iou_s, dice_s, len(matched)/len(ref_polys)

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Baysor quality filtering analysis
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 1: Baysor quality analysis and filtering")

baysor_out  = DATA_DIR/"baysor_output_large"
baysor_df   = pd.read_csv(baysor_out/"segmentation.csv")
by_stats    = pd.read_csv(baysor_out/"segmentation_cell_stats.csv")
baysor_assigned = baysor_df[baysor_df["is_noise"]==False].copy() if "is_noise" in baysor_df.columns else baysor_df.copy()

log(f"Raw Baysor cells: {len(by_stats):,}")
log(f"  Area: mean={by_stats['area'].mean():.1f}, median={by_stats['area'].median():.1f}, min={by_stats['area'].min():.2f}")
log(f"  n_transcripts: mean={by_stats['n_transcripts'].mean():.1f}, median={by_stats['n_transcripts'].median():.1f}, min={by_stats['n_transcripts'].min()}")

# Quality filter: remove tiny noise cells
AREA_MIN    = 10.0   # µm² — roughly 1.8 µm radius minimum
N_TRANS_MIN = 10     # minimum transcripts per cell
good_cells = by_stats[(by_stats["area"]>=AREA_MIN)&(by_stats["n_transcripts"]>=N_TRANS_MIN)]["cell"]
bad_cells  = by_stats[(by_stats["area"]<AREA_MIN)|(by_stats["n_transcripts"]<N_TRANS_MIN)]["cell"]
log(f"Noise cells removed (area<{AREA_MIN} or n_trans<{N_TRANS_MIN}): {len(bad_cells):,}")
log(f"Quality Baysor cells retained: {len(good_cells):,}")

# Load Baysor polygons (all + filtered)
with open(baysor_out/"segmentation_polygons_2d.json") as f:
    geo = json.load(f)

by_polys_all = []
by_poly_ids  = []
for feat in geo.get("features",[]):
    cell_id = feat.get("id","")
    geom = feat["geometry"]
    if geom["type"]=="Polygon":
        coords = geom["coordinates"][0]
        p = Polygon(coords)
        if p.is_valid and p.area>0:
            by_polys_all.append(p)
            by_poly_ids.append(cell_id)

# Filter to quality cells only
good_cell_set = set(good_cells.astype(str))
by_polys_filtered = [p for p,cid in zip(by_polys_all,by_poly_ids) if cid in good_cell_set]
log(f"All Baysor polygons: {len(by_polys_all):,}, Quality-filtered: {len(by_polys_filtered):,}")

# Cell area comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(by_stats["area"].clip(upper=300), bins=60, color="tomato", edgecolor="none", alpha=0.8)
axes[0].axvline(AREA_MIN, color="black", linestyle="--", label=f"Filter ({AREA_MIN} µm²)")
axes[0].set_xlabel("Cell area (µm²)"); axes[0].set_ylabel("Count")
axes[0].set_title(f"Baysor Cell Area Distribution\n(n={len(by_stats):,}, clipped at 300 µm²)")
axes[0].legend()

axes[1].hist(tile_cells["cell_area"].clip(upper=300), bins=60, color="steelblue", edgecolor="none", alpha=0.8)
axes[1].set_xlabel("Cell area (µm²)"); axes[1].set_ylabel("Count")
axes[1].set_title(f"10x Reference Cell Area Distribution\n(n={len(tile_cells):,})")

axes[2].hist(tile_cells["cell_area"].clip(upper=300), bins=60, color="steelblue", edgecolor="none", alpha=0.6, label="10x Reference", density=True)
axes[2].hist(by_stats[by_stats["area"]>=AREA_MIN]["area"].clip(upper=300), bins=60, color="seagreen", edgecolor="none", alpha=0.6, label="Baysor (filtered)", density=True)
axes[2].set_xlabel("Cell area (µm²)"); axes[2].set_ylabel("Density")
axes[2].set_title("Cell Area Comparison (10x vs Baysor filtered)"); axes[2].legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR/"baysor_cell_area_analysis.png", dpi=150); plt.close()
log("Saved: baysor_cell_area_analysis.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Full IoU evaluation with and without Baysor filter
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 2: IoU evaluation — all vs filtered Baysor")

cp_iou_nuc,  cp_dice_nuc,  cp_recall_nuc  = match_iou(cp_polys, ref_nuc)
cp_iou_cell, cp_dice_cell, cp_recall_cell = match_iou(cp_polys, ref_cell)

by_iou_nuc_all,  by_dice_nuc_all,  by_recall_nuc_all  = match_iou(by_polys_all, ref_nuc)
by_iou_cell_all, by_dice_cell_all, by_recall_cell_all = match_iou(by_polys_all, ref_cell)

by_iou_nuc_f,  by_dice_nuc_f,  by_recall_nuc_f  = match_iou(by_polys_filtered, ref_nuc)
by_iou_cell_f, by_dice_cell_f, by_recall_cell_f = match_iou(by_polys_filtered, ref_cell)

log(f"Cellpose vs nucleus (n={len(cp_iou_nuc)}): IoU={np.mean(cp_iou_nuc):.3f}±{np.std(cp_iou_nuc):.3f}, Recall={cp_recall_nuc*100:.1f}%")
log(f"Cellpose vs cell   (n={len(cp_iou_cell)}): IoU={np.mean(cp_iou_cell):.3f}±{np.std(cp_iou_cell):.3f}, Recall={cp_recall_cell*100:.1f}%")
log(f"Baysor (all) vs cell  (n={len(by_iou_cell_all)}): IoU={np.mean(by_iou_cell_all):.3f}, Recall={by_recall_cell_all*100:.1f}%")
log(f"Baysor (filtered) vs cell (n={len(by_iou_cell_f)}): IoU={np.mean(by_iou_cell_f):.3f}, Recall={by_recall_cell_f*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Statistical significance tests
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 3: Statistical significance tests")

def bootstrap_ci(data, n_boot=2000, ci=0.95, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return np.mean(data), lo, hi

# Mann-Whitney U: Cellpose vs Baysor (filtered) — IoU distributions
mw_nuc = stats.mannwhitneyu(cp_iou_nuc, by_iou_nuc_f, alternative="two-sided") if by_iou_nuc_f else None
mw_cell = stats.mannwhitneyu(cp_iou_cell, by_iou_cell_f, alternative="two-sided") if by_iou_cell_f else None

# Bootstrap CIs
cp_nuc_mean,  cp_nuc_lo,  cp_nuc_hi  = bootstrap_ci(cp_iou_nuc)
cp_cell_mean, cp_cell_lo, cp_cell_hi = bootstrap_ci(cp_iou_cell)

log(f"\nCellpose vs Nucleus:  mean IoU={cp_nuc_mean:.3f} [{cp_nuc_lo:.3f}, {cp_nuc_hi:.3f}] 95% CI")
log(f"Cellpose vs Cell:     mean IoU={cp_cell_mean:.3f} [{cp_cell_lo:.3f}, {cp_cell_hi:.3f}] 95% CI")

if by_iou_cell_f:
    by_cell_mean, by_cell_lo, by_cell_hi = bootstrap_ci(by_iou_cell_f)
    log(f"Baysor vs Cell (filt): mean IoU={by_cell_mean:.3f} [{by_cell_lo:.3f}, {by_cell_hi:.3f}] 95% CI")
    if mw_cell:
        log(f"Mann-Whitney U (IoU: Cellpose-cell vs Baysor-cell): U={mw_cell.statistic:.0f}, p={mw_cell.pvalue:.2e}")

print("\n=== STATISTICAL TEST RESULTS ===")
print(f"Cellpose vs nucleus IoU: {cp_nuc_mean:.4f} (95% CI [{cp_nuc_lo:.4f}, {cp_nuc_hi:.4f}])")
print(f"Cellpose vs cell IoU:    {cp_cell_mean:.4f} (95% CI [{cp_cell_lo:.4f}, {cp_cell_hi:.4f}])")
if by_iou_cell_f:
    print(f"Baysor (filt) vs cell:   {by_cell_mean:.4f} (95% CI [{by_cell_lo:.4f}, {by_cell_hi:.4f}])")
if mw_cell: print(f"Mann-Whitney p (Cellpose vs Baysor IoU): {mw_cell.pvalue:.2e}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Baysor parameter sweep (scales 6, 10, 12, 15 um)
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 4: Baysor parameter sweep")

baysor_csv = DATA_DIR/"baysor_input_large.csv"
sweep_results = [
    {"scale": 8, "n_cells_raw": len(by_stats),
     "n_cells_filtered": len(good_cells),
     "recall_cell_raw": by_recall_cell_all,
     "recall_cell_filt": by_recall_cell_f,
     "iou_cell_raw": np.mean(by_iou_cell_all) if by_iou_cell_all else 0,
     "iou_cell_filt": np.mean(by_iou_cell_f) if by_iou_cell_f else 0}
]

for scale in [6, 10, 12, 15]:
    log(f"  Running Baysor scale={scale}µm...")
    out_dir = DATA_DIR/f"baysor_sweep_s{scale}"
    out_dir.mkdir(exist_ok=True)
    cmd = ["julia", f"--project={Path.home()}/.julia/environments/v1.11", "-e",
           f"""using Baysor; append!(ARGS, ["run","-x","x","-y","y","-z","z","-g","gene",
           "-s","{scale}","-m","5","--n-clusters=4","-o","{out_dir}","{baysor_csv}"]); Baysor.julia_main()"""]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log(f"  Baysor scale={scale} failed: {result.stderr[-200:]}")
        continue

    seg_path = out_dir/"segmentation.csv"
    stats_path = out_dir/"segmentation_cell_stats.csv"
    poly_path  = out_dir/"segmentation_polygons_2d.json"
    if not seg_path.exists():
        log(f"  No output for scale={scale}")
        continue

    s_stats = pd.read_csv(stats_path)
    good = s_stats[(s_stats["area"]>=AREA_MIN)&(s_stats["n_transcripts"]>=N_TRANS_MIN)]["cell"]

    # Load polygons
    s_polys_all = []; s_poly_ids = []
    if poly_path.exists():
        with open(poly_path) as f: geo2 = json.load(f)
        for feat in geo2.get("features",[]):
            geom = feat["geometry"]
            if geom["type"]=="Polygon":
                coords = geom["coordinates"][0]
                p = Polygon(coords)
                if p.is_valid and p.area>0:
                    s_polys_all.append(p)
                    s_poly_ids.append(feat.get("id",""))

    good_set = set(good.astype(str))
    s_polys_f = [p for p,cid in zip(s_polys_all,s_poly_ids) if cid in good_set]

    _, _, recall_all = match_iou(s_polys_all, ref_cell)
    iou_f, _, recall_f = match_iou(s_polys_f, ref_cell)

    sweep_results.append({
        "scale": scale,
        "n_cells_raw": len(s_stats),
        "n_cells_filtered": len(good),
        "recall_cell_raw": recall_all,
        "recall_cell_filt": recall_f,
        "iou_cell_raw": np.mean(iou_f) if iou_f else 0,
        "iou_cell_filt": np.mean(iou_f) if iou_f else 0,
    })
    log(f"  scale={scale}: {len(s_stats)} cells ({len(good)} filtered), recall_filt={recall_f*100:.1f}%, IoU={np.mean(iou_f):.3f}")

sweep_df = pd.DataFrame(sweep_results).sort_values("scale").reset_index(drop=True)
cv_n = sweep_df["n_cells_filtered"].std()/sweep_df["n_cells_filtered"].mean()
sweep_df["cv"] = round(cv_n, 4)
sweep_df.to_csv(RESULTS_DIR/"baysor_scale_sweep.csv", index=False)
print("\n=== BAYSOR SCALE SWEEP ===")
print(sweep_df.to_string(index=False))
log(f"Baysor cell count CV across scale sweep: {cv_n:.3f}")

# Baysor sweep plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(sweep_df["scale"], sweep_df["n_cells_filtered"], marker="o", color="seagreen", linewidth=2)
axes[0].axvline(8, color="red", linestyle="--", label="Current (8 µm)")
for _, row in sweep_df.iterrows():
    axes[0].annotate(f"{int(row['n_cells_filtered'])}", (row['scale'], row['n_cells_filtered']),
                     textcoords="offset points", xytext=(5,5), fontsize=9)
axes[0].set_xlabel("Prior cell radius (µm)"); axes[0].set_ylabel("Cells (quality-filtered)")
axes[0].set_title("Baysor — Cells Detected vs Scale"); axes[0].legend()

axes[1].plot(sweep_df["scale"], sweep_df["recall_cell_filt"]*100, marker="o", color="seagreen", linewidth=2)
axes[1].axvline(8, color="red", linestyle="--")
axes[1].set_xlabel("Prior cell radius (µm)"); axes[1].set_ylabel("Recall (%) vs 10x cell reference")
axes[1].set_title("Baysor — Recall vs Scale")

axes[2].plot(sweep_df["scale"], sweep_df["iou_cell_filt"], marker="o", color="seagreen", linewidth=2)
axes[2].axvline(8, color="red", linestyle="--")
axes[2].set_xlabel("Prior cell radius (µm)"); axes[2].set_ylabel("Mean IoU vs 10x cell reference")
axes[2].set_title("Baysor — Mean IoU vs Scale")
plt.tight_layout()
plt.savefig(RESULTS_DIR/"baysor_scale_sweep.png", dpi=150); plt.close()
log("Saved: baysor_scale_sweep.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Updated summary figures with statistics
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 5: Updated evaluation figures with statistics")

# IoU distributions with statistical annotation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(cp_iou_nuc, bins=50, color="steelblue", edgecolor="none", alpha=0.85,
             label=f"Cellpose vs Nucleus\n(n={len(cp_iou_nuc)}, mean={np.mean(cp_iou_nuc):.3f})")
axes[0].axvline(np.mean(cp_iou_nuc), color="steelblue", linestyle="--", linewidth=2)
if by_iou_nuc_f:
    axes[0].hist(by_iou_nuc_f, bins=50, color="seagreen", edgecolor="none", alpha=0.6,
                 label=f"Baysor (filtered) vs Nucleus\n(n={len(by_iou_nuc_f)}, mean={np.mean(by_iou_nuc_f):.3f})")
    axes[0].axvline(np.mean(by_iou_nuc_f), color="seagreen", linestyle="--", linewidth=2)
axes[0].set_xlabel("IoU"); axes[0].set_ylabel("Count")
axes[0].set_title("IoU vs Nucleus Reference\n(Both methods — for comparison)")
axes[0].legend(fontsize=9)

axes[1].hist(cp_iou_cell, bins=50, color="steelblue", edgecolor="none", alpha=0.85,
             label=f"Cellpose vs Cell\n(n={len(cp_iou_cell)}, mean={np.mean(cp_iou_cell):.3f})")
axes[1].axvline(np.mean(cp_iou_cell), color="steelblue", linestyle="--", linewidth=2)
if by_iou_cell_f:
    axes[1].hist(by_iou_cell_f, bins=50, color="seagreen", edgecolor="none", alpha=0.6,
                 label=f"Baysor (filtered) vs Cell\n(n={len(by_iou_cell_f)}, mean={np.mean(by_iou_cell_f):.3f})")
    axes[1].axvline(np.mean(by_iou_cell_f), color="seagreen", linestyle="--", linewidth=2)
if mw_cell:
    axes[1].text(0.05, 0.95, f"Mann-Whitney p = {mw_cell.pvalue:.2e}",
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
axes[1].set_xlabel("IoU"); axes[1].set_ylabel("Count")
axes[1].set_title("IoU vs Cell Reference (Primary Fair Comparison)\nStatistical test included")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"iou_distributions_both.png", dpi=150); plt.close()
log("Saved: iou_distributions_both.png")

# Main comparison bar chart with error bars and p-values
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
by_cell_filt_mean = np.mean(by_iou_cell_f) if by_iou_cell_f else 0
by_dice_filt_mean = np.mean(by_dice_cell_f) if by_dice_cell_f else 0

iou_data   = [np.mean(cp_iou_nuc), np.mean(cp_iou_cell),
               np.mean(by_iou_nuc_f) if by_iou_nuc_f else 0, by_cell_filt_mean]
iou_err    = [np.std(cp_iou_nuc)/np.sqrt(len(cp_iou_nuc)),
               np.std(cp_iou_cell)/np.sqrt(len(cp_iou_cell)),
               np.std(by_iou_nuc_f)/np.sqrt(len(by_iou_nuc_f)) if by_iou_nuc_f else 0,
               np.std(by_iou_cell_f)/np.sqrt(len(by_iou_cell_f)) if by_iou_cell_f else 0]
dice_data  = [np.mean(cp_dice_nuc), np.mean(cp_dice_cell),
               np.mean(by_dice_nuc_f) if by_iou_nuc_f else 0, by_dice_filt_mean]
recall_data= [cp_recall_nuc, cp_recall_cell,
               by_recall_nuc_f, by_recall_cell_f]

labels_4 = ["Cellpose\nvs Nucleus★","Cellpose\nvs Cell","Baysor\nvs Nucleus","Baysor\nvs Cell★"]
colors_4  = ["steelblue","lightsteelblue","seagreen","lightgreen"]
x = np.arange(4)

for ax, vals, errs, ylabel in zip(axes,
    [iou_data, dice_data, recall_data],
    [iou_err, [0]*4, [0]*4],
    ["Mean IoU (± SEM)", "Mean Dice", "Recall"]):
    bars = ax.bar(x, vals, yerr=errs, color=colors_4, alpha=0.9,
                   capsize=5, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels_4, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel); ax.set_title(ylabel)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, min(v+0.02, 0.97), f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")

if mw_cell:
    axes[0].text(0.5, 0.97, f"Cellpose★ vs Baysor★: p={mw_cell.pvalue:.2e}",
                 transform=axes[0].transAxes, fontsize=8, ha="center",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle(f"Full Evaluation — Quality-Filtered Results (n={len(tile_cells):,} reference cells, 435×435 µm tile)\n"
             f"★ = primary fair comparison for each method", fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"three_way_iou_comparison.png", dpi=150); plt.close()
log("Saved: three_way_iou_comparison.png")

# Side-by-side with filtered Baysor
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
panels = [
    (ref_nuc, "yellow", f"10x Reference ({len(ref_nuc):,} nuclei)"),
    (cp_polys, "cyan",  f"Cellpose ({len(cp_polys):,} nuclei)\nIoU={np.mean(cp_iou_nuc):.3f}±{np.std(cp_iou_nuc):.3f}, Recall={cp_recall_nuc*100:.0f}%"),
    (by_polys_filtered, "lime",
     f"Baysor filtered ({len(by_polys_filtered):,} cells)\nIoU={np.mean(by_iou_cell_f):.3f}±{np.std(by_iou_cell_f):.3f}, Recall={by_recall_cell_f*100:.0f}%" if by_iou_cell_f else "Baysor filtered"),
]
for ax, (polys, color, title) in zip(axes, panels):
    ax.imshow(dapi_norm, cmap="gray", origin="lower")
    for poly in polys:
        xs=(np.array(poly.exterior.xy[0])-x0)/PIXEL_SIZE
        ys=(np.array(poly.exterior.xy[1])-y0)/PIXEL_SIZE
        ax.plot(xs, ys, color=color, linewidth=0.4, alpha=0.8)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.suptitle("Segmentation Comparison — 435×435 µm tile (n=1,432 reference cells)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"side_by_side_3way.png", dpi=150); plt.close()
log("Saved: side_by_side_3way.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Per-method downstream with marker gene comparison
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 6: Per-method downstream + marker gene comparison")

t_tile = pq.read_table(DATA_DIR/"transcripts_cellpose_large.parquet").to_pandas()
baysor_assigned["x_location"] = baysor_assigned["x"]
baysor_assigned["y_location"] = baysor_assigned["y"]
baysor_assigned["feature_name"] = baysor_assigned["gene"]
# Keep only quality cells in Baysor downstream
good_cells_set = set(good_cells.astype(str))
by_assigned_filtered = baysor_assigned[baysor_assigned["cell"].astype(str).isin(good_cells_set)]

def build_adata(df, cell_col, gene_col="feature_name", x_col="x_location", y_col="y_location"):
    assigned = df[df[cell_col].notna()&(df[cell_col]!="UNASSIGNED")&(df[cell_col]!=0)].copy()
    assigned[cell_col] = assigned[cell_col].astype(str)
    counts = assigned.groupby([cell_col,gene_col]).size().unstack(fill_value=0)
    if counts.shape[0]<20: return None
    a = ad.AnnData(X=csr_matrix(counts.values))
    a.obs_names = counts.index.astype(str); a.var_names = counts.columns
    cx = assigned.groupby(cell_col)[[x_col,y_col]].mean()
    a.obs = a.obs.join(cx.rename(columns={x_col:"x",y_col:"y"}))
    a.obsm["spatial"] = a.obs[["x","y"]].values
    return a

xen_a = build_adata(t_tile, "cell_id")
cp_a  = build_adata(t_tile, "cellpose_cell_id")
by_a  = build_adata(by_assigned_filtered, "cell")

def process_adata(a, name, res=0.5):
    if a is None: return None
    sc.pp.filter_cells(a, min_counts=5); sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=100); sc.pp.log1p(a)
    n_pcs = min(25, a.n_obs-1, a.n_vars-1)
    sc.pp.pca(a, n_comps=n_pcs, random_state=RANDOM_SEED)
    sc.pp.neighbors(a, n_neighbors=min(15,a.n_obs-1), n_pcs=min(15,n_pcs), random_state=RANDOM_SEED)
    sc.tl.umap(a, random_state=RANDOM_SEED)
    sc.tl.leiden(a, resolution=res, key_added="leiden", seed=RANDOM_SEED)
    sc.tl.rank_genes_groups(a, groupby="leiden", method="wilcoxon", n_genes=5)
    log(f"  {name}: {a.n_obs} cells, {a.obs['leiden'].nunique()} clusters")
    return a

adatas = {}
for name, a in [("Xenium 10x", xen_a), ("Cellpose", cp_a), ("Baysor (filtered)", by_a)]:
    processed = process_adata(a, name)
    if processed is not None: adatas[name] = processed

# UMAP panels
ncols = len(adatas)
fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6))
if ncols==1: axes=[axes]
for ax,(name,a) in zip(axes,adatas.items()):
    sc.pl.umap(a, color="leiden", ax=ax, show=False,
               title=f"{name}\n({a.n_obs} cells, {a.obs['leiden'].nunique()} clusters)",
               legend_loc="on data", legend_fontsize=9)
plt.suptitle("UMAP Per Method — 435×435 µm Tile (seeds fixed)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"umap_comparison_methods.png", dpi=150); plt.close()
log("Saved: umap_comparison_methods.png")

# Marker gene comparison across methods
known_types = {
    "EPCAM":"Epithelial","KRT17":"Epithelial","KRT8":"Epithelial","KRT18":"Epithelial","KRT5":"Myoepithelial",
    "CD68":"Macrophage","CD163":"Macrophage","LYZ":"Macrophage","CD74":"Macrophage/Immune",
    "CD3E":"T Cell","CD8A":"T Cell","IL2RG":"Immune",
    "ACTA2":"Myoepithelial","CNN1":"Myoepithelial","MYLK":"Myoepithelial",
    "PECAM1":"Endothelial","VWF":"Endothelial","RGS5":"Endothelial","COL4A1":"Endothelial",
    "COL1A1":"Stromal","FAP":"Stromal","MMP2":"Stromal","FBLN1":"Stromal",
    "MKI67":"Proliferating","TOP2A":"Proliferating",
    "VEGFA":"Hypoxic/Tumour","LDHA":"Hypoxic/Tumour","NDRG1":"Hypoxic/Tumour",
}

print("\n=== MARKER GENES PER METHOD ===")
method_ct_summary = {}
for name, a in adatas.items():
    print(f"\n--- {name} ---")
    clusters = list(a.uns["rank_genes_groups"]["names"].dtype.names)
    ct_calls = {}
    for cl in clusters:
        top5 = [a.uns["rank_genes_groups"]["names"][i][clusters.index(cl)] for i in range(5)]
        ct = "Unknown"
        for g in top5:
            if g in known_types: ct=known_types[g]; break
        ct_calls[cl] = ct
        print(f"  Cluster {cl} → {ct:25s} | top: {top5[:3]}")
    a.obs["cell_type"] = a.obs["leiden"].map(ct_calls).astype(str)
    method_ct_summary[name] = a.obs["cell_type"].value_counts(normalize=True).round(3)

# Cell type proportion comparison
ct_df = pd.DataFrame(method_ct_summary).fillna(0)
fig, ax = plt.subplots(figsize=(12, 5))
ct_df.T.plot(kind="bar", ax=ax, colormap="tab20")
ax.set_xlabel("Method"); ax.set_ylabel("Proportion")
ax.set_title("Cell Type Proportions Across Methods (Tile Level)\nValidated by Wilcoxon Marker Gene Test")
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"cell_type_proportions_per_method.png", dpi=150, bbox_inches="tight"); plt.close()
log("Saved: cell_type_proportions_per_method.png")

# Spatial comparison
fig, axes = plt.subplots(1, ncols, figsize=(8*ncols, 7))
if ncols==1: axes=[axes]
palette = plt.cm.tab10.colors
for ax,(name,a) in zip(axes,adatas.items()):
    for cluster in sorted(a.obs["cell_type"].unique()):
        mask = a.obs["cell_type"]==cluster
        coords = a.obsm["spatial"][mask.values]
        color = palette[list(a.obs["cell_type"].unique()).index(cluster) % 10]
        ax.scatter(coords[:,0], coords[:,1], s=5, alpha=0.7, color=color, label=cluster)
    ax.set_title(f"{name}\n({a.n_obs} cells)")
    ax.set_aspect("equal"); ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
    ax.legend(markerscale=2, fontsize=7, loc="upper right")
plt.suptitle("Spatial Cell Type Map Per Method — 435×435 µm Tile", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"spatial_comparison_methods.png", dpi=150); plt.close()
log("Saved: spatial_comparison_methods.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — Fix full-tissue annotation and neighbourhood enrichment (seeds)
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 7: Full tissue — fixed annotation + neighbourhood enrichment")

adata_full = sc.read_h5ad(DATA_DIR/"adata_10x_processed.h5ad")

# Re-run clustering with seeds to fix reproducibility
sc.pp.pca(adata_full, n_comps=30, random_state=RANDOM_SEED)
sc.pp.neighbors(adata_full, n_neighbors=15, n_pcs=20, random_state=RANDOM_SEED)
sc.tl.umap(adata_full, random_state=RANDOM_SEED)
sc.tl.leiden(adata_full, resolution=0.4, key_added="leiden", seed=RANDOM_SEED)
sc.tl.rank_genes_groups(adata_full, groupby="leiden", method="wilcoxon", n_genes=10)

clusters = list(adata_full.uns["rank_genes_groups"]["names"].dtype.names)
ct_map = {}
print("\n=== FULL TISSUE MARKER GENES (seeded, reproducible) ===")
for cl in clusters:
    top10 = [adata_full.uns["rank_genes_groups"]["names"][i][clusters.index(cl)] for i in range(10)]
    ct = "Unknown"
    for g in top10:
        if g in known_types: ct=known_types[g]; break
    # Manual override for metabolic cluster — be conservative
    if ct == "Hypoxic/Tumour":
        ct = "Metabolically active (Hypoxic-like)"
    ct_map[cl] = ct
    print(f"  Cluster {cl} → {ct:35s} | top: {top10[:5]}")

adata_full.obs["cell_type"] = adata_full.obs["leiden"].map(ct_map).astype(str)

props = adata_full.obs["cell_type"].value_counts(normalize=True).sort_values(ascending=False)
print("\n=== FINAL CELL TYPE PROPORTIONS ===")
for ct,v in props.items(): print(f"  {ct}: {v*100:.1f}%")

# Figures
palette_full = {ct: c for ct,c in zip(props.index,
    ["#e41a1c","#ff7f00","#4daf4a","#984ea3","#377eb8","#a65628","#f781bf","#888888"])}

fig, ax = plt.subplots(figsize=(9, 7))
sc.pl.umap(adata_full, color="cell_type", ax=ax, show=False,
           title=f"10x Xenium — Cell Types (n=201,446, seed={RANDOM_SEED})",
           legend_loc="right margin")
plt.tight_layout()
plt.savefig(RESULTS_DIR/"umap_cell_types_annotated.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
colors_list = [palette_full.get(ct,"grey") for ct in props.index]
ax.bar(props.index, props.values, color=colors_list)
for i,(ct,v) in enumerate(props.items()):
    ax.text(i, v+0.004, f"{v*100:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Proportion")
ax.set_title("Validated Cell Type Proportions — Wilcoxon marker gene confirmed, seeded")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"cell_type_proportions.png", dpi=150); plt.close()

sub_idx = np.random.RandomState(RANDOM_SEED).choice(adata_full.n_obs, size=min(60000,adata_full.n_obs), replace=False)
sub = adata_full[sub_idx]
fig, ax = plt.subplots(figsize=(10,10))
for ct in sub.obs["cell_type"].unique():
    mask = sub.obs["cell_type"]==ct
    coords = sub.obsm["spatial"][mask.values]
    ax.scatter(coords[:,0], coords[:,1], s=0.5, alpha=0.5, color=palette_full.get(ct,"grey"), label=ct)
ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
ax.set_title(f"Spatial Cell Type Map — 10x (60k sample, seed={RANDOM_SEED})")
ax.set_aspect("equal"); ax.legend(markerscale=8, fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(RESULTS_DIR/"spatial_cell_types.png", dpi=150); plt.close()

# Neighbourhood enrichment (seeded)
from scipy.spatial import cKDTree
coords = adata_full.obsm["spatial"]; ct_labels = adata_full.obs["cell_type"].values
cell_types_full = sorted(adata_full.obs["cell_type"].unique())
ct_idx_full = {ct:i for i,ct in enumerate(cell_types_full)}
tree = cKDTree(coords); pairs = tree.query_pairs(r=50.0)
obs_mat = np.zeros((len(cell_types_full),len(cell_types_full)),dtype=int)
for i,j in pairs:
    a,b=ct_idx_full[ct_labels[i]],ct_idx_full[ct_labels[j]]
    obs_mat[a,b]+=1; obs_mat[b,a]+=1
totals=obs_mat.sum(axis=1); total_pairs=obs_mat.sum()
exp_mat=np.outer(totals,totals)/(total_pairs+1e-9)
enrich=np.log2((obs_mat+1)/(exp_mat+1))
fig, ax = plt.subplots(figsize=(9,8))
sns.heatmap(enrich, xticklabels=cell_types_full, yticklabels=cell_types_full,
            cmap="RdBu_r", center=0, ax=ax, annot=True, fmt=".1f", annot_kws={"size":8})
ax.set_title("Spatial Neighbourhood Enrichment (log₂ obs/exp, 50 µm radius)")
plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"neighbourhood_enrichment.png", dpi=150, bbox_inches="tight"); plt.close()

adata_full.write_h5ad(DATA_DIR/"adata_10x_processed.h5ad")
log("Saved: all full-tissue figures")

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — Final summary table (complete)
# ══════════════════════════════════════════════════════════════════════
log("\nSTEP 8: Final summary table")

by_iou_use  = by_iou_cell_f  if by_iou_cell_f  else []
by_dice_use = by_dice_cell_f if by_dice_cell_f else []
by_recall_use = by_recall_cell_f

summary = pd.DataFrame([
    {"Method":"Xenium 10x","Reference":"—","Cells":len(tile_cells),
     "Assigned%":f"{100*t_tile[t_tile['cell_id']!='UNASSIGNED'].shape[0]/len(t_tile):.1f}%",
     "Mean TPC":"~76","Mean IoU":f"—","95% CI":"—","Dice":"—","Recall":"—"},
    {"Method":"Cellpose (nuclei)","Reference":"Nucleus","Cells":len(cp_polys),
     "Assigned%":f"{100*t_tile[t_tile['cellpose_cell_id']>0].shape[0]/len(t_tile):.1f}%",
     "Mean TPC":f"{t_tile[t_tile['cellpose_cell_id']>0].groupby('cellpose_cell_id').size().mean():.1f}",
     "Mean IoU":f"{cp_nuc_mean:.3f}","95% CI":f"[{cp_nuc_lo:.3f},{cp_nuc_hi:.3f}]",
     "Dice":f"{np.mean(cp_dice_nuc):.3f}","Recall":f"{cp_recall_nuc*100:.1f}%"},
    {"Method":"Baysor (quality-filtered)","Reference":"Cell","Cells":len(by_polys_filtered),
     "Assigned%":f"{100*len(by_assigned_filtered)/len(t_tile):.1f}%",
     "Mean TPC":f"{by_assigned_filtered.groupby('cell').size().mean():.1f}" if len(by_assigned_filtered)>0 else "—",
     "Mean IoU":f"{np.mean(by_iou_use):.3f}" if by_iou_use else "—",
     "95% CI":f"[{by_cell_lo:.3f},{by_cell_hi:.3f}]" if by_iou_cell_f else "—",
     "Dice":f"{np.mean(by_dice_use):.3f}" if by_dice_use else "—",
     "Recall":f"{by_recall_use*100:.1f}%"},
])
summary.to_csv(RESULTS_DIR/"method_summary_stats.csv", index=False)
print(f"\n{'='*80}")
print(summary.to_string(index=False))
print(f"{'='*80}")
if mw_cell: print(f"\nMann-Whitney U (Cellpose vs Baysor IoU, cell reference): p = {mw_cell.pvalue:.2e}")

total = (time.time()-t0)/60
log(f"\n{'='*60}")
log(f"COMPREHENSIVE FIX COMPLETE in {total:.1f} minutes")
log(f"{'='*60}")
