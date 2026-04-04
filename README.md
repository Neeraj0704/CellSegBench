# CellSegBench

A benchmarking framework for evaluating cell segmentation methods on subcellular spatial transcriptomics data from cancer tissue.

## Overview

Cell segmentation is a foundational step in spatial transcriptomics analysis — it determines which transcripts belong to which cell. Different segmentation methods can produce substantially different cell boundaries on the same tissue section, which propagates into every downstream analysis: cell typing, spatial neighborhood analysis, and biological interpretation.

This project systematically compares three segmentation approaches on a public human breast cancer Xenium dataset, quantifying both their technical differences and their downstream biological consequences. A key focus is evaluating segmentation quality using the **ground truth nuclear boundaries** provided by the Xenium platform's DAPI nuclear staining, enabling accuracy assessment beyond summary statistics alone.

## Data

**Dataset:** [Xenium In Situ Gene Expression — Human Breast Cancer (FFPE)](https://www.10xgenomics.com/datasets/xenium-ffpe-human-breast-biomarkers), 10x Genomics.

The Xenium platform outputs:
- Sub-micron resolution transcript coordinates (x, y, z)
- DAPI nuclear staining images
- 10x-provided nucleus segmentation masks (used as **ground truth reference boundaries**)
- Cell-by-gene count matrices from the platform's own segmentation pipeline

The 10x-provided nucleus segmentation (derived from DAPI) serves as the geometric ground truth for evaluating how well each method recovers true nuclear boundaries. Since nuclei are unambiguously stained, this gives a principled reference for spatial accuracy without requiring manual annotation.

## Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| **Xenium default** | Image-based | 10x's built-in nucleus segmentation from DAPI; used as the ground truth reference |
| **Cellpose** | Deep learning | Generalist neural network for cell/nucleus segmentation (Stringer et al., 2021) |
| **Baysor** | Transcript-based | Bayesian assignment of transcripts to cells using spatial and expression context (Petukhov et al., 2022) |

## Evaluation Strategy

### Ground Truth for Accuracy

The Xenium platform provides DAPI-stained nuclear masks that serve as a geometry reference. These are used to compute:

- **IoU (Intersection over Union)** between predicted cell boundaries and DAPI nuclei
- **Boundary F1 score** — precision/recall of boundary pixels
- **Matched cell rate** — fraction of DAPI nuclei that have a corresponding segmented cell (detection recall)

This allows direct spatial accuracy measurement rather than relying only on downstream proxies.

### Transcript-level Metrics

- Transcripts per cell (mean, median, distribution)
- Fraction of transcripts left unassigned
- Cell count and cell size distribution

### Robustness

- Sensitivity to key parameters (Cellpose diameter, Baysor prior cell size) across a grid of values
- Coefficient of variation in cell count and transcript assignment across parameter settings

### Biological Downstream Impact

- UMAP and Leiden clustering on each segmentation's count matrix
- Marker gene expression per cluster (e.g., EPCAM for epithelial, CD68 for macrophages)
- Spatial visualization of cell type assignments per method using Squidpy

## Repository Structure

```
CellSegBench/
├── data/                  # Raw and processed Xenium data (not tracked in git)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_cellpose_segmentation.ipynb
│   ├── 03_baysor_segmentation.ipynb
│   ├── 04_evaluation_metrics.ipynb
│   └── 05_downstream_analysis.ipynb
├── src/
│   ├── segmentation/      # Wrappers for each segmentation method
│   ├── evaluation/        # IoU, boundary F1, transcript assignment metrics
│   └── visualization/     # Spatial plotting utilities
├── results/               # Output figures and summary tables
├── environment.yml
└── README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate cellsegbench
```

Key dependencies: `squidpy`, `cellpose`, `anndata`, `scanpy`, `shapely`, `scikit-image`, `matplotlib`.

Baysor is run as a standalone Julia binary — see the [Baysor installation guide](https://github.com/kharchenkolab/Baysor).

## References

1. Stringer, C. et al. *Cellpose: a generalist algorithm for cellular segmentation.* Nature Methods, 2021.
2. Petukhov, V. et al. *Baysor: Bayesian segmentation of spatial transcriptomics data.* Nature Biotechnology, 2022.
3. Palla, G. et al. *Squidpy: a scalable framework for spatial omics analysis.* Nature Methods, 2022.
4. Janesick, A. et al. *High resolution mapping of the breast cancer tumor microenvironment using integrated single cell, spatial and in situ analysis.* Nature Communications, 2023.
