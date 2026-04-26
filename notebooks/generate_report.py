"""
Generate full project report as PDF.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path
import os

RESULTS_DIR = Path("../results")
OUT_PDF     = RESULTS_DIR / "CellSegBench_Report.pdf"

# ── Styles ────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle("Title", parent=styles["Title"],
    fontSize=20, textColor=colors.HexColor("#1a1a2e"),
    spaceAfter=8, alignment=TA_CENTER, fontName="Helvetica-Bold")

subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
    fontSize=11, textColor=colors.HexColor("#444444"),
    spaceAfter=4, alignment=TA_CENTER)

h1_style = ParagraphStyle("H1", parent=styles["Heading1"],
    fontSize=14, textColor=colors.HexColor("#1a1a2e"),
    spaceBefore=16, spaceAfter=6, fontName="Helvetica-Bold",
    borderPad=4)

h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
    fontSize=12, textColor=colors.HexColor("#16213e"),
    spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")

body_style = ParagraphStyle("Body", parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY,
    textColor=colors.HexColor("#2d2d2d"))

bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"],
    fontSize=10, leading=14, spaceAfter=3, leftIndent=16,
    textColor=colors.HexColor("#2d2d2d"))

caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
    fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER,
    spaceAfter=8, spaceBefore=2)

def h1(text): return Paragraph(text, h1_style)
def h2(text): return Paragraph(text, h2_style)
def body(text): return Paragraph(text, body_style)
def bullet(text): return Paragraph(f"• {text}", bullet_style)
def caption(text): return Paragraph(f"<i>{text}</i>", caption_style)
def sp(n=6): return Spacer(1, n)
def hr(): return HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"), spaceAfter=6)

def fig(name, width=5.5*inch, caption_text=None):
    path = RESULTS_DIR / name
    if not path.exists():
        return [body(f"[Figure not found: {name}]")]
    items = [Image(str(path), width=width,
                   height=width * 0.72, kind="proportional")]
    if caption_text:
        items.append(caption(caption_text))
    return items

def table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t

# ══════════════════════════════════════════════════════════════════════════
# Build document
# ══════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter,
    leftMargin=0.9*inch, rightMargin=0.9*inch,
    topMargin=0.9*inch, bottomMargin=0.9*inch)

story = []

# ── Title page ─────────────────────────────────────────────────────────────
story += [
    sp(40),
    Paragraph("Benchmarking Cell Segmentation Methods for", title_style),
    Paragraph("Subcellular Spatial Transcriptomics Data in Cancer Tissue", title_style),
    sp(12),
    Paragraph("Neeraj Vijayakumar Pattanashetti", subtitle_style),
    Paragraph("University of Illinois Chicago  |  npatt@uic.edu", subtitle_style),
    Paragraph("CellSegBench Project Report  |  April 2026", subtitle_style),
    sp(20),
    hr(),
    sp(10),
    body(
        "This report documents the complete methodology, computational workflow, "
        "dataset characteristics, segmentation results, evaluation metrics, and "
        "biological downstream analysis for the CellSegBench project. Three "
        "cell segmentation approaches — the 10x Genomics Xenium platform baseline, "
        "Cellpose (deep learning image-based), and Baysor (transcript-based Bayesian) "
        "— are compared on a public human breast cancer spatial transcriptomics dataset."
    ),
    PageBreak(),
]

# ── 1. Introduction ────────────────────────────────────────────────────────
story += [
    h1("1. Introduction and Motivation"),
    body(
        "Cell segmentation is the process of delineating individual cell boundaries "
        "in tissue data — it is the foundational step that determines which transcripts "
        "belong to which cell in spatial transcriptomics. Because all downstream "
        "analysis (cell typing, marker gene identification, spatial organization) "
        "operates on cell-level summaries, the choice of segmentation method "
        "directly affects biological conclusions."
    ),
    sp(),
    body(
        "Different segmentation methods approach this problem very differently: "
        "image-based methods like Cellpose detect nuclei from staining images using "
        "deep learning, while transcript-based methods use the spatial coordinates "
        "and gene identities of RNA molecules themselves. There is no universal "
        "consensus on which approach is more accurate or consistent for cancer tissue "
        "data, motivating this benchmarking study."
    ),
    sp(),
    body("The specific research questions addressed are:"),
    bullet("How accurately does Cellpose recover true nucleus boundaries compared to the 10x platform reference?"),
    bullet("How does Baysor, a transcript-based Bayesian segmentation method, compare to image-based segmentation?"),
    bullet("How stable is Cellpose across different diameter parameter settings?"),
    bullet("Do different segmentation methods produce different downstream biological signals (clusters, marker genes)?"),
    bullet("Which cell types co-localise spatially in the breast cancer tumour microenvironment?"),
    sp(12),
]

# ── 2. Dataset ─────────────────────────────────────────────────────────────
story += [
    h1("2. Dataset"),
    h2("2.1 Source"),
    body(
        "The dataset used is the <b>10x Genomics Xenium In Situ Gene Expression — "
        "Human Breast Cancer (FFPE), Section 1 Top</b>, publicly available from "
        "10x Genomics. It accompanies the preprint <i>\"Biomarker Quantification in "
        "Breast Cancer using Xenium In Situ\"</i> (Janesick et al., 2023). "
        "The full output bundle (16.7 GB) was downloaded, containing all raw "
        "transcript coordinates, nucleus and cell boundary polygons, morphology "
        "images, and the platform's own cell-by-gene count matrix."
    ),
    sp(),
    h2("2.2 Technology"),
    body(
        "Xenium In Situ is a subcellular spatial transcriptomics platform by 10x Genomics. "
        "It decodes gene expression at single-molecule resolution by imaging "
        "fluorescently labeled RNA molecules directly in tissue sections. "
        "Each detected transcript is assigned (x, y, z) coordinates in microns, "
        "a gene identity, and a quality score (Q-score). The platform uses a custom "
        "280-gene panel targeting breast cancer biomarkers and housekeeping genes."
    ),
    sp(),
    h2("2.3 Dataset Characteristics"),
    sp(4),
    table([
        ["Parameter", "Value"],
        ["Total transcripts (raw)", "64,581,006"],
        ["Gene transcripts after Q≥20 filter", "55,283,583"],
        ["Gene panel size", "280 genes"],
        ["Cells (10x platform segmentation)", "209,467"],
        ["Tissue area", "60,267,079 µm² (~60 mm²)"],
        ["DAPI image dimensions", "27,420 × 53,994 pixels"],
        ["Pixel size", "0.2125 µm/pixel"],
        ["Transcripts assigned by 10x", "43,538,027 (78.8%)"],
        ["Mean transcripts per cell (10x)", "209.1"],
        ["Median transcripts per cell (10x)", "137.0"],
        ["Mean cell area (10x)", "71.0 µm²"],
        ["Median cell area (10x)", "56.7 µm²"],
        ["10x segmentation method", "Boundary stain: ATP1A1 + CD45 + E-Cadherin"],
    ], col_widths=[3.2*inch, 3.2*inch]),
    sp(10),
    h2("2.4 Top Expressed Genes"),
    body("The 15 most detected genes across the tissue after quality filtering:"),
    sp(4),
    table([
        ["Rank", "Gene", "Transcripts", "Biological Role"],
        ["1",  "MALAT1",   "3,536,599", "Nuclear long non-coding RNA, ubiquitous"],
        ["2",  "EEF1G",    "2,190,949", "Translation elongation factor"],
        ["3",  "MTCO1P40", "1,576,027", "Mitochondrial pseudogene"],
        ["4",  "EEF2",     "1,551,420", "Translation elongation factor"],
        ["5",  "H3F3B",    "1,443,450", "Histone H3.3"],
        ["6",  "SFRP1",    "1,420,749", "Wnt signaling antagonist, cancer suppressor"],
        ["7",  "VEGFA",    "1,416,894", "Vascular endothelial growth factor, angiogenesis"],
        ["8",  "LDHA",     "1,087,032", "Lactate dehydrogenase, Warburg effect"],
        ["9",  "RPLP0",      "978,586", "Ribosomal protein"],
        ["10", "PPIB",       "928,566", "Cyclophilin B, protein folding"],
    ], col_widths=[0.5*inch, 1.1*inch, 1.3*inch, 3.5*inch]),
    sp(10),
]

story += [f for f in fig("top_genes.png", width=5.5*inch,
    caption_text="Figure 1. Top 30 most detected genes after Q≥20 quality filtering across the full tissue section.")]
story.append(sp(10))

story += [f for f in fig("transcript_spatial_distribution.png", width=4.5*inch,
    caption_text="Figure 2. Spatial distribution of 150,000 sampled transcripts across the tissue section (Q≥20). Color intensity reflects local transcript density.")]
story.append(sp(10))

story += [f for f in fig("marker_genes_spatial.png", width=5.5*inch,
    caption_text="Figure 3. Spatial expression patterns of key cell-type marker genes: EPCAM (epithelial), CD68 (macrophage), CD3E (T cell), KRT17 (basal epithelial).")]

story.append(PageBreak())

# ── 3. Methods ─────────────────────────────────────────────────────────────
story += [
    h1("3. Methods"),
    h2("3.1 Data Preprocessing"),
    body(
        "Raw transcript data was loaded from the Xenium parquet output using PyArrow "
        "with predicate pushdown filtering to avoid loading all 64.5M rows into RAM. "
        "Two filters were applied sequentially: (1) retain only real gene transcripts "
        "(excluding control probes and blank codewords, is_gene=True), and (2) retain "
        "transcripts with Q-score ≥ 20, the standard Xenium quality threshold. "
        "This reduced the dataset from 64,581,006 to 55,283,583 transcripts — "
        "retaining 85.6% of gene-level transcripts."
    ),
    sp(),
    body(
        "Given the large dataset size (800 MB parquet file), all processing was "
        "performed in a memory-efficient manner: floats were downcast to float32, "
        "string columns were stored as categoricals, and segmentation experiments "
        "were restricted to a representative 512×512 pixel (109×109 µm) tile from "
        "the centre of the tissue section. This tile contained 12,848 transcripts "
        "and 97 reference cells — sufficient for rigorous method comparison while "
        "remaining computationally tractable on a CPU-only machine."
    ),
    sp(),
    h2("3.2 Segmentation Methods"),
    sp(4),
    body("<b>Method 1 — Xenium 10x Platform Baseline</b>"),
    body(
        "The 10x Genomics Xenium platform performs its own cell segmentation using "
        "a multi-stain boundary approach (ATP1A1 + CD45 + E-Cadherin membrane markers) "
        "combined with DAPI nuclear staining. This produces cell boundary polygons and "
        "nucleus boundary polygons stored as vertex coordinate files. The nucleus "
        "boundary polygons serve as the <b>geometric ground truth reference</b> for "
        "evaluating other methods, since they are derived from validated laboratory "
        "staining and represent the best available approximation of true cell boundaries. "
        "Across the full tissue, 10x detected 209,467 cells with a mean area of 71.0 µm²."
    ),
    sp(6),
    body("<b>Method 2 — Cellpose (Deep Learning, Image-Based)</b>"),
    body(
        "Cellpose (v4.1.1) is a generalist deep learning model for cell and nucleus "
        "segmentation trained on a large and diverse dataset of biological microscopy "
        "images (Stringer et al., 2021). It operates on the DAPI nuclear staining "
        "image and detects nucleus boundaries by predicting spatial flow fields and "
        "cell probability maps. The pretrained 'nuclei' model (1.15 GB) was used "
        "without fine-tuning. The DAPI image (morphology_focus/ch0000_dapi.ome.tif, "
        "399 MB, 27,420×53,994 px) was cropped to the 512×512 tile region. "
        "A default diameter of 30 pixels (6.4 µm) was used, corresponding to the "
        "expected nucleus diameter in breast tissue. Masks were converted to Shapely "
        "polygon boundaries in micron coordinates for comparison against reference polygons."
    ),
    sp(6),
    body("<b>Method 3 — Baysor (Transcript-Based Bayesian Segmentation)</b>"),
    body(
        "Baysor (v0.7.1) is a segmentation algorithm specifically designed for "
        "imaging-based spatial transcriptomics (Petukhov et al., 2022). It uses "
        "a Bayesian framework to assign transcripts to cells based on both spatial "
        "proximity and gene expression similarity, modelling each cell as a Gaussian "
        "mixture component in transcript space. Unlike Cellpose, Baysor operates "
        "purely on transcript coordinates without requiring a staining image. "
        "It was run via Julia v1.11 with parameters: scale=15 µm (expected cell radius), "
        "min-molecules-per-cell=10, n-clusters=4 (major cell types). "
        "Baysor converged after 500 iterations with a noise level of 1.81%, "
        "detecting 38 cells and assigning 99.8% of transcripts. Cell boundaries "
        "were exported as GeoJSON polygons for evaluation."
    ),
    sp(),
    h2("3.3 Evaluation Metrics"),
    body("Three types of metrics were computed:"),
    sp(4),
    body("<b>Geometric accuracy (vs. 10x nucleus reference):</b>"),
    bullet("IoU (Intersection over Union): area of overlap / area of union between predicted and reference polygon"),
    bullet("Dice coefficient: 2 × intersection / (area_pred + area_ref)"),
    bullet("Matched cell recall: fraction of reference nuclei that have a matching predicted polygon (IoU ≥ 0.3 threshold)"),
    bullet("Matching was done greedily using a Shapely STRtree spatial index for efficiency"),
    sp(4),
    body("<b>Transcript-level metrics:</b>"),
    bullet("Transcripts per cell (TPC): mean and median across all assigned cells"),
    bullet("Unassigned transcript fraction: transcripts not captured by any predicted cell"),
    sp(4),
    body("<b>Parameter robustness:</b>"),
    bullet("Cellpose diameter sweep: 20, 30, 40 pixels (4.25, 6.38, 8.5 µm)"),
    bullet("Coefficient of Variation (CV) of cell count across the sweep"),
    sp(),
    h2("3.4 Downstream Biological Analysis"),
    body(
        "Cell-by-gene count matrices were constructed from each method's transcript "
        "assignments by pivoting transcript-level data into a cells × genes matrix. "
        "For the 10x baseline, the full tissue count matrix (209,467 cells × 280 genes) "
        "was used from the platform's cell_feature_matrix.h5 file. "
        "Standard Scanpy preprocessing was applied identically to all: "
        "total-count normalisation (target sum 100), log1p transformation, "
        "PCA (30 components), k-nearest neighbour graph (k=15), UMAP embedding, "
        "and Leiden clustering (resolution=0.4 for full tissue, 0.5 for tile). "
        "Marker gene expression was visualised using dot plots."
    ),
    PageBreak(),
]

# ── 4. Results ─────────────────────────────────────────────────────────────
story += [
    h1("4. Results"),
    h2("4.1 Cellpose Segmentation"),
    body(
        "Cellpose was applied to a 512×512 pixel (109×109 µm) DAPI tile centred on "
        "the tissue. At the default diameter of 30 pixels (6.4 µm), Cellpose detected "
        "<b>91 nuclei</b> in this region. The 10x reference contains 97 nuclei in "
        "the same area, meaning Cellpose detected slightly fewer cells (94% of reference count). "
        "Detected nuclei had a mean area of <b>21.7 µm²</b> and median of <b>20.6 µm²</b>."
    ),
    sp(),
    body(
        "Transcript assignment via pixel-coordinate lookup (mapping transcript µm "
        "coordinates to the Cellpose mask array) assigned <b>4,138 of 12,848 transcripts "
        "(32.2%)</b> to cells. The remaining 67.8% fell outside predicted nucleus "
        "boundaries. This lower assignment rate compared to 10x (73.9%) reflects that "
        "Cellpose segments only nuclei, while 10x uses expanded cell boundaries that "
        "capture cytoplasmic transcripts. Mean transcripts per cell was <b>45.5</b> "
        "(median 35.0)."
    ),
    sp(8),
]

story += [f for f in fig("cellpose_segmentation_map.png", width=5.5*inch,
    caption_text="Figure 4. Cellpose nuclei segmentation (red overlay) on the DAPI 512×512 tile. Left: raw DAPI input. Right: detected nucleus masks.")]
story.append(sp(8))

story += [f for f in fig("cellpose_area_distribution.png", width=4.5*inch,
    caption_text="Figure 5. Distribution of detected nucleus areas for Cellpose (µm²). Mean=21.7, Median=20.6 µm².")]
story.append(sp(12))

story += [
    h2("4.2 Evaluation Against 10x Reference"),
    body(
        "Cellpose nucleus boundaries were matched to 10x reference nucleus polygons "
        "using greedy IoU-based matching (threshold ≥ 0.3). Of 97 reference nuclei, "
        "<b>77 were successfully matched (79.4% recall)</b>. Among matched pairs, "
        "the mean IoU was <b>0.780</b> and mean Dice coefficient was <b>0.872</b> — "
        "indicating very strong geometric alignment between Cellpose predictions and "
        "the 10x reference boundaries."
    ),
    sp(8),
]

story += [f for f in fig("side_by_side_3way.png", width=6.0*inch,
    caption_text="Figure 6. Side-by-side segmentation comparison on the 109×109 µm tile. Left: 10x reference (yellow). Centre: Cellpose (cyan). Right: Baysor transcript-based (lime).")]
story.append(sp(8))

story += [f for f in fig("three_way_iou_comparison.png", width=5.5*inch,
    caption_text="Figure 7. Quantitative comparison — IoU, Dice, and recall vs. 10x nucleus reference for Cellpose and Baysor.")]
story.append(sp(8))

story += [f for f in fig("cellpose_iou_distribution.png", width=4.5*inch,
    caption_text="Figure 8. Distribution of per-cell IoU scores for Cellpose vs. 10x reference. Mean IoU = 0.780 (red dashed line).")]
story.append(sp(12))

story += [
    h2("4.3 Baysor Transcript-Based Segmentation"),
    body(
        "Baysor was run on 12,848 transcripts in the tile using a prior cell radius "
        "of 15 µm and minimum 10 transcripts per cell. The algorithm converged after "
        "500 iterations with a noise fraction of only <b>1.81%</b>, detecting "
        "<b>38 cells</b> and assigning <b>99.8% of transcripts</b>. "
        "Mean transcripts per cell was <b>337.4</b> (median 248.5) — substantially "
        "higher than Cellpose (45.5) because Baysor segments full cells including "
        "cytoplasm, while Cellpose segments only nuclei."
    ),
    sp(),
    body(
        "Against the 10x nucleus reference, Baysor achieved a mean IoU of <b>0.392</b> "
        "and Dice of <b>0.563</b>, with a recall of only <b>2.1%</b> at IoU ≥ 0.3. "
        "The low recall is expected: Baysor's cell boundaries are much larger than "
        "nucleus boundaries (it segments the full cell body), so they do not overlap "
        "well with compact nucleus polygons at the IoU=0.3 threshold. This is an "
        "evaluation limitation rather than a Baysor failure — comparing full-cell "
        "boundaries against nucleus-only references inherently penalises methods that "
        "correctly capture cytoplasmic transcripts."
    ),
    sp(8),
]

story += [f for f in fig("transcripts_per_cell_comparison.png", width=5.0*inch,
    caption_text="Figure 9. Transcripts-per-cell distribution for all three methods. Baysor assigns nearly all transcripts; Cellpose captures nucleus-only subsets.")]
story.append(sp(12))

story += [
    h2("4.4 Summary Comparison Table"),
    sp(6),
    table([
        ["Method", "Cells", "Assigned %", "Mean TPC", "Median TPC", "Mean IoU", "Mean Dice", "Recall*"],
        ["Xenium 10x", "97",  "73.9%", "209.1†", "137.0†", "—",     "—",     "—"],
        ["Cellpose",   "91",  "32.2%",  "45.5",   "35.0",  "0.780", "0.872", "79.4%"],
        ["Baysor",     "38",  "99.8%", "337.4",  "248.5",  "0.392", "0.563",  "2.1%"],
    ], col_widths=[1.3*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.7*inch]),
    sp(4),
    body("* Recall computed vs. 10x nucleus boundaries at IoU ≥ 0.3. Baysor's low recall reflects boundary type mismatch (full cell vs. nucleus), not poor performance."),
    body("† 10x TPC from full tissue; tile-level values differ."),
    sp(12),
]

story += [
    h2("4.5 Cellpose Parameter Robustness"),
    body(
        "The Cellpose diameter parameter controls the expected nucleus size and "
        "strongly influences cell detection. A sweep over three diameter values "
        "was performed on the same tile:"
    ),
    sp(6),
    table([
        ["Diameter (px)", "Diameter (µm)", "Cells Detected"],
        ["20",  "4.25", "87"],
        ["30",  "6.38", "91"],
        ["40",  "8.50", "109"],
    ], col_widths=[1.8*inch, 1.8*inch, 1.8*inch]),
    sp(6),
    body(
        "The coefficient of variation (CV) across these three settings was <b>0.122</b> "
        "(12.2%), indicating moderate sensitivity to the diameter parameter. Cell "
        "counts ranged from 87 to 109 — a 25% variation — suggesting that parameter "
        "choice meaningfully affects results. The default diameter (30px) produced "
        "an intermediate count consistent with the 10x reference (97 cells)."
    ),
    sp(8),
]

story += [f for f in fig("cellpose_diameter_sweep.png", width=5.0*inch,
    caption_text="Figure 11. Cellpose cell count as a function of diameter parameter. Cell count increases with diameter as larger expected sizes merge adjacent nuclei.")]
story.append(PageBreak())

# ── 5. Downstream ──────────────────────────────────────────────────────────
story += [
    h1("5. Downstream Biological Analysis"),
    h2("5.1 Full Tissue Clustering (10x Xenium Baseline)"),
    body(
        "The 10x platform's cell-by-gene count matrix (209,467 cells × 280 genes) "
        "was used for full-tissue downstream analysis. After quality filtering "
        "(minimum 10 counts per cell, minimum 10 cells per gene), <b>201,446 cells</b> "
        "remained. Standard Scanpy preprocessing (normalisation, log1p, PCA 30 components, "
        "k-NN graph k=15, UMAP, Leiden resolution=0.4) identified <b>7 distinct clusters</b>."
    ),
    sp(),
    body(
        "Marker genes available in the panel include EPCAM (epithelial), KRT17 (basal "
        "epithelial), CD68 (macrophage), CD3E (T cell), ACTA2 (smooth muscle/myoepithelial), "
        "PECAM1 (endothelial), and MKI67 (proliferating cells). The dot plot shows "
        "differential expression across clusters, consistent with the expected cellular "
        "heterogeneity of breast cancer tissue."
    ),
    sp(8),
]

story += [f for f in fig("umap_leiden_10x.png", width=5.5*inch,
    caption_text="Figure 12. UMAP of 201,446 cells from the 10x Xenium full tissue segmentation, coloured by Leiden cluster (resolution=0.4, 7 clusters).")]
story.append(sp(8))

story += [f for f in fig("marker_dotplot_10x.png", width=5.5*inch,
    caption_text="Figure 13. Dot plot of marker gene expression across 7 Leiden clusters. Dot size = fraction of cells expressing the gene; colour = normalised expression level.")]
story.append(sp(8))

story += [f for f in fig("umap_cell_types_annotated.png", width=5.5*inch,
    caption_text="Figure 14. UMAP coloured by annotated cell type. Seven biologically interpretable populations identified using marker gene expression.")]
story.append(sp(8))

story += [f for f in fig("cell_type_proportions.png", width=5.0*inch,
    caption_text="Figure 15. Cell type proportions across 201,446 cells. Epithelial cells are the dominant population (27.5%), consistent with breast cancer tissue.")]
story.append(sp(8))

story += [f for f in fig("spatial_cell_types.png", width=5.0*inch,
    caption_text="Figure 16. Spatial distribution of annotated cell types across the tissue (60,000 cell sample). Epithelial cells cluster in tumour nests; immune cells are dispersed.")]
story.append(sp(8))

story += [f for f in fig("neighbourhood_enrichment.png", width=5.5*inch,
    caption_text="Figure 17. Spatial neighbourhood enrichment heatmap (log₂ observed/expected co-occurrence within 50 µm). Warm colours indicate enriched co-localisation; cool colours indicate avoidance.")]
story.append(sp(12))

story += [
    h2("5.2 Per-Method Clustering Comparison (Tile Level)"),
    body(
        "On the 109×109 µm tile, count matrices were built from each method's "
        "transcript assignments and the same Scanpy pipeline was applied. The tile "
        "contains few cells per method, so clustering is limited but allows direct "
        "comparison of what biological signal each method captures."
    ),
    sp(6),
    table([
        ["Method", "Cells (tile)", "Leiden Clusters"],
        ["Xenium 10x", "115", "3"],
        ["Cellpose",    "90", "3"],
        ["Baysor",      "36", "2"],
    ], col_widths=[2.2*inch, 2.0*inch, 2.0*inch]),
    sp(8),
]

story += [f for f in fig("umap_comparison_methods.png", width=6.0*inch,
    caption_text="Figure 18. UMAP comparison across three segmentation methods on the tile. All three produce 2–3 clusters with broadly consistent structure.")]
story.append(sp(8))

story += [f for f in fig("spatial_comparison_methods.png", width=6.0*inch,
    caption_text="Figure 19. Spatial cluster maps per method on the tile. Cluster assignments differ in granularity but the spatial organisation is broadly preserved.")]
story.append(PageBreak())

# ── 6. Discussion ──────────────────────────────────────────────────────────
story += [
    h1("6. Discussion"),
    h2("6.1 Cellpose Accuracy"),
    body(
        "Cellpose achieved strong geometric accuracy against the 10x nucleus reference, "
        "with a mean IoU of 0.780 and Dice of 0.872. These values indicate substantial "
        "overlap between Cellpose predictions and reference boundaries. A recall of 79.4% "
        "means that approximately 1 in 5 reference nuclei was missed — likely small or "
        "closely packed nuclei where Cellpose merges adjacent cells or misses low-contrast "
        "nuclei. The 20 unmatched nuclei (out of 97) represent cases where Cellpose "
        "either missed the cell or produced a significantly different boundary shape."
    ),
    sp(),
    h2("6.2 Baysor and the Nucleus Reference Problem"),
    body(
        "Baysor's low recall (2.1%) against the 10x nucleus reference reflects a fundamental "
        "evaluation mismatch: Baysor segments full cells (nucleus + cytoplasm) while the "
        "reference polygons represent nucleus boundaries only. A Baysor cell boundary is "
        "typically 3–5× larger in area than the corresponding nucleus, so IoU ≥ 0.3 is "
        "almost never achievable against a compact nucleus polygon even when Baysor correctly "
        "identifies the cell. Baysor's 99.8% transcript assignment rate and biologically "
        "plausible transcript-per-cell counts (mean 337, consistent with full-cell capture) "
        "suggest it is functioning correctly. Future evaluation should use full cell boundary "
        "polygons (available in the Xenium output) as the reference rather than nucleus-only polygons."
    ),
    sp(),
    h2("6.3 Parameter Sensitivity"),
    body(
        "The Cellpose diameter sweep (CV=0.122) reveals moderate parameter sensitivity. "
        "Users who do not carefully tune the diameter may obtain cell counts varying by "
        "up to 25%. For breast tissue with heterogeneous cell sizes (epithelial cells "
        "~15–50 µm², stromal cells larger), a single fixed diameter is a known limitation. "
        "Future work could explore adaptive diameter estimation or per-region parameter tuning."
    ),
    sp(),
    h2("6.4 Downstream Impact"),
    body(
        "Despite differences in geometric accuracy and transcript assignment rates, "
        "all three methods produced 2–3 Leiden clusters on the tile, "
        "suggesting that the coarse-level biological signal is robust to segmentation choice. "
        "On the full tissue (10x baseline), 7 biologically interpretable clusters were "
        "identified: Epithelial (27.5%), Stromal/Fibroblast (20.6%), Macrophage (16.1%), "
        "Epithelial basal (10.8%), T Cell (9.3%), Endothelial (9.0%), and Myoepithelial (6.7%). "
        "Neighbourhood enrichment analysis revealed expected spatial patterns: Epithelial and "
        "Myoepithelial cells are strongly co-localised, Macrophages neighbour T cells "
        "(consistent with immune infiltration), and Endothelial cells cluster together "
        "in vessel structures."
    ),
    sp(12),
]

# ── 7. Challenges ──────────────────────────────────────────────────────────
story += [
    h1("7. Challenges Encountered"),
    bullet("<b>Memory constraints:</b> Loading 64.5M transcripts (~1.7 GB after filtering) on a 16 GB machine with minimal free RAM required PyArrow predicate pushdown to filter before loading, float32 downcasting, and categorical encoding."),
    bullet("<b>DAPI image size:</b> The full DAPI image (27,420×53,994 px, 399 MB) could not be loaded fully; analysis was restricted to a representative 109×109 µm tile centred on the tissue."),
    bullet("<b>Cellpose API changes:</b> Cellpose v4.x renamed the main class from Cellpose to CellposeModel and changed return values from a 4-tuple to a 3-tuple, requiring code debugging."),
    bullet("<b>CPU-only inference:</b> Without a GPU, a single Cellpose run on the 512×512 tile took ~2 minutes. The 3-value diameter sweep added ~7 minutes."),
    bullet("<b>Baysor installation complexity:</b> Baysor requires Julia. The pre-built Docker image (vpetukhov/baysor) was compiled for Intel x86 and failed on Apple Silicon under Rosetta. Julia 1.12 (brew) was also incompatible with Baysor's plotting dependencies. Resolution: installed Julia 1.11 via juliaup, which successfully installed and ran Baysor."),
    bullet("<b>Evaluation reference mismatch:</b> Baysor segments full cells while the 10x reference provides nucleus boundaries only, making direct IoU comparison misleading. This required careful interpretation of Baysor's low recall metric."),
    sp(12),
]

# ── 8. Remaining Work ──────────────────────────────────────────────────────
story += [
    h1("8. Future Work"),
    bullet("<b>Full cell boundary evaluation:</b> Re-evaluate Baysor using the 10x cell boundary polygons (not nucleus boundaries) as reference, which would give a fair geometric comparison."),
    bullet("<b>Larger tile analysis:</b> Extend Cellpose and Baysor to a larger region (2048×2048 px, ~870×870 µm) for statistically robust IoU/recall estimates across hundreds of cells."),
    bullet("<b>Full-tissue Baysor:</b> Run Baysor on all 55M transcripts using HTCondor or a cloud GPU instance to enable tissue-wide cell type mapping and comparison against 10x."),
    bullet("<b>GPU acceleration:</b> Running Cellpose on a GPU would reduce inference from ~2 min to ~5 sec per tile, enabling full-tissue segmentation in under an hour."),
    bullet("<b>Baysor parameter sweep:</b> Evaluate sensitivity of Baysor to the scale parameter (expected cell radius) analogous to the Cellpose diameter sweep."),
    sp(12),
]

# ── 9. References ──────────────────────────────────────────────────────────
story += [
    h1("9. References"),
    body("[1] Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. <i>Nature Methods</i>, 18(1), 100–106."),
    sp(4),
    body("[2] Petukhov, V., et al. (2022). Cell segmentation in imaging-based spatial transcriptomics. <i>Nature Biotechnology</i>, 40, 345–354."),
    sp(4),
    body("[3] Palla, G., et al. (2022). Squidpy: a scalable framework for spatial omics analysis. <i>Nature Methods</i>, 19(2), 171–178."),
    sp(4),
    body("[4] Janesick, A., et al. (2023). High resolution mapping of the breast cancer tumor microenvironment using integrated single cell, spatial and in situ analysis. <i>Nature Communications</i>, 14, 8353."),
    sp(4),
    body("[5] Wolf, F.A., Angerer, P., & Theis, F.J. (2018). SCANPY: large-scale single-cell gene expression data analysis. <i>Genome Biology</i>, 19(1), 15."),
]

# ── Build PDF ─────────────────────────────────────────────────────────────
doc.build(story)
print(f"Report saved: {OUT_PDF}")
print(f"Size: {OUT_PDF.stat().st_size / 1024 / 1024:.1f} MB")
