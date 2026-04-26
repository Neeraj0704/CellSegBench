"""Prepare transcript CSV for Baysor from the tile region."""
import json, sys
sys.path.insert(0, "..")
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path("../data/Human_Breast_Biomarkers_S1_Top_outs")

with open(DATA_DIR / "tile_meta.json") as f:
    tile = json.load(f)

t = pq.read_table(
    DATA_DIR / "transcripts_filtered.parquet",
    columns=["transcript_id", "feature_name", "x_location", "y_location", "z_location"],
    filters=[
        ("x_location", ">=", tile["x0_um"]), ("x_location", "<", tile["x1_um"]),
        ("y_location", ">=", tile["y0_um"]), ("y_location", "<", tile["y1_um"]),
    ]
).to_pandas()

out = t[["x_location", "y_location", "z_location", "feature_name"]].copy()
out.columns = ["x", "y", "z", "gene"]

out_path = DATA_DIR / "baysor_input.csv"
out.to_csv(out_path, index=False)
print(f"Baysor input: {len(out):,} transcripts → {out_path}")
print(f"Tile: x=[{tile['x0_um']:.0f}, {tile['x1_um']:.0f}] µm, "
      f"y=[{tile['y0_um']:.0f}, {tile['y1_um']:.0f}] µm")
print(out.head())
