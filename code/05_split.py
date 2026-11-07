"""
Step 5 — spatially-blocked train/test split.

GOAL: split tiles into train (~75%) and test (~25%) by CONTIGUOUS geographic
blocks (14×14 tiles ≈ 3.5 km), never randomly — random splits leak via spatial
autocorrelation between neighbouring tiles.

Reads:   data/labels/tile_labels.parquet
Writes:  data/processed/split.parquet  (tile_id, block, is_test)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LBL = ROOT / "data" / "labels"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

BLOCK = 14                     # tiles per block side (~3.5 km at 250 m)
TEST_FRACTION = 0.25
SEED = 42

# reconstruct block ids from tile_id = t{row}c{col}
tiles = pd.read_parquet(LBL / "tile_labels.parquet").copy()
tiles = tiles[tiles.label != "other"].reset_index(drop=True)
tiles["row"] = tiles.tile_id.str.slice(1, 4).astype(int)
tiles["col"] = tiles.tile_id.str.slice(5, 8).astype(int)
tiles["block"] = (tiles.row // BLOCK).astype(str) + "_" + (tiles.col // BLOCK).astype(str)

blocks = sorted(tiles.block.unique())
rng = np.random.default_rng(SEED)
rng.shuffle(blocks)
n_test = max(1, int(round(len(blocks) * TEST_FRACTION)))
test_blocks = set(blocks[:n_test])
tiles["is_test"] = tiles.block.isin(test_blocks)

split = tiles[["tile_id", "block", "is_test"]]
split.to_parquet(OUT / "split.parquet")

print(f"blocks: {len(blocks)} ({BLOCK}×{BLOCK} tiles each), "
      f"test blocks: {len(test_blocks)}")
print("\ntile counts  train/test:")
print(tiles.groupby(["label", "is_test"]).size().unstack(fill_value=0))
