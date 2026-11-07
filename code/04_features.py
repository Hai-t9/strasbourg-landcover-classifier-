"""
Step 4 — spectral features per 250 m tile.

GOAL: one numeric feature row per tile from the Sentinel-2 stack:
  • per-band mean + std (10 + 10 features)
  • NDVI mean/std (vegetation), NDWI mean (water), NDBI mean (built-up)
  • brightness mean, red/blue ratio, nir/red ratio
Merged with tile labels → data/features/features.parquet
(the model step drops 'other'/'unlabelled' tiles)
"""

import json as _json
from pathlib import Path

import numpy as np
import pandas as pd

import rasterio

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "img"
LBL = ROOT / "data" / "labels"
FEAT_DIR = ROOT / "data" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

TILE_PX = 25
EPS = 1e-6

with rasterio.open(IMG / "s2_stack.tif") as src:
    bands = list(src.descriptions)
    H, W = src.height, src.width
    data = {b: src.read(i + 1).astype("float32") for i, b in enumerate(bands)}
print(f"→ stack: {len(bands)} bands {W}×{H}")

bidx = {b: i for i, b in enumerate(bands)}
tH, tW = H // TILE_PX, W // TILE_PX


def tile_view(a):
    """(H,W) -> (n_tiles, TILE_PX*TILE_PX) matching tile_id ordering t{r}c{c}."""
    t = a[:tH * TILE_PX, :tW * TILE_PX].reshape(tH, TILE_PX, tW, TILE_PX)
    return t.transpose(0, 2, 1, 3).reshape(tH * tW, TILE_PX * TILE_PX)


feats = {}
for b in bands:
    tv = tile_view(data[b])
    feats[f"{b}_mean"] = tv.mean(axis=1)
    feats[f"{b}_std"] = tv.std(axis=1)

# indices computed pixel-wise, then aggregated per tile (correcter than on means)
nir = tile_view(data["nir"])
red = tile_view(data["red"])
green = tile_view(data["green"])
sw16 = tile_view(data["swir16"])

ndvi = np.clip((nir - red) / (nir + red + EPS), -1, 1)
ndwi = np.clip((green - nir) / (green + nir + EPS), -1, 1)
ndbi = np.clip((sw16 - nir) / (sw16 + nir + EPS), -1, 1)

for name, arr in [("ndvi", ndvi), ("ndwi", ndwi), ("ndbi", ndbi)]:
    feats[f"{name}_mean"] = arr.mean(axis=1)
    feats[f"{name}_std"] = arr.std(axis=1)

tv_sum = np.sum([tile_view(data[b]) for b in bands], axis=0) + EPS
normalized = [tile_view(data[b]) / tv_sum for b in bands]
feats["brightness"] = (tv_sum / len(bands)).mean(axis=1)
fracs = {f"frac_{b}": normalized[i].mean(axis=1)
         for i, b in enumerate(bands)}          # relative band contributions
feats.update(fracs)

df = pd.DataFrame(feats)
df.insert(0, "tile_id", [f"t{r:03d}c{c:03d}" for r in range(tH) for c in range(tW)])

# merge labels
labels = pd.read_parquet(LBL / "tile_labels.parquet")
df = df.merge(labels[["tile_id", "label", "major_pct"]], on="tile_id", how="left")
print(f"→ features: {df.shape[1]-2} per tile × {len(df)} tiles; labelled: "
      f"{df.label.notna().sum()}")

out = FEAT_DIR / "features.parquet"
df.to_parquet(out)
print(f"✓ saved {out}")

# quick per-class feature signature sanity print
sig = df[df.label.isin(["water", "forest", "farmland", "built_up", "grass"])] \
    .groupby("label")[["ndvi_mean", "ndwi_mean", "ndbi_mean", "nir_mean"]].mean()
print("\nmean spectral signature per class (sanity check):")
print(sig.round(3))
