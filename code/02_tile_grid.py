"""
Step 2 — build the 250 m tile grid over the downloaded image.

GOAL: one row per image tile (25×25 px = 250 m at 10 m resolution),
each with its id, (row, col), and EPSG:32632 bbox.

Reads:   data/img/s2_stack.tif
Writes:  data/img/tile_grid.parquet
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

import rasterio

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "img"
STACK = IMG / "s2_stack.tif"
OUT = IMG / "tile_grid.parquet"

TILE_PX = 25          # 25 × 25 px = 250 m at 10 m/pixel

with rasterio.open(STACK) as src:
    W, H = src.width, src.height
    tr, crs = src.transform, src.crs

nrows, ncols = H // TILE_PX, W // TILE_PX
used_h, used_w = nrows * TILE_PX, ncols * TILE_PX
if (used_h, used_w) != (H, W):
    print(f"note: cropping {H-used_h}×{W-used_w} px edge sliver (not divisible by {TILE_PX})")
print(f"image {W}×{H} px  →  {nrows}×{ncols} tiles of {TILE_PX}×{TILE_PX} px")

rows = []
for r in range(nrows):
    for c in range(ncols):
        x0, ymax = tr * (c * TILE_PX, r * TILE_PX)
        x1, ymin = tr * ((c + 1) * TILE_PX, (r + 1) * TILE_PX)
        rows.append({
            "tile_id": f"t{r:03d}c{c:03d}",
            "row": r, "col": c,
            "pixel_x": c * TILE_PX, "pixel_y": r * TILE_PX,  # top-left px in stack
            "geometry": shapely.box(min(x0, x1), min(ymin, ymax),
                                    max(x0, x1), max(ymin, ymax)),
        })

gdf = gpd.GeoDataFrame(pd.DataFrame(rows), crs=crs)
gdf.to_parquet(OUT)
print(f"✓ saved {OUT}  ({len(gdf)} tiles)")
