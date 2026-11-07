"""
Step 3 — label the tiles from the OSM vector data.

GOAL: rasterize the OSM layers to the exact 10 m pixel grid of the image,
point-sample it on a lattice, and assign each 250 m tile a class label
(majority class, must cover ≥ 60 % of the tile — otherwise the tile is
 unlabeled and dropped from training later).

Reads:   data/img/s2_stack.tif            (grid reference only)
         data/shape/*.shp                 (OSM labels)
Writes:  data/labels/tile_labels.parquet  (tile_id, label, coverage%)
         data/labels/label_raster.tif     (10 m class map, QC + features)
         data/labels/label_preview.jpg
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import shapely

import rasterio
import rasterio.features

ROOT = Path(__file__).resolve().parent.parent
SHAPE = ROOT / "data" / "shape"
IMG = ROOT / "data" / "img"
STACK = IMG / "s2_stack.tif"
LBL_DIR = ROOT / "data" / "labels"
LBL_DIR.mkdir(parents=True, exist_ok=True)

TILE_PX = 25
MIN_COVERAGE = 60.0   # % of the tile covered by the majority class

CLASS_MAP = {           # OSM 'type' -> target class
    "water":    ["water", "river", "stream", "canal", "riverbank", "basin",
                 "pond", "dock"],
    "forest":   ["forest", "wood"],
    "grass":    ["grass", "meadow", "park", "recreation_ground",
                 "village_green"],
    "farmland": ["farmland", "orchard", "vineyard", "allotments",
                 "plant_nursery"],
    "built_up": ["residential", "commercial", "industrial", "retail",
                 "cemetery", "quarry", "construction", "farmyard"],
}
FALLBACK = "other"

# ------------------------------------------------------------- grid reference
with rasterio.open(STACK) as src:
    H, W, TR, CRS = src.height, src.width, src.transform, src.crs

# ------------------------------------------------------------- 1) vectors → classes
def to_class(layer):
    layer = layer.copy()
    layer["cls"] = layer["type"].map(
        lambda t: next((k for k, v in CLASS_MAP.items() if t in v), FALLBACK))
    return layer


def fill_holes_polygons(layer):
    """drop internal holes — OSM polygons often have them and they leak 'other'."""
    return gpd.GeoSeries(layer.geometry.values.make_valid().record_holes() if
                         hasattr(layer.geometry, "make_valid") else
                         layer.geometry.values)


lu = to_class(gpd.read_file(SHAPE / "landuse.shp"))
na = to_class(gpd.read_file(SHAPE / "natural.shp"))
bd = gpd.read_file(SHAPE / "buildings.shp")
bd["cls"] = "built_up"
ww = gpd.read_file(SHAPE / "waterways.shp")

# water = natural.water polygons + thin 15 m waterline corridor
# (thick buffers overstate water: bank pixels are fields/meadows, poisoning the class)
water_buf = ww.to_crs(CRS).buffer(15)

frames = [gpd.GeoDataFrame(pd.concat([lu, na])[["geometry", "cls"]], crs=4326)]
frames.append(gpd.GeoDataFrame({"cls": ["built_up"] * len(bd)},
                               geometry=bd.geometry.values, crs=4326))
frames.append(gpd.GeoDataFrame({"cls": ["water"] * len(water_buf)},
                               geometry=water_buf.values, crs=CRS).to_crs(4326))
poly = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326).to_crs(CRS)
poly["geometry"] = poly.geometry.make_valid()
poly = poly[~poly.geometry.is_empty].reset_index(drop=True)
poly = poly[~poly.cls.isna()].reset_index(drop=True)
print(f"→ {len(poly)} labelled polygons (per-class):")
print(poly.cls.value_counts())

# ------------------------------------------------------------- 2) rasterize to 10m grid
CLASS_ORDER = ["water", "forest", "grass", "farmland", "built_up", "other"]
CLASS_IDS = {c: i + 1 for i, c in enumerate(CLASS_ORDER)}
raster_shapes = [(geom, CLASS_IDS[c])
                 for geom, c in zip(poly.geometry.values, poly.cls.values)]
raster = rasterio.features.rasterize(
    raster_shapes, out_shape=(H, W), transform=TR, fill=0, dtype="uint8")
LABELS = {i: c for c, i in CLASS_IDS.items()}
np.save(LBL_DIR / "label_raster.npy", raster)
with rasterio.open(LBL_DIR / "label_raster.tif", "w", driver="GTiff",
                   height=H, width=W, count=1, dtype="uint8",
                   crs=CRS, transform=TR, nodata=0, compress="deflate") as dst:
    dst.write(raster, 1)
print(f"✓ wrote {LBL_DIR / 'label_raster.tif'}  ({H}×{W})")

# ------------------------------------------------------------- 3) per-tile majority class
tH, tW = H // TILE_PX, W // TILE_PX
raster = raster[:tH * TILE_PX, :tW * TILE_PX]     # drop the 18-px edge sliver
tiles = raster.reshape(tH, TILE_PX, tW, TILE_PX)
tiles = tiles.transpose(0, 2, 1, 3).reshape(tH * tW, TILE_PX * TILE_PX)

n = TILE_PX * TILE_PX
frac = np.zeros((tH * tW, len(LABELS) + 1), dtype="float32")
for k, cls_id in enumerate(sorted(LABELS.keys())):
    frac[:, k] = (tiles == cls_id).mean(axis=1)
frac[:, len(LABELS)] = (tiles == 0).mean(axis=1)  # coverage fraction of 'unlabelled'
cols = [LABELS[i] for i in sorted(LABELS)] + ["unlabelled"]

mp = pd.DataFrame(frac, columns=cols)
mp["tile_id"] = [f"t{r:03d}c{c:03d}" for r in range(tH) for c in range(tW)]
mp["major"] = mp[cols].idxmax(axis=1)
mp["major_pct"] = 100 * mp[cols].max(axis=1)
clean = mp[(mp.major_pct >= MIN_COVERAGE) & (mp.major != "unlabelled")]
clean["label"] = clean["major"]
clean[["tile_id", "label", "major_pct"]].to_parquet(LBL_DIR / "tile_labels.parquet")

print(f"\n=== clean tiles per class (≥ {MIN_COVERAGE:.0f}% coverage) ===")
print(clean["major"].value_counts().sort_index())
print(f"total: {len(clean)} of {tH * tW} tiles")

# ------------------------------------------------------------- 4) QC preview
colour = np.array([[0, 0, 0], [41, 96, 209], [17, 90, 28], [140, 197, 91],
                   [217, 165, 37], [200, 40, 40], [150, 150, 150]], dtype="uint8")
img = colour[raster]
import PIL.Image
PIL.Image.fromarray(img).resize((W // 6, H // 6)).save(
    LBL_DIR / "label_preview.jpg", quality=90)
print(f"✓ wrote {LBL_DIR / 'label_preview.jpg'}")
