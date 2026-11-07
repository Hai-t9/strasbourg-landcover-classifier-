"""
Step 8 — classify the WHOLE study area.

GOAL: apply the trained model to every 250 m tile of the image (labelled or
not) and write the predicted land-cover map.

Reads:   data/img/s2_stack.tif, data/features/features.parquet
         results/model.pkl
Writes:  results/predicted_map.tif   (10 m grid, class id per 25×25 tile block)
         results/predicted_map_preview.jpg
         results/prediction_table.parquet (tile_id, predicted class, confidence)
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rasterio

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "img"
FEAT = ROOT / "data" / "features" / "features.parquet"
OUT = ROOT / "results"

CLASSES = ["water", "forest", "grass", "farmland", "built_up"]
COLOURS = np.array([[72, 115, 224], [19, 104, 38], [150, 205, 99],
                    [222, 181, 74], [212, 68, 62]], dtype="uint8")

with rasterio.open(IMG / "s2_stack.tif") as src:
    H, W, TR, CRS = src.height, src.width, src.transform, src.crs

df = pd.read_parquet(FEAT)          # all 18396 tiles
feat_cols = [c for c in df.columns if c.endswith(("_mean", "_std"))
             or c.startswith("frac_") or c in ("brightness",)]

with open(OUT / "model.pkl", "rb") as f:
    model = pickle.load(f)

X = df[feat_cols].to_numpy("float32")
proba = model.predict_proba(X)
pred_labels = model.predict(X)             # actual class strings
preds = np.array([CLASSES.index(c) for c in pred_labels])   # colour-map index
conf = proba.max(axis=1)

out_tab = pd.DataFrame({"tile_id": df.tile_id, "pred": pred_labels,
                        "confidence": conf.round(3),
                        "osm_label": df.label})
out_tab.to_parquet(OUT / "prediction_table.parquet")
print("predicted-class distribution over the WHOLE area:")
print(out_tab.pred.value_counts())

# ---- map raster: one 10 m pixel grid upsampled from per-tile class ids
def rowcol(tid):
    return int(tid[1:4]), int(tid[5:8])

tH, tW = H // 25, W // 25
tile_map = np.zeros((tH, tW), dtype="uint8")
tile_conf = np.zeros((tH, tW), dtype="float32")
for tid, p, c in zip(out_tab.tile_id, preds, conf):
    r, cc = rowcol(tid)
    if 0 <= r < tH and 0 <= cc < tW:
        tile_map[r, cc] = p + 1
        tile_conf[r, cc] = c

# save GeoTIFF (1 = water ... 5 = built_up, 0 = n/a)
with rasterio.open(OUT / "predicted_map.tif", "w", driver="GTiff",
                   height=H, width=W, count=1, dtype="uint8", crs=CRS,
                   transform=TR, nodata=0, compress="deflate") as dst:
    class_img = np.kron(tile_map, np.ones((25, 25), dtype="uint8"))
    dst.set_band_description(1, "predicted_landcover")
    # values: 0=unclassified, 1 water, 2 forest, 3 grass, 4 farmland, 5 built_up
    dst.write(class_img[:H, :W], 1)

# preview jpg
img = COLOURS[tile_map - 1] * (tile_map[..., None] > 0)
PIL_zoom = tuple(d // 2 for d in tile_map.shape[::-1])
import PIL.Image
PIL.Image.fromarray(img).resize(PIL_zoom, PIL.Image.NEAREST).save(
    OUT / "predicted_map_preview.jpg", quality=90)

# confidence preview side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
axes[0].imshow(img, interpolation="nearest")
axes[0].set_title("predicted land cover (all tiles)")
axes[1].imshow(tile_conf, cmap="magma", vmin=0.5, vmax=1)
axes[1].set_title("prediction confidence")
for a in axes:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
fig.savefig(OUT / "predicted_map_overview.png", dpi=140)
print(f"✓ saved {OUT / 'predicted_map.tif'}, predicted_map_preview.jpg, "
      f"predicted_map_overview.png, prediction_table.parquet")
