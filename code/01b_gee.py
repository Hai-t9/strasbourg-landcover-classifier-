"""
Step 1 — download Sentinel-2 imagery via Google Earth Engine (GEE path).

GOAL: a cloud-masked, 2025-summer median composite of Sentinel-2 of the study
area on disk, 10 m, EPSG:32632, as a multi-band GeoTIFF.

Approach: 4 quadrant tiles × 10 bands requested as NPY tiles
(GEE caps one request at ~48 MB; the full region is ~60 MB/band).
Resumable: every fetched tile is cached in data/img/_gee_tiles/*.npy
and re-runs only fetch missing tiles. Network hiccups retry with backoff.
"""

import io
import json as _json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.transform
import shapely.geometry

import ee

# ---------------------------------------------------------------- study area
BBOX_WGS84 = (7.55, 48.43, 8.04, 48.72)
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "img"
TILE_DIR = OUT_DIR / "_gee_tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {"blue": "B2", "green": "B3", "red": "B4",
         "rededge70": "B5", "rededge71": "B6", "rededge72": "B7",
         "nir": "B8", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}
DT = "2025-06-01/2025-09-30"
SCALES = 10                       # m/pixel
TARGET_CRS = "EPSG:32632"

# ------------------------------------------------------------- 1) connect
KEY = next(ROOT.glob("strasbourg-classification-*.json"))
svc = _json.load(open(KEY))
creds = ee.ServiceAccountCredentials(svc["client_email"], str(KEY))
ee.Initialize(credentials=creds, project=svc["project_id"])
print("→ GEE authorised:", svc["client_email"], flush=True)

# ------------------------------------------------------------- 2) composite
region_wgs = ee.Geometry.Rectangle(list(BBOX_WGS84))
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(region_wgs).filterDate(*DT.split("/")))


def mask(it):
    """drop clouds, shadows, cirrus, nodata via the SCL band."""
    scl = it.select("SCL")
    keep = (scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5))
            .Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)))
    return (it.select(list(BANDS.values()))
              .updateMask(keep)
              .divide(10000)                                        # refl 0..1
              .copyProperties(it, ["system:time_start"]))


composite = s2.map(mask).median()
n_scenes = s2.size().getInfo()
print(f"→ composite over {n_scenes} scenes   ({DT})", flush=True)

# ------------------------------------------------------------- 3) grid
tr = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
x0w, y0w = tr.transform(BBOX_WGS84[0], BBOX_WGS84[1])
x1w, y1w = tr.transform(BBOX_WGS84[2], BBOX_WGS84[3])
x0 = int(np.floor(min(x0w, x1w) / SCALES) * SCALES)
x1 = int(np.ceil(max(x0w, x1w) / SCALES) * SCALES) + SCALES
y0 = int(np.floor(min(y0w, y1w) / SCALES) * SCALES)
y1 = int(np.ceil(max(y0w, y1w) / SCALES) * SCALES) + SCALES
XMID = int(round((x0 + x1) / 2 / SCALES) * SCALES)
YMID = int(round((y0 + y1) / 2 / SCALES) * SCALES)
QUADS = {"sw": (x0, y0, XMID, YMID), "se": (XMID, y0, x1, YMID),
         "nw": (x0, YMID, XMID, y1), "ne": (XMID, YMID, x1, y1)}
print(f"→ area (UTM 32N): x {x0:,}–{x1:,},  y {y0:,}–{y1:,}", flush=True)


def fetch_tile(quad, band_asset, retries=6):
    """download one quadrant × one band with retry/backoff; cached on disk.

    NOTE: the download URL must be regenerated on every attempt — GEE ticket
    URLs are short-lived and reusing one across retries yields 401s.
    """
    cache = TILE_DIR / f"{quad}-{band_asset}.npy"
    if cache.exists() and cache.stat().st_size > 256:
        return np.load(cache)
    qx0, qy0, qx1, qy1 = QUADS[quad]
    last = None
    for attempt in range(1, retries + 1):
        try:
            rect = ee.Geometry.Rectangle([qx0, qy0, qx1, qy1], TARGET_CRS,
                                         True, False)
            url = composite.select(band_asset).getDownloadURL(
                {"scale": SCALES, "crs": TARGET_CRS, "region": rect,
                 "format": "NPY"})
            with urllib.request.urlopen(url, timeout=180) as r:
                arr = np.load(io.BytesIO(r.read()), allow_pickle=True)
            if arr.dtype.fields is not None:                # structured NPY
                arr = arr[arr.dtype.names[0]]
            np.save(cache, arr)
            return arr
        except Exception as e:
            last = e
            wait = 5 + min(60, 5 * attempt)
            print(f"      ⚠ retry {attempt}/{retries} for {quad}/{band_asset}"
                  f" ({type(e).__name__}) — sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise last


tiles = {}
total = len(QUADS) * len(BANDS)
done = 0
for quad in QUADS:
    tiles[quad] = {}
    for name_band, band in BANDS.items():
        arr = fetch_tile(quad, band)
        tiles[quad][name_band] = arr
        done += 1
        print(f"   [{done:2d}/{total}] {quad} {name_band:10s} {arr.shape}"
              f" {arr.dtype}", flush=True)

# ------------------------------------------------------------- 4) assemble stack
# tolerate a ±1 px mismatch between quadrant tiles
def same(a, b):
    return (abs(a.shape[0] - b.shape[0]) <= 2 and abs(a.shape[1] - b.shape[1]) <= 2)


for name_band in BANDS:
    b = tiles
    assert same(b["sw"][name_band], b["se"][name_band]), name_band
    assert same(b["nw"][name_band], b["ne"][name_band]), name_band

rows_sw = tiles["sw"]["blue"].shape[0]
rows_nw = tiles["nw"]["blue"].shape[0]
cols_w = tiles["sw"]["blue"].shape[1]
cols_e = tiles["se"]["blue"].shape[1]
H_total, W_total = rows_sw + rows_nw, cols_w + cols_e
print(f"→ final stack: {len(BANDS)} bands, {H_total}×{W_total} px", flush=True)


def pad_to(a, shape):
    out = np.zeros(shape, dtype=a.dtype)
    a_h, a_w = a.shape
    out[:a_h, :a_w] = a
    return out


out_tif = OUT_DIR / "s2_stack.tif"
profile = dict(driver="GTiff", compress="deflate", count=len(BANDS),
               width=W_total, height=H_total, crs=TARGET_CRS,
               transform=rasterio.transform.from_origin(x0, y1, SCALES, SCALES),
               nodata=0, dtype="float32")
with rasterio.open(out_tif, "w", **profile) as dst:
    for i, name_band in enumerate(BANDS.keys(), start=1):
        nw = pad_to(tiles["nw"][name_band], (rows_nw, cols_w))
        sw = pad_to(tiles["sw"][name_band], (rows_sw, cols_w))
        ne = pad_to(tiles["ne"][name_band], (rows_nw, cols_e))
        se = pad_to(tiles["se"][name_band], (rows_sw, cols_e))
        band_stack = np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])
        dst.write(band_stack.astype("float32"), i)
        dst.set_band_description(i, name_band)
print(f"✓ saved {out_tif}", flush=True)

# ------------------------------------------------------------- 5) QC preview
with rasterio.open(out_tif) as src:
    r = src.read(list(BANDS).index("red") + 1)
    g = src.read(list(BANDS).index("green") + 1)
    bl = src.read(list(BANDS).index("blue") + 1)
rgb = np.stack([r, g, bl], axis=-1)
for c in range(3):
    hi = np.nanpercentile(rgb[..., c], 99.5)
    rgb[..., c] = np.clip(rgb[..., c] / max(hi, 1) * 255, 0, 255)
import PIL.Image
PIL.Image.fromarray(rgb.astype("uint8")).resize(
    (W_total // 4, H_total // 4)).save(OUT_DIR / "s2_rgb_preview.jpg", quality=85)
print("✓ saved data/img/s2_rgb_preview.jpg", flush=True)

(OUT_DIR / "gee_scene.json").write_text(_json.dumps(
    {"scenes": n_scenes, "date_range": DT, "bands": list(BANDS.keys()),
     "crs": TARGET_CRS, "scale": SCALES,
     "grid": {"x0": x0, "y1": y1, "w": W_total, "h": H_total}}, indent=2))
print("✓ saved data/img/gee_scene.json", flush=True)
