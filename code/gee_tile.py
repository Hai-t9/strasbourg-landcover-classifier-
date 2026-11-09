"""
gee_tile.py — classify any lat/lon in the study area (and slightly beyond) with
the trained tile classifier, by fetching the Sentinel-2 tile on demand from GEE.

Used by the Streamlit app (code/app.py); can also be called from the CLI:
    python3 code/gee_tile.py 48.5734 7.7521
"""

import io
import json as _json
import pickle
import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCALE = 10                      # m/pixel
TILE_PX = 25                    # 250 m tile
EPS = 1e-6

# same band mapping as the training pipeline
BANDS = {"blue": "B2", "green": "B3", "red": "B4",
         "rededge70": "B5", "rededge71": "B6", "rededge72": "B7",
         "nir": "B8", "nir08": "B8A", "swir16": "B11", "swir22": "B12"}

CLASSES = ["water", "forest", "grass", "farmland", "built_up"]
CLASS_COLOURS = {"water": "#4873e0", "forest": "#136826", "grass": "#96cd63",
                 "farmland": "#deb54a", "built_up": "#d4443e"}

# NOTE: the classifier is valid for the 2025-summer composite period
DATE_RANGE = "2025-06-01/2025-09-30"

_composite = None
_model = None


def _init_ee():
    global _composite
    if _composite is not None:
        return _composite
    import ee
    import os
    # key sources, in priority order
    raw = os.environ.get("GEE_KEY_JSON")          # raw JSON text (cloud secret)
    path = os.environ.get("GEE_KEY_FILE") or None  # path to the key json
    if raw is None:
        candidates = [Path(path)] if path else sorted(ROOT.glob("strasbourg-classification-*.json"))
        key_path = next((c for c in candidates if c.exists()), None)
    else:
        key_path = None
    if key_path is not None:
        svc = _json.load(open(key_path))
        ee.Initialize(credentials=ee.ServiceAccountCredentials(svc["client_email"], str(key_path)),
                      project=svc["project_id"])
    elif raw is not None:
        import tempfile
        tmp = Path(tempfile.mkstemp(suffix=".json")[1])
        tmp.write_text(raw)
        svc = _json.loads(raw)
        ee.Initialize(credentials=ee.ServiceAccountCredentials(svc["client_email"], str(tmp)),
                      project=svc["project_id"])
    else:
        ee.Initialize(project="strasbourg-classification")   # on-machine default creds
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(ee.Geometry.Rectangle([7.2, 48.1, 8.8, 49.1]))
          .filterDate(*DATE_RANGE.split("/")))

    def mask(it):
        scl = it.select("SCL")
        keep = (scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5))
                .Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)))
        return (it.select(list(BANDS.values()))
                  .updateMask(keep).divide(10000)
                  .copyProperties(it, ["system:time_start"]))

    _composite = s2.map(mask).median()
    return _composite


def load_env_key():
    import os
    raw = os.environ.get("GEE_KEY_JSON")
    return raw


def load_model():
    global _model
    if _model is None:
        with open(ROOT / "results" / "model.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model


def load_feature_cols():
    """the exact feature-column list the model was trained on."""
    sidecar = ROOT / "results" / "feature_cols.json"
    if sidecar.exists():
        return _json.loads(sidecar.read_text())
    # fallback: parse from step-06's report (kept for reproducibility)
    import ast
    rep = (ROOT / "results" / "report.txt").read_text()
    return ast.literal_eval(rep.split("feature_cols=", 1)[1].split("\n", 1)[0])


def fetch_tile(lat: float, lon: float) -> dict:
    """download the 25×25 px Sentinel-2 tile centered on (lat, lon).
    returns {band_name: (25,25) float32 ndarray}."""
    import ee
    img = _init_ee()
    import pyproj
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    cx, cy = tr.transform(lon, lat)
    half = TILE_PX * SCALE / 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    rect = ee.Geometry.Rectangle([x0, y0, x1, y1], "EPSG:32632", True, False)
    url = img.getDownloadURL({"scale": SCALE, "crs": "EPSG:32632",
                              "region": rect, "format": "NPY"})
    with urllib.request.urlopen(url, timeout=300) as r:
        arr = np.load(io.BytesIO(r.read()), allow_pickle=True)
    out = {}
    if arr.dtype.fields is not None:     # multi-band structured NPY
        for fname in arr.dtype.names:
            band = next((k for k, v in BANDS.items() if v == fname), None)
            if band is not None:
                out[band] = arr[fname].astype("float32")
    else:                                 # single band fallback
        out["blue"] = arr
    return out


def compute_features(tiles: dict) -> dict:
    """same 38 features per tile as code/04_features.py"""
    feats = {}
    for b, a in tiles.items():
        feats[f"{b}_mean"] = a.mean()
        feats[f"{b}_std"] = a.std()
    nir, red = tiles["nir"], tiles["red"]
    green, sw16 = tiles["green"], tiles["swir16"]
    ndvi = np.clip((nir - red) / (nir + red + EPS), -1, 1)
    ndwi = np.clip((green - nir) / (green + nir + EPS), -1, 1)
    ndbi = np.clip((sw16 - nir) / (sw16 + nir + EPS), -1, 1)
    for name, arr in [("ndvi", ndvi), ("ndwi", ndwi), ("ndbi", ndbi)]:
        feats[f"{name}_mean"] = arr.mean()
        feats[f"{name}_std"] = arr.std()
    tv_sum = np.sum([a for a in tiles.values()], axis=0) + EPS
    feats["brightness"] = (tv_sum / len(tiles)).mean()
    for b, a in tiles.items():
        feats[f"frac_{b}"] = (a / tv_sum).mean()
    return feats


CLASSIFY_CACHE = {}


def classify(lat: float, lon: float):
    """returns (class_label, confidence, tiles_dict, proba_dict)."""
    tiles = fetch_tile(lat, lon)
    feats = compute_features(tiles)
    cols = load_feature_cols()
    X = np.array([[feats[c] for c in cols]], dtype="float32")
    model = load_model()
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    probs = {c: float(p) for c, p in zip(model.classes_, proba)}
    return str(pred), float(proba.max()), tiles, probs


def rgb_preview(tiles: dict) -> np.ndarray:
    """(25,25,3) uint8 true-colour crop, 2%/98% stretched."""
    rgb = np.stack([tiles["red"], tiles["green"], tiles["blue"]], -1)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")


# ----------------------------------------------------------------- CLI test
if __name__ == "__main__":
    import sys
    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    label, conf, tiles, probs = classify(lat, lon)
    print(f"({lat}, {lon}) → {label}  conf={conf:.2f}")
    for c in sorted(probs, key=probs.get, reverse=True):
        print(f"   {c:9s} {probs[c]:.3f}")
