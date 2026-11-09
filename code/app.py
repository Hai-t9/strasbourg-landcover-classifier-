"""
app.py — Streamlit demo: click a point (or type lat/lon) → the app fetches the
250 m Sentinel-2 tile at that spot from Google Earth Engine and predicts its
land-cover class.

Run:    streamlit run code/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gee_tile as geotile

st.set_page_config(page_title="Strasbourg land-cover classifier",
                   page_icon="🛰️", layout="wide")

st.title("🛰️ Sentinel-2 land-cover tile classifier")
st.caption("Click the map (or type coordinates) → a 250 m Sentinel-2 tile "
           "(summer 2025 median, 10 bands) is fetched from Google Earth Engine "
           "and classified into **water / forest / grass / farmland / built-up** "
           "with the model trained on OSM labels.")

# ------------------------------------------------------------- inputs
DEFAULT = {"lat": 48.5734, "lon": 7.7521}

with st.sidebar:
    st.header("1 · Choose a location")
    st.write("click the map to move the marker ↓")
    markers = st.map(pd.DataFrame([DEFAULT]), zoom=10)

    lat_in = st.number_input("latitude", min_value=47.8, max_value=49.4,
                             value=DEFAULT["lat"], step=0.001, format="%.5f")
    lon_in = st.number_input("longitude", min_value=6.9, max_value=8.9,
                             value=DEFAULT["lon"], step=0.001, format="%.5f")

    lat, lon = lat_in, lon_in
    # if the user clicked the map, take the *last clicked* point
    try:
        last = None
        if isinstance(markers, pd.DataFrame) and len(markers):
            last = markers.iloc[-1]
        elif hasattr(markers, "get_last_row"):
            last_df = markers.get_last_row()
            last = last_df.iloc[-1] if isinstance(last_df, pd.DataFrame) and len(last_df) else last_df
        if last is not None:
            if isinstance(last, (dict,)):
                last = pd.Series(last)
            la, lo = last.get("lat", np.nan), last.get("lon", np.nan)
            if not pd.isna(la) and not pd.isna(lo) and (la != DEFAULT["lat"] or lo != DEFAULT["lon"]):
                lat, lon = float(la), float(lo)
                st.write(f"map point: **{lat:.5f}, {lon:.5f}**")
    except Exception:
        pass
    st.caption(f"current point: {lat:.5f}, {lon:.5f}")

    st.divider()
    st.subheader("2 · Classify")
    go = st.button("🚀 classify this point", type="primary",
                   use_container_width=True)

# ------------------------------------------------------------- classify
if go:
    with st.spinner("fetching Sentinel-2 tile from Google Earth Engine …"):
        label, conf, tiles, probs = geotile.classify(lat, lon)

    st.subheader(f"({lat:.5f}, {lon:.5f})")
    col_img, col_res, col_bar = st.columns([1.3, 1.1, 1.1])

    with col_img:
        st.image(geotile.rgb_preview(tiles), output_format="PNG",
                 caption="the 250 m × 250 m Sentinel-2 tile fed to the model")
        ndvi = (tiles["nir"] - tiles["red"]) / (tiles["nir"] + tiles["red"] + 1e-6)
        ramp = np.array([[80, 40, 40], [140, 140, 100], [90, 170, 90],
                         [20, 140, 20], [15, 110, 15]], dtype="uint8")
        nd = ramp[np.clip(((ndvi + 0.4) / 0.7 * 4), 0, 4).astype("int8")]
        st.image(nd, output_format="PNG", caption="per-pixel NDVI (red = low, green = high)")

    with col_res:
        st.markdown("### prediction")
        st.markdown(
            f"<h1 style='color:{geotile.CLASS_COLOURS[label]};margin:0'>"
            f"{label.upper()}</h1>"
            f"<p style='color:{geotile.CLASS_COLOURS[label]};font-size:1.05em'>"
            f"{int(conf * 100)} % confidence</p>",
            unsafe_allow_html=True)
        st.progress(min(max(conf, 0.0), 1.0), text=f"model confidence {conf:.2f}")
        st.caption("⚠️ gradient boosting tends to be overconfident; the confidence "
                   "is best read as a *ranking between tiles*, not an absolute probability.")

    with col_bar:
        st.markdown("### class probabilities")
        pdf = pd.DataFrame({"class": list(probs), "p": list(probs.values())}) \
            .sort_values("p", ascending=False)
        for _, row in pdf.iterrows():
            st.metric(f"{row['class']}", f"{row['p'] * 100:.1f} %")
else:
    st.info("👈 pick a point on the map (or type coordinates) and press "
            "**classify this point**.")
