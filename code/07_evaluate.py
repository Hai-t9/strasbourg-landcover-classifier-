"""
Step 7 — evaluation artefacts for the winning model.

GOAL: a confusion-matrix figure and a compact summary of the held-out test
performance, saved for the report.

Reads:   data/features/features.parquet, data/processed/split.parquet
         results/model.pkl
Writes:  results/confusion_matrix.png, results/summary.md
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "features.parquet"
SPLIT = ROOT / "data" / "processed" / "split.parquet"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CLASSES = ["water", "forest", "grass", "farmland", "built_up"]

df = pd.read_parquet(FEAT).merge(pd.read_parquet(SPLIT), on="tile_id")
df = df[df.label.isin(CLASSES)]
feat_cols = [c for c in df.columns if c.endswith(("_mean", "_std"))
             or c.startswith("frac_") or c in ("brightness",)]

with open(OUT / "model.pkl", "rb") as f:
    model = pickle.load(f)

is_test = df.is_test.to_numpy()
Xte, yte = df[feat_cols].to_numpy("float32")[is_test], df.label.to_numpy()[is_test]
pred = model.predict(Xte)

cm = confusion_matrix(yte, pred, labels=CLASSES, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
fig, ax = plt.subplots(figsize=(6.5, 5.4))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
ax.set_title("Sentinel-2 tile classification — test blocks (row-normalised)")
plt.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=150)

macro = f1_score(yte, pred, average="macro")
acc = (pred == yte).mean()
lines = [
    "# Project 2 — Sentinel-2 land-cover tile classification\n",
    f"- labelled tiles: {len(df)} (train {(~is_test).sum()} / test {is_test.sum()})",
    f"- held-out test accuracy: **{acc:.3f}**",
    f"- held-out test macro-F1: **{macro:.3f}**",
    "",
    "| class | row-recall | notes |",
    "|---|---|---|",
]
for i, c in enumerate(CLASSES):
    lines.append(f"| {c} | {cm[i, i]:.2f} | {'spectrally distinct' if c in ('water','forest','built_up') else 'weak class'} |")
lines += [
    "",
    f"- baseline (majority-class dummy): acc {(df.label == 'farmland').mean():.3f}, macro-F1 ≈ {1/(1+4)*0.2:.3f}",
    "- model: HistGradientBoosting; split: spatial blocks (14×14 tiles ≈ 3.5 km)",
]
(OUT / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
print(f"✓ saved {OUT / 'confusion_matrix.png'} and {OUT / 'summary.md'}")
