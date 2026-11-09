"""
Step 6 — model benchmark on spectral features.

GOAL: compare Dummy baseline / LogisticRegression / RandomForest /
HistGradientBoosting on the spatially-held-out test blocks, save metrics.

Reads:   data/features/features.parquet
         data/processed/split.parquet
Writes:  results/metrics.json, results/report.txt, results/model.pkl
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             make_scorer)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "features.parquet"
SPLIT = ROOT / "data" / "processed" / "split.parquet"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["water", "forest", "grass", "farmland", "built_up"]
SEED = 42

df = pd.read_parquet(FEAT)
sp = pd.read_parquet(SPLIT)
df = df.merge(sp, on="tile_id")
df = df[df.label.isin(CLASSES)].reset_index(drop=True)

feat_cols = [c for c in df.columns if c.endswith(("_mean", "_std"))
             or c.startswith("frac_") or c in ("brightness",)]
X = df[feat_cols].to_numpy(dtype="float32")
y = df.label.to_numpy()
is_test = df.is_test.to_numpy()

Xtr, ytr = X[~is_test], y[~is_test]
Xte, yte = X[is_test], y[is_test]
print(f"train: {len(Xtr)}   test: {len(Xte)}   features: {len(feat_cols)}")

models = {
    "dummy_majority": DummyClassifier(strategy="most_frequent"),
    "logreg": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0,
                                               class_weight="balanced")),
    "random_forest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1,
        class_weight="balanced", random_state=SEED),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, l2_regularization=1.0,
        random_state=SEED, class_weight="balanced"),
}

results = {}
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    macro = f1_score(yte, pred, average="macro")
    results[name] = {
        "macro_f1": float(macro),
        "accuracy": float(accuracy_score(yte, pred)),
    }
    print(f"{name:24s} macro-F1={macro:.3f}  acc={accuracy_score(yte, pred):.3f}")

best = max(results, key=lambda k: results[k]["macro_f1"])
print(f"\nbest model: {best}  (macro-F1 {results[best]['macro_f1']:.3f})")

best_model = models[best]
pred = best_model.predict(Xte)
report = classification_report(yte, pred, digits=3, zero_division=0)
print("\nper-class report (test):\n", report)

(OUT / "metrics.json").write_text(json.dumps(results, indent=2))
(OUT / "report.txt").write_text(
    f"best={best}\nfeature_cols={feat_cols}\n\n{report}")
(OUT / "feature_cols.json").write_text(json.dumps(feat_cols, indent=1))

with open(OUT / "model.pkl", "wb") as f:
    pickle.dump(best_model, f)

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feat_cols) \
        .sort_values(ascending=False)
    print("\ntop-12 feature importances:")
    print(imp.head(12).round(4))
    imp.to_csv(OUT / "feature_importance.csv")
