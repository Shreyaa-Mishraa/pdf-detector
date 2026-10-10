"""Train and evaluate the PDF malware classifier.

Run this INSIDE the isolated, network-less VM/container that holds the samples.

  1. Load data/manifest.csv and check each file's SHA-256 against it.
  2. Extract static features (features.py) for every file.
  3. Fit a RandomForestClassifier on the train split only.
  4. Evaluate once on the held-out test split. No tuning happens here.
  5. Save model.joblib, feature_names.json and results.json.

Every number in results.json comes from this run. Nothing is hardcoded.
"""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from features import FEATURE_NAMES, extract_features, to_vector

ROOT = Path(__file__).resolve().parent
SEED = 42

# Fixed and simple on purpose, with no search over these values.
#   n_estimators=200   enough trees for stable probabilities on a small dataset
#   max_depth=None     trees grow until leaves are pure (the sklearn default)
#   min_samples_leaf=1 sklearn default
#   max_features="sqrt" sklearn default for classification
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": SEED,
}
THRESHOLD = 0.5  # P(malicious) >= THRESHOLD means "malicious"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(rows, raw_dir, split):
    X, y = [], []
    for row in rows:
        if row["split"] != split:
            continue
        path = raw_dir / row["filename"]
        if not path.is_file():
            raise SystemExit(f"missing sample: {path}")
        if sha256_of(path) != row["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {path}; rebuild the manifest")
        X.append(to_vector(extract_features(path)))
        y.append(int(row["label"]))
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT)
    args = ap.parse_args()

    with open(args.data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("manifest.csv is empty; run build_manifest.py first")

    raw_dir = args.data_dir / "raw"
    X_train, y_train = load_split(rows, raw_dir, "train")
    X_test, y_test = load_split(rows, raw_dir, "test")

    model = RandomForestClassifier(**HYPERPARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    roc_auc = (float(roc_auc_score(y_test, proba))
               if len(set(y_test)) == 2 else None)  # undefined with one class

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "train": {"benign": int((y_train == 0).sum()), "malicious": int((y_train == 1).sum())},
            "test": {"benign": int((y_test == 0).sum()), "malicious": int((y_test == 1).sum())},
        },
        "model": "RandomForestClassifier",
        "hyperparameters": HYPERPARAMS,
        "threshold": THRESHOLD,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": roc_auc,
        },
        "confusion_matrix": {
            "labels": ["benign", "malicious"],
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "feature_importances": {name: round(float(v), 6) for name, v in importances},
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib")
    (args.out_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2))
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("counts", "test_metrics", "confusion_matrix")},
                     indent=2))
    print(f"saved model.joblib, feature_names.json, results.json to {args.out_dir}")


if __name__ == "__main__":
    main()
