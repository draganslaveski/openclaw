#!/usr/bin/env python3
"""Fast ROI-focused queue line classifier with 80/20 split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog


LABEL_MAP = {"light": 0, "medium": 1, "high": 2, "extreme": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_and_deduplicate_labels(labels_path: Path, camera_id: str) -> list[dict]:
    """Load labels, keep only latest per sample."""
    by_sample: dict[str, dict] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("camera_id") != camera_id:
            continue

        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue

        old = by_sample.get(sample_id)
        if old is None or str(row.get("labeled_at", "")) > str(old.get("labeled_at", "")):
            by_sample[sample_id] = row

    rows = list(by_sample.values())
    rows = [r for r in rows if bool(r.get("is_usable", True))]
    rows = [r for r in rows if str(r.get("line_bucket", "")) in LABEL_MAP]
    rows = [r for r in rows if Path(str(r.get("file", ""))).exists()]
    return rows


def extract_roi_raw(img_path: Path, roi_fraction: float = 0.4) -> np.ndarray:
    """Extract bottom ROI as raw pixels, resized to 32x32."""
    with Image.open(img_path) as im:
        g = im.convert("L")
        w, h = g.size
        top = int(h * (1 - roi_fraction))
        roi = g.crop((0, top, w, h))
        arr = np.asarray(roi.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return arr.reshape(-1)


def extract_roi_hog(img_path: Path, roi_fraction: float = 0.4) -> np.ndarray:
    """Extract bottom ROI HOG features."""
    with Image.open(img_path) as im:
        g = im.convert("L")
        w, h = g.size
        top = int(h * (1 - roi_fraction))
        roi = g.crop((0, top, w, h))
        arr = np.asarray(roi.resize((64, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hf = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return hf.astype(np.float32)


def main() -> int:
    labels_path = Path("border-dataset/data/labels/line_size_labels.jsonl")
    rows = load_and_deduplicate_labels(labels_path, "hak_bajakovo_entry")
    print(f"Loaded {len(rows)} deduped, usable samples")

    # Extract features
    X_raw = []
    X_hog = []
    y = []
    for r in rows:
        path = Path(r["file"])
        X_raw.append(extract_roi_raw(path))
        X_hog.append(extract_roi_hog(path))
        y.append(LABEL_MAP[r["line_bucket"]])

    X_raw = np.vstack(X_raw)
    X_hog = np.vstack(X_hog)
    y = np.array(y)
    print(f"Feature matrix: raw={X_raw.shape} hog={X_hog.shape} labels={y.shape}")

    # Split
    Xr_tr, Xr_te, Xh_tr, Xh_te, y_tr, y_te = train_test_split(
        X_raw, X_hog, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/test split: train={len(y_tr)}/{len(y_te)}")

    # Build ensemble
    m1 = Pipeline(
        [
            ("sc", StandardScaler()),
            ("svm", LinearSVC(C=2.0, class_weight="balanced", max_iter=8000, random_state=42)),
        ]
    )
    m2 = Pipeline(
        [
            ("sc", StandardScaler()),
            ("svm", LinearSVC(C=1.5, class_weight="balanced", max_iter=8000, random_state=42)),
        ]
    )
    m3 = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=1)
    m4 = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, max_depth=20, class_weight="balanced_subsample", random_state=42, n_jobs=1)

    # Fit individually
    m1.fit(Xr_tr, y_tr)
    m2.fit(Xh_tr, y_tr)
    m3.fit(Xr_tr, y_tr)
    m4.fit(Xh_tr, y_tr)

    # Get predictions on test
    p1 = m1.predict(Xr_te)
    p2 = m2.predict(Xh_te)
    p3 = m3.predict(Xr_te)
    p4 = m4.predict(Xh_te)

    # Majority vote
    ensemble_pred = np.round(np.mean([p1, p2, p3, p4], axis=0)).astype(int)
    ensemble_pred = np.clip(ensemble_pred, 0, 3)

    acc1 = accuracy_score(y_te, p1)
    acc2 = accuracy_score(y_te, p2)
    acc3 = accuracy_score(y_te, p3)
    acc4 = accuracy_score(y_te, p4)
    ens_acc = accuracy_score(y_te, ensemble_pred)

    print(f"\nModel accuracies on test set (20%={len(y_te)} samples):")
    print(f"  raw+svm_c2.0:       {acc1:.4f}")
    print(f"  hog+svm_c1.5:       {acc2:.4f}")
    print(f"  raw+rf300:          {acc3:.4f}")
    print(f"  hog+rf250:          {acc4:.4f}")
    print(f"  ensemble (4-way):   {ens_acc:.4f}")

    report = classification_report(
        y_te,
        ensemble_pred,
        target_names=[INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_te, ensemble_pred)

    print("\nClassification report (ensemble):")
    print(report)
    print("Confusion matrix:")
    print(cm)

    return 0 if ens_acc >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
