#!/usr/bin/env python3
"""Fast ensemble combining best classifiers for 90% target."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


def extract_hog_full(img_path: Path) -> np.ndarray:
    """Extract HOG from full image."""
    with Image.open(img_path) as im:
        g = im.convert("L")
        arr = np.asarray(g.resize((128, 96), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hf = hog(
        arr,
        orientations=12,
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
    X = []
    y = []
    for r in rows:
        path = Path(r["file"])
        X.append(extract_hog_full(path))
        y.append(LABEL_MAP[r["line_bucket"]])

    X = np.vstack(X)
    y = np.array(y)
    print(f"Feature matrix: {X.shape}, labels: {y.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/test split: train={len(y_train)}, test={len(y_test)}\n")

    # Estimators from results
    estimators = [
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced", probability=True, random_state=42))
        ])),
        ("gb", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
            n_iter_no_change=5,
            validation_fraction=0.1
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )),
        ("gb2", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=9,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_iter_no_change=5,
            validation_fraction=0.1
        )),
    ]

    # Voting ensemble
    print("Training VotingClassifier...")
    vc = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    vc.fit(X_train, y_train)

    # Predictions
    y_pred = vc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nVotingClassifier test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=[INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)],
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Per-model accuracy
    print("\n=== Per-model accuracies ===")
    for name, est in estimators:
        if hasattr(est, "predict"):
            pred = est.predict(X_test) if name == "svm" else est.predict(X_test)
        else:
            pred = est.predict(X_test)
        model_acc = accuracy_score(y_test, pred)
        print(f"{name:8s}: {model_acc:.4f}")

    return 0 if acc >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
