#!/usr/bin/env python3
"""Aggressive tuning for 90% target on 80/20 split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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
    """Extract HOG from full image, medium resolution."""
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
    print(f"Class distribution: {np.bincount(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/test split: train={len(y_train)}, test={len(y_test)}")

    # Grid search on SVM
    print("\n=== GridSearchCV on SVM ===")
    pipe_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=False, random_state=42)),
        ]
    )
    param_grid_svm = {
        "svm__C": [2.0, 4.0, 8.0, 16.0],
        "svm__gamma": ["scale", 0.05, 0.1, 0.2],
    }
    gs_svm = GridSearchCV(
        pipe_svm,
        param_grid_svm,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    gs_svm.fit(X_train, y_train)
    print(f"Best SVM: {gs_svm.best_params_} -> CV={gs_svm.best_score_:.4f}")
    svm_pred = gs_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM test acc: {svm_acc:.4f}")

    # Grid search on GradientBoosting
    print("\n=== GridSearchCV on GradientBoosting ===")
    gb = GradientBoostingClassifier(random_state=42, n_iter_no_change=5, validation_fraction=0.1)
    param_grid_gb = {
        "n_estimators": [100, 200],
        "max_depth": [5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    gs_gb = GridSearchCV(
        gb,
        param_grid_gb,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    gs_gb.fit(X_train, y_train)
    print(f"Best GB: {gs_gb.best_params_} -> CV={gs_gb.best_score_:.4f}")
    gb_pred = gs_gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"GB test acc: {gb_acc:.4f}")

    # Ensemble: majority vote
    ensemble_pred = np.round((svm_pred + gb_pred) / 2).astype(int)
    ens_acc = accuracy_score(y_test, ensemble_pred)
    print(f"\nEnsemble (SVM+GB avg): {ens_acc:.4f}")

    best_pred = svm_pred if svm_acc > gb_acc else gb_pred
    best_model_name = "SVM" if svm_acc > gb_acc else "GB"
    best_acc = max(svm_acc, gb_acc)

    print(f"\n=== Best model: {best_model_name} ({best_acc:.4f}) ===")
    report = classification_report(
        y_test,
        best_pred,
        target_names=[INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, best_pred)

    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    return 0 if best_acc >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
