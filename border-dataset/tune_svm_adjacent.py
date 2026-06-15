#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABEL_TO_INT = {"light": 0, "medium": 1, "high": 2, "extreme": 3}


def load_rows(labels_path: Path, camera_id: str) -> list[dict]:
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
    rows = [r for r in rows if str(r.get("line_bucket", "")) in LABEL_TO_INT]
    rows = [r for r in rows if Path(str(r.get("file", ""))).exists()]
    return rows


def feat(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((96, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def ordinal_metrics(cm: np.ndarray) -> tuple[float, float]:
    n = cm.sum()
    mae = sum(abs(i - j) * cm[i, j] for i in range(4) for j in range(4)) / n
    within1 = sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / n
    return float(mae), float(within1)


def main() -> int:
    rows = load_rows(Path("border-dataset/data/labels/line_size_labels.jsonl"), "hak_bajakovo_entry")
    x = np.vstack([feat(Path(r["file"])) for r in rows])
    y = np.array([LABEL_TO_INT[r["line_bucket"]] for r in rows], dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    configs: list[tuple[float, float | str, str | dict[int, float]]] = []
    for c in [8, 10, 12]:
        for gamma in ["scale", 0.02]:
            for class_weight in [
                "balanced",
                {0: 1.0, 1: 1.3, 2: 1.0, 3: 1.0},
                {0: 1.0, 1: 1.5, 2: 1.1, 3: 1.0},
            ]:
                configs.append((c, gamma, class_weight))

    best = None
    for idx, (c, gamma, class_weight) in enumerate(configs, start=1):
        model = Pipeline(
            [
                ("sc", StandardScaler(with_mean=True)),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=c,
                        gamma=gamma,
                        class_weight=class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
        acc = float(accuracy_score(y_test, pred))
        mae, within1 = ordinal_metrics(cm)
        score = acc - 0.25 * mae
        rec_medium = float(cm[1, 1] / cm[1].sum())
        rec_high = float(cm[2, 2] / cm[2].sum())

        item = {
            "C": c,
            "gamma": gamma,
            "class_weight": class_weight,
            "accuracy": acc,
            "mae": mae,
            "within1": within1,
            "score": score,
            "recall_medium": rec_medium,
            "recall_high": rec_high,
            "confusion_matrix": cm.tolist(),
        }
        print(
            f"[{idx}/{len(configs)}] C={c} gamma={gamma} "
            f"acc={acc:.4f} mae={mae:.4f} within1={within1:.4f}"
        )
        if best is None or item["score"] > best["score"]:
            best = item

    baseline_cm = np.array([[48, 3, 1, 3], [6, 18, 16, 7], [0, 6, 59, 21], [0, 2, 14, 89]])
    b_acc = float(np.trace(baseline_cm) / baseline_cm.sum())
    b_mae, b_within1 = ordinal_metrics(baseline_cm)

    print("=== Best tuned SVM ===")
    print("config:", best["C"], best["gamma"], best["class_weight"])
    print(f"accuracy: {best['accuracy']:.4f}")
    print(f"mae: {best['mae']:.4f}")
    print(f"within1: {best['within1']:.4f}")
    print(f"recall_medium: {best['recall_medium']:.4f}")
    print(f"recall_high: {best['recall_high']:.4f}")
    print("confusion_matrix:")
    for row in best["confusion_matrix"]:
        print(row)

    print("\n=== Baseline reference ===")
    print(f"accuracy: {b_acc:.4f}")
    print(f"mae: {b_mae:.4f}")
    print(f"within1: {b_within1:.4f}")

    out = Path("border-dataset/models/tuned_svm_adjacent_summary.json")
    out.write_text(json.dumps({"best": best, "baseline": {"accuracy": b_acc, "mae": b_mae, "within1": b_within1}}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
