#!/usr/bin/env python3
"""Train an ordinal queue-size classifier for hak_bajakovo_entry.

This models ordered classes via three cumulative binary models:
- P(y > 0), P(y > 1), P(y > 2)
Then reconstructs class probabilities for 4 ordered classes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


LABEL_TO_INT = {
    "light": 0,
    "medium": 1,
    "high": 2,
    "extreme": 3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class OrdinalMetrics:
    accuracy: float
    mae: float
    off_by_1_or_less: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train ordinal queue-size classifier")
    parser.add_argument(
        "--labels",
        default=str(repo_root / "border-dataset" / "data" / "labels" / "line_size_labels.jsonl"),
        help="Path to labels JSONL",
    )
    parser.add_argument("--camera-id", default="hak_bajakovo_entry", help="Camera ID")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "border-dataset" / "models"),
        help="Output directory",
    )
    return parser.parse_args()


def load_latest_labels(labels_path: Path, camera_id: str) -> list[dict]:
    by_sample: dict[str, dict] = {}
    for raw_line in labels_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
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


def extract_hog_feature(path: Path, size: tuple[int, int] = (96, 48)) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    feats = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)


def build_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    ids: list[str] = []

    for row in rows:
        path = Path(row["file"])
        x_list.append(extract_hog_feature(path))
        y_list.append(LABEL_TO_INT[row["line_bucket"]])
        ids.append(row["sample_id"])

    return np.vstack(x_list), np.asarray(y_list, dtype=np.int64), np.asarray(ids)


class OrdinalLogit:
    """Cumulative-link style ordinal model built from binary logits."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: list[Pipeline] = []

    def _make_model(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=2.0,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OrdinalLogit":
        self.models = []
        for threshold in (0, 1, 2):
            target = (y > threshold).astype(np.int64)
            model = self._make_model()
            model.fit(x, target)
            self.models.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model is not fit")

        # Cumulative probabilities p_k = P(y > k)
        p = np.column_stack([m.predict_proba(x)[:, 1] for m in self.models])

        # Enforce monotonicity p0 >= p1 >= p2 to keep valid ordinal probabilities.
        p[:, 1] = np.minimum(p[:, 1], p[:, 0])
        p[:, 2] = np.minimum(p[:, 2], p[:, 1])

        probs = np.zeros((x.shape[0], 4), dtype=np.float64)
        probs[:, 0] = 1.0 - p[:, 0]
        probs[:, 1] = p[:, 0] - p[:, 1]
        probs[:, 2] = p[:, 1] - p[:, 2]
        probs[:, 3] = p[:, 2]

        # Numerical stability
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> OrdinalMetrics:
    abs_err = np.abs(y_true - y_pred)
    return OrdinalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        mae=float(np.mean(abs_err)),
        off_by_1_or_less=float(np.mean(abs_err <= 1)),
    )


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_latest_labels(labels_path, args.camera_id)
    if not rows:
        raise SystemExit("No usable labeled rows found.")

    x_all, y_all, sample_ids = build_features(rows)

    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
        x_all,
        y_all,
        sample_ids,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all,
    )

    model = OrdinalLogit(random_state=args.seed).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("=== Ordinal Model Evaluation ===")
    print(f"Samples: {len(rows)}")
    print(f"Split: train={len(y_train)} test={len(y_test)}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"MAE (class distance): {metrics.mae:.4f}")
    print(f"Within 1 category: {metrics.off_by_1_or_less:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    joblib.dump(
        {
            "camera_id": args.camera_id,
            "label_to_int": LABEL_TO_INT,
            "int_to_label": INT_TO_LABEL,
            "model": model,
        },
        out_dir / "queue_model_ordinal.joblib",
    )

    summary = {
        "camera_id": args.camera_id,
        "num_samples_after_filter": len(rows),
        "test_size": args.test_size,
        "seed": args.seed,
        "accuracy": metrics.accuracy,
        "mae": metrics.mae,
        "off_by_1_or_less": metrics.off_by_1_or_less,
        "confusion_matrix": cm.tolist(),
        "class_order": ["light", "medium", "high", "extreme"],
        "train_ids": ids_train.tolist(),
        "test_ids": ids_test.tolist(),
    }
    (out_dir / "training_summary_ordinal.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    (out_dir / "classification_report_ordinal.txt").write_text(
        report + "\n\nConfusion matrix:\n" + np.array2string(cm),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
