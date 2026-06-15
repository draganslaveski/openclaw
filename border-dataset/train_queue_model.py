#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog


LABEL_TO_INT = {
    "light": 0,
    "medium": 1,
    "high": 2,
    "extreme": 3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class ExperimentResult:
    name: str
    cv_mean: float
    cv_std: float
    test_acc: float
    feature_kind: str
    crop_kind: str
    model_name: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train queue-size model from labeled border camera samples.")
    parser.add_argument(
        "--labels",
        default=str(repo_root / "border-dataset" / "data" / "labels" / "line_size_labels.jsonl"),
        help="Path to labels JSONL file",
    )
    parser.add_argument(
        "--camera-id",
        default="hak_bajakovo_entry",
        help="Camera ID to train on",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller experiment set without hyperparameter tuning",
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "border-dataset" / "models"),
        help="Output directory for model and reports",
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


def crop_image(image: Image.Image, crop_kind: str) -> Image.Image:
    width, height = image.size
    if crop_kind == "full":
        return image
    if crop_kind == "lower60":
        top = int(height * 0.40)
        return image.crop((0, top, width, height))
    raise ValueError(f"Unknown crop_kind: {crop_kind}")


def extract_raw_feature(path: Path, crop_kind: str, size: tuple[int, int] = (40, 40)) -> np.ndarray:
    with Image.open(path) as img:
        gray = crop_image(img.convert("L"), crop_kind)
        arr = np.asarray(gray.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return arr.reshape(-1)


def extract_hog_feature(path: Path, crop_kind: str, size: tuple[int, int] = (96, 48)) -> np.ndarray:
    with Image.open(path) as img:
        gray = crop_image(img.convert("L"), crop_kind)
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


def build_features(rows: list[dict], feature_fn: Callable[[Path, str], np.ndarray], crop_kind: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    ids: list[str] = []

    for row in rows:
        path = Path(row["file"])
        x_list.append(feature_fn(path, crop_kind))
        y_list.append(LABEL_TO_INT[row["line_bucket"]])
        ids.append(row["sample_id"])

    return np.vstack(x_list), np.asarray(y_list, dtype=np.int64), ids


def run_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    estimator,
    name: str,
    feature_kind: str,
    crop_kind: str,
    model_name: str,
    seed: int,
    n_splits: int,
) -> ExperimentResult:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(estimator, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

    fitted = clone(estimator)
    fitted.fit(x_train, y_train)
    test_pred = fitted.predict(x_test)
    test_acc = float(accuracy_score(y_test, test_pred))

    return ExperimentResult(
        name=name,
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        test_acc=test_acc,
        feature_kind=feature_kind,
        crop_kind=crop_kind,
        model_name=model_name,
    )


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_latest_labels(labels_path, args.camera_id)
    if not rows:
        raise SystemExit("No usable labeled rows found.")

    splits_written = []

    feature_builders: dict[str, Callable[[Path, str], np.ndarray]] = {
        "raw40": extract_raw_feature,
        "hog": extract_hog_feature,
    }
    crop_options = ["full", "lower60"]

    estimators = {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        C=1.0,
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "model",
                    LinearSVC(
                        C=1.5,
                        class_weight="balanced",
                        random_state=args.seed,
                        max_iter=6000,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=8.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=False,
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=1,
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=10,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            random_state=args.seed,
        ),
    }

    if args.quick:
        estimators = {
            "linear_svm": estimators["linear_svm"],
            "svm_rbf": estimators["svm_rbf"],
            "rf": estimators["rf"],
        }

    results: list[ExperimentResult] = []
    trained_models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], object]] = {}

    for feature_kind, feature_fn in feature_builders.items():
        for crop_kind in crop_options:
            x_all, y_all, sample_ids = build_features(rows, feature_fn, crop_kind)
            sample_ids_np = np.asarray(sample_ids)

            x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
                x_all,
                y_all,
                sample_ids_np,
                test_size=args.test_size,
                random_state=args.seed,
                stratify=y_all,
            )

            split_tag = f"{feature_kind}_{crop_kind}"
            split_json = {
                "camera_id": args.camera_id,
                "feature_kind": feature_kind,
                "crop_kind": crop_kind,
                "test_size": args.test_size,
                "seed": args.seed,
                "train_ids": ids_train.tolist(),
                "test_ids": ids_test.tolist(),
            }
            split_path = out_dir / f"split_{split_tag}.json"
            split_path.write_text(json.dumps(split_json, indent=2), encoding="utf-8")
            splits_written.append(str(split_path))

            for model_name, estimator in estimators.items():
                name = f"{feature_kind}+{crop_kind}+{model_name}"
                result = run_experiment(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    estimator,
                    name=name,
                    feature_kind=feature_kind,
                    crop_kind=crop_kind,
                    model_name=model_name,
                    seed=args.seed,
                    n_splits=3 if args.quick else 5,
                )
                results.append(result)
                trained_models[name] = (x_train, y_train, x_test, y_test, ids_train.tolist(), ids_test.tolist(), estimator)
                print(f"[EXP] {name} cv={result.cv_mean:.4f}±{result.cv_std:.4f} test={result.test_acc:.4f}")

    results.sort(key=lambda r: (r.test_acc, r.cv_mean), reverse=True)

    best = results[0]
    print(f"\nBest baseline: {best.name} test={best.test_acc:.4f} cv={best.cv_mean:.4f}")

    x_train, y_train, x_test, y_test, ids_train, ids_test, best_estimator = trained_models[best.name]

    tune_search = None
    tuned_model = None
    if best.model_name == "svm_rbf":
        param_dist = {
            "model__C": [1, 2, 4, 8, 12, 16],
            "model__gamma": ["scale", 0.1, 0.05, 0.02, 0.01],
        }
        tune_search = RandomizedSearchCV(
            estimator=best_estimator,
            param_distributions=param_dist,
            n_iter=10,
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
            n_jobs=1,
            random_state=args.seed,
            verbose=1,
        )
    elif best.model_name == "logreg":
        param_dist = {
            "model__C": [0.25, 0.5, 1, 2, 4, 8],
        }
        tune_search = RandomizedSearchCV(
            estimator=best_estimator,
            param_distributions=param_dist,
            n_iter=6,
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
            n_jobs=1,
            random_state=args.seed,
            verbose=1,
        )
    elif best.model_name == "rf":
        param_dist = {
            "n_estimators": [200, 400, 700],
            "max_depth": [None, 12, 16, 22],
            "min_samples_leaf": [2, 3, 4, 6],
            "max_features": ["sqrt", 0.5, 0.7],
        }
        tune_search = RandomizedSearchCV(
            estimator=best_estimator,
            param_distributions=param_dist,
            n_iter=12,
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
            n_jobs=1,
            random_state=args.seed,
            verbose=1,
        )
    elif best.model_name == "hgb":
        param_dist = {
            "learning_rate": [0.03, 0.05, 0.08, 0.12],
            "max_depth": [6, 8, 10, 12],
            "max_leaf_nodes": [31, 63, 127],
            "min_samples_leaf": [15, 20, 30, 40],
        }
        tune_search = RandomizedSearchCV(
            estimator=best_estimator,
            param_distributions=param_dist,
            n_iter=12,
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
            n_jobs=1,
            random_state=args.seed,
            verbose=1,
        )

    if args.quick:
        tuned_model = clone(best_estimator).fit(x_train, y_train)
    elif tune_search is not None:
        tune_search.fit(x_train, y_train)
        tuned_model = tune_search.best_estimator_
        print(f"Tuned best CV={tune_search.best_score_:.4f} params={tune_search.best_params_}")
    else:
        tuned_model = clone(best_estimator).fit(x_train, y_train)

    final_model = tuned_model
    final_model.fit(x_train, y_train)
    y_pred = final_model.predict(x_test)

    test_acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        target_names=[INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    model_path = out_dir / "queue_model_best.joblib"
    joblib.dump(
        {
            "camera_id": args.camera_id,
            "label_to_int": LABEL_TO_INT,
            "int_to_label": INT_TO_LABEL,
            "model": final_model,
            "best_baseline": best.__dict__,
        },
        model_path,
    )

    summary = {
        "camera_id": args.camera_id,
        "num_samples_after_filter": len(rows),
        "test_size": args.test_size,
        "seed": args.seed,
        "best_baseline": best.__dict__,
        "test_accuracy_final": test_acc,
        "model_path": str(model_path),
        "split_files": splits_written,
    }

    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report + "\n\nConfusion matrix:\n" + np.array2string(cm), encoding="utf-8")

    all_results_path = out_dir / "experiment_results.json"
    all_results_path.write_text(
        json.dumps([r.__dict__ for r in results], indent=2),
        encoding="utf-8",
    )

    print("\n=== Final Evaluation ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(report)
    print("Confusion matrix:")
    print(cm)
    print(f"\nSaved model: {model_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved all experiments: {all_results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
