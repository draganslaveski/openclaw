#!/usr/bin/env python3
"""Run CNN experiments for ordered queue-size classes.

Experiments covered:
1) CNN baseline with class balancing
2) CNN with ordinal-aware distance penalty
3) Stronger architecture with ordinal-aware loss

Outputs JSON summary + per-experiment confusion matrices.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

LABEL_TO_INT = {"light": 0, "medium": 1, "high": 2, "extreme": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class ExperimentConfig:
    name: str
    arch: str
    epochs: int
    lr: float
    batch_size: int
    image_size: int
    use_weighted_sampler: bool
    use_class_weight: bool
    ordinal_lambda: float


@dataclass
class ExperimentResult:
    name: str
    arch: str
    best_epoch: int
    val_accuracy: float
    val_mae: float
    val_within1: float
    confusion_matrix: list[list[int]]


class QueueImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform: transforms.Compose):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = Path(row["file"])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            x = self.transform(rgb)
        y = LABEL_TO_INT[row["line_bucket"]]
        return x, y


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CNN + ordinal queue model experiments")
    parser.add_argument(
        "--labels",
        default=str(repo_root / "border-dataset" / "data" / "labels" / "line_size_labels.jsonl"),
        help="Path to labels JSONL",
    )
    parser.add_argument("--camera-id", default="hak_bajakovo_entry", help="Camera ID")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "border-dataset" / "models"),
        help="Output directory",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def make_loaders(
    train_rows: list[dict],
    val_rows: list[dict],
    cfg: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    train_tf, val_tf = build_transforms(cfg.image_size)
    train_ds = QueueImageDataset(train_rows, train_tf)
    val_ds = QueueImageDataset(val_rows, val_tf)

    y_train = np.array([LABEL_TO_INT[r["line_bucket"]] for r in train_rows], dtype=np.int64)
    class_counts = np.bincount(y_train, minlength=4)

    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        sample_weights = np.array([1.0 / class_counts[y] for y in y_train], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, class_counts


def ordinal_penalty(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    device = logits.device
    class_idx = torch.arange(num_classes, device=device).view(1, -1)
    target_idx = targets.view(-1, 1)
    dist = torch.abs(class_idx - target_idx).float()
    # Expected class distance under predicted distribution
    return (probs * dist).sum(dim=1).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            all_true.extend(yb.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    acc = float((y_true == y_pred).mean())
    abs_err = np.abs(y_true - y_pred)
    mae = float(abs_err.mean())
    within1 = float((abs_err <= 1).mean())
    return acc, mae, within1, cm


def run_experiment(
    cfg: ExperimentConfig,
    train_rows: list[dict],
    val_rows: list[dict],
    seed: int,
    device: torch.device,
) -> ExperimentResult:
    set_seed(seed)
    train_loader, val_loader, class_counts = make_loaders(train_rows, val_rows, cfg)

    model = make_model(cfg.arch).to(device)

    class_weight = None
    if cfg.use_class_weight:
        inv = 1.0 / np.maximum(class_counts, 1)
        inv = inv / inv.sum() * len(inv)
        class_weight = torch.tensor(inv, dtype=torch.float32, device=device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_state = None
    best_acc = -1.0
    best_epoch = 0
    best_metrics = (0.0, 0.0, 0.0, np.zeros((4, 4), dtype=np.int64))

    print(f"\n=== Running {cfg.name} ({cfg.arch}) ===")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce_loss(logits, yb)
            if cfg.ordinal_lambda > 0:
                loss = loss + cfg.ordinal_lambda * ordinal_penalty(logits, yb, num_classes=4)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        val_acc, val_mae, val_within1, val_cm = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(n_batches, 1)
        print(
            f"epoch={epoch:02d} loss={avg_loss:.4f} "
            f"val_acc={val_acc:.4f} val_mae={val_mae:.4f} val_within1={val_within1:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = (val_acc, val_mae, val_within1, val_cm)

    if best_state is not None:
        model.load_state_dict(best_state)

    best_acc, best_mae, best_within1, best_cm = best_metrics
    return ExperimentResult(
        name=cfg.name,
        arch=cfg.arch,
        best_epoch=best_epoch,
        val_accuracy=best_acc,
        val_mae=best_mae,
        val_within1=best_within1,
        confusion_matrix=best_cm.tolist(),
    )


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_latest_labels(labels_path, args.camera_id)
    if not rows:
        raise SystemExit("No usable labeled rows found.")

    y_all = np.array([LABEL_TO_INT[r["line_bucket"]] for r in rows], dtype=np.int64)
    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples: total={len(rows)} train={len(train_rows)} val={len(val_rows)}")

    experiments = [
        ExperimentConfig(
            name="cnn_baseline_balanced",
            arch="mobilenet_v3_small",
            epochs=8,
            lr=2e-4,
            batch_size=32,
            image_size=160,
            use_weighted_sampler=True,
            use_class_weight=True,
            ordinal_lambda=0.0,
        ),
        ExperimentConfig(
            name="cnn_ordinal_balanced",
            arch="mobilenet_v3_small",
            epochs=8,
            lr=2e-4,
            batch_size=32,
            image_size=160,
            use_weighted_sampler=True,
            use_class_weight=True,
            ordinal_lambda=0.35,
        ),
        ExperimentConfig(
            name="resnet18_ordinal_balanced",
            arch="resnet18",
            epochs=8,
            lr=1.5e-4,
            batch_size=32,
            image_size=160,
            use_weighted_sampler=True,
            use_class_weight=True,
            ordinal_lambda=0.35,
        ),
        ExperimentConfig(
            name="resnet18_ordinal_balanced_long",
            arch="resnet18",
            epochs=12,
            lr=1.0e-4,
            batch_size=32,
            image_size=160,
            use_weighted_sampler=True,
            use_class_weight=True,
            ordinal_lambda=0.35,
        ),
        ExperimentConfig(
            name="resnet18_ordinal_weighted_only",
            arch="resnet18",
            epochs=10,
            lr=1.5e-4,
            batch_size=32,
            image_size=160,
            use_weighted_sampler=False,
            use_class_weight=True,
            ordinal_lambda=0.35,
        ),
        ExperimentConfig(
            name="resnet18_ordinal_sampler_only",
            arch="resnet18",
            epochs=10,
            lr=1.5e-4,
            batch_size=32,
            image_size=160,
            use_weighted_sampler=True,
            use_class_weight=False,
            ordinal_lambda=0.35,
        ),
    ]

    results: list[ExperimentResult] = []
    for cfg in experiments:
        res = run_experiment(cfg, train_rows, val_rows, args.seed, device)
        results.append(res)

    results_sorted = sorted(results, key=lambda r: (r.val_accuracy, -r.val_mae), reverse=True)
    best = results_sorted[0]

    print("\n=== Ranked Results ===")
    for r in results_sorted:
        print(
            f"{r.name:28s} acc={r.val_accuracy:.4f} "
            f"mae={r.val_mae:.4f} within1={r.val_within1:.4f} best_epoch={r.best_epoch}"
        )

    print("\nBest experiment:")
    print(
        f"{best.name} ({best.arch}) acc={best.val_accuracy:.4f} "
        f"mae={best.val_mae:.4f} within1={best.val_within1:.4f}"
    )
    print("Confusion matrix:")
    for row in best.confusion_matrix:
        print(row)

    summary = {
        "camera_id": args.camera_id,
        "seed": args.seed,
        "test_size": args.test_size,
        "num_samples_after_filter": len(rows),
        "results": [asdict(r) for r in results_sorted],
        "best": asdict(best),
    }

    out_json = out_dir / "cnn_ordinal_experiments_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {out_json}")

    # Keep CLI status aligned with user's 90% goal.
    return 0 if best.val_accuracy >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
