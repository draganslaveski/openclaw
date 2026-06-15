#!/usr/bin/env python3
"""Train and persist the current best queue model (ResNet18 ordinal + class weights)."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABEL_TO_INT = {"light": 0, "medium": 1, "high": 2, "extreme": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


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


class QueueDataset(Dataset):
    def __init__(self, rows: list[dict], transform: transforms.Compose):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(Path(row["file"])) as image:
            x = self.transform(image.convert("RGB"))
        y = LABEL_TO_INT[row["line_bucket"]]
        return x, y


def make_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 4)
    return model


def ordinal_penalty(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    class_idx = torch.arange(num_classes, device=logits.device).view(1, -1)
    target_idx = targets.view(-1, 1)
    dist = torch.abs(class_idx - target_idx).float()
    return (probs * dist).sum(dim=1).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    yt = np.array(y_true)
    yp = np.array(y_pred)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    acc = float((yt == yp).mean())
    abs_err = np.abs(yt - yp)
    mae = float(abs_err.mean())
    within1 = float((abs_err <= 1).mean())
    return acc, mae, within1, cm


def main() -> int:
    seed = 42
    camera_id = "hak_bajakovo_entry"
    test_size = 0.2
    epochs = 10
    batch_size = 32
    image_size = 160
    lr = 1.5e-4
    ordinal_lambda = 0.35

    set_seed(seed)

    repo_root = Path(__file__).resolve().parents[1]
    labels_path = repo_root / "border-dataset" / "data" / "labels" / "line_size_labels.jsonl"
    out_dir = repo_root / "border-dataset" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_latest_labels(labels_path, camera_id)
    y_all = np.array([LABEL_TO_INT[r["line_bucket"]] for r in rows], dtype=np.int64)
    train_rows, val_rows = train_test_split(rows, test_size=test_size, random_state=seed, stratify=y_all)

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

    train_ds = QueueDataset(train_rows, train_tf)
    val_ds = QueueDataset(val_rows, val_tf)

    y_train = np.array([LABEL_TO_INT[r["line_bucket"]] for r in train_rows], dtype=np.int64)
    class_counts = np.bincount(y_train, minlength=4)
    inv = 1.0 / np.maximum(class_counts, 1)
    inv = inv / inv.sum() * len(inv)
    class_weight = torch.tensor(inv, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_epoch = 0
    best_acc = -1.0
    best_mae = 0.0
    best_within1 = 0.0
    best_cm = np.zeros((4, 4), dtype=np.int64)

    print(f"Device: {device}")
    print(f"Samples: total={len(rows)} train={len(train_rows)} val={len(val_rows)}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce_loss(logits, yb) + ordinal_lambda * ordinal_penalty(logits, yb)
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
            best_mae = val_mae
            best_within1 = val_within1
            best_cm = val_cm
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise SystemExit("No model state captured.")

    checkpoint = {
        "camera_id": camera_id,
        "arch": "resnet18",
        "num_classes": 4,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "image_size": image_size,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }
    model_path = out_dir / "current_queue_model.pt"
    torch.save(checkpoint, model_path)

    meta = {
        "model_file": str(model_path),
        "camera_id": camera_id,
        "arch": "resnet18",
        "recipe": "ordinal_loss_plus_class_weights",
        "best_epoch": best_epoch,
        "val_accuracy": best_acc,
        "val_mae": best_mae,
        "val_within1": best_within1,
        "confusion_matrix": best_cm.tolist(),
        "class_order": ["light", "medium", "high", "extreme"],
        "seed": seed,
        "test_size": test_size,
        "num_samples_after_filter": len(rows),
    }
    meta_path = out_dir / "current_queue_model.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved current model:", model_path)
    print("Saved metadata:", meta_path)
    print("Best confusion matrix:")
    print(best_cm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
