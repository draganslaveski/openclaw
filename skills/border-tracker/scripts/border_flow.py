#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
from PIL import Image
from skimage.feature import hog

LABELS = ["light", "medium", "high", "extreme"]
SYSTEM_CRON_TAG_PREFIX = "OPENCLAW_BORDER_MONITOR"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def iso_to_filename_stamp(iso_ts: str) -> str:
    ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return ts.strftime("%Y%m%dT%H%M%S")


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip()).strip("-") or "unknown"


def _norm(text: str) -> str:
    return text.strip().lower()


def _build_capture_command(camera_name: str, camera_id: str) -> str:
    python_exe = "/home/dragan-slaveski/.openclaw/.venv/bin/python"
    return " ".join(
        [
            shlex.quote(python_exe),
            "/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py",
            "capture-snapshot",
            "--flow-name",
            "monitor",
            "--camera",
            shlex.quote(camera_name),
            "--cameras-file",
            "/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json",
            "--snapshots-dir",
            "/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshots",
            "--snapshot-index-file",
            "/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl",
            "--output-json",
            f"/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/latest_capture_{safe_name(camera_id)}.json",
        ]
    )


def _load_user_crontab_lines() -> list[str]:
    proc = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if proc.returncode == 0:
        text = proc.stdout
    else:
        stderr = (proc.stderr or "").lower()
        if "no crontab" in stderr:
            return []
        raise SystemExit(f"Failed to read user crontab: {proc.stderr.strip() or 'unknown error'}")
    return text.splitlines()


def _write_user_crontab_lines(lines: list[str]) -> None:
    text = "\n".join(lines).rstrip("\n") + "\n"
    proc = subprocess.run(["crontab", "-"], input=text, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"Failed to update user crontab: {proc.stderr.strip() or 'unknown error'}")


def _system_cron_tag_for_camera(camera_id: str) -> str:
    return f"{SYSTEM_CRON_TAG_PREFIX}:{safe_name(camera_id)}"


def _strip_system_cron_entries(lines: list[str], tag_markers: set[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        if any(marker in line for marker in tag_markers):
            continue
        kept.append(line)
    return kept


def load_cameras(cameras_file: Path, camera_selector: str) -> list[dict[str, Any]]:
    data = json.loads(cameras_file.read_text(encoding="utf-8"))
    cameras = [c for c in data.get("cameras", []) if c.get("enabled", True)]
    selector = _norm(camera_selector)
    if selector == "all":
        return cameras

    by_id = [c for c in cameras if _norm(str(c.get("id", ""))) == selector]
    if by_id:
        return by_id

    by_name = [c for c in cameras if _norm(str(c.get("name", ""))) == selector]
    if by_name:
        return by_name

    # Allow convenient partial name matching if unambiguous.
    partial = [c for c in cameras if selector in _norm(str(c.get("name", "")))]
    if len(partial) == 1:
        return partial
    if len(partial) > 1:
        names = ", ".join(str(c.get("name", c.get("id", "unknown"))) for c in partial)
        raise SystemExit(f"Camera selector '{camera_selector}' matched multiple cameras: {names}")

    return []


def fetch_snapshot(url: str, timeout_sec: int) -> Image.Image:
    response = requests.get(url, timeout=timeout_sec, headers={"User-Agent": "BorderFlow/1.0"})
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as tmp:
        return tmp.convert("RGB")


def is_unavailable_placeholder(image: Image.Image) -> bool:
    # HAK unavailable frames are mostly uniform color with centered white text.
    arr = np.asarray(image.resize((320, 180), Image.Resampling.BILINEAR), dtype=np.uint8)
    body = arr[18:, :, :]

    median_color = np.median(body.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(body.astype(np.float32) - median_color.astype(np.float32), axis=2)
    flat_ratio = float(np.mean(dist < 24.0))

    h, w, _ = arr.shape
    center = arr[int(h * 0.35) : int(h * 0.65), int(w * 0.2) : int(w * 0.8), :]
    center_med = np.median(center.reshape(-1, 3), axis=0)
    center_dist = np.linalg.norm(center.astype(np.float32) - center_med.astype(np.float32), axis=2)
    center_flat_ratio = float(np.mean(center_dist < 20.0))
    white_mask = (center[:, :, 0] > 220) & (center[:, :, 1] > 220) & (center[:, :, 2] > 220)
    white_ratio = float(np.mean(white_mask))

    return flat_ratio > 0.92 and center_flat_ratio > 0.88 and white_ratio > 0.001


class CameraUnavailableError(RuntimeError):
    pass


def to_hog_feature(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    arr = np.asarray(gray.resize((96, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    vec = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return vec.astype(np.float32)


@dataclass
class Prediction:
    label: str
    score: float | None
    model_name: str


class QueuePredictor:
    def __init__(self, model_root: Path):
        self.model_root = model_root
        self.joblib_model_path = model_root / "queue_model_best.joblib"
        self.torch_model_path = model_root / "current_queue_model.pt"
        self._mode = "none"
        self._joblib_model = None
        self._torch_model = None
        self._torch_device = None

        self._try_load_torch_model()
        if self._mode == "none":
            self._try_load_joblib_model()

    def _try_load_torch_model(self) -> None:
        if not self.torch_model_path.exists():
            return
        try:
            import torch
            from torchvision import models

            ckpt = torch.load(self.torch_model_path, map_location="cpu")
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 4)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            self._torch_model = model
            self._torch_device = torch.device("cpu")
            self._mode = "torch"
        except Exception:
            self._mode = "none"

    def _try_load_joblib_model(self) -> None:
        if not self.joblib_model_path.exists():
            return
        self._joblib_model = joblib.load(self.joblib_model_path)
        self._mode = "joblib"

    def available(self) -> bool:
        return self._mode in {"torch", "joblib"}

    def predict(self, image: Image.Image) -> Prediction:
        if self._mode == "torch":
            return self._predict_torch(image)
        if self._mode == "joblib":
            return self._predict_joblib(image)
        raise RuntimeError("No model available. Expected current_queue_model.pt or queue_model_best.joblib")

    def _predict_joblib(self, image: Image.Image) -> Prediction:
        x = to_hog_feature(image).reshape(1, -1)
        pred = int(self._joblib_model.predict(x)[0])
        score = None
        if hasattr(self._joblib_model, "decision_function"):
            raw = self._joblib_model.decision_function(x)
            if isinstance(raw, np.ndarray):
                score = float(np.max(raw))
        return Prediction(label=LABELS[pred], score=score, model_name="queue_model_best.joblib")

    def _predict_torch(self, image: Image.Image) -> Prediction:
        import torch

        arr = np.asarray(image.resize((160, 160), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).unsqueeze(0)

        with torch.no_grad():
            logits = self._torch_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            score = float(np.max(probs))

        return Prediction(label=LABELS[pred], score=score, model_name="current_queue_model.pt")


def append_history(history_file: Path, row: dict[str, Any]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=True) for r in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def run_status(args: argparse.Namespace) -> int:
    cameras = load_cameras(args.cameras_file, args.camera)
    if not cameras:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")

    predictor = QueuePredictor(args.models_dir)
    if not predictor.available():
        raise SystemExit("No model file found in models dir")

    results: list[dict[str, Any]] = []
    for cam in cameras:
        cam_id = str(cam["id"])
        cam_name = str(cam.get("name", cam_id))
        url = str(cam["url"])
        captured_at = now_iso()
        try:
            image = fetch_snapshot(url, args.timeout_sec)
            if is_unavailable_placeholder(image):
                raise CameraUnavailableError("camera image unavailable")
            pred = predictor.predict(image)
            row = {
                "captured_at": captured_at,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "url": url,
                "status": "ok",
                "line_bucket": pred.label,
                "score": pred.score,
                "model": pred.model_name,
                "flow": args.flow_name,
            }
            results.append(row)
            if args.history_file:
                append_history(args.history_file, row)
            if args.save_debug_dir:
                args.save_debug_dir.mkdir(parents=True, exist_ok=True)
                out = args.save_debug_dir / f"{safe_name(cam_id)}-{iso_to_filename_stamp(captured_at)}.jpg"
                image.save(out, format="JPEG", quality=90)
                row["snapshot_file"] = str(out)
        except CameraUnavailableError as exc:
            row = {
                "captured_at": captured_at,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "url": url,
                "status": "unavailable",
                "unavailable_reason": str(exc),
                "flow": args.flow_name,
            }
            results.append(row)
            if args.history_file:
                append_history(args.history_file, row)
        except Exception as exc:
            row = {
                "captured_at": captured_at,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "url": url,
                "status": "error",
                "error": str(exc),
                "flow": args.flow_name,
            }
            results.append(row)
            if args.history_file:
                append_history(args.history_file, row)

    ok = [r for r in results if r["status"] == "ok"]
    unavailable = [r for r in results if r["status"] == "unavailable"]
    err = [r for r in results if r["status"] == "error"]
    print(f"BORDER STATUS ({args.flow_name})")
    print(f"Checked: {len(results)} camera(s), ok={len(ok)}, unavailable={len(unavailable)}, error={len(err)}")
    for row in ok:
        score_txt = "" if row.get("score") is None else f" ({row['score']:.3f})"
        print(f"- {row['camera_name']} [{row['camera_id']}]: {row['line_bucket']}{score_txt}")
        if row.get("snapshot_file"):
            print(f"  snapshot_file: {row['snapshot_file']}")
        else:
            print("  snapshot_file: (not saved; run with --save-debug-dir)")
    for row in unavailable:
        print(f"- {row['camera_name']} [{row['camera_id']}]: UNAVAILABLE - {row.get('unavailable_reason', 'camera image unavailable')}")
    for row in err:
        print(f"- {row['camera_name']} [{row['camera_id']}]: ERROR - {row['error']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": now_iso(),
            "flow": args.flow_name,
            "camera": args.camera,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if (ok or unavailable) else 1


def run_capture_snapshot(args: argparse.Namespace) -> int:
    cameras = load_cameras(args.cameras_file, args.camera)
    if not cameras:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")

    args.snapshots_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for cam in cameras:
        cam_id = str(cam["id"])
        cam_name = str(cam.get("name", cam_id))
        url = str(cam["url"])
        captured_at = now_iso()

        try:
            image = fetch_snapshot(url, args.timeout_sec)
            if is_unavailable_placeholder(image):
                raise CameraUnavailableError("camera image unavailable")
            out = args.snapshots_dir / f"{safe_name(cam_id)}-{iso_to_filename_stamp(captured_at)}.jpg"
            image.save(out, format="JPEG", quality=90)

            row = {
                "captured_at": captured_at,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "url": url,
                "status": "ok",
                "snapshot_file": str(out),
                "flow": args.flow_name,
            }
            results.append(row)
            if args.snapshot_index_file:
                append_jsonl(args.snapshot_index_file, row)
        except CameraUnavailableError as exc:
            row = {
                "captured_at": captured_at,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "url": url,
                "status": "unavailable",
                "unavailable_reason": str(exc),
                "flow": args.flow_name,
            }
            results.append(row)
            if args.snapshot_index_file:
                append_jsonl(args.snapshot_index_file, row)
        except Exception as exc:
            row = {
                "captured_at": captured_at,
                "camera_id": cam_id,
                "camera_name": cam_name,
                "url": url,
                "status": "error",
                "error": str(exc),
                "flow": args.flow_name,
            }
            results.append(row)
            if args.snapshot_index_file:
                append_jsonl(args.snapshot_index_file, row)

    ok = [r for r in results if r["status"] == "ok"]
    unavailable = [r for r in results if r["status"] == "unavailable"]
    err = [r for r in results if r["status"] == "error"]
    print(f"BORDER SNAPSHOT CAPTURE ({args.flow_name})")
    print(f"Checked: {len(results)} camera(s), ok={len(ok)}, unavailable={len(unavailable)}, error={len(err)}")
    for row in ok:
        print(f"- {row['camera_name']} [{row['camera_id']}]: snapshot_file: {row['snapshot_file']}")
    for row in unavailable:
        print(f"- {row['camera_name']} [{row['camera_id']}]: UNAVAILABLE - {row.get('unavailable_reason', 'camera image unavailable')}")
    for row in err:
        print(f"- {row['camera_name']} [{row['camera_id']}]: ERROR - {row['error']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": now_iso(),
            "flow": args.flow_name,
            "camera": args.camera,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if (ok or unavailable) else 1


def run_patterns(args: argparse.Namespace) -> int:
    selected = load_cameras(args.cameras_file, args.camera)
    if not selected:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")
    selected_ids = {str(c.get("id")) for c in selected}

    rows: list[dict[str, Any]] = []
    unavailable_rows = 0
    unavailable_by_hour: dict[int, int] = {}
    snapshot_status_counts: dict[str, int] = {}
    snapshot_rows_in_window = 0
    snapshot_inference_enabled = False

    cutoff_local: datetime | None = None
    if args.hours is not None:
        if args.hours <= 0:
            raise SystemExit("--hours must be a positive number")
        cutoff_local = datetime.now().astimezone() - timedelta(hours=float(args.hours))

    def parse_row_ts(row: dict[str, Any]) -> datetime | None:
        raw = row.get("captured_at")
        if not raw:
            return None
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(datetime.now().astimezone().tzinfo)

    if args.history_file and args.history_file.exists():
        for line in args.history_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            status = row.get("status")
            if status == "unavailable":
                if str(row.get("camera_id")) not in selected_ids:
                    continue
                local_ts = parse_row_ts(row)
                if local_ts is None:
                    continue
                if cutoff_local is not None and local_ts < cutoff_local:
                    continue
                unavailable_rows += 1
                unavailable_by_hour[local_ts.hour] = unavailable_by_hour.get(local_ts.hour, 0) + 1
                continue
            if status != "ok":
                continue
            if str(row.get("camera_id")) not in selected_ids:
                continue
            if not row.get("line_bucket"):
                continue
            local_ts = parse_row_ts(row)
            if local_ts is None:
                continue
            if cutoff_local is not None and local_ts < cutoff_local:
                continue
            row_copy = dict(row)
            row_copy["source"] = "history"
            rows.append(row_copy)

    if args.snapshot_index_file and args.snapshot_index_file.exists():
        predictor = QueuePredictor(args.models_dir)
        snapshot_inference_enabled = predictor.available()
        for line in args.snapshot_index_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("camera_id")) not in selected_ids:
                continue
            local_ts = parse_row_ts(row)
            if local_ts is None:
                continue
            if cutoff_local is not None and local_ts < cutoff_local:
                continue

            status = str(row.get("status", "unknown"))
            snapshot_rows_in_window += 1
            snapshot_status_counts[status] = snapshot_status_counts.get(status, 0) + 1

            if status == "unavailable":
                unavailable_rows += 1
                unavailable_by_hour[local_ts.hour] = unavailable_by_hour.get(local_ts.hour, 0) + 1
                continue
            if status != "ok":
                continue

            snapshot_path = Path(str(row.get("snapshot_file", "")))
            if not snapshot_path.exists():
                continue
            if not snapshot_inference_enabled:
                continue
            try:
                with Image.open(snapshot_path) as img:
                    rgb = img.convert("RGB")
                    if is_unavailable_placeholder(rgb):
                        unavailable_rows += 1
                        unavailable_by_hour[local_ts.hour] = unavailable_by_hour.get(local_ts.hour, 0) + 1
                        continue
                    pred = predictor.predict(rgb)
                rows.append(
                    {
                        "captured_at": row.get("captured_at", now_iso()),
                        "camera_id": row.get("camera_id"),
                        "camera_name": row.get("camera_name"),
                        "status": "ok",
                        "line_bucket": pred.label,
                        "score": pred.score,
                        "model": pred.model_name,
                        "source": "snapshot_inference",
                        "snapshot_file": str(snapshot_path),
                    }
                )
            except Exception:
                continue

    if not rows:
        print("No matching samples for patterns (history or snapshot inference).")
        print(f"Snapshot records in window: {snapshot_rows_in_window}")
        if snapshot_status_counts:
            parts = ", ".join(f"{k}={snapshot_status_counts[k]}" for k in sorted(snapshot_status_counts))
            print(f"Snapshot status split: {parts}")
        if not snapshot_inference_enabled and args.snapshot_index_file and args.snapshot_index_file.exists():
            print("Snapshot inference: disabled (no model available)")
        if unavailable_rows:
            print(f"Unavailable captures (filtered): {unavailable_rows}")
        return 0

    local_tz = datetime.now().astimezone().tzinfo
    tz_label = str(local_tz) if local_tz is not None else "local"

    history_rows = sum(1 for r in rows if r.get("source") == "history")
    inferred_rows = sum(1 for r in rows if r.get("source") == "snapshot_inference")

    by_hour: dict[int, dict[str, int]] = {}
    total_extreme = 0
    for row in rows:
        ts = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        local_ts = ts.astimezone(local_tz)
        hour = local_ts.hour
        label = str(row.get("line_bucket", "unknown"))
        by_hour.setdefault(hour, {"samples": 0, "extreme": 0})
        by_hour[hour]["samples"] += 1
        if label == "extreme":
            by_hour[hour]["extreme"] += 1
            total_extreme += 1

    insufficient_hours = sorted(h for h, n in unavailable_by_hour.items() if by_hour.get(h, {}).get("samples", 0) == 0 and n > 0)

    print("BORDER PATTERNS")
    print(f"Camera: {args.camera}")
    if cutoff_local is not None:
        print(f"Window: last {args.hours:g} hour(s)")
    print(f"Data sources: history={history_rows}, inferred_from_snapshots={inferred_rows}")
    print(f"Snapshot records in window: {snapshot_rows_in_window}")
    if snapshot_status_counts:
        parts = ", ".join(f"{k}={snapshot_status_counts[k]}" for k in sorted(snapshot_status_counts))
        print(f"Snapshot status split: {parts}")
    if not snapshot_inference_enabled and args.snapshot_index_file and args.snapshot_index_file.exists():
        print("Snapshot inference: disabled (no model available)")
    print(f"Unavailable captures (filtered): {unavailable_rows}")
    if unavailable_by_hour:
        print("Unavailable by hour (local):")
        for hour in sorted(unavailable_by_hour):
            print(f"- {hour:02d}:00-{hour:02d}:59 -> unavailable {unavailable_by_hour[hour]}")
    if insufficient_hours:
        hour_ranges = ", ".join(f"{h:02d}:00-{h:02d}:59" for h in insufficient_hours)
        print(f"Insufficient coverage hours (unavailable with no usable samples): {hour_ranges}")
        print("Trend interpretation note: do not treat insufficient coverage hours as quiet/low-traffic windows.")
    print(f"Samples: {len(rows)}, extreme samples: {total_extreme}")
    print(f"Extreme by hour ({tz_label}):")
    for hour in sorted(by_hour):
        samples = by_hour[hour]["samples"]
        ext = by_hour[hour]["extreme"]
        ratio = 100.0 * ext / max(samples, 1)
        print(f"- {hour:02d}:00-{hour:02d}:59 -> extreme {ext}/{samples} ({ratio:.1f}%)")
    return 0


def run_backfill_unavailable(args: argparse.Namespace) -> int:
    run_ts = now_iso()
    reason = "camera image unavailable"
    cache: dict[str, bool] = {}

    def check_unavailable(snapshot_path: Path) -> bool:
        key = str(snapshot_path)
        if key in cache:
            return cache[key]
        try:
            with Image.open(snapshot_path) as img:
                cache[key] = is_unavailable_placeholder(img.convert("RGB"))
        except Exception:
            cache[key] = False
        return cache[key]

    snapshot_total = 0
    snapshot_marked = 0
    snapshot_rows: list[dict[str, Any]] = []
    if args.snapshot_index_file.exists():
        for line in args.snapshot_index_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            snapshot_total += 1
            if row.get("status") == "ok":
                p = Path(str(row.get("snapshot_file", "")))
                if p.exists() and check_unavailable(p):
                    row["status"] = "unavailable"
                    row["unavailable_reason"] = reason
                    row["unavailable_detected_at"] = run_ts
                    row["availability_source"] = "backfill"
                    row.pop("error", None)
                    snapshot_marked += 1
            snapshot_rows.append(row)

    history_total = 0
    history_marked = 0
    history_rows: list[dict[str, Any]] = []
    if args.history_file.exists():
        for line in args.history_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            history_total += 1
            if row.get("status") == "ok":
                snapshot_path = None
                raw_snapshot = row.get("snapshot_file")
                if raw_snapshot:
                    snapshot_path = Path(str(raw_snapshot))
                else:
                    cam_id = str(row.get("camera_id", "")).strip()
                    captured_at = row.get("captured_at")
                    if cam_id and captured_at:
                        try:
                            stamp = iso_to_filename_stamp(str(captured_at))
                            snapshot_path = args.snapshots_dir / f"{safe_name(cam_id)}-{stamp}.jpg"
                        except Exception:
                            snapshot_path = None

                if snapshot_path and snapshot_path.exists() and check_unavailable(snapshot_path):
                    row["status"] = "unavailable"
                    row["unavailable_reason"] = reason
                    row["unavailable_detected_at"] = run_ts
                    row["availability_source"] = "backfill"
                    row.pop("line_bucket", None)
                    row.pop("score", None)
                    row.pop("model", None)
                    history_marked += 1
            history_rows.append(row)

    if args.apply:
        if snapshot_rows:
            rewrite_jsonl(args.snapshot_index_file, snapshot_rows)
        if history_rows:
            rewrite_jsonl(args.history_file, history_rows)
        print("Backfill mode: APPLY")
    else:
        print("Backfill mode: DRY-RUN (no files written)")

    print(
        f"Snapshot index: total_rows={snapshot_total}, marked_unavailable={snapshot_marked}, file={args.snapshot_index_file}"
    )
    print(f"History: total_rows={history_total}, marked_unavailable={history_marked}, file={args.history_file}")
    print(f"Checked unique snapshot files: {len(cache)}")
    return 0


def run_unavailable_summary(args: argparse.Namespace) -> int:
    selected = load_cameras(args.cameras_file, args.camera)
    if not selected:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")
    selected_ids = {str(c.get("id")) for c in selected}

    cutoff_local: datetime | None = None
    if args.hours is not None:
        if args.hours <= 0:
            raise SystemExit("--hours must be a positive number")
        cutoff_local = datetime.now().astimezone() - timedelta(hours=float(args.hours))

    def parse_row_ts(row: dict[str, Any]) -> datetime | None:
        raw = row.get("captured_at")
        if not raw:
            return None
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(datetime.now().astimezone().tzinfo)

    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def ingest_row(row: dict[str, Any], source: str) -> None:
        status = str(row.get("status", ""))
        if status != "unavailable":
            return
        cam_id = str(row.get("camera_id", ""))
        if cam_id not in selected_ids:
            return
        local_ts = parse_row_ts(row)
        if local_ts is None:
            return
        if cutoff_local is not None and local_ts < cutoff_local:
            return
        key = (cam_id, local_ts.isoformat(), source)
        if key in seen:
            return
        seen.add(key)
        events.append(
            {
                "captured_at": local_ts,
                "camera_id": cam_id,
                "camera_name": row.get("camera_name", cam_id),
                "reason": row.get("unavailable_reason", "camera image unavailable"),
                "source": source,
            }
        )

    if args.history_file.exists():
        for line in args.history_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ingest_row(json.loads(line), "history")

    if args.snapshot_index_file.exists():
        for line in args.snapshot_index_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ingest_row(json.loads(line), "snapshot_index")

    events.sort(key=lambda e: e["captured_at"])

    print("BORDER UNAVAILABLE SUMMARY")
    print(f"Camera: {args.camera}")
    if cutoff_local is not None:
        print(f"Window: last {args.hours:g} hour(s)")
    print(f"Unavailable events: {len(events)}")

    if not events:
        return 0

    by_hour: dict[int, int] = {}
    for e in events:
        hour = int(e["captured_at"].hour)
        by_hour[hour] = by_hour.get(hour, 0) + 1

    first = events[0]
    last = events[-1]
    print(f"First unavailable: {first['captured_at'].isoformat()} [{first['camera_name']}]")
    print(f"Last unavailable: {last['captured_at'].isoformat()} [{last['camera_name']}]")
    print("Unavailable by hour (local):")
    for hour in sorted(by_hour):
        print(f"- {hour:02d}:00-{hour:02d}:59 -> {by_hour[hour]}")

    preview = events[-min(10, len(events)) :]
    print("Most recent unavailable events:")
    for e in preview:
        print(
            f"- {e['captured_at'].isoformat()} | {e['camera_name']} [{e['camera_id']}] | {e['reason']} | source={e['source']}"
        )
    return 0


def run_snapshot_summary(args: argparse.Namespace) -> int:
    selected = load_cameras(args.cameras_file, args.camera)
    if not selected:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")
    selected_ids = {str(c.get("id")) for c in selected}

    cutoff_local: datetime | None = None
    if args.hours is not None:
        if args.hours <= 0:
            raise SystemExit("--hours must be a positive number")
        cutoff_local = datetime.now().astimezone() - timedelta(hours=float(args.hours))

    def parse_row_ts(row: dict[str, Any]) -> datetime | None:
        raw = row.get("captured_at")
        if not raw:
            return None
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(datetime.now().astimezone().tzinfo)

    rows: list[dict[str, Any]] = []
    if args.snapshot_index_file.exists():
        for line in args.snapshot_index_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("camera_id")) not in selected_ids:
                continue
            ts = parse_row_ts(row)
            if ts is None:
                continue
            if cutoff_local is not None and ts < cutoff_local:
                continue
            row_copy = dict(row)
            row_copy["_ts"] = ts
            rows.append(row_copy)

    rows.sort(key=lambda r: r["_ts"])

    status_counts: dict[str, int] = {}
    for r in rows:
        status = str(r.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print("BORDER SNAPSHOT SUMMARY")
    print(f"Camera: {args.camera}")
    if cutoff_local is not None:
        print(f"Window: last {args.hours:g} hour(s)")
    print(f"Total snapshot records: {len(rows)}")
    for key in sorted(status_counts):
        print(f"- {key}: {status_counts[key]}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        print(f"First OK snapshot: {ok_rows[0]['captured_at']}")
        print(f"Last OK snapshot: {ok_rows[-1]['captured_at']}")

    preview = rows[-min(5, len(rows)) :]
    if preview:
        print("Most recent snapshot records:")
        for r in preview:
            cam_name = r.get("camera_name", r.get("camera_id", "unknown"))
            print(
                f"- {r.get('captured_at')} | {cam_name} [{r.get('camera_id')}] | status={r.get('status')}"
            )
    return 0


def run_upsert_monitor_job(args: argparse.Namespace) -> int:
    if args.interval_min <= 0:
        raise SystemExit("--interval-min must be a positive integer")

    selected = load_cameras(args.cameras_file, args.camera)
    if not selected:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")
    if len(selected) != 1:
        raise SystemExit("Monitoring job requires exactly one camera. Use a specific camera name or id (not 'all').")

    camera = selected[0]
    camera_id = str(camera.get("id"))
    camera_name = str(camera.get("name", camera_id))

    jobs_file = args.jobs_file
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    if jobs_file.exists():
        jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
    else:
        jobs = {"version": 1, "jobs": []}

    jobs.setdefault("version", 1)
    jobs.setdefault("jobs", [])

    name = f"border-monitor-{safe_name(camera_id)}-{int(args.interval_min)}m"
    command = _build_capture_command(camera_name, camera_id)

    entry = {
        "name": name,
        "enabled": True,
        "schedule": {
            "kind": "every",
            "everyMs": int(args.interval_min * 60 * 1000),
            "anchorMs": int(datetime.now(timezone.utc).timestamp() * 1000),
        },
        "sessionTarget": "background",
        "payload": {
            "kind": "exec",
            "command": command,
            "cwd": "/home/dragan-slaveski/.openclaw/workspace",
            "timeoutMs": 60000,
        },
    }

    replaced = False
    for i, job in enumerate(jobs["jobs"]):
        if job.get("name") == name:
            jobs["jobs"][i] = entry
            replaced = True
            break
    if not replaced:
        jobs["jobs"].append(entry)

    jobs_file.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"Upserted job: {name}")
    print(f"Interval: every {args.interval_min} minute(s)")
    print(f"Target camera: {camera_name} [{camera_id}]")
    print("Mode: local exec snapshot capture (no LLM)")
    return 0


def run_upsert_system_cron(args: argparse.Namespace) -> int:
    if args.interval_min <= 0:
        raise SystemExit("--interval-min must be a positive integer")
    if args.interval_min > 59:
        raise SystemExit("--interval-min must be <= 59 for Linux crontab minute scheduling")

    selected = load_cameras(args.cameras_file, args.camera)
    if not selected:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")
    if len(selected) != 1:
        raise SystemExit("System cron monitoring requires exactly one camera. Use a specific camera name or id (not 'all').")

    camera = selected[0]
    camera_id = str(camera.get("id"))
    camera_name = str(camera.get("name", camera_id))

    command = _build_capture_command(camera_name, camera_id)
    cron_tag = _system_cron_tag_for_camera(camera_id)
    cron_line = f"*/{int(args.interval_min)} * * * * {command} # {cron_tag}"

    current = _load_user_crontab_lines()
    cleaned = _strip_system_cron_entries(current, {cron_tag})
    cleaned.append(cron_line)
    _write_user_crontab_lines(cleaned)

    print(f"Upserted system cron entry: {cron_tag}")
    print(f"Interval: every {args.interval_min} minute(s)")
    print(f"Target camera: {camera_name} [{camera_id}]")
    print("Mode: Linux crontab local exec snapshot capture (no LLM)")
    return 0


def run_disable_monitor_job(args: argparse.Namespace) -> int:
    selected = load_cameras(args.cameras_file, args.camera)
    if not selected:
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")

    camera_ids = {str(c.get("id")) for c in selected}
    if "all" in {_norm(args.camera)}:
        camera_ids = {
            str(c.get("id"))
            for c in load_cameras(args.cameras_file, "all")
        }

    jobs_file = args.jobs_file
    if not jobs_file.exists():
        print(f"Jobs file not found: {jobs_file}")
        return 0

    jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
    jobs.setdefault("jobs", [])

    disabled = 0
    for job in jobs["jobs"]:
        name = str(job.get("name", ""))
        if not name.startswith("border-monitor-"):
            continue
        for camera_id in camera_ids:
            prefix = f"border-monitor-{safe_name(camera_id)}-"
            if name.startswith(prefix) and job.get("enabled", True):
                job["enabled"] = False
                disabled += 1
                break

    jobs_file.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"Disabled monitor job(s): {disabled}")
    print(f"Target camera selector: {args.camera}")
    return 0


def run_disable_system_cron(args: argparse.Namespace) -> int:
    selected = load_cameras(args.cameras_file, args.camera)
    if not selected and _norm(args.camera) != "all":
        raise SystemExit(f"No enabled cameras matched camera={args.camera}")

    if _norm(args.camera) == "all":
        markers = {SYSTEM_CRON_TAG_PREFIX}
    else:
        markers = {_system_cron_tag_for_camera(str(c.get("id"))) for c in selected}

    current = _load_user_crontab_lines()
    cleaned = _strip_system_cron_entries(current, markers)
    removed = len(current) - len(cleaned)
    _write_user_crontab_lines(cleaned)

    print(f"Disabled system cron monitor entry/entries: {removed}")
    print(f"Target camera selector: {args.camera}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Border tracker flow utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="Run one-shot status classification")
    status.add_argument("--flow-name", default="status")
    status.add_argument("--camera", default="Bajakovo Entry")
    status.add_argument("--cameras-file", type=Path, required=True)
    status.add_argument("--models-dir", type=Path, required=True)
    status.add_argument("--timeout-sec", type=int, default=20)
    status.add_argument("--history-file", type=Path)
    status.add_argument("--output-json", type=Path)
    status.add_argument("--save-debug-dir", type=Path)

    capture = sub.add_parser("capture-snapshot", help="Capture snapshots only (no model inference)")
    capture.add_argument("--flow-name", default="monitor")
    capture.add_argument("--camera", default="Bajakovo Entry")
    capture.add_argument("--cameras-file", type=Path, required=True)
    capture.add_argument("--timeout-sec", type=int, default=20)
    capture.add_argument("--snapshots-dir", type=Path, required=True)
    capture.add_argument("--snapshot-index-file", type=Path)
    capture.add_argument("--output-json", type=Path)

    patterns = sub.add_parser("patterns", help="Summarize patterns from history")
    patterns.add_argument(
        "--history-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl"),
    )
    patterns.add_argument("--camera", default="Bajakovo Entry")
    patterns.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )
    patterns.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/models"),
    )
    patterns.add_argument(
        "--snapshot-index-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl"),
    )
    patterns.add_argument("--hours", type=float, help="Restrict analysis to the last N hours")

    backfill = sub.add_parser(
        "backfill-unavailable",
        help="Retroactively mark unavailable placeholder frames in history/snapshot index",
    )
    backfill.add_argument(
        "--history-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl"),
    )
    backfill.add_argument(
        "--snapshot-index-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl"),
    )
    backfill.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshots"),
    )
    backfill.add_argument("--apply", action="store_true", help="Write updated statuses to JSONL files")

    unavailable = sub.add_parser(
        "unavailable-summary",
        help="Summarize when selected camera(s) were unavailable",
    )
    unavailable.add_argument("--camera", default="Bajakovo Entry")
    unavailable.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )
    unavailable.add_argument(
        "--history-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl"),
    )
    unavailable.add_argument(
        "--snapshot-index-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl"),
    )
    unavailable.add_argument("--hours", type=float, help="Restrict summary to the last N hours")

    snapshot_summary = sub.add_parser(
        "snapshot-summary",
        help="Summarize snapshot records and statuses for selected camera(s)",
    )
    snapshot_summary.add_argument("--camera", default="Bajakovo Entry")
    snapshot_summary.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )
    snapshot_summary.add_argument(
        "--snapshot-index-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl"),
    )
    snapshot_summary.add_argument("--hours", type=float, help="Restrict summary to the last N hours")

    monitor = sub.add_parser("upsert-monitor-job", help="Create/update monitoring cron entry")
    monitor.add_argument("--camera", default="Bajakovo Entry")
    monitor.add_argument("--interval-min", type=int, required=True)
    monitor.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )
    monitor.add_argument("--jobs-file", type=Path, default=Path("/home/dragan-slaveski/.openclaw/cron/jobs.json"))

    monitor_system = sub.add_parser("upsert-system-cron", help="Create/update Linux crontab monitoring entry")
    monitor_system.add_argument("--camera", default="Bajakovo Entry")
    monitor_system.add_argument("--interval-min", type=int, required=True)
    monitor_system.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )

    disable = sub.add_parser("disable-monitor-job", help="Disable existing monitoring cron entry/entries")
    disable.add_argument("--camera", default="Bajakovo Entry")
    disable.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )
    disable.add_argument("--jobs-file", type=Path, default=Path("/home/dragan-slaveski/.openclaw/cron/jobs.json"))

    disable_system = sub.add_parser("disable-system-cron", help="Disable Linux crontab monitoring entry/entries")
    disable_system.add_argument("--camera", default="Bajakovo Entry")
    disable_system.add_argument(
        "--cameras-file",
        type=Path,
        default=Path("/home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "status":
        return run_status(args)
    if args.cmd == "capture-snapshot":
        return run_capture_snapshot(args)
    if args.cmd == "patterns":
        return run_patterns(args)
    if args.cmd == "backfill-unavailable":
        return run_backfill_unavailable(args)
    if args.cmd == "unavailable-summary":
        return run_unavailable_summary(args)
    if args.cmd == "snapshot-summary":
        return run_snapshot_summary(args)
    if args.cmd == "upsert-monitor-job":
        return run_upsert_monitor_job(args)
    if args.cmd == "upsert-system-cron":
        return run_upsert_system_cron(args)
    if args.cmd == "disable-monitor-job":
        return run_disable_monitor_job(args)
    if args.cmd == "disable-system-cron":
        return run_disable_system_cron(args)
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
