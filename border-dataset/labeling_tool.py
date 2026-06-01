#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


BUCKETS = [
    (0, 25, "light"),
    (26, 50, "medium"),
    (51, 75, "high"),
    (76, 100, "extreme"),
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sample_id_for(path_text: str) -> str:
    return hashlib.sha1(path_text.encode("utf-8")).hexdigest()[:16]


def bucket_for_percent(value: int | None) -> str | None:
    if value is None:
        return None
    for low, high, name in BUCKETS:
        if low <= value <= high:
            return name
    return "extreme"


@dataclass
class Sample:
    sample_id: str
    camera_id: str
    captured_at: str
    file: str
    relative_file: str
    content_type: str
    bytes_size: int
    url: str

    def to_dict(self, label: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "camera_id": self.camera_id,
            "captured_at": self.captured_at,
            "file": self.file,
            "relative_file": self.relative_file,
            "content_type": self.content_type,
            "bytes": self.bytes_size,
            "source_url": self.url,
            "image_url": f"/api/image/{self.sample_id}",
            "label": label,
        }


class LabelingStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.data_root = self.root_dir / "data"
        self.raw_root = self.data_root / "raw"
        self.labels_dir = self.data_root / "labels"
        self.labels_path = self.labels_dir / "line_size_labels.jsonl"
        self.ui_path = self.root_dir / "labeling_ui.html"
        self.samples_by_id: dict[str, Sample] = {}
        self.samples_in_order: list[Sample] = []
        self.reload_samples()

    def reload_samples(self) -> None:
        manifest_path = self.raw_root / "manifest.jsonl"
        samples: list[Sample] = []

        if not manifest_path.exists():
            self.samples_by_id = {}
            self.samples_in_order = []
            return

        with manifest_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("status") != "ok":
                    continue

                file_text = str(event.get("file", "")).strip()
                if not file_text:
                    continue

                file_path = Path(file_text)
                if not file_path.exists():
                    continue

                sample_id = sample_id_for(file_text)
                try:
                    relative_file = str(file_path.resolve().relative_to(self.raw_root.resolve()))
                except ValueError:
                    relative_file = str(file_path)

                samples.append(
                    Sample(
                        sample_id=sample_id,
                        camera_id=str(event.get("camera_id", "unknown")),
                        captured_at=str(event.get("captured_at", "")),
                        file=str(file_path.resolve()),
                        relative_file=relative_file,
                        content_type=str(event.get("content_type", "image/jpeg")),
                        bytes_size=int(event.get("bytes", 0) or 0),
                        url=str(event.get("url", "")),
                    )
                )

        samples.sort(key=lambda sample: (sample.captured_at, sample.camera_id, sample.relative_file))
        self.samples_in_order = samples
        self.samples_by_id = {sample.sample_id: sample for sample in samples}

    def load_labels(self) -> dict[str, dict[str, Any]]:
        labels: dict[str, dict[str, Any]] = {}
        if not self.labels_path.exists():
            return labels

        with self.labels_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id", "")).strip()
                if sample_id:
                    labels[sample_id] = record
        return labels

    def build_dataset(self) -> dict[str, Any]:
        labels = self.load_labels()
        samples = [sample.to_dict(labels.get(sample.sample_id)) for sample in self.samples_in_order]
        cameras = sorted({sample.camera_id for sample in self.samples_in_order})

        labeled_count = sum(1 for sample in samples if sample.get("label"))
        unusable_count = sum(
            1
            for sample in samples
            if sample.get("label") and not bool(sample["label"].get("is_usable", True))
        )

        per_camera: dict[str, dict[str, int]] = {}
        for sample in samples:
            camera_id = sample["camera_id"]
            camera_stats = per_camera.setdefault(camera_id, {"total": 0, "labeled": 0, "unusable": 0})
            camera_stats["total"] += 1
            if sample.get("label"):
                camera_stats["labeled"] += 1
                if not bool(sample["label"].get("is_usable", True)):
                    camera_stats["unusable"] += 1

        return {
            "generated_at": now_iso(),
            "label_file": str(self.labels_path),
            "samples": samples,
            "cameras": cameras,
            "stats": {
                "total": len(samples),
                "labeled": labeled_count,
                "unlabeled": max(len(samples) - labeled_count, 0),
                "unusable": unusable_count,
                "per_camera": per_camera,
            },
        }

    def save_label(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(payload.get("sample_id", "")).strip()
        sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError("Unknown sample_id")

        is_usable = bool(payload.get("is_usable", True))
        percent_raw = payload.get("line_percent")
        if percent_raw in ("", None):
            percent = None
        else:
            percent = int(percent_raw)

        if is_usable:
            if percent is None:
                raise ValueError("line_percent is required for usable samples")
            if percent < 0 or percent > 100:
                raise ValueError("line_percent must be between 0 and 100")
        else:
            percent = None

        notes = str(payload.get("notes", "")).strip()
        if len(notes) > 1000:
            raise ValueError("notes must be 1000 characters or fewer")

        record = {
            "sample_id": sample.sample_id,
            "camera_id": sample.camera_id,
            "captured_at": sample.captured_at,
            "file": sample.file,
            "relative_file": sample.relative_file,
            "line_percent": percent,
            "line_bucket": bucket_for_percent(percent),
            "is_usable": is_usable,
            "notes": notes,
            "labeled_at": now_iso(),
        }

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        return record

    def image_bytes(self, sample_id: str) -> bytes:
        sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(sample_id)
        return Path(sample.file).read_bytes()


class LabelingServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], store: LabelingStore) -> None:
        super().__init__(address, LabelingRequestHandler)
        self.store = store


class LabelingRequestHandler(BaseHTTPRequestHandler):
    server: LabelingServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._serve_ui()
            return

        if parsed.path == "/api/dataset":
            self._send_json(self.server.store.build_dataset())
            return

        if parsed.path.startswith("/api/image/"):
            sample_id = parsed.path.removeprefix("/api/image/")
            try:
                payload = self.server.store.image_bytes(sample_id)
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown sample")
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path == "/api/ping":
            self._send_json({"ok": True, "samples": len(self.server.store.samples_in_order)})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/labels":
            self._handle_save_label()
            return

        if parsed.path == "/api/reload":
            self.server.store.reload_samples()
            self._send_json({"ok": True, "samples": len(self.server.store.samples_in_order)})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {fmt % args}")

    def _serve_ui(self) -> None:
        ui_bytes = self.server.store.ui_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(ui_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(ui_bytes)

    def _handle_save_label(self) -> None:
        payload = self._read_json_body()
        try:
            record = self.server.store.save_label(payload)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"ok": True, "label": record}, status=HTTPStatus.CREATED)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the border dataset labeling tool.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port to bind")
    parser.add_argument(
        "--root-dir",
        default=str(root_dir),
        help="Directory containing data/raw, data/labels, and labeling_ui.html",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = LabelingStore(Path(args.root_dir))
    server = LabelingServer((args.host, int(args.port)), store)
    print(f"Labeling tool listening on http://{args.host}:{args.port}")
    print(f"Loaded {len(store.samples_in_order)} samples from {store.raw_root}")
    print(f"Labels will be appended to {store.labels_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeling tool.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())