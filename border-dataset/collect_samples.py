#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import requests


def now_local() -> datetime:
    return datetime.now().astimezone()


def local_stamp(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%S%z")


def safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "camera"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_jpeg(content_type: str) -> bool:
    c = (content_type or "").lower()
    return "jpeg" in c or "jpg" in c or "image/" in c


def save_sample(output_root: Path, camera_id: str, ts: datetime, payload: bytes) -> Path:
    day_dir = output_root / camera_id / ts.strftime("%Y") / ts.strftime("%m") / ts.strftime("%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    out_file = day_dir / f"{local_stamp(ts)}.jpg"
    out_file.write_bytes(payload)
    return out_file


def append_manifest(output_root: Path, event: dict) -> None:
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def collect_once(config_path: Path, output_root: Path, timeout_sec: int, user_agent: str) -> int:
    cfg = load_config(config_path)
    cameras = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]

    if not cameras:
        print("No enabled cameras found in config.")
        return 1

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    ok_count = 0
    fail_count = 0
    run_ts = now_local()

    for cam in cameras:
        cam_id = safe_id(str(cam.get("id", "camera")))
        cam_type = str(cam.get("type", "jpeg_snapshot"))
        url = str(cam.get("url", "")).strip()

        if cam_type != "jpeg_snapshot" or not url:
            fail_count += 1
            append_manifest(
                output_root,
                {
                    "captured_at": run_ts.isoformat(),
                    "camera_id": cam_id,
                    "status": "error",
                    "error": "Unsupported camera type or missing URL",
                    "camera_type": cam_type,
                    "url": url,
                },
            )
            continue

        try:
            response = session.get(url, timeout=timeout_sec)
            response.raise_for_status()
            payload = response.content
            content_type = response.headers.get("Content-Type", "")

            if not payload or not ensure_jpeg(content_type):
                raise ValueError(f"Unexpected content type: {content_type or 'unknown'}")

            ts = now_local()
            out_file = save_sample(output_root, cam_id, ts, payload)
            append_manifest(
                output_root,
                {
                    "captured_at": ts.isoformat(),
                    "camera_id": cam_id,
                    "status": "ok",
                    "camera_type": cam_type,
                    "url": url,
                    "content_type": content_type,
                    "bytes": len(payload),
                    "file": str(out_file),
                },
            )
            ok_count += 1
            print(f"OK  {cam_id}: {out_file}")
        except Exception as exc:
            fail_count += 1
            ts = now_local()
            append_manifest(
                output_root,
                {
                    "captured_at": ts.isoformat(),
                    "camera_id": cam_id,
                    "status": "error",
                    "camera_type": cam_type,
                    "url": url,
                    "error": str(exc),
                },
            )
            print(f"ERR {cam_id}: {exc}")

    print(f"Done. ok={ok_count} failed={fail_count} total={len(cameras)}")
    return 0 if ok_count > 0 else 1


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Collect raw border camera samples for later labeling.")
    parser.add_argument(
        "--config",
        default=str(default_root / "cameras.json"),
        help="Path to camera config JSON",
    )
    parser.add_argument(
        "--output-root",
        default=str(default_root / "data" / "raw"),
        help="Directory where raw images and manifest are stored",
    )
    parser.add_argument("--timeout-sec", type=int, default=20, help="HTTP timeout per camera")
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (X11; Linux x86_64) BorderDatasetCollector/1.0",
        help="User-Agent header for camera requests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return collect_once(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        timeout_sec=int(args.timeout_sec),
        user_agent=str(args.user_agent),
    )


if __name__ == "__main__":
    raise SystemExit(main())
