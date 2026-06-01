#!/usr/bin/env python3
"""Safely append a trip entry to workspace/trips.json."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import unicodedata
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parent / "trips.json"


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug or "trip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a trip entry to trips.json")
    parser.add_argument("--entry-date", required=True, help="Entry date (YYYY-MM-DD)")
    parser.add_argument("--exit-date", required=True, help="Exit date (YYYY-MM-DD)")
    parser.add_argument("--destination", required=True, help="Destination")
    parser.add_argument("--description", required=True, help="Trip description")
    parser.add_argument(
        "--people",
        required=True,
        help="Comma-separated person IDs (example: dragan,jelena)",
    )
    parser.add_argument(
        "--id",
        dest="trip_id",
        default=None,
        help="Optional custom trip ID",
    )
    parser.add_argument(
        "--file",
        default=str(DEFAULT_PATH),
        help="Path to trips.json (default: workspace/trips.json)",
    )
    return parser.parse_args()


def _parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def _next_unique_id(existing_ids: set[str], base_id: str) -> str:
    if base_id not in existing_ids:
        return base_id
    idx = 2
    while f"{base_id}-{idx}" in existing_ids:
        idx += 1
    return f"{base_id}-{idx}"


def main() -> int:
    args = _parse_args()

    entry_date = _parse_date(args.entry_date)
    exit_date = _parse_date(args.exit_date)
    if exit_date < entry_date:
        raise SystemExit("exit-date must be on or after entry-date")

    people = [p.strip() for p in args.people.split(",") if p.strip()]
    if not people:
        raise SystemExit("At least one person ID is required in --people")

    trips_path = Path(args.file).resolve()
    if not trips_path.exists():
        raise SystemExit(f"File not found: {trips_path}")

    data = json.loads(trips_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "trips" not in data or "people" not in data:
        raise SystemExit("Invalid schema in trips.json (expected top-level people and trips)")

    known_people = {person.get("id") for person in data.get("people", []) if isinstance(person, dict)}
    unknown_people = [pid for pid in people if pid not in known_people]
    if unknown_people:
        raise SystemExit(f"Unknown person IDs: {', '.join(unknown_people)}")

    base_id = args.trip_id or f"trip-{entry_date.isoformat()}-{_slugify(args.destination)}"
    existing_ids = {
        trip.get("id") for trip in data.get("trips", []) if isinstance(trip, dict) and trip.get("id")
    }
    trip_id = _next_unique_id(existing_ids, base_id)

    new_trip = {
        "id": trip_id,
        "entryDate": entry_date.isoformat(),
        "exitDate": exit_date.isoformat(),
        "destination": args.destination,
        "description": args.description,
        "personIds": people,
    }

    data["trips"].append(new_trip)
    data["lastUpdated"] = dt.date.today().isoformat()

    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    trips_path.write_text(serialized + "\n", encoding="utf-8")

    print(f"Added trip: {trip_id}")
    print(f"File updated: {trips_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
