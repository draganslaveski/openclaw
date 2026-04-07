"""
Unit tests for pure functions in flight_tracker.py.
Run with: venv/bin/python -m pytest scripts/test_flight_tracker.py -v
"""
import sys
import os
from datetime import datetime, timezone, timedelta

# Bootstrap venv path so imports work without activating the venv
VENV = os.path.join(os.path.dirname(__file__), '..', 'venv', 'lib', 'python3.12', 'site-packages')
if os.path.isdir(VENV) and VENV not in sys.path:
    sys.path.insert(0, VENV)

# Import the module under test
sys.path.insert(0, os.path.dirname(__file__))
import flight_tracker as ft


# ---------------------------------------------------------------------------
# parse_time_any
# ---------------------------------------------------------------------------

class TestParseTimeAny:
    def test_aerodatabox_format(self):
        result = ft.parse_time_any("2026-04-07 21:55Z")
        assert result == datetime(2026, 4, 7, 21, 55, tzinfo=timezone.utc)

    def test_aerodatabox_format_with_seconds(self):
        result = ft.parse_time_any("2026-04-07 21:55:30Z")
        assert result == datetime(2026, 4, 7, 21, 55, 30, tzinfo=timezone.utc)

    def test_iso_with_z(self):
        result = ft.parse_time_any("2026-04-07T21:55:00Z")
        assert result == datetime(2026, 4, 7, 21, 55, tzinfo=timezone.utc)

    def test_iso_with_offset(self):
        result = ft.parse_time_any("2026-04-07T23:55:00+02:00")
        assert result.utctimetuple()[:5] == (2026, 4, 7, 21, 55)

    def test_none_returns_none(self):
        assert ft.parse_time_any(None) is None

    def test_empty_string_returns_none(self):
        assert ft.parse_time_any("") is None

    def test_garbage_returns_none(self):
        assert ft.parse_time_any("not-a-date") is None


# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert ft.haversine(44.0, 20.0, 44.0, 20.0) == 0.0

    def test_belgrade_larnaca_approx(self):
        # BEG ~44.8°N 20.3°E  →  LCA ~34.9°N 33.6°E  ≈ 1490 km
        dist = ft.haversine(44.8, 20.3, 34.9, 33.6)
        assert 1400 < dist < 1600

    def test_symmetry(self):
        d1 = ft.haversine(44.8, 20.3, 34.9, 33.6)
        d2 = ft.haversine(34.9, 33.6, 44.8, 20.3)
        assert abs(d1 - d2) < 0.001


# ---------------------------------------------------------------------------
# classify_buffer
# ---------------------------------------------------------------------------

class TestClassifyBuffer:
    def test_on_time(self):
        assert ft.classify_buffer(60) == "ON_TIME"
        assert ft.classify_buffer(30) == "ON_TIME"

    def test_tight(self):
        assert ft.classify_buffer(29) == "TIGHT"
        assert ft.classify_buffer(0) == "TIGHT"

    def test_likely_delay(self):
        assert ft.classify_buffer(-1) == "LIKELY_DELAY"
        assert ft.classify_buffer(-60) == "LIKELY_DELAY"

    def test_none(self):
        assert ft.classify_buffer(None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# select_best_aerodatabox_flight
# ---------------------------------------------------------------------------

def _adb_record(dep_utc):
    """Build a minimal AeroDataBox-style flight record."""
    return {
        "departure": {
            "scheduledTime": {"utc": dep_utc},
            "iata": "BEG"
        },
        "arrival": {"iata": "LCA"}
    }


class TestSelectBestAerodataboxFlight:
    def test_returns_none_for_empty(self):
        assert ft.select_best_aerodatabox_flight([]) is None
        assert ft.select_best_aerodatabox_flight(None) is None

    def test_single_dict_returned_as_is(self):
        record = _adb_record("2026-04-07 21:55Z")
        assert ft.select_best_aerodatabox_flight(record) is record

    def test_picks_upcoming_over_past(self):
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%MZ")
        past   = (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%MZ")
        records = [_adb_record(past), _adb_record(future)]
        result = ft.select_best_aerodatabox_flight(records)
        assert result["departure"]["scheduledTime"]["utc"] == future

    def test_picks_closest_upcoming_among_multiple_future(self):
        now = datetime.now(timezone.utc)
        soon   = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%MZ")
        later  = (now + timedelta(hours=8)).strftime("%Y-%m-%d %H:%MZ")
        records = [_adb_record(later), _adb_record(soon)]
        result = ft.select_best_aerodatabox_flight(records)
        assert result["departure"]["scheduledTime"]["utc"] == soon

    def test_tolerates_recent_past_within_3h(self):
        # A flight that left 2h ago should still be preferred over one 48h ago
        now = datetime.now(timezone.utc)
        recent  = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%MZ")
        old     = (now - timedelta(hours=48)).strftime("%Y-%m-%d %H:%MZ")
        records = [_adb_record(old), _adb_record(recent)]
        result = ft.select_best_aerodatabox_flight(records)
        assert result["departure"]["scheduledTime"]["utc"] == recent

    def test_fallback_to_first_if_no_times(self):
        records = [{"departure": {}, "arrival": {}}, {"departure": {}, "arrival": {}}]
        result = ft.select_best_aerodatabox_flight(records)
        assert result is records[0]


# ---------------------------------------------------------------------------
# select_best_aviationstack_flight
# ---------------------------------------------------------------------------

def _avs_record(dep_iata, arr_iata, dep_scheduled):
    return {
        "departure": {"iata": dep_iata, "scheduled": dep_scheduled},
        "arrival":   {"iata": arr_iata},
    }


class TestSelectBestAviationstackFlight:
    def test_returns_none_for_empty(self):
        assert ft.select_best_aviationstack_flight([], {}) is None

    def test_picks_matching_route_over_wrong_route(self):
        hint = {"departure_iata": "BEG", "arrival_iata": "LCA", "departure_scheduled": "2026-04-07 21:55Z"}
        correct = _avs_record("BEG", "LCA", "2026-04-07T21:55:00Z")
        wrong   = _avs_record("LHR", "JFK", "2026-04-07T21:55:00Z")
        result = ft.select_best_aviationstack_flight([wrong, correct], hint)
        assert result["departure"]["iata"] == "BEG"

    def test_picks_closest_time_when_routes_match(self):
        hint = {"departure_iata": "BEG", "arrival_iata": "LCA", "departure_scheduled": "2026-04-07 21:55Z"}
        close = _avs_record("BEG", "LCA", "2026-04-07T21:55:00Z")
        far   = _avs_record("BEG", "LCA", "2026-04-07T08:00:00Z")
        result = ft.select_best_aviationstack_flight([far, close], hint)
        assert result["departure"]["scheduled"] == "2026-04-07T21:55:00Z"

    def test_no_hint_falls_back_to_first(self):
        records = [_avs_record("BEG", "LCA", "2026-04-07T21:55:00Z")]
        result = ft.select_best_aviationstack_flight(records, None)
        assert result is records[0]

    def test_wrong_arr_penalised(self):
        hint = {"departure_iata": "BEG", "arrival_iata": "LCA", "departure_scheduled": "2026-04-07 21:55Z"}
        correct = _avs_record("BEG", "LCA", "2026-04-07T21:55:00Z")
        wrong_arr = _avs_record("BEG", "JFK", "2026-04-07T21:55:00Z")
        result = ft.select_best_aviationstack_flight([wrong_arr, correct], hint)
        assert result["arrival"]["iata"] == "LCA"


# ---------------------------------------------------------------------------
# humanize_scenario / humanize_risk
# ---------------------------------------------------------------------------

class TestHumanize:
    def test_known_scenario(self):
        assert "heading" in ft.humanize_scenario("inbound_to_departure")

    def test_unknown_scenario_returns_fallback(self):
        result = ft.humanize_scenario("something_new")
        assert isinstance(result, str) and len(result) > 0

    def test_known_risk(self):
        assert "on track" in ft.humanize_risk("ON_TIME").lower()

    def test_unknown_risk_returns_key(self):
        assert ft.humanize_risk("WHATEVER") == "WHATEVER"
