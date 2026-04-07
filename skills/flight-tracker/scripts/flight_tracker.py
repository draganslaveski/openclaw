#!/usr/bin/env python3
import sys
import os

VENV_SITE_PACKAGES = '/home/dragan-slaveski/.openclaw/workspace/skills/flight-tracker/venv/lib/python3.12/site-packages'
if VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

import math
import requests
from datetime import datetime, timezone, timedelta

OPENSKY_CLIENT_ID = os.environ.get("OPENSKY_CLIENT_ID")
OPENSKY_CLIENT_SECRET = os.environ.get("OPENSKY_CLIENT_SECRET")
AVIATIONSTACK_API_KEY = os.environ.get("AVIATIONSTACK_API_KEY")
AERODATABOX_API_KEY = os.environ.get("AERODATABOX_API_KEY")


def parse_time_any(value):
    """Parse provider timestamps with graceful fallback."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    # AeroDataBox UTC often comes as "YYYY-MM-DD HH:MMZ"
    for fmt in ("%Y-%m-%d %H:%MZ", "%Y-%m-%d %H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # ISO formats from providers (sometimes with trailing Z)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def select_best_aerodatabox_flight(data):
    """Pick the schedule entry that best matches the upcoming/current leg."""
    if not data:
        return None
    if isinstance(data, dict):
        return data

    now = datetime.now(timezone.utc)
    best = None
    best_score = None

    for item in data:
        dep = item.get("departure", {}) if isinstance(item, dict) else {}
        dep_utc = dep.get("scheduledTime", {}).get("utc") if isinstance(dep, dict) else None
        dep_time = parse_time_any(dep_utc)
        if not dep_time:
            continue

        # Prefer future departures; tolerate recent past if flight just left.
        minutes = (dep_time - now).total_seconds() / 60
        if minutes >= -180:
            score = abs(minutes)
        else:
            score = abs(minutes) + 10000

        if best is None or score < best_score:
            best = item
            best_score = score

    return best or data[0]


def select_best_aviationstack_flight(items, schedule_hint):
    """Pick aviationstack record matching the selected schedule leg."""
    if not items:
        return None

    hint_dep = (schedule_hint or {}).get("departure_iata")
    hint_arr = (schedule_hint or {}).get("arrival_iata")
    hint_dep_time = parse_time_any((schedule_hint or {}).get("departure_scheduled"))

    now = datetime.now(timezone.utc)
    best = None
    best_score = None

    for f in items:
        dep_iata = ((f.get("departure") or {}).get("iata") or "").upper()
        arr_iata = ((f.get("arrival") or {}).get("iata") or "").upper()
        dep_time = parse_time_any((f.get("departure") or {}).get("scheduled"))

        score = 0
        if hint_dep and dep_iata and dep_iata != hint_dep.upper():
            score += 5000
        if hint_arr and arr_iata and arr_iata != hint_arr.upper():
            score += 5000

        if dep_time and hint_dep_time:
            score += abs((dep_time - hint_dep_time).total_seconds()) / 60
        elif dep_time:
            score += abs((dep_time - now).total_seconds()) / 60
        else:
            score += 1000

        if best is None or score < best_score:
            best = f
            best_score = score

    return best or items[0]


def get_fr24_api():
    sys.path.insert(0, VENV_SITE_PACKAGES)
    from FlightRadar24 import FlightRadar24API
    return FlightRadar24API()

def get_token():
    response = requests.post(
        "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": OPENSKY_CLIENT_ID,
            "client_secret": OPENSKY_CLIENT_SECRET
        },
        timeout=10
    )
    response.raise_for_status()
    return response.json()["access_token"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def iata_to_icao(iata):
    if not iata:
        return None
    try:
        response = requests.get(
            f"https://airport-data.com/api/ap_info.json?iata={iata.upper()}",
            timeout=10
        )
        data = response.json()
        if data.get("icao"):
            return data["icao"]
    except Exception:
        pass
    return None

def icao_to_iata(icao):
    if not icao or icao == '?':
        return '?'
    if len(icao) == 3:
        return icao
    try:
        response = requests.get(
            f"https://airport-data.com/api/ap_info.json?icao={icao.upper()}",
            timeout=10
        )
        data = response.json()
        if data.get("iata"):
            return data["iata"]
    except Exception:
        pass
    return icao[-3:]

def get_airport_coords(icao):
    if not icao:
        return None
    try:
        response = requests.get(
            f"https://airport-data.com/api/ap_info.json?icao={icao}",
            timeout=10
        )
        data = response.json()
        if data.get("latitude") and data.get("longitude"):
            return {
                "name": data.get("name", icao),
                "lat": float(data["latitude"]),
                "lon": float(data["longitude"])
            }
    except Exception:
        pass
    return None

def estimate_arrival(position, airport_icao):
    airport = get_airport_coords(airport_icao)
    if not airport or not position.get("latitude") or not position.get("longitude"):
        return None
    distance = haversine(
        position["latitude"], position["longitude"],
        airport["lat"], airport["lon"]
    )
    speed_kmh = position["velocity"] * 3.6 if position.get("velocity") else 0
    if speed_kmh < 10:
        return None
    minutes = (distance / speed_kmh) * 60
    return {
        "airport": airport["name"],
        "airport_icao": airport_icao,
        "distance_km": round(distance),
        "speed_kmh": round(speed_kmh),
        "eta_minutes": round(minutes)
    }

def _resolve_scheduled_utc(time_block):
    """
    Derive a UTC ISO string from an AeroDataBox scheduledTime block.

    AeroDataBox's "utc" field is sometimes populated with local wall-clock time
    (a known data quality issue), which causes departure countdowns to be off by
    the local UTC offset (e.g. +2 h for CET/CEST).  The "local" field always
    carries an explicit offset (+HH:MM), so converting that to UTC is more
    reliable.  We prefer that path and fall back to the raw "utc" string only
    when the local field is unavailable or lacks an offset.
    """
    local_str = time_block.get("local") if isinstance(time_block, dict) else None
    utc_str = (time_block.get("utc", "") or "") if isinstance(time_block, dict) else ""

    if local_str:
        local_dt = parse_time_any(local_str)
        if local_dt and local_dt.utcoffset() is not None:
            utc_dt = local_dt.astimezone(timezone.utc)
            return utc_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Fall back to raw UTC field
    if utc_str:
        return utc_str.replace("Z", "+00:00").replace(" ", "T")
    return ""


def get_flight_schedule_aerodatabox(flight_number):
    """Get today's flight schedule from AeroDataBox."""
    try:
        response = requests.get(
            f"https://aerodatabox.p.rapidapi.com/flights/number/{flight_number.upper()}",
            headers={
                "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
                "x-rapidapi-key": AERODATABOX_API_KEY
            },
            params={"withLocation": "true"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            return None

        flight = select_best_aerodatabox_flight(data)
        dep = flight.get("departure", {})
        arr = flight.get("arrival", {})

        departure_scheduled = _resolve_scheduled_utc(dep.get("scheduledTime", {}))
        arrival_scheduled = _resolve_scheduled_utc(arr.get("scheduledTime", {}))

        # Calculate flight duration from resolved UTC times
        duration_min = None
        d = parse_time_any(departure_scheduled)
        a = parse_time_any(arrival_scheduled)
        if d and a:
            duration_min = round((a - d).total_seconds() / 60)

        return {
            "flight_iata": flight.get("number", "").replace(" ", ""),
            "departure_iata": dep.get("airport", {}).get("iata"),
            "departure_icao": dep.get("airport", {}).get("icao"),
            "departure_airport": dep.get("airport", {}).get("name"),
            "departure_scheduled": departure_scheduled,
            "departure_scheduled_local": dep.get("scheduledTime", {}).get("local"),
            "arrival_iata": arr.get("airport", {}).get("iata"),
            "arrival_icao": arr.get("airport", {}).get("icao"),
            "arrival_airport": arr.get("airport", {}).get("name"),
            "arrival_scheduled": arrival_scheduled,
            "arrival_scheduled_local": arr.get("scheduledTime", {}).get("local"),
            "flight_duration_min": duration_min,
            "status": flight.get("status"),
            "aircraft_model": flight.get("aircraft", {}).get("model"),
            "aircraft_icao24": None,
            "aircraft_registration": None,
            "live": None
        }
    except Exception as e:
        print(f"AeroDataBox error: {e}")
        return None


def get_flight_schedule_aviationstack(flight_number, schedule_hint=None):
    """Get flight schedule + aircraft icao24 from AviationStack."""
    try:
        response = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params={
                "access_key": AVIATIONSTACK_API_KEY,
                "flight_iata": flight_number.upper(),
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("data") or []
        if not items:
            return None

        flight = select_best_aviationstack_flight(items, schedule_hint or {})
        return {
            "aircraft_icao24": flight.get("aircraft", {}).get("icao24"),
            "aircraft_registration": flight.get("aircraft", {}).get("registration"),
            "live": flight.get("live"),
        }
    except Exception as e:
        print(f"AviationStack error: {e}")
        return None

def get_position_fr24(flight_number):
    """Get current position using FlightRadar24."""
    try:
        fr_api = get_fr24_api()
        all_flights = fr_api.get_flights(airline="ASL")
        flights = [f for f in all_flights if f.number == flight_number.upper()]
        if not flights:
            # Try broader search
            all_flights2 = fr_api.get_flights()
            flights = [f for f in all_flights2 if f.number == flight_number.upper()]
        if not flights:
            return None
        f = flights[0]
        return {
            "callsign": f.callsign,
            "longitude": f.longitude,
            "latitude": f.latitude,
            "altitude": f.altitude * 0.3048,  # feet to meters
            "velocity": f.ground_speed * 0.514444,  # knots to m/s
            "heading": f.heading,
            "on_ground": bool(f.on_ground),
            "last_update": datetime.fromtimestamp(f.time).strftime("%H:%M:%S"),
            "registration": f.registration,
            "origin": f.origin_airport_iata,
            "destination": f.destination_airport_iata,
        }
    except Exception as e:
        print(f"FlightRadar24 error: {e}")
        return None

def get_position_opensky(icao24):
    """Get current position from OpenSky."""
    if not icao24:
        return None
    try:
        token = get_token()
        response = requests.get(
            "https://opensky-network.org/api/states/all",
            headers={"Authorization": f"Bearer {token}"},
            params={"icao24": icao24.lower()},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("states"):
            return None
        state = data["states"][0]
        return {
            "callsign": state[1].strip() if state[1] else "Unknown",
            "longitude": state[5],
            "latitude": state[6],
            "altitude": state[7],
            "velocity": state[9],
            "heading": state[10],
            "on_ground": state[8],
            "last_update": datetime.fromtimestamp(state[3]).strftime("%H:%M:%S")
        }
    except Exception as e:
        print(f"OpenSky error: {e}")
        return None

def enrich_position_from_registration_fr24(position):
    """Fill origin/destination from FR24 using aircraft registration when missing."""
    registration = (position or {}).get("registration")
    if not registration:
        return position
    try:
        fr_api = get_fr24_api()
        reg_upper = registration.upper()
        for f in fr_api.get_flights():
            if (f.registration or "").upper() == reg_upper:
                if not position.get("origin") and getattr(f, "origin_airport_iata", None):
                    position["origin"] = f.origin_airport_iata
                if not position.get("destination") and getattr(f, "destination_airport_iata", None):
                    position["destination"] = f.destination_airport_iata
                if not position.get("callsign") and getattr(f, "callsign", None):
                    position["callsign"] = f.callsign
                break
    except Exception as e:
        print(f"FR24 registration lookup error: {e}")
    return position

def classify_buffer(buffer_minutes):
    if buffer_minutes is None:
        return "UNKNOWN"
    if buffer_minutes >= 30:
        return "ON_TIME"
    if buffer_minutes >= 0:
        return "TIGHT"
    return "LIKELY_DELAY"

def humanize_scenario(scenario):
    labels = {
        "already_departed": "Your flight has already started boarding/departure flow",
        "inbound_to_departure": "The aircraft is already heading to your departure airport",
        "full_rotation": "The aircraft still needs to finish another leg before your departure",
        "missing_schedule": "Schedule details are incomplete right now",
    }
    return labels.get(scenario, "Current aircraft situation")

def humanize_risk(risk):
    labels = {
        "ON_TIME": "Looks good: on track to arrive on time",
        "TIGHT": "Could still make it, but timing is tight",
        "LIKELY_DELAY": "High chance of delay based on current position",
        "IN_AIR_OR_DEPARTED": "Flight is airborne — no pre-departure risk check needed",
        "UNKNOWN": "Not enough live data yet for a confident estimate",
    }
    return labels.get(risk, risk)

def assess_inflight_delay(position, schedule):
    """
    Assess arrival delay for a flight that is currently airborne.

    Uses the aircraft's current coordinates + ground speed to estimate ETA to
    the arrival airport, then compares that against the scheduled arrival time.
    """
    arrival_iata = schedule.get("arrival_iata")
    arrival_icao = schedule.get("arrival_icao") or iata_to_icao(arrival_iata)
    arrival_scheduled = schedule.get("arrival_scheduled")

    # Derive scheduled arrival from departure + duration if direct field missing
    if not arrival_scheduled:
        dep_scheduled = schedule.get("departure_scheduled")
        duration_min = schedule.get("flight_duration_min")
        if dep_scheduled and duration_min:
            dep_dt = parse_time_any(dep_scheduled)
            if dep_dt:
                arr_dt = dep_dt + timedelta(minutes=duration_min)
                arrival_scheduled = arr_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    if not arrival_icao or not arrival_scheduled:
        print("   ⚠️  Cannot assess in-flight delay (missing arrival airport or schedule)")
        return None

    if position.get("on_ground"):
        print("   ℹ️  Aircraft is on the ground — skipping in-flight ETA check")
        return None

    eta = estimate_arrival(position, arrival_icao)
    if not eta:
        print(f"   ⚠️  Cannot calculate ETA to {arrival_iata or arrival_icao} (missing position/speed)")
        return None

    arr_time = parse_time_any(arrival_scheduled)
    if not arr_time:
        print("   ⚠️  Cannot parse scheduled arrival time")
        return None

    now = datetime.now(timezone.utc)
    scheduled_in_min = (arr_time - now).total_seconds() / 60
    buffer = scheduled_in_min - eta["eta_minutes"]
    risk = classify_buffer(buffer)

    print(f"\n✈️  In-flight delay assessment:")
    print(f"   ETA to {eta['airport']}: ~{eta['eta_minutes']} min ({eta['distance_km']} km at {eta['speed_kmh']} km/h)")
    print(f"   Scheduled arrival in: ~{round(scheduled_in_min)} min")
    if buffer >= 30:
        print(f"   ✅ ON TIME — {round(buffer)} min ahead of schedule")
    elif buffer >= 0:
        print(f"   ⚠️  TIGHT — only {round(buffer)} min to spare")
    else:
        print(f"   🔴 LIKELY DELAY — ~{round(abs(buffer))} min late based on current position")

    return {
        "scenario": "inflight",
        "risk": risk,
        "buffer_minutes": round(buffer),
        "eta_to_arrival_minutes": eta["eta_minutes"],
        "scheduled_arrival_in_minutes": round(scheduled_in_min),
        "arrival_iata": arrival_iata,
    }

def assess_rotation_delay(position, schedule):
    """Assess delay risk based on aircraft rotation scenario."""
    departure_iata = schedule.get("departure_iata")
    departure_scheduled = schedule.get("departure_scheduled")
    if not departure_iata or not departure_scheduled:
        return {
            "scenario": "missing_schedule",
            "risk": "UNKNOWN",
            "buffer_minutes": None,
        }

    dep_time = datetime.fromisoformat(departure_scheduled.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    flight_dest = position.get("destination") or ""
    flight_origin = position.get("origin") or ""

    # Scenario 1: Flight already departed — assess in-flight delay if airborne
    if flight_origin == departure_iata and now > dep_time - timedelta(minutes=30):
        print(f"\n✅ Flight has departed from {departure_iata}")
        inflight = assess_inflight_delay(position, schedule)
        if inflight:
            return inflight
        return {
            "scenario": "already_departed",
            "risk": "IN_AIR_OR_DEPARTED",
            "buffer_minutes": None,
        }

    # Scenario 2: Aircraft inbound to departure airport
    dep_icao = schedule.get("departure_icao") or iata_to_icao(departure_iata)
    if flight_dest == departure_iata and not position.get("on_ground"):
        print(f"\n📡 Aircraft inbound to {departure_iata}")
        eta = estimate_arrival(position, dep_icao)
        if eta:
            print(f"🛬 ETA to {eta['airport']}: ~{eta['eta_minutes']} min ({eta['distance_km']} km)")
            minutes_until_dep = (dep_time - now).total_seconds() / 60
            turnaround = 50
            buffer = minutes_until_dep - eta["eta_minutes"] - turnaround
            risk = classify_buffer(buffer)
            if buffer >= 30:
                print(f"   ✅ ON TIME — {round(buffer)} min buffer")
            elif buffer >= 0:
                print(f"   ⚠️  TIGHT — only {round(buffer)} min buffer")
            else:
                print(f"   🔴 LIKELY DELAY — {round(abs(buffer))} min short")
            return {
                "scenario": "inbound_to_departure",
                "risk": risk,
                "buffer_minutes": round(buffer),
                "eta_to_departure_minutes": eta["eta_minutes"],
                "turnaround_minutes": turnaround,
                "minutes_until_departure": round(minutes_until_dep),
            }
        return {
            "scenario": "inbound_to_departure",
            "risk": "UNKNOWN",
            "buffer_minutes": None,
        }

    # Scenario 3: Full rotation analysis
    print(f"\n📡 Aircraft not yet inbound to {departure_iata}, full rotation analysis...")
    return assess_full_rotation_delay(position, schedule)

def assess_full_rotation_delay(position, schedule):
    """Full rotation: ETA to current dest + turnaround + return flight + turnaround."""
    departure_iata = schedule.get("departure_iata")
    departure_scheduled = schedule.get("departure_scheduled")
    current_dest_iata = position.get("destination")
    registration = position.get("registration")

    if not all([departure_iata, departure_scheduled, current_dest_iata, registration]):
        missing = [k for k, v in {
            "departure_iata": departure_iata,
            "departure_scheduled": departure_scheduled,
            "current_dest_iata": current_dest_iata,
            "registration": registration
        }.items() if not v]
        print(f"   ⚠️  Missing data for rotation analysis: {', '.join(missing)}")
        return {
            "scenario": "full_rotation",
            "risk": "UNKNOWN",
            "buffer_minutes": None,
            "missing": missing,
        }

    dep_time = datetime.fromisoformat(departure_scheduled.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    minutes_until_departure = (dep_time - now).total_seconds() / 60

    local_tz = timezone(timedelta(hours=2))
    print(f"\n⏱️  Full rotation analysis:")
    print(f"   Scheduled departure: {dep_time.astimezone(local_tz).strftime('%H:%M')} local")
    print(f"   Time until departure: {round(minutes_until_departure)} min")

    # Step 1: ETA to current destination
    dest_icao = iata_to_icao(current_dest_iata)
    eta = estimate_arrival(position, dest_icao) if dest_icao else None
    if not eta:
        print(f"   ⚠️  Cannot calculate ETA to {current_dest_iata}")
        return {
            "scenario": "full_rotation",
            "risk": "UNKNOWN",
            "buffer_minutes": None,
            "missing": ["eta_to_current_destination"],
        }
    eta_to_dest = eta["eta_minutes"]
    print(f"\n   1️⃣  ETA to {current_dest_iata}: ~{eta_to_dest} min")

    # Step 2: Turnaround at destination
    turnaround_dest = 55
    print(f"   2️⃣  Turnaround at {current_dest_iata}: ~{turnaround_dest} min")

    # Step 3: Return flight duration
    return_duration = None
    try:
        fr_api = get_fr24_api()
        dest_airport = fr_api.get_airport(current_dest_iata, details=True)
        departures_data = dest_airport.departures.get('data', [])
        for item in departures_data:
            if not item:
                continue
            f = item.get('flight', {})
            reg = f.get('aircraft', {}).get('registration', '')
            dest_info = f.get('airport', {}).get('destination', {})
            dest_arr_iata = dest_info.get('code', {}).get('iata', '') if isinstance(dest_info, dict) else ''
            if reg == registration and dest_arr_iata == departure_iata:
                duration_sec = f.get('time', {}).get('other', {}).get('duration')
                if duration_sec:
                    return_duration = duration_sec // 60
                    number = f.get('identification', {}).get('number', {}).get('default', '')
                    dep_ts = f.get('time', {}).get('scheduled', {}).get('departure')
                    dep_str = datetime.fromtimestamp(dep_ts).strftime('%H:%M') if dep_ts else '?'
                    print(f"   3️⃣  Return flight {number} {current_dest_iata}→{departure_iata} dep:{dep_str} duration:~{return_duration} min")
                    break
    except Exception as e:
        print(f"   ⚠️  FR24 departures error: {e}")

    if not return_duration:
        return_duration = schedule.get("flight_duration_min")
        if not return_duration:
            dep_coords = get_airport_coords(iata_to_icao(departure_iata))
            dest_coords = get_airport_coords(dest_icao)
            if dep_coords and dest_coords:
                dist = haversine(dep_coords["lat"], dep_coords["lon"],
                               dest_coords["lat"], dest_coords["lon"])
                return_duration = round((dist / 800) * 60)
        print(f"   3️⃣  Return {current_dest_iata}→{departure_iata}: ~{return_duration} min (estimated)")

    # Step 4: Turnaround at departure airport
    turnaround_dep = 55
    print(f"   4️⃣  Turnaround at {departure_iata}: ~{turnaround_dep} min")

    # Final assessment
    total_needed = eta_to_dest + turnaround_dest + (return_duration or 0) + turnaround_dep
    buffer = minutes_until_departure - total_needed
    risk = classify_buffer(buffer)
    print(f"\n   📊 Total needed: ~{total_needed} min | Available: ~{round(minutes_until_departure)} min")
    if buffer >= 30:
        print(f"   ✅ ON TIME — {round(buffer)} min buffer")
    elif buffer >= 0:
        print(f"   ⚠️  TIGHT — only {round(buffer)} min buffer")
    else:
        print(f"   🔴 LIKELY DELAY — {round(abs(buffer))} min short")
    return {
        "scenario": "full_rotation",
        "risk": risk,
        "buffer_minutes": round(buffer),
        "eta_to_current_destination_minutes": eta_to_dest,
        "return_duration_minutes": return_duration,
        "turnaround_destination_minutes": turnaround_dest,
        "turnaround_departure_minutes": turnaround_dep,
        "minutes_until_departure": round(minutes_until_departure),
        "current_destination_iata": current_dest_iata,
        "departure_iata": departure_iata,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: flight_tracker.py <flight_number> [departure_iata]")
        print("Example: flight_tracker.py JU242 BEG")
        sys.exit(1)

    flight_number = sys.argv[1].upper()
    departure_override = sys.argv[2].upper() if len(sys.argv) >= 3 else None

    print(f"\n{'='*50}")
    print(f"🔍 Analyzing flight {flight_number}")
    print(f"{'='*50}\n")

    # Step 1: Schedule from AeroDataBox + icao24 from AviationStack
    print("📋 Fetching flight schedule...")
    schedule = get_flight_schedule_aerodatabox(flight_number)
    if not schedule:
        print("❌ Could not find schedule")
        sys.exit(1)

    av = get_flight_schedule_aviationstack(flight_number, schedule)
    if av:
        schedule["aircraft_icao24"] = av.get("aircraft_icao24")
        schedule["aircraft_registration"] = av.get("aircraft_registration")
        schedule["live"] = av.get("live")

    if departure_override:
        schedule["departure_iata"] = departure_override
        schedule["departure_icao"] = iata_to_icao(departure_override)
        print(f"🧭 Departure override: {departure_override}")

    print(f"✈️  Flight: {schedule['flight_iata']}")
    print(f"🛫 From: {schedule['departure_airport']} ({schedule['departure_iata']})")
    print(f"🛬 To: {schedule['arrival_airport']} ({schedule['arrival_iata']})")
    print(f"🕐 Scheduled departure: {schedule.get('departure_scheduled_local') or schedule.get('departure_scheduled')}")
    print(f"📊 Status: {schedule['status']}")
    if schedule.get('aircraft_registration'):
        print(f"🛩️  Aircraft: {schedule['aircraft_registration']}")
    if schedule.get('aircraft_model'):
        print(f"✈️  Model: {schedule['aircraft_model']}")

    # Step 2: Get current position
    # Priority: AviationStack live → FR24 → OpenSky
    position = None
    live = schedule.get("live")
    if live and live.get("latitude"):
        use_live = True
        dep_dt = parse_time_any(schedule.get("departure_scheduled"))
        dep_iata = schedule.get("departure_iata")
        dep_icao = schedule.get("departure_icao") or iata_to_icao(dep_iata)

        # Guard against stale live snapshots from a different leg with same flight number.
        if dep_dt and dep_dt > datetime.now(timezone.utc) + timedelta(minutes=20) and dep_icao:
            dep_airport = get_airport_coords(dep_icao)
            if dep_airport and live.get("longitude") is not None and live.get("latitude") is not None:
                distance_from_dep = haversine(
                    float(live.get("latitude")),
                    float(live.get("longitude")),
                    dep_airport["lat"],
                    dep_airport["lon"],
                )
                if distance_from_dep > 150:
                    use_live = False
                    print(f"📡 Ignoring stale live record ({round(distance_from_dep)} km from departure airport)")

        if use_live:
            position = {
                "callsign": schedule.get("callsign", ""),
                "longitude": live.get("longitude"),
                "latitude": live.get("latitude"),
                "altitude": live.get("altitude") or 0,
                "velocity": (live.get("speed_horizontal") or 0) / 3.6,
                "heading": live.get("direction", 0),
                "on_ground": live.get("is_ground", False),
                "last_update": live.get("updated", ""),
                "registration": schedule.get("aircraft_registration"),
                "origin": schedule.get("departure_iata"),
                "destination": schedule.get("arrival_iata"),
            }
            print("📡 Using AviationStack live data")

    if not position:
        print("📡 Trying FlightRadar24...")
        position = get_position_fr24(flight_number)
        if not position:
            print("📡 Falling back to OpenSky...")
            position = get_position_opensky(schedule.get("aircraft_icao24"))

    # Enrich position with schedule data if missing
    if position:
        if not position.get("registration"):
            position["registration"] = schedule.get("aircraft_registration")
        if not position.get("destination"):
            position["destination"] = schedule.get("arrival_iata")
        if not position.get("origin"):
            position["origin"] = schedule.get("departure_iata")

        # OpenSky often lacks route fields; enrich from FR24 by registration.
        if (not position.get("origin") or not position.get("destination")) and position.get("registration"):
            position = enrich_position_from_registration_fr24(position)

    # Step 3: Display position
    if position:
        alt_ft = round(position["altitude"] * 3.28084) if position.get("altitude") else 0
        speed_kmh = round(position["velocity"] * 3.6) if position.get("velocity") else 0

        if position.get("on_ground"):
            print(f"🅿️  Aircraft on the ground")
            print(f"📍 Position: {round(position['latitude'], 4)}°N, {round(position['longitude'], 4)}°E")
        else:
            print(f"✈️  Aircraft airborne")
            print(f"📍 Position: {round(position['latitude'], 4)}°N, {round(position['longitude'], 4)}°E")
            print(f"🔼 Altitude: {alt_ft} ft")
            print(f"💨 Speed: {speed_kmh} km/h")
            print(f"🧭 Heading: {round(position['heading'])}°")
            print(f"🕐 Last update: {position['last_update']}")

        # Step 4: Delay risk assessment
        risk_result = assess_rotation_delay(position, schedule)
        if risk_result:
            scenario = risk_result.get('scenario', 'unknown')
            risk = risk_result.get('risk', 'UNKNOWN')
            if scenario == "already_departed" and risk == "IN_AIR_OR_DEPARTED":
                # Couldn't produce in-flight estimate (e.g. on ground, no arrival data)
                print("\n🧾 Delay outlook")
                print(f"   {humanize_scenario(scenario)}")
                print(f"   {humanize_risk(risk)}")
            elif scenario not in ("already_departed", "inflight"):
                print("\n🧾 Delay outlook")
                print(f"   {humanize_scenario(scenario)}")
                print(f"   {humanize_risk(risk)}")
                if risk_result.get("buffer_minutes") is not None:
                    print(f"   Buffer: {risk_result['buffer_minutes']} min")
            else:
                # inflight scenario: summary already printed inside assess_inflight_delay
                print("\n🧾 Delay outlook")
                print(f"   {humanize_risk(risk)}")
                if risk_result.get("buffer_minutes") is not None:
                    print(f"   Buffer: {risk_result['buffer_minutes']} min")
    else:
        print("⚠️  Could not get current position")

    print(f"\n{'='*50}\n")

if __name__ == "__main__":
    main()