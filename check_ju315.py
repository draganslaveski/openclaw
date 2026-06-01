#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/dragan-slaveski/.openclaw/workspace/skills/flight-tracker/venv/lib/python3.12/site-packages')

from FlightRadar24 import FlightRadar24API

fr_api = FlightRadar24API()

# Search for JU315
try:
    all_flights = fr_api.get_flights(airline="ASL")  # Air Serbia
    ju315_flights = [f for f in all_flights if f.number == "JU315"]
    
    if not ju315_flights:
        print("Not found in Air Serbia flights, trying broader search...")
        all_flights = fr_api.get_flights()
        ju315_flights = [f for f in all_flights if f.number == "JU315"]
    
    if ju315_flights:
        f = ju315_flights[0]
        print(f"✈️  Flight: {f.number}")
        print(f"🛩️  Aircraft Registration: {f.registration}")
        print(f"📍 Position: {f.latitude}, {f.longitude}")
        print(f"🛫 From: {f.origin_airport_iata}")
        print(f"🛬 To: {f.destination_airport_iata}")
        print(f"⛰️  Altitude: {f.altitude} ft")
        print(f"💨 Speed: {f.ground_speed} kt")
        print(f"🅿️  On ground: {f.on_ground}")
    else:
        print("JU315 not found on FR24")
except Exception as e:
    print(f"Error: {e}")
