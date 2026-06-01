#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/dragan-slaveski/.openclaw/workspace/skills/flight-tracker/venv/lib/python3.12/site-packages')

from FlightRadar24 import FlightRadar24API

fr_api = FlightRadar24API()

# Get all ASL (Air Serbia) flights
try:
    print("Fetching ASL flights...\n")
    all_flights = fr_api.get_flights(airline="ASL")
    
    # Filter for flights to BEG
    beg_flights = [f for f in all_flights if f.destination_airport_iata == "BEG"]
    
    print(f"Found {len(beg_flights)} ASL flights to BEG:\n")
    for f in beg_flights:
        print(f"Flight: {f.number}")
        print(f"  Registration: {f.registration}")
        print(f"  From: {f.origin_airport_iata}")
        print(f"  To: {f.destination_airport_iata}")
        print(f"  Altitude: {f.altitude} ft")
        print(f"  Speed: {f.ground_speed} kt")
        print(f"  On ground: {f.on_ground}")
        print(f"  Position: {f.latitude:.4f}, {f.longitude:.4f}")
        print()
    
    if not beg_flights:
        print("No ASL flights to BEG found. Showing all ASL flights:\n")
        for f in all_flights[:20]:
            print(f"{f.number} ({f.registration}) {f.origin_airport_iata}→{f.destination_airport_iata}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
