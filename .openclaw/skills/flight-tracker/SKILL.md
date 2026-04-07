---
name: flight-tracker
description: Use this skill to track live position of any aircraft or flight in real-time, get current altitude speed heading and ETA to destination airport using OpenSky Network ADS-B data. Use when user asks to track a flight, where is my plane, how far is my flight, when will my plane arrive, is my flight close to airport.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
      env: ["OPENSKY_CLIENT_ID", "OPENSKY_CLIENT_SECRET"]
---

## Cron Job Workflow
When user asks to track a flight and receive periodic updates:

1. Run scripts/flight_tracker.py to get current status
2. Create a cron job entry in ~/.openclaw/cron/jobs.json with this structure:
```json
{
  "name": "flight-tracker-W61234",
  "schedule": {
    "kind": "cron",
    "expr": "*/30 * * * *"
  },
  "sessionTarget": "isolated",
  "payload": {
    "kind": "agentTurn",
    "message": "Use flight-tracker skill to check flight W61234 to BEG and send me WhatsApp update with current position and ETA. If flight has landed, delete this cron job."
  }
}
```
3. Confirm to user that tracking has started and they will receive updates every 30 minutes.
4. When flight lands (on_ground=true), delete the cron job automatically.