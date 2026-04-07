---
name: flight-tracker
description: Use this skill to evaluate delay risk for a planned flight by identifying the assigned aircraft, checking where that aircraft is now, and estimating whether it can finish the remaining rotation and return to the departure airport on time.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
      env: ["OPENSKY_CLIENT_ID", "OPENSKY_CLIENT_SECRET", "AVIATIONSTACK_API_KEY", "AERODATABOX_API_KEY"]
---

## One-shot Delay Risk Workflow
When user asks to track a flight and estimate risk of delay:

1. Run scripts/flight_tracker.py with the flight number.
2. If user explicitly says a departure airport, pass it as second argument.
3. Summarize:
  - aircraft registration/sign (if available)
  - current aircraft position and current route leg
  - required remaining rotation to reach departure airport
  - delay risk verdict: ON_TIME, TIGHT, LIKELY_DELAY, or UNKNOWN

Examples:

```bash
python3 scripts/flight_tracker.py W61234
python3 scripts/flight_tracker.py W61234 BEG
```

## Cron Job Workflow
When user asks to track a flight and receive periodic updates:

1. Run scripts/flight_tracker.py to get current status and delay risk summary.
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
    "message": "Use flight-tracker skill to check flight W61234 (departure BEG), report delay risk based on aircraft current position + required return rotation, and send WhatsApp update. If flight has landed, delete this cron job."
  }
}
```
3. Confirm to user that tracking has started and they will receive updates every 30 minutes.
4. When flight lands (on_ground=true), delete the cron job automatically.