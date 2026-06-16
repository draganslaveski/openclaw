---
name: border-tracker
description: Check border camera queue status and run interval monitoring flows that can be triggered from WhatsApp.
metadata:
  openclaw:
    requires:
      bins: ["python3", "crontab"]
---

## Flow 1: Border Status (One-shot)
Use this when user asks for current border status for one camera or all cameras.

Run:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py status \
  --flow-name status \
  --camera "Bajakovo Entry" \
  --cameras-file /home/dragan-slaveski/.openclaw/workspace/border-dataset/cameras.json \
  --models-dir /home/dragan-slaveski/.openclaw/workspace/border-dataset/models \
  --history-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl \
  --output-json /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/latest_all.json \
  --save-debug-dir /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshots
```

Notes:
- You can use camera names (preferred), camera ids, or `all`.
- Default camera is `Bajakovo Entry`.
- The latest available model is auto-selected (current_queue_model.pt, fallback queue_model_best.joblib).
- For status responses, return concise queue status only.
- Do not include snapshot file path in user-facing response unless explicitly requested.

## Flow 2: Border Monitoring (Interval)
Use this when user asks to keep monitoring a border camera periodically.

Monitoring must use local snapshot capture only (no LLM summarization on each run).

1. Create or update Linux system cron job:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py upsert-system-cron \
  --camera "Bajakovo Entry" \
  --interval-min 30
```

2. Confirm to user that periodic monitoring is enabled.

What gets scheduled:
- A Linux crontab entry for the current user.
- It runs `capture-snapshot` and writes snapshots to `workspace/skills/border-tracker/state/snapshots`.
- It appends capture metadata to `workspace/skills/border-tracker/state/snapshot_index.jsonl`.
- No per-run channel message should be sent.

## Flow 3: Disable Border Monitoring
Use this when user asks to stop monitoring for a camera.

Run:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py disable-system-cron \
  --camera "Bajakovo Entry"
```

Notes:
- This disables existing system cron entries tagged for the selected camera.
- Use `--camera all` to disable all border monitor cron entries.

## WhatsApp Intent Routing (Start/Stop)
For WhatsApp messages, map intents directly to system cron commands:

- Start monitoring intents (examples: "start monitoring", "monitor every 30 min", "enable border monitoring"):

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py upsert-system-cron \
  --camera "Bajakovo Entry" \
  --interval-min 30
```

- Stop monitoring intents (examples: "stop monitoring", "disable border monitoring", "stop 30-min checks"):

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py disable-system-cron \
  --camera "Bajakovo Entry"
```

- Stop all monitoring intents (examples: "stop all border monitoring"):

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py disable-system-cron \
  --camera all
```

Always confirm to the user whether monitoring is now enabled or disabled.

## Pattern Query (Later)
When user asks for observed patterns, run local model inference over saved snapshots first, then summarize:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py patterns \
  --history-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl \
  --snapshot-index-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl \
  --models-dir /home/dragan-slaveski/.openclaw/workspace/border-dataset/models \
  --camera "Bajakovo Entry"
```

This prints intervals where extreme queue predictions are more common, using:
- existing labeled history rows when available, and
- fresh local inference from saved snapshots.

Patterns output also reports unavailable captures filtered from analysis:
- `Unavailable captures (filtered): N`
- `Snapshot records in window: N`
- `Snapshot status split: ok=X, unavailable=Y, error=Z`

When present, these lines must be treated as data-quality guards:
- `Insufficient coverage hours ...`
- `Trend interpretation note: do not treat insufficient coverage hours as quiet/low-traffic windows.`

For user-facing summaries, do not call those hours "quietest" or "best time"; report them as unknown due to downtime.

Time handling for pattern responses:
- Hour buckets must be interpreted and reported in local machine timezone (user timezone), not UTC.

Only after this local analysis should LLM be used to format/summarize the response for WhatsApp.
Do not include snapshot links/files for patterns responses.

## Retroactive Unavailable Backfill
If historical snapshots were saved before unavailable-frame filtering was added, retroactively patch statuses:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py backfill-unavailable \
  --history-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl \
  --snapshot-index-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl \
  --snapshots-dir /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshots \
  --apply
```

This marks matching historical rows as `status=unavailable` with reason metadata.

## Unavailable Camera Query
When user asks when camera was unavailable/downtime periods, run:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py unavailable-summary \
  --camera "Bajakovo Entry" \
  --history-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl \
  --snapshot-index-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl
```

For explicit windows, add `--hours`:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py unavailable-summary \
  --camera "Bajakovo Entry" \
  --history-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl \
  --snapshot-index-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl \
  --hours 24
```

## Snapshot Availability Query
When user asks if snapshots exist / how many snapshots exist in a window, run:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py snapshot-summary \
  --camera "Bajakovo Entry" \
  --snapshot-index-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl \
  --hours 12
```

This reports:
- total snapshot records in the window,
- status split (`ok`, `unavailable`, `error`),
- first/last OK snapshot timestamps,
- most recent snapshot records.

For combined asks (trend + unavailable in one message):
- Run both `patterns` and `unavailable-summary` with the same camera/window.
- Return a combined response that includes both trend and downtime sections.

For explicit time-window trend requests, pass `--hours`:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py patterns \
  --history-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/history.jsonl \
  --snapshot-index-file /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/state/snapshot_index.jsonl \
  --models-dir /home/dragan-slaveski/.openclaw/workspace/border-dataset/models \
  --camera "Bajakovo Entry" \
  --hours 8
```

Intent precedence for WhatsApp:
- If user asks for trend/pattern/history (for example "trend for last 8h"), use `patterns` (with `--hours` when provided).
- Use `status` only for current-state requests (for example "border status now").
