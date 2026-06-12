---
name: border-tracker
description: Check border camera queue status and run interval monitoring flows that can be triggered from WhatsApp.
metadata:
  openclaw:
    requires:
      bins: ["python3"]
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
- For status responses, include only the saved snapshot file path (no URL).

## Flow 2: Border Monitoring (Interval)
Use this when user asks to keep monitoring a border camera periodically.

Monitoring must use local snapshot capture only (no LLM summarization on each run).

1. Create or update cron job:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py upsert-monitor-job \
  --camera "Bajakovo Entry" \
  --interval-min 30 \
  --jobs-file /home/dragan-slaveski/.openclaw/cron/jobs.json
```

2. Confirm to user that periodic monitoring is enabled.

What gets scheduled:
- A regular cron runner entry with local `exec` payload.
- It runs `capture-snapshot` and writes snapshots to `workspace/skills/border-tracker/state/snapshots`.
- It appends capture metadata to `workspace/skills/border-tracker/state/snapshot_index.jsonl`.
- No per-run channel message should be sent.

## Flow 3: Disable Border Monitoring
Use this when user asks to stop monitoring for a camera.

Run:

```bash
python3 /home/dragan-slaveski/.openclaw/workspace/skills/border-tracker/scripts/border_flow.py disable-monitor-job \
  --camera "Bajakovo Entry" \
  --jobs-file /home/dragan-slaveski/.openclaw/cron/jobs.json
```

Notes:
- This disables existing `border-monitor-*` jobs for the selected camera.
- Use `--camera all` to disable all border monitor jobs.

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

Only after this local analysis should LLM be used to format/summarize the response for WhatsApp.
Do not include snapshot links/files for patterns responses.
