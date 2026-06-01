# Border dataset labeling tool

This folder now includes a local annotation tool for estimating queue size in each collected border camera frame.

## What it does

- Reads samples from `data/raw/manifest.jsonl`
- Serves the captured JPEG frames in a browser UI
- Saves labels to `data/labels/line_size_labels.jsonl`
- Lets you mark bad frames as unusable so they stay out of training

Each saved label contains:

- `sample_id`
- `camera_id`
- `captured_at`
- `relative_file`
- `line_percent` (0-100)
- `line_bucket` (`none`, `very_short`, `short`, `medium`, `long`, `extreme`)
- `is_usable`
- `notes`
- `labeled_at`

## Run it

From the repository root:

```bash
source .venv/bin/activate
python workspace/border-dataset/labeling_tool.py
```

Then open:

```text
http://127.0.0.1:8765
```

## Workflow

- Leave the filter on `Only unlabeled` to work through fresh frames
- Use the slider or preset buttons to estimate queue size
- Use `Save + next` for usable frames
- Use `Mark unusable` for blocked, broken, or irrelevant frames
- Use the labels JSONL later to build train/validation/test splits for modeling