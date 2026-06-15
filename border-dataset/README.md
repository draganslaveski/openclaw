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

## Prevent sleep during training

On Linux, run training through the provided wrapper so the PC does not go to sleep while training is active:

```bash
./workspace/border-dataset/run_training_no_sleep.sh /home/dragan-slaveski/.openclaw/.venv/bin/python workspace/border-dataset/train_queue_fast.py
```

You can wrap any training command the same way. Sleep blocking is released automatically when the command exits.

If your system policy denies sleep inhibition in this shell, run strict mode to fail fast:

```bash
REQUIRE_INHIBIT=1 ./workspace/border-dataset/run_training_no_sleep.sh /home/dragan-slaveski/.openclaw/.venv/bin/python workspace/border-dataset/train_queue_fast.py
```

If strict mode fails with access denied, run the same command from your local logged-in desktop terminal (or with `sudo` if your machine policy requires it).