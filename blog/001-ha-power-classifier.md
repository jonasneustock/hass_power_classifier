# Taming Unknown Loads in Home Assistant with a Diff-Based Power Classifier

Smart homes get messy when appliances don’t report their own power. Washers, dishwashers, and “dumb” plugs often leave you guessing what’s actually running. This post walks through an open-source FastAPI app I’m building that learns appliance start/stop signatures from your Home Assistant power sensors, labels segments with your help, and feeds power estimates back into Home Assistant (or MQTT) — without needing per-device smart plugs.

## What it does
- Polls up to 10 Home Assistant power sensors (multi‑phase friendly) on a configurable interval.
- Works on **power diffs** (deltas between samples) to detect meaningful changes with relative or absolute thresholds.
- Builds candidate segments around changes, shows them in a web UI, and lets you label them as `start`, `stop`, or `base`.
- Trains two models once you’ve labeled enough:
  - A classifier to recognize start/stop events per appliance.
  - A regression model to estimate per‑appliance power draw after a start, resetting to 0 on stop.
- Publishes per‑appliance power via MQTT discovery or Home Assistant REST; base draw is tracked separately.
- Optional **activity sensors**: map binary sensors (on/off) to appliances to pre-hint start/stop labels.

## How it works (high level)
1) **Polling & diffs**: The poller reads configured power sensors, stores raw totals, and computes diffs (sample-to-sample deltas). Segmentation runs on these diffs to better isolate changes, especially in multi-load scenarios.

2) **Segmentation**: When a relative change threshold (or absolute fallback) is crossed, the app captures a window of samples (pre/post) and creates a segment. Each segment records simple stats (mean, std, slope, change score, flank direction) and the timestamp range.

3) **Labeling UI**: A Tailwind-based dashboard shows recent diff graphs, multi-sensor overlays, and candidate segments. You label segments as `start`, `stop`, or `base`. Old unlabeled segments can auto-clean after a TTL.

4) **Activity hints**: If you provide binary sensors per appliance, their on/off transitions are recorded and applied as hints to newly created segments (pre-labeling assist).

5) **Training**:
   - **Classifier**: RandomForest on segment features; skips classes with fewer than 5 labels to avoid underfitting; 80/20 split when data allows. Metrics captured: accuracy, precision, recall, F1.
   - **Regression**: Per-appliance linear regression on diff samples vs. time since `start`; metrics: MSE, MAPE. Only positive flanks are used for draw estimation.

6) **Publishing**: When an appliance is “active” (start seen, stop not yet), the app pushes predicted watts to Home Assistant via REST or MQTT discovery topics. On stop, it resets that appliance’s published power to 0 W.

## Why diffs?
Absolute power alone is noisy in multi-appliance homes. By looking at *changes* between samples, the app can isolate spikes or drops that align with appliance state changes, even when multiple appliances run in parallel. Diffs also make it easier to learn per-appliance power shapes without full isolation.

## Setup snapshot
- Dockerized (Python 3.12) with a simple `.env`:
  - `HA_BASE_URL`, `HA_TOKEN`
  - `HA_POWER_SENSORS` (comma-separated, up to 10)
  - `RELATIVE_CHANGE_THRESHOLD` or `ABSOLUTE_CHANGE_THRESHOLD`
  - `SEGMENT_PRE_SAMPLES`, `SEGMENT_POST_SAMPLES`
  - `MQTT_ENABLED`, `MQTT_HOST`, `MQTT_BASE_TOPIC`, `MQTT_DISCOVERY_PREFIX`
- Runs a FastAPI UI on port 8000 by default.
- Stores data/models under `/data` (SQLite + joblib).

## Roadmap ideas
- Better per-appliance curve fitting for variable cycles.
- Automated threshold tuning.
- Export/import labeled datasets.
- Tighter HA integration (config flow) and live metric charts.

## Why this matters
If you’ve ever tried to piece together washer/dryer states from a single “sum of watts” sensor, you know the pain. This app aims to make unlabeled power streams usable: detect changes, let you label quickly, then do the heavy lifting to keep Home Assistant informed — all without installing smart plugs on every device.

*Have feedback or want to try it? Grab the Docker image, point it at your HA, and start labeling. Contributions welcome!*
