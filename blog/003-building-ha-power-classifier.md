# Building the HA Power Classifier: From “Mystery Watts” to Actionable Signals

Home Assistant is fantastic at pulling device state into one place, but it’s a different story when your appliances don’t report their own usage. Plenty of big loads—washers, dryers, dishwashers, EV chargers, even gaming PCs—sit behind non-reporting plugs or on shared circuits. You see the total watts for a circuit (or a sum across phases) but have no idea which appliance is responsible for each bump in consumption. This post walks through why I built the HA Power Classifier, how it works under the hood, and what I learned along the way. It’s a deep dive into segmentation, labeling, model training, MQTT/REST publishing, and the pragmatic UI decisions that make it usable day to day.

> TL;DR: The app watches one or more Home Assistant power sensors, works on diffs (not absolutes), detects meaningful changes, lets you label segments as `start`/`stop`/`base`, trains a classifier + regression model, and pushes per-appliance power estimates back to Home Assistant or MQTT discovery. It’s meant for people who want appliance-level insight without buying a dozen smart plugs.

## The pain this solves

If you’ve ever tried to infer appliance status from a single “total power” sensor, you’ve felt the pain:

- **No per-appliance telemetry**: A washer and dryer on the same circuit create overlapping power signatures.
- **Noisy signals**: Fridges, pumps, chargers, and standby draw all create micro-spikes that swamp simple thresholds.
- **False starts**: A random spike might look like a start event, but it’s actually just a compressor kick.
- **No feedback loop**: Even if you eyeball the graph and guess, Home Assistant doesn’t get clean “appliance X is running” entities unless you wire them up yourself.

The goal was to create a loop where the app:
1) Listens to total (or per-phase) power sensors.
2) Detects meaningful changes using diffs.
3) Surfaces candidate segments in a UI for labeling.
4) Learns from those labels (classification + regression).
5) Publishes per-appliance power back to Home Assistant (via REST or MQTT discovery), zeroing the value on stop.
6) Uses optional activity sensors (binary on/off) to hint at start/stop labels.

## Why diffs instead of absolutes?

Looking at absolute power is deceptively hard when multiple appliances share a circuit or panel. Diffs (delta between consecutive samples) isolate the *change* rather than the total:

- **Change isolation**: A +700 W spike followed by a plateau is easier to treat as a start event than an absolute value that depends on prior background load.
- **Parallel loads**: When two appliances overlap, the combined absolute value is ambiguous; the diffs still show distinct jumps.
- **Noise control**: You can threshold relative or absolute changes in the diff stream, and even adapt those thresholds based on recent noise (rolling stddev).

The app stores both absolute totals and per-sensor readings, but segmentation and training happen on diffs.

## Architecture in plain English

**Stack**: FastAPI + SQLite + joblib (for model persistence) + MQTT/REST publishing. Tailwind for UI, vanilla JS for charts. Dockerized on Python 3.12.

**Main pieces**:

- **HA client**: Talks to Home Assistant’s REST API with a masked token in logs. Fetches sensor states; can also set states when MQTT is off.
- **Poller**: Reads up to 10 power sensors every N seconds. Computes diffs and looks for significant changes (relative or absolute thresholds; optional adaptive thresholds based on noise). Builds a rolling window to extract segments (pre/post samples).
- **Segments**: Each segment stores stats (mean, std, max, min, slope, change score, flank) and timestamps. A “flank” indicates whether the segment trend is positive, negative, or flat.
- **Labeling UI**: Shows recent diff graphs (overall and per sensor), lets you label segments as `start`, `stop`, or `base`. A batch labeling flow jumps to the latest unlabeled segment after you submit. Top-3 model predictions appear as quick buttons; the old single “proposed” block is gone.
- **Activity sensors**: Optional comma-separated binary sensors (on/off). On/off transitions are recorded and displayed as dashed lines on charts, and used as hints to prefill predicted appliance/phase on new segments.
- **Models**:
  - **Classifier**: RandomForest on segment features. Hyperparameter tuning (optional) tries a few parameter sets. Classes with <5 labels are skipped to avoid underfitting. 80/20 split when enough data exists; else train on all.
  - **Regression**: Per-appliance decision-tree regressor (swapped in from linear regression). Trains on diff samples vs. time since start, *per sensor* when multiple power sensors are defined. Only positive flanks feed regression to avoid negative power predictions. Metrics include MSE and MAPE.
- **Publishing**: When an appliance is in an “active session” (start seen, stop not yet), the app publishes predicted watts via MQTT discovery topics or Home Assistant REST. On stop, it resets that appliance’s published power to 0 W. Current published power is stored and shown in the dashboard.
- **Training manager**: Handles training triggers (manual or auto on interval), metrics history persistence, and scheduler loop. Metrics history drives sparklines on the Models page.
- **Automation API**: JSON endpoints for retraining, metrics/history, segments, and appliances to integrate with your own scripts/automations.

## Data flow, step by step

1) **Poll**: Every `POLL_INTERVAL_SECONDS`, read HA power sensors. If none are reachable, skip the cycle.
2) **Diff**: Compute total diff and per-sensor diffs, append to a deque.
3) **Threshold**: Check relative threshold (or absolute if relative is disabled). If adaptive mode is on, compute a dynamic relative threshold from rolling stddev, clamped by min/max bounds.
4) **Segment**: On trigger, capture a window of pre/post samples (configurable). Require at least 30 samples. Compute features and flank.
5) **Hints**: If activity sensors fired recently, attach their start/stop hint to the segment as a prediction.
6) **Predict**: Run classifier on the segment; store prediction. (Top-3 are also available for quick labeling.)
7) **Label**: In the UI, pick appliance/phase or use quick buttons. Batch mode jumps to the next unlabeled segment after saving.
8) **Train**: Once enough labels exist (per-appliance min + >=5 samples per class), kick off classifier + regression training in a background thread (optionally on a schedule).
9) **Publish**: Track active sessions; push predicted watts (or mean/running watts fallback) to MQTT/REST; update `current_power` in the datastore. On stop, publish 0 W.
10) **Cleanup**: Periodically delete unlabeled segments older than a TTL.

## UI choices that matter

- **Diff-first graphs**: The dashboard and per-sensor charts show diff values, not absolute watts. Event markers (start/stop) render as vertical lines; activity sensors as dashed lines.
- **Top-3 predictions**: Instead of a single “proposed” label, you see three quick buttons with probabilities. Faster, and you can still override manually.
- **Batch labeling**: A “Batch label” button on the segments page jumps you to the latest unlabeled segment; “Save & Next” moves you forward as you label.
- **Models page**: Shows current metrics plus history sparklines for classifier accuracy and regression MSE. Also shows regression MAPE, and the training status (running/failed/idle).
- **Appliances page**: Inline edit/rename/delete; shows regression min/mean/max per appliance, current published power, and activity sensors.

## Adaptive thresholds in practice

Power data is noisy. A static relative threshold (e.g., 20%) might miss small but real events on a quiet line, or trigger too often on a noisy line. Adaptive mode:

- Looks at a rolling window of recent diffs (default 50 samples).
- Computes stddev as a “noise” estimate.
- Sets relative threshold = noise * multiplier (default 3), clamped between min/max bounds.
- Falls back to absolute threshold only if relative is disabled.

This yields fewer false positives on noisy feeds and more sensitivity on quiet feeds without manual retuning.

## Hyperparameter tuning, but pragmatic

There’s a toggle (`HYPERPARAM_TUNING`) to try a handful of parameter combinations for both the classifier and regression tree. It’s intentionally small to keep training fast in a home environment. If disabled, it uses sane defaults. Models and metrics are persisted via joblib.

## Per-sensor regression

European/multi-phase setups often have multiple power sensors. Regression now trains on per-sensor diffs: for each labeled segment, it pulls samples per sensor, computes diffs, and fits the decision tree on those combined points. This helps the model learn shape across phases rather than just the summed total.

## Activity sensors as labeling hints

If you map binary sensors to an appliance (comma-separated), their on/off transitions get recorded. When a new segment is created, recent activity events within a time window are used as hints to prefill predicted appliance/phase. They’re also drawn on the dashboard charts as dashed lines, so you can visually compare detected events with actual device states (if available).

## Publishing back to Home Assistant

You pick per-appliance power entities when creating appliances (or let MQTT discovery set them up). When an appliance transitions from start to stop, the app publishes 0 W immediately. Current published power is shown on the dashboard and appliance page. Base draw is kept separate (label segments as `base` so they don’t trigger pushes).

## Automation API

New JSON endpoints make it easy to script around the app:

- `POST /api/retrain` – trigger training.
- `GET /api/metrics` – current training state, latest metrics, and history.
- `GET /api/segments` – list segments (limit param).
- `GET /api/appliances` – list appliances and their stats.

A future improvement is to add token-based auth for these endpoints.

## Export/import labeled data

You can export labeled segments as JSON and re-import them later (or on another instance). This makes it easier to share or back up your training data. Future work: add CSV and simple diffing for conflicts.

## Scheduling retrains

Set `RETRAIN_INTERVAL_SECONDS` to a non-zero value to enable periodic training. The training manager starts a scheduler thread on startup and triggers training when not already running. Training logs go into the UI’s log tab and the metrics history file.

## The build process: lessons learned

### 1) Keep the first model simple, then iterate
The very first version used static thresholds and a simple classifier. Moving to diff-based segmentation, adding flanks, and then adding regression and adaptive thresholds were incremental. Shipping something small early helped uncover real-world pain (like noisy sensors) before over-engineering.

### 2) UI matters for labeling speed
Batch labeling, top-3 quick buttons, and the removal of a single “proposed” label made labeling faster and less confusing. Seeing activity sensor hints overlaid on the same chart reduced “guess the appliance” moments.

### 3) MQTT vs REST
MQTT discovery is great for “it just appears” UX in Home Assistant, but REST is a good fallback. The app supports both; when MQTT is enabled, it only publishes power topics (no status entities) to stay lightweight.

### 4) Persistence and portability
Everything lives in `/data`: SQLite for samples/segments/appliances, joblib for models, and JSON for metrics history. Export/import of labeled segments emerged as a must-have once multiple users asked for “backup” and “sharing.”

### 5) Multi-phase reality
Supporting up to 10 sensors forced the regression to train on per-sensor diffs. This also meant the UI had to show combined per-sensor graphs and keep the dashboards useful when phases don’t align perfectly.

### 6) Noise is the real boss
Adaptive thresholds were added after seeing how a single compressor or a noisy EV charger could spam segments. Clamping the adaptive threshold avoids swinging from too sensitive to too insensitive.

### 7) Testing pays off
Pytest coverage now includes HA client with a mock server, config/logging utilities, datastore imports/updates, and the model services. Staying in a Dockerized environment with clear env vars made it easier to reproduce issues.

## Current feature set (snapshot)

- Diff-based segmentation with adaptive/relative/absolute thresholds.
- Multi-sensor support (up to 10); per-sensor regression training.
- Labeling UI with top-3 predictions and batch mode.
- Activity sensor hints and overlays.
- Classifier (RandomForest) and regression (decision tree) with optional tuning.
- MQTT discovery or REST publishing of per-appliance power; power reset on stop.
- Metrics history with sparklines; training scheduler.
- Export/import of labeled segments; automation API endpoints.
- Inline appliance management (edit/rename/delete) and HA validation.

## What’s next

Some ideas already captured in the codebase and future plans:

- Secure the automation endpoints with tokens or HA auth.
- Add CSV export/import for labeled segments.
- Smarter sequence models for classification.
- Active learning to prioritize uncertain segments.
- Config flow add-on for Home Assistant.
- Health endpoints and structured logs for ops visibility.

## How to try it

1) Copy `.env.example` to `.env`; set `HA_BASE_URL`, `HA_TOKEN`, and sensors (`HA_POWER_SENSORS` or `HA_POWER_SENSOR_ENTITY`).
2) Optional: enable MQTT (`MQTT_ENABLED=true`) and set broker details; turn on adaptive thresholds or hyperparam tuning if desired.
3) `docker compose up --build` and open `http://localhost:8000`.
4) Label segments as they appear; watch metrics update on the Models page; see per-appliance power in HA.

## Closing thoughts

Building this app was about making unlabeled power streams useful without extra hardware. The core loop—detect, label, train, publish—needed to be fast and forgiving. Diffs, adaptive thresholds, activity hints, and lightweight models struck a balance between accuracy and simplicity. There’s plenty to polish (auth, better models, smarter UI), but the current build already turns “mystery watts” into actionable appliance insights in Home Assistant. If you’re wrestling with non-reporting appliances, give it a try, and let me know what you’d like to see next.***
