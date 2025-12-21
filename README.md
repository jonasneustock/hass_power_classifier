# HA Power Classifier

Dockerized FastAPI app that connects to Home Assistant, records a power sensor, surfaces segments for labeling, trains a classifier, and pushes appliance state + estimated per-appliance wattage back into Home Assistant.

## What it does
- Connects to a Home Assistant instance via REST API
- Polls a power sensor on a fixed interval
- Builds rolling segments and flags candidate changes
- Web UI for labeling segments as `start`, `running`, or `stop` per appliance
- Trains a classifier once enough labels exist per appliance
- Pushes status updates and per-appliance power estimates back to Home Assistant entities

## Quick start
1. Copy `.env.example` to `.env` and fill in the Home Assistant details.
2. Start the container:

```bash
docker compose up --build
```

3. Open the UI at `http://localhost:8000`.

## Configuration
All configuration is handled via environment variables in `.env`:

- `HA_BASE_URL`: URL of Home Assistant (e.g. `http://homeassistant.local:8123`)
- `HA_TOKEN`: Home Assistant long-lived access token
- `HA_POWER_SENSOR_ENTITY`: Source power sensor entity ID
- `POLL_INTERVAL_SECONDS`: Polling interval
- `SEGMENT_WINDOW_SECONDS`: Window size for each segment
- `RELATIVE_CHANGE_THRESHOLD`: Relative change threshold (e.g. `0.2` for 20%)
- `SEGMENT_PRE_SAMPLES`: Samples to include before a detected change
- `SEGMENT_POST_SAMPLES`: Samples to include after a detected change
- `MIN_LABELS_PER_APPLIANCE`: Minimum labels per appliance before training
- `STATUS_TTL_SECONDS`: Time to keep a running status valid

## Notes
- Create per-appliance entities in Home Assistant for `status` and `power` and paste their IDs into the app.
- The classifier uses simple window features. Improve accuracy by labeling consistent segments across appliances.
- Per-appliance wattage is computed from each appliance's learned running average and split proportionally across active appliances.
