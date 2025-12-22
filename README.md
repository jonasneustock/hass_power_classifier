# HA Power Classifier

Dockerized FastAPI app that connects to Home Assistant, records a power sensor, surfaces segments for labeling, trains a classifier, and pushes appliance state + estimated per-appliance wattage back into Home Assistant.

## What it does
- Connects to a Home Assistant instance via REST API
- Polls one or more power sensors on a fixed interval and works on power **diffs** (delta between samples)
- Builds rolling segments and flags candidate changes
- Web UI for labeling segments as `start`, `stop`, or `base` per appliance
- Trains a classifier once enough labels exist per appliance and a regression model to predict per-appliance power over time after `start`
- Pushes per-appliance power estimates back to Home Assistant via REST or MQTT discovery (no status entities)

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
- `HA_POWER_SENSOR_ENTITY`: Source power sensor entity ID (single)
- `HA_POWER_SENSORS`: Comma-separated list of power sensors (up to 10) for multi-phase setups; summed for segmentation
- `POLL_INTERVAL_SECONDS`: Polling interval
- `RELATIVE_CHANGE_THRESHOLD`: Relative change threshold (e.g. `0.2` for 20%); leave empty to disable
- `ABSOLUTE_CHANGE_THRESHOLD`: Absolute delta trigger between consecutive diffs; used only when relative threshold is empty
- `SEGMENT_PRE_SAMPLES`: Samples to include before a detected change
- `SEGMENT_POST_SAMPLES`: Samples to include after a detected change
- `MIN_LABELS_PER_APPLIANCE`: Minimum labels per appliance before training
- `STATUS_TTL_SECONDS`: Time to keep a running status valid
- `MQTT_ENABLED`: Enable MQTT output for Home Assistant discovery
- `MQTT_HOST`: MQTT broker hostname
- `MQTT_PORT`: MQTT broker port
- `MQTT_USERNAME`: MQTT username
- `MQTT_PASSWORD`: MQTT password
- `MQTT_BASE_TOPIC`: Base topic for per-appliance state
- `MQTT_DISCOVERY_PREFIX`: Home Assistant discovery prefix (default `homeassistant`)
- `MQTT_CLIENT_ID`: Client ID for the MQTT connection
- `MQTT_DEVICE_ID`: Device identifier used for discovery unique IDs
## Model training
- Classifier: RandomForest on diff-based segment features; uses an 80/20 train/test split when there are at least 5 samples and more than 1 class, otherwise trains on all data. Metrics recorded: accuracy, precision, recall, F1, sample and class counts.
- Regression: per-appliance LinearRegression on diff samples vs. time since start; uses an 80/20 split when there are at least 10 samples, otherwise trains on all data. Metrics recorded: MSE and MAPE.
- Base appliance: auto-created to hold `base` labels; base segments are excluded from training and power pushes.

## Test coverage (local snapshot)
```
pytest --cov=app --cov-report=term
```
| Module | Stmts | Miss | Cover |
| --- | ---:| ---:| ---:|
| app/__init__.py | 0 | 0 | 100% |
| app/classifier.py | 124 | 25 | 80% |
| app/data_store.py | 177 | 35 | 80% |
| app/ha_client.py | 32 | 32 | 0% |
| app/main.py | 458 | 458 | 0% |
| app/mqtt_client.py | 53 | 53 | 0% |
| app/utils.py | 59 | 1 | 98% |
| **Total** | 903 | 604 | **33%** |

## Notes
- When MQTT is enabled, the app publishes MQTT discovery topics for each appliance and pushes power to `MQTT_BASE_TOPIC/<appliance>/power`.
- The classifier/regression use simple statistical features; improve accuracy by labeling consistent segments across appliances.
- Power estimation is split per active appliance using learned regression/means; `base` labels capture ambient draw and don’t trigger pushes.
