import logging
import os
import sqlite3
import threading
import time
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.classifier import ClassifierService, RegressionService
from app.data_store import DataStore
from app.ha_client import HAClient
from app.mqtt_client import MqttPublisher
from app.utils import build_mqtt_topics, compute_features, normalize_object_id, slugify


def load_config():
    load_dotenv()
    sensors_env = os.getenv("HA_POWER_SENSORS", "")
    sensors = [s.strip() for s in sensors_env.split(",") if s.strip()]
    if not sensors:
        fallback = os.getenv("HA_POWER_SENSOR_ENTITY", "")
        if fallback:
            sensors = [fallback]
    return {
        "ha_base_url": os.getenv("HA_BASE_URL", "http://homeassistant.local:8123"),
        "ha_token": os.getenv("HA_TOKEN", ""),
        "power_sensors": sensors[:10],
        "poll_interval": float(os.getenv("POLL_INTERVAL_SECONDS", "5")),
        "relative_change_threshold": float(
            os.getenv("RELATIVE_CHANGE_THRESHOLD", "0.2")
        ),
        "segment_pre_samples": int(os.getenv("SEGMENT_PRE_SAMPLES", "15")),
        "segment_post_samples": int(os.getenv("SEGMENT_POST_SAMPLES", "15")),
        "min_labels": int(os.getenv("MIN_LABELS_PER_APPLIANCE", "5")),
        "status_ttl": int(os.getenv("STATUS_TTL_SECONDS", "300")),
        "unlabeled_ttl": int(os.getenv("UNLABELED_TTL_SECONDS", "7200")),
        "cleanup_interval": int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300")),
        "app_title": os.getenv("APP_TITLE", "HA Power Classifier"),
        "data_dir": os.getenv("DATA_DIR", "/data"),
        "mqtt_enabled": os.getenv("MQTT_ENABLED", "false").lower()
        in ("1", "true", "yes", "on"),
        "mqtt_host": os.getenv("MQTT_HOST", "localhost"),
        "mqtt_port": int(os.getenv("MQTT_PORT", "1883")),
        "mqtt_username": os.getenv("MQTT_USERNAME", ""),
        "mqtt_password": os.getenv("MQTT_PASSWORD", ""),
        "mqtt_base_topic": os.getenv("MQTT_BASE_TOPIC", "ha_power_classifier").rstrip(
            "/"
        ),
        "mqtt_discovery_prefix": os.getenv(
            "MQTT_DISCOVERY_PREFIX", "homeassistant"
        ).rstrip("/"),
        "mqtt_client_id": os.getenv("MQTT_CLIENT_ID", "ha-power-classifier"),
        "mqtt_device_id": os.getenv("MQTT_DEVICE_ID", "ha_power_classifier"),
    }


class PowerPoller:
    def __init__(self, store, ha_client, classifier, regression_service, config, mqtt_publisher=None):
        self.store = store
        self.ha_client = ha_client
        self.classifier = classifier
        self.regression_service = regression_service
        self.config = config
        self.mqtt_publisher = mqtt_publisher
        self.sensors = config["power_sensors"]
        min_window = config["segment_pre_samples"] + config["segment_post_samples"] + 1
        max_samples = max(min_window + 20, 50)
        self.samples_diff = deque(maxlen=max_samples)
        self.pending_segment = None
        self.sample_count = 0
        self.last_cleanup_ts = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.active_sessions = {}
        self.prev_total = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def run(self):
        while not self.stop_event.is_set():
            ts = int(time.time())
            total_value = 0.0
            read_success = 0
            for sensor in self.sensors:
                try:
                    state = self.ha_client.get_state(sensor)
                    value = float(state["state"])
                    self.store.add_sensor_sample(ts, sensor, value)
                    total_value += value
                    read_success += 1
                except Exception as exc:
                    logging.warning("Failed to read HA sensor %s: %s", sensor, exc)
                    log_event(f"Failed to read HA sensor {sensor}: {exc}", level="warning")

            if read_success == 0:
                time.sleep(self.config["poll_interval"])
                continue

            self.store.add_sample(ts, total_value)
            if self.prev_total is None:
                self.prev_total = total_value
                time.sleep(self.config["poll_interval"])
                continue

            diff_value = total_value - self.prev_total
            self.prev_total = total_value
            self.samples_diff.append((ts, diff_value))
            self.sample_count += 1

            if len(self.samples_diff) >= 2 and self.pending_segment is None:
                prev_value = self.samples_diff[-2][1]
                denom = abs(prev_value) if abs(prev_value) > 1e-6 else 1.0
                relative_change = abs(diff_value - prev_value) / denom
                if (
                    relative_change >= self.config["relative_change_threshold"]
                    and len(self.samples_diff) >= self.config["segment_pre_samples"] + 1
                ):
                    self.pending_segment = {
                        "trigger_count": self.sample_count,
                    }

            if self.pending_segment:
                post_samples = self.config["segment_post_samples"]
                if self.sample_count - self.pending_segment["trigger_count"] >= post_samples:
                    samples_list = list(self.samples_diff)
                    segment_length = (
                        self.config["segment_pre_samples"]
                        + self.config["segment_post_samples"]
                        + 1
                    )
                    if len(samples_list) >= segment_length:
                        segment_samples = samples_list[-segment_length:]
                        if len(segment_samples) >= 30:
                            features = compute_features(segment_samples)
                            delta = segment_samples[-1][1] - segment_samples[0][1]
                            if delta > 0:
                                flank = "positive"
                            elif delta < 0:
                                flank = "negative"
                            else:
                                flank = "flat"
                            segment = {
                                "start_ts": segment_samples[0][0],
                                "end_ts": segment_samples[-1][0],
                                "mean": features["mean"],
                                "std": features["std"],
                                "max": features["max"],
                                "min": features["min"],
                                "duration": features["duration"],
                                "slope": features["slope"],
                                "change_score": features["change_score"],
                                "candidate": True,
                                "flank": flank,
                                "created_ts": ts,
                            }
                            segment_id = self.store.add_segment(segment)
                            log_event(
                                f"Segment #{segment_id} created change={round(features['change_score']*100,1)}%"
                            )
                            prediction = self.classifier.predict(segment)
                            if prediction:
                                appliance, phase = prediction
                                self.store.update_segment_prediction(
                                    segment_id, appliance, phase
                                )
                                log_event(
                                    f"Prediction for segment #{segment_id}: {appliance}/{phase}"
                                )
                                self._record_phase(appliance, phase, ts)
                    self.pending_segment = None

            if ts - self.last_cleanup_ts >= self.config["cleanup_interval"]:
                cutoff = ts - self.config["unlabeled_ttl"]
                deleted = self.store.delete_unlabeled_before(cutoff)
                if deleted:
                    logging.info("Deleted %s unlabeled segments older than %s", deleted, cutoff)
                    log_event(f"Cleanup removed {deleted} stale segments")
                self.last_cleanup_ts = ts

            self._push_power_allocations(ts, total_value)
            time.sleep(self.config["poll_interval"])

    def _record_phase(self, appliance, phase, ts):
        appliance_row = self.store.get_appliance(appliance)
        if not appliance_row:
            return
        last_status = appliance_row.get("last_status")
        if last_status == phase:
            return
        if phase == "start":
            self.active_sessions[appliance] = ts
        if phase == "stop":
            self.active_sessions.pop(appliance, None)
        self.store.update_appliance_status(appliance, phase, ts)
        log_event(f"Phase update for {appliance}: {phase}")
        if last_status == "start" and phase == "stop":
            # push zero watts to power entity
            if self.config.get("mqtt_enabled") and self.mqtt_publisher:
                topics = build_mqtt_topics(appliance_row, self.config)
                if not self.mqtt_publisher.publish_value(
                    topics["power_state_topic"], 0, retain=True
                ):
                    logging.warning("Failed to publish MQTT power 0 for %s", appliance)
                else:
                    log_event(f"Power reset to 0 for {appliance}")
            else:
                try:
                    self.ha_client.set_state(
                        appliance_row["power_entity_id"],
                        0,
                        {
                            "appliance": appliance,
                            "source": "ha_power_classifier",
                        },
                    )
                    log_event(f"Power reset to 0 for {appliance}")
                except Exception as exc:
                    logging.warning("Failed to reset power for %s: %s", appliance, exc)

    def _push_power_allocations(self, ts, total_power):
        appliances = {a["name"]: a for a in self.store.list_appliances()}
        if not self.active_sessions:
            return

        for appliance_name, start_ts in list(self.active_sessions.items()):
            appliance = appliances.get(appliance_name)
            if not appliance:
                continue
            elapsed = max(0, ts - start_ts)
            predicted = self.regression_service.predict(appliance_name, elapsed)
            watts = predicted
            if watts is None or watts <= 0:
                watts = appliance.get("mean_power") or appliance.get("running_watts") or 0
            if watts <= 0:
                continue
            if self.config.get("mqtt_enabled") and self.mqtt_publisher:
                topics = build_mqtt_topics(appliance, self.config)
                if not self.mqtt_publisher.publish_value(
                    topics["power_state_topic"], round(watts, 2), retain=True
                ):
                    logging.warning(
                        "Failed to publish MQTT power for %s", appliance_name
                    )
                else:
                    log_event(f"Power published for {appliance_name}: {round(watts,2)} W")
                continue
            try:
                self.ha_client.set_state(
                    appliance["power_entity_id"],
                    round(watts, 2),
                    {
                        "appliance": appliance_name,
                        "source": "ha_power_classifier",
                    },
                )
                log_event(f"Power set for {appliance_name}: {round(watts,2)} W")
            except Exception as exc:
                logging.warning("Failed to push power for %s: %s", appliance_name, exc)


config = load_config()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=config["app_title"])
base_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))


def format_ts(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


templates.env.filters["format_ts"] = format_ts


def publish_mqtt_discovery(appliance_row, config, mqtt_publisher):
    if not mqtt_publisher:
        return
    topics = build_mqtt_topics(appliance_row, config)
    device = {
        "identifiers": [config["mqtt_device_id"]],
        "name": config["app_title"],
        "manufacturer": "custom",
        "model": "ha_power_classifier",
    }
    power_payload = {
        "name": f"{appliance_row['name']} power",
        "state_topic": topics["power_state_topic"],
        "unique_id": f"{config['mqtt_device_id']}_{topics['power_object_id']}",
        "object_id": topics["power_object_id"],
        "device": device,
        "unit_of_measurement": "W",
        "device_class": "power",
        "state_class": "measurement",
    }
    mqtt_publisher.publish_json(topics["power_config_topic"], power_payload, retain=True)


data_dir = Path(config["data_dir"])
data_dir.mkdir(parents=True, exist_ok=True)
store = DataStore(str(data_dir / "power_classifier.sqlite"))
ha_client = HAClient(config["ha_base_url"], config["ha_token"])
classifier = ClassifierService(str(data_dir / "model.pkl"))
regression_service = RegressionService()
mqtt_publisher = None
if config["mqtt_enabled"]:
    mqtt_publisher = MqttPublisher(
        host=config["mqtt_host"],
        port=config["mqtt_port"],
        username=config["mqtt_username"],
        password=config["mqtt_password"],
        client_id=config["mqtt_client_id"],
    )

poller = PowerPoller(store, ha_client, classifier, regression_service, config, mqtt_publisher)
recent_logs = deque(maxlen=200)
training_state = {
    "running": False,
    "error": None,
    "last_started": None,
    "last_finished": None,
}
training_lock = threading.Lock()
metrics_file = data_dir / "model_metrics.json"
metrics_history = []


def log_event(message, level="info"):
    ts = int(time.time())
    entry = {"ts": ts, "message": message, "level": level}
    recent_logs.appendleft(entry)
    getattr(logging, level, logging.info)(message)


def load_metrics_history():
    if metrics_file.exists():
        try:
            return json.loads(metrics_file.read_text())
        except Exception:
            return []
    return []


def save_metrics_entry(entry):
    history = load_metrics_history()
    history.append(entry)
    history = history[-100:]
    metrics_file.write_text(json.dumps(history, indent=2))
    return history


def ensure_base_appliance():
    base = store.get_appliance("base")
    if not base:
        store.add_appliance("base", "", "")
        log_event("Base appliance created for baseline labeling")


def _run_training():
    with training_lock:
        if training_state["running"]:
            log_event("Training already running, skipping", level="info")
            return
        training_state["running"] = True
    training_state["error"] = None
    training_state["last_started"] = int(time.time())
    log_event("Training started")
    try:
        appliances = store.list_appliances()
        if not appliances:
            log_event("Training skipped: no appliances", level="warning")
            training_state["last_finished"] = int(time.time())
            return
        counts = store.get_label_counts_by_appliance()
        eligible = {
            appliance["name"]
            for appliance in appliances
            if counts.get(appliance["name"], 0) >= config["min_labels"]
        }
        if not eligible:
            log_event("Training skipped: no appliance meets label threshold", level="warning")
            training_state["last_finished"] = int(time.time())
            return
        labeled_segments = store.get_labeled_segments()
        labeled_segments = [
            seg
            for seg in labeled_segments
            if seg["label_appliance"] in eligible and seg["label_phase"] != "base"
        ]
        if not labeled_segments:
            log_event("Training skipped: no labeled segments", level="warning")
            training_state["last_finished"] = int(time.time())
            return
        clf_metrics = classifier.train(labeled_segments, eligible_appliances=eligible)
        regression_service.train(labeled_segments, store)
        reg_metrics = regression_service.last_metrics
        power_stats = compute_power_stats_by_appliance()
        for name, stats in power_stats.items():
            store.update_appliance_power_stats(
                name, stats["min_power"], stats["mean_power"], stats["max_power"]
            )
        training_state["last_finished"] = int(time.time())
        metrics_entry = {
            "ts": training_state["last_finished"],
            "classifier": clf_metrics,
            "regression": reg_metrics,
        }
        global metrics_history
        metrics_history = save_metrics_entry(metrics_entry)
        log_event("Training finished")
    except Exception as exc:
        training_state["error"] = str(exc)
        training_state["last_finished"] = int(time.time())
        log_event(f"Training failed: {exc}", level="error")
    finally:
        training_state["running"] = False


def maybe_train_classifier():
    log_event("Training trigger requested")
    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()


def compute_power_stats_by_appliance():
    appliances = store.list_appliances()
    labeled_segments = store.get_labeled_segments()
    stats = {}
    for appliance in appliances:
        name = appliance["name"]
        labeled = [
            seg
            for seg in labeled_segments
            if seg["label_appliance"] == name and seg["label_phase"] != "base"
        ]
        values = []
        for seg in labeled:
            samples = store.get_samples_between(seg["start_ts"], seg["end_ts"])
            values.extend([s["value"] for s in samples])
        if values:
            stats[name] = {
                "min_power": float(np.min(values)),
                "mean_power": float(np.mean(values)),
                "max_power": float(np.max(values)),
            }
    return stats


def check_ha_connection():
    if not config["ha_token"] or not config["power_sensors"]:
        logging.warning("HA_TOKEN or HA_POWER_SENSORS not set")
        log_event("HA config missing token or sensors", level="warning")
        return False
    ok = True
    for sensor in config["power_sensors"]:
        try:
            state = ha_client.get_state(sensor)
            float(state["state"])
            logging.info("HA connection check succeeded for %s", sensor)
            log_event(f"HA check ok for {sensor}")
        except Exception as exc:
            logging.warning("HA connection check failed for %s: %s", sensor, exc)
            log_event(f"HA check failed for {sensor}: {exc}", level="warning")
            ok = False
    return ok


@app.on_event("startup")
def on_startup():
    ensure_base_appliance()
    check_ha_connection()
    if mqtt_publisher:
        try:
            mqtt_publisher.connect()
        except Exception as exc:
            logging.warning("MQTT connection failed: %s", exc)
        else:
            for appliance in store.list_appliances():
                publish_mqtt_discovery(appliance, config, mqtt_publisher)
            log_event("MQTT connected")
    log_event("Application started")
    poller.start()


@app.on_event("shutdown")
def on_shutdown():
    poller.stop()
    if mqtt_publisher:
        mqtt_publisher.close()
    log_event("Application stopped")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    latest_sample = store.get_latest_sample()
    latest_per_sensor = store.get_latest_sensor_samples()
    appliances = store.list_appliances()
    segments = store.list_segments(limit=10, unlabeled_only=True)
    recent_samples_raw = store.get_recent_samples(limit=200)
    recent_samples = []
    for idx in range(1, len(recent_samples_raw)):
        prev = recent_samples_raw[idx - 1]
        curr = recent_samples_raw[idx]
        recent_samples.append(
            {"ts": curr["ts"], "value": curr["value"] - prev["value"]}
        )

    recent_by_sensor = {}
    for sensor in config["power_sensors"]:
        raw = store.get_recent_sensor_samples(sensor, limit=200)
        diffs = []
        for idx in range(1, len(raw)):
            prev = raw[idx - 1]
            curr = raw[idx]
            diffs.append({"ts": curr["ts"], "value": curr["value"] - prev["value"]})
        recent_by_sensor[sensor] = diffs
    detection_events = [
        {
            "ts": seg["start_ts"],
            "appliance": seg.get("predicted_appliance") or seg.get("label_appliance"),
            "phase": seg.get("predicted_phase") or seg.get("label_phase"),
        }
        for seg in store.list_segments(limit=100, unlabeled_only=False)
        if (seg.get("predicted_phase") or seg.get("label_phase")) in ("start", "stop")
    ]
    training = classifier.last_metrics
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "config": config,
            "latest_sample": latest_sample,
            "latest_per_sensor": latest_per_sensor,
            "appliances": appliances,
            "segments": segments,
            "recent_samples": recent_samples,
            "recent_by_sensor": recent_by_sensor,
            "detection_events": detection_events,
            "training": training,
            "training_state": training_state,
        },
    )


@app.get("/appliances", response_class=HTMLResponse)
def appliances_page(request: Request):
    appliances = store.list_appliances()
    if config.get("mqtt_enabled"):
        enriched = []
        for appliance in appliances:
            item = dict(appliance)
            item.update(build_mqtt_topics(appliance, config))
            enriched.append(item)
        appliances = enriched
    log_event("Appliances page viewed")
    return templates.TemplateResponse(
        "appliances.html",
        {"request": request, "appliances": appliances, "config": config},
    )


@app.post("/appliances")
def create_appliance(
    request: Request,
    name: str = Form(...),
    power_entity_id: str = Form(...),
):
    errors = []
    if not config.get("mqtt_enabled"):
        try:
            ha_client.get_state(power_entity_id)
        except Exception as exc:
            logging.warning("Power entity check failed: %s", exc)
            errors.append("Power entity ID not available in Home Assistant.")
    if errors:
        appliances = store.list_appliances()
        if config.get("mqtt_enabled"):
            enriched = []
            for appliance in appliances:
                item = dict(appliance)
                item.update(build_mqtt_topics(appliance, config))
                enriched.append(item)
            appliances = enriched
        return templates.TemplateResponse(
            "appliances.html",
            {
                "request": request,
                "appliances": appliances,
                "config": config,
                "error": " ".join(errors),
                "form": {
                    "name": name,
                    "power_entity_id": power_entity_id,
                },
            },
            status_code=400,
        )
    try:
        store.add_appliance(name, "", power_entity_id)
        log_event(f"Appliance created: {name}")
    except sqlite3.IntegrityError:
        appliances = store.list_appliances()
        if config.get("mqtt_enabled"):
            enriched = []
            for appliance in appliances:
                item = dict(appliance)
                item.update(build_mqtt_topics(appliance, config))
                enriched.append(item)
            appliances = enriched
        return templates.TemplateResponse(
            "appliances.html",
            {
                "request": request,
                "appliances": appliances,
                "config": config,
                "error": "Appliance name already exists.",
                "form": {
                    "name": name,
                    "power_entity_id": power_entity_id,
                },
            },
            status_code=400,
        )
    appliance_row = store.get_appliance(name)
    if appliance_row and mqtt_publisher:
        publish_mqtt_discovery(appliance_row, config, mqtt_publisher)
    return RedirectResponse(url="/appliances", status_code=303)


@app.get("/segments", response_class=HTMLResponse)
def segments_page(
    request: Request,
    candidate: int = 1,
    unlabeled: int = 1,
    min_change: float = 0.0,
):
    segments = store.list_segments(
        limit=200,
        unlabeled_only=bool(unlabeled),
        candidate_only=bool(candidate),
    )
    if min_change > 0:
        segments = [s for s in segments if s["change_score"] >= min_change]
    return templates.TemplateResponse(
        "segments.html",
        {
            "request": request,
            "segments": segments,
            "candidate": candidate,
            "unlabeled": unlabeled,
            "min_change": min_change,
            "config": config,
        },
    )


@app.get("/segments/{segment_id}", response_class=HTMLResponse)
def segment_detail(request: Request, segment_id: int):
    segment = store.get_segment(segment_id)
    if not segment:
        return RedirectResponse(url="/segments", status_code=303)
    samples = store.get_samples_between(segment["start_ts"], segment["end_ts"])
    appliances = store.list_appliances()
    return templates.TemplateResponse(
        "segment_detail.html",
        {
            "request": request,
            "segment": segment,
            "samples": samples,
            "appliances": appliances,
            "config": config,
        },
    )


@app.post("/segments/{segment_id}/label")
def label_segment(
    segment_id: int,
    appliance: str = Form(...),
    phase: str = Form(...),
):
    store.update_segment_label(segment_id, appliance, phase)
    maybe_train_classifier()
    log_event(f"Labeled segment #{segment_id} as {appliance}/{phase}")
    return RedirectResponse(url="/segments", status_code=303)


@app.post("/segments/{segment_id}/accept_prediction")
def accept_prediction(segment_id: int):
    segment = store.get_segment(segment_id)
    if not segment or not segment.get("predicted_appliance") or not segment.get("predicted_phase"):
        return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)
    store.update_segment_label(
        segment_id, segment["predicted_appliance"], segment["predicted_phase"]
    )
    log_event(
        f"Accepted prediction for segment #{segment_id}: {segment['predicted_appliance']}/{segment['predicted_phase']}"
    )
    maybe_train_classifier()
    return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)


@app.post("/segments/{segment_id}/reject_prediction")
def reject_prediction(segment_id: int):
    store.clear_segment_prediction(segment_id)
    log_event(f"Rejected prediction for segment #{segment_id}")
    return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)


@app.post("/segments/{segment_id}/delete")
def delete_segment(segment_id: int):
    store.delete_segment(segment_id)
    log_event(f"Deleted segment #{segment_id}")
    return RedirectResponse(url="/segments", status_code=303)


@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    labeled_segments = store.get_labeled_segments()
    counts = store.get_label_counts_by_appliance()
    appliances = store.list_appliances()
    current_metrics = metrics_history[-1] if metrics_history else None
    return templates.TemplateResponse(
        "models.html",
        {
            "request": request,
            "config": config,
            "training": classifier.last_metrics,
            "labeled_segments": labeled_segments,
            "counts": counts,
            "appliances": appliances,
            "training_state": training_state,
            "metrics_history": metrics_history,
            "current_metrics": current_metrics,
        },
    )


@app.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request):
    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "config": config,
        },
    )


@app.get("/logs/feed")
def get_logs():
    return {"logs": list(recent_logs)[:100]}


@app.post("/models/retrain")
def retrain_models():
    maybe_train_classifier()
    log_event("Manual retrain requested")
    return RedirectResponse(url="/models", status_code=303)
