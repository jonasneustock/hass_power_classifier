import logging
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.classifier import ClassifierService
from app.data_store import DataStore
from app.ha_client import HAClient


def load_config():
    load_dotenv()
    return {
        "ha_base_url": os.getenv("HA_BASE_URL", "http://homeassistant.local:8123"),
        "ha_token": os.getenv("HA_TOKEN", ""),
        "power_sensor": os.getenv("HA_POWER_SENSOR_ENTITY", ""),
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
    }


def compute_features(samples):
    timestamps = np.array([s[0] for s in samples], dtype=np.float64)
    values = np.array([s[1] for s in samples], dtype=np.float64)
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    baseline = float(values[0]) if len(values) else 0.0
    denom = abs(baseline) if abs(baseline) > 1e-6 else 1.0
    relative_values = (values - baseline) / denom

    mean = float(np.mean(relative_values))
    std = float(np.std(relative_values))
    min_val = float(np.min(relative_values))
    max_val = float(np.max(relative_values))
    change_score = max_val - min_val

    if len(timestamps) > 1:
        t_centered = timestamps - np.mean(timestamps)
        v_centered = relative_values - np.mean(relative_values)
        denom = np.sum(t_centered ** 2)
        slope = float(np.sum(t_centered * v_centered) / denom) if denom else 0.0
    else:
        slope = 0.0

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "duration": duration,
        "slope": slope,
        "change_score": change_score,
    }


class PowerPoller:
    def __init__(self, store, ha_client, classifier, config):
        self.store = store
        self.ha_client = ha_client
        self.classifier = classifier
        self.config = config
        min_window = config["segment_pre_samples"] + config["segment_post_samples"] + 1
        max_samples = max(min_window + 20, 50)
        self.samples = deque(maxlen=max_samples)
        self.pending_segment = None
        self.last_cleanup_ts = 0
        self.stop_event = threading.Event()
        self.thread = None

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
            try:
                state = self.ha_client.get_state(self.config["power_sensor"])
                value = float(state["state"])
            except Exception as exc:
                logging.warning("Failed to read HA sensor: %s", exc)
                time.sleep(self.config["poll_interval"])
                continue

            self.store.add_sample(ts, value)
            self.samples.append((ts, value))

            if len(self.samples) >= 2 and self.pending_segment is None:
                prev_value = self.samples[-2][1]
                denom = abs(prev_value) if abs(prev_value) > 1e-6 else 1.0
                relative_change = abs(value - prev_value) / denom
                if (
                    relative_change >= self.config["relative_change_threshold"]
                    and len(self.samples) >= self.config["segment_pre_samples"] + 1
                ):
                    start_index = len(self.samples) - (self.config["segment_pre_samples"] + 1)
                    self.pending_segment = {
                        "start_index": start_index,
                        "remaining_after": self.config["segment_post_samples"],
                        "trigger_ts": ts,
                    }

            if self.pending_segment:
                if ts > self.pending_segment["trigger_ts"]:
                    self.pending_segment["remaining_after"] -= 1
                if self.pending_segment["remaining_after"] <= 0:
                    samples_list = list(self.samples)
                    start_index = self.pending_segment["start_index"]
                    segment_length = (
                        self.config["segment_pre_samples"]
                        + self.config["segment_post_samples"]
                        + 1
                    )
                    end_index = start_index + segment_length
                    if end_index <= len(samples_list):
                        segment_samples = samples_list[start_index:end_index]
                        if len(segment_samples) >= 30:
                            features = compute_features(segment_samples)
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
                                "created_ts": ts,
                            }
                            segment_id = self.store.add_segment(segment)
                            prediction = self.classifier.predict(segment)
                            if prediction:
                                appliance, phase = prediction
                                self.store.update_segment_prediction(
                                    segment_id, appliance, phase
                                )
                                self._push_status(appliance, phase, ts)
                    self.pending_segment = None

            if ts - self.last_cleanup_ts >= self.config["cleanup_interval"]:
                cutoff = ts - self.config["unlabeled_ttl"]
                deleted = self.store.delete_unlabeled_before(cutoff)
                if deleted:
                    logging.info("Deleted %s unlabeled segments older than %s", deleted, cutoff)
                self.last_cleanup_ts = ts

            self._push_power_allocations(ts, value)
            time.sleep(self.config["poll_interval"])

    def _push_status(self, appliance, phase, ts):
        appliance_row = self.store.get_appliance(appliance)
        if not appliance_row:
            return
        last_status = appliance_row.get("last_status")
        if last_status == phase:
            return
        try:
            self.ha_client.set_state(
                appliance_row["status_entity_id"],
                phase,
                {
                    "appliance": appliance,
                    "phase": phase,
                    "source": "ha_power_classifier",
                },
            )
            self.store.update_appliance_status(appliance, phase, ts)
        except Exception as exc:
            logging.warning("Failed to push status for %s: %s", appliance, exc)

    def _push_power_allocations(self, ts, total_power):
        appliances = self.store.list_appliances()
        active = []
        for appliance in appliances:
            last_ts = appliance.get("last_status_ts") or 0
            if appliance.get("last_status") != "running":
                continue
            if ts - last_ts > self.config["status_ttl"]:
                continue
            if (appliance.get("running_watts") or 0) <= 0:
                continue
            active.append(appliance)

        if not active:
            return

        total_running = sum(a["running_watts"] for a in active)
        if total_running <= 0:
            return

        for appliance in active:
            proportion = appliance["running_watts"] / total_running
            watts = proportion * total_power
            try:
                self.ha_client.set_state(
                    appliance["power_entity_id"],
                    round(watts, 2),
                    {
                        "appliance": appliance["name"],
                        "proportion": round(proportion, 3),
                        "source": "ha_power_classifier",
                    },
                )
            except Exception as exc:
                logging.warning("Failed to push power for %s: %s", appliance["name"], exc)


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


data_dir = Path(config["data_dir"])
data_dir.mkdir(parents=True, exist_ok=True)
store = DataStore(str(data_dir / "power_classifier.sqlite"))
ha_client = HAClient(config["ha_base_url"], config["ha_token"])
classifier = ClassifierService(str(data_dir / "model.pkl"))

poller = PowerPoller(store, ha_client, classifier, config)


def maybe_train_classifier():
    appliances = store.list_appliances()
    if not appliances:
        return None
    counts = store.get_label_counts_by_appliance()
    for appliance in appliances:
        if counts.get(appliance["name"], 0) < config["min_labels"]:
            return None
    labeled_segments = store.get_labeled_segments()
    if not labeled_segments:
        return None
    metrics = classifier.train(labeled_segments)
    running_watts = store.get_running_segments_by_appliance()
    for name, avg_watts in running_watts.items():
        store.update_appliance_running_watts(name, avg_watts)
    return metrics


def check_ha_connection():
    if not config["ha_token"] or not config["power_sensor"]:
        logging.warning("HA_TOKEN or HA_POWER_SENSOR_ENTITY not set")
        return False
    try:
        state = ha_client.get_state(config["power_sensor"])
        float(state["state"])
    except Exception as exc:
        logging.warning("HA connection check failed: %s", exc)
        return False
    logging.info("HA connection check succeeded for %s", config["power_sensor"])
    return True


@app.on_event("startup")
def on_startup():
    check_ha_connection()
    poller.start()


@app.on_event("shutdown")
def on_shutdown():
    poller.stop()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    latest_sample = store.get_latest_sample()
    appliances = store.list_appliances()
    segments = store.list_segments(limit=10, unlabeled_only=True)
    recent_samples = store.get_recent_samples(limit=200)
    training = classifier.last_metrics
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "config": config,
            "latest_sample": latest_sample,
            "appliances": appliances,
            "segments": segments,
            "recent_samples": recent_samples,
            "training": training,
        },
    )


@app.get("/appliances", response_class=HTMLResponse)
def appliances_page(request: Request):
    appliances = store.list_appliances()
    return templates.TemplateResponse(
        "appliances.html",
        {"request": request, "appliances": appliances, "config": config},
    )


@app.post("/appliances")
def create_appliance(
    request: Request,
    name: str = Form(...),
    status_entity_id: str = Form(...),
    power_entity_id: str = Form(...),
):
    errors = []
    try:
        ha_client.get_state(status_entity_id)
    except Exception as exc:
        logging.warning("Status entity check failed: %s", exc)
        errors.append("Status entity ID not available in Home Assistant.")
    try:
        ha_client.get_state(power_entity_id)
    except Exception as exc:
        logging.warning("Power entity check failed: %s", exc)
        errors.append("Power entity ID not available in Home Assistant.")
    if errors:
        appliances = store.list_appliances()
        return templates.TemplateResponse(
            "appliances.html",
            {
                "request": request,
                "appliances": appliances,
                "config": config,
                "error": " ".join(errors),
                "form": {
                    "name": name,
                    "status_entity_id": status_entity_id,
                    "power_entity_id": power_entity_id,
                },
            },
            status_code=400,
        )
    try:
        store.add_appliance(name, status_entity_id, power_entity_id)
    except sqlite3.IntegrityError:
        appliances = store.list_appliances()
        return templates.TemplateResponse(
            "appliances.html",
            {
                "request": request,
                "appliances": appliances,
                "config": config,
                "error": "Appliance name already exists.",
                "form": {
                    "name": name,
                    "status_entity_id": status_entity_id,
                    "power_entity_id": power_entity_id,
                },
            },
            status_code=400,
        )
    return RedirectResponse(url="/appliances", status_code=303)


@app.get("/segments", response_class=HTMLResponse)
def segments_page(request: Request, candidate: int = 1, unlabeled: int = 1):
    segments = store.list_segments(
        limit=200,
        unlabeled_only=bool(unlabeled),
        candidate_only=bool(candidate),
    )
    return templates.TemplateResponse(
        "segments.html",
        {
            "request": request,
            "segments": segments,
            "candidate": candidate,
            "unlabeled": unlabeled,
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
    return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)
