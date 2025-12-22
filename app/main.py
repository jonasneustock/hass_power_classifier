import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.classifier import ClassifierService, RegressionService
from app.config import load_config
from app.data_store import DataStore
from app.ha_client import HAClient
from app.logging_utils import log_event, recent_logs
from app.mqtt_client import MqttPublisher
from app.poller import PowerPoller
from app.training import TrainingManager
from app.utils import build_mqtt_topics


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

training_manager = TrainingManager(
    store, classifier, regression_service, config, data_dir
)
poller = PowerPoller(
    store, ha_client, classifier, regression_service, config, mqtt_publisher
)


def maybe_train_classifier():
    training_manager.trigger_training()


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
    training_manager.ensure_base_appliance()
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
            "training_state": training_manager.training_state,
        },
        headers={"X-Partial": request.headers.get("X-Partial", "")},
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
    current_metrics = (
        training_manager.metrics_history[-1]
        if training_manager.metrics_history
        else None
    )
    return templates.TemplateResponse(
        "models.html",
        {
            "request": request,
            "config": config,
            "training": classifier.last_metrics,
            "labeled_segments": labeled_segments,
            "counts": counts,
            "appliances": appliances,
            "training_state": training_manager.training_state,
            "metrics_history": training_manager.metrics_history,
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
