from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import context

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    latest_sample = context.store.get_latest_sample()
    latest_per_sensor = context.store.get_latest_sensor_samples()
    appliances = context.store.list_appliances()
    segments = context.store.list_segments(limit=10, unlabeled_only=True)
    recent_samples_raw = context.store.get_recent_samples(limit=200)
    recent_samples = []
    for idx in range(1, len(recent_samples_raw)):
        prev = recent_samples_raw[idx - 1]
        curr = recent_samples_raw[idx]
        recent_samples.append({"ts": curr["ts"], "value": curr["value"] - prev["value"]})

    recent_by_sensor = {}
    for sensor in context.config["power_sensors"]:
        raw = context.store.get_recent_sensor_samples(sensor, limit=200)
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
        for seg in context.store.list_segments(limit=100, unlabeled_only=False)
        if (seg.get("predicted_phase") or seg.get("label_phase")) in ("start", "stop")
    ]
    activity_events = list(getattr(context.poller, "activity_events", []))
    training = context.classifier.last_metrics
    return context.templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "config": context.config,
            "latest_sample": latest_sample,
            "latest_per_sensor": latest_per_sensor,
            "appliances": appliances,
            "segments": segments,
            "recent_samples": recent_samples,
            "recent_by_sensor": recent_by_sensor,
            "detection_events": detection_events,
            "activity_events": activity_events,
            "training": training,
            "training_state": context.training_manager.training_state,
        },
        headers={"X-Partial": request.headers.get("X-Partial", "")},
    )

