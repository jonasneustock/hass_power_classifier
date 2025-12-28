from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import context
from app.logging_utils import log_event

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    log_event("Dashboard requested")
    latest_sample = context.store.get_latest_sample()
    latest_per_sensor = context.store.get_latest_sensor_samples()
    appliances = context.store.list_appliances()
    segments = context.store.list_segments(limit=5, unlabeled_only=True)
    recent_samples = list(getattr(context.poller, "recent_total_diffs", []))
    sensor_diffs = getattr(context.poller, "recent_sensor_diffs", {})
    recent_by_sensor = {k: list(v) for k, v in sensor_diffs.items()}
    detection_events = []
    for seg in context.store.list_detection_events(limit=30):
        event_phase = seg.get("predicted_phase") or seg.get("label_phase")
        if event_phase not in ("start", "stop", None):
            continue
        detection_events.append(
            {
                "ts": seg["start_ts"],
                "appliance": seg.get("predicted_appliance") or seg.get("label_appliance"),
                "phase": event_phase,
            }
        )
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
