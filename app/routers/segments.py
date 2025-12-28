import json

from fastapi import APIRouter, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app import context
from app.logging_utils import log_event
from app.utils import compute_segment_delta, push_power_value

router = APIRouter(prefix="/segments")


@router.get("", response_class=HTMLResponse)
def segments_page(
    request: Request,
    candidate: int = 1,
    unlabeled: int = 1,
    min_change: float = 0.0,
    page: int = 1,
    limit: int = 50,
):
    page = max(1, page)
    limit = max(1, min(200, limit))
    offset = (page - 1) * limit
    log_event(f"Segments page viewed (candidate={candidate}, unlabeled={unlabeled}, min_change={min_change})")
    segments = context.store.list_segments(
        limit=limit + 1,
        unlabeled_only=bool(unlabeled),
        candidate_only=bool(candidate),
        offset=offset,
    )
    if min_change > 0:
        segments = [s for s in segments if s["change_score"] >= min_change]
    has_next = len(segments) > limit
    segments = segments[:limit]
    has_prev = page > 1
    return context.templates.TemplateResponse(
        "segments.html",
        {
            "request": request,
            "segments": segments,
            "candidate": candidate,
            "unlabeled": unlabeled,
            "min_change": min_change,
            "page": page,
            "limit": limit,
            "has_next": has_next,
            "has_prev": has_prev,
            "config": context.config,
        },
    )


@router.get("/export")
def export_segments():
    segments = context.store.get_labeled_segments()
    log_event(f"Segments export: {len(segments)} items")
    return JSONResponse(content={"segments": segments})


@router.post("/import")
def import_segments(file: UploadFile = File(...)):
    if not file:
        return RedirectResponse(url="/segments", status_code=303)
    try:
        data = file.file.read()
        payload = json.loads(data)
        segments = payload.get("segments") if isinstance(payload, dict) else payload
        if not isinstance(segments, list):
            raise ValueError("Invalid payload")
        inserted = context.store.import_segments(segments)
        log_event(f"Imported {inserted} segments")
    except Exception as exc:
        log_event(f"Import failed: {exc}", level="error")
    return RedirectResponse(url="/segments", status_code=303)


@router.get("/next_unlabeled")
def next_segment():
    seg = context.store.get_latest_unlabeled_segment()
    log_event("Next unlabeled segment requested")
    if seg:
        return RedirectResponse(url=f"/segments/{seg['id']}", status_code=303)
    return RedirectResponse(url="/segments", status_code=303)


@router.get("/{segment_id}", response_class=HTMLResponse)
def segment_detail(request: Request, segment_id: int):
    segment = context.store.get_segment(segment_id)
    if not segment:
        log_event(f"Segment {segment_id} not found", level="warning")
        return RedirectResponse(url="/segments", status_code=303)
    log_event(f"Segment detail requested for #{segment_id}")
    samples = context.store.get_samples_between(segment["start_ts"], segment["end_ts"])
    appliances = [a for a in context.store.list_appliances() if not a.get("learning_appliance")]
    predictions = context.classifier.top_predictions(segment, top_n=3)
    return context.templates.TemplateResponse(
        "segment_detail.html",
        {
            "request": request,
            "segment": segment,
            "samples": samples,
            "appliances": appliances,
            "predictions": predictions,
            "config": context.config,
        },
    )


@router.post("/{segment_id}/label")
def label_segment(
    segment_id: int,
    appliance: str = Form(...),
    next_segment: int = Form(0),
):
    context.store.update_segment_label(segment_id, appliance, None)
    context.training_manager.trigger_training()
    segment = context.store.get_segment(segment_id)
    if segment:
        flank = segment.get("flank")
        delta = compute_segment_delta(context.store, segment)
        current = context.store.get_appliance(appliance).get("current_power") or 0
        if flank == "negative":
            new_power = max(0.0, current - delta)
        else:
            new_power = current + delta
        push_power_value(
            appliance, new_power, context.store, context.ha_client, context.mqtt_publisher, context.config
        )
    log_event(f"Labeled segment #{segment_id} as {appliance}")
    if next_segment:
        next_seg = context.store.get_latest_unlabeled_segment()
        if next_seg:
            return RedirectResponse(url=f"/segments/{next_seg['id']}", status_code=303)
    return RedirectResponse(url="/segments", status_code=303)


@router.post("/{segment_id}/accept_prediction")
def accept_prediction(segment_id: int):
    segment = context.store.get_segment(segment_id)
    if not segment or not segment.get("predicted_appliance"):
        return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)
    appliance = segment["predicted_appliance"]
    appliance_row = context.store.get_appliance(appliance)
    if not appliance_row:
        return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)
    context.store.update_segment_label(segment_id, appliance, None)
    delta = compute_segment_delta(context.store, segment)
    current = appliance_row.get("current_power") or 0
    flank = segment.get("flank")
    if flank == "negative":
        new_power = max(0.0, current - delta)
    else:
        new_power = current + delta
    push_power_value(
        appliance, new_power, context.store, context.ha_client, context.mqtt_publisher, context.config
    )
    log_event(
        f"Accepted prediction for segment #{segment_id}: {segment['predicted_appliance']}"
    )
    context.training_manager.trigger_training()
    return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)


@router.post("/{segment_id}/reject_prediction")
def reject_prediction(segment_id: int):
    context.store.clear_segment_prediction(segment_id)
    log_event(f"Rejected prediction for segment #{segment_id}")
    return RedirectResponse(url=f"/segments/{segment_id}", status_code=303)


@router.post("/{segment_id}/delete")
def delete_segment(segment_id: int):
    context.store.delete_segment(segment_id)
    log_event(f"Deleted segment #{segment_id}")
    return RedirectResponse(url="/segments", status_code=303)
