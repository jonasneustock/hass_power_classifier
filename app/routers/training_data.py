import logging

import numpy as np
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app import context
from app.logging_utils import log_event


router = APIRouter(prefix="/training-data")


def _cluster_segments(segments, eps=0.2, min_samples=2):
    if not segments:
        return []
    feats = []
    for seg in segments:
        feats.append(
            [
                seg.get("mean", 0.0),
                seg.get("std", 0.0),
                seg.get("max", 0.0),
                seg.get("min", 0.0),
                seg.get("duration", 0.0),
                seg.get("slope", 0.0),
                seg.get("change_score", 0.0),
            ]
        )
    X = np.array(feats)
    try:
        X_scaled = StandardScaler().fit_transform(X)
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
        labels = model.fit_predict(X_scaled)
    except Exception as exc:
        logging.warning("Clustering failed: %s", exc)
        log_event(f"Clustering failed: {exc}", level="warning")
        labels = np.array([-1] * len(segments))
    with_labels = []
    for seg, label in zip(segments, labels):
        seg_copy = dict(seg)
        seg_copy["cluster"] = int(label)
        with_labels.append(seg_copy)
    return with_labels


@router.get("", response_class=HTMLResponse)
def training_data_page(request: Request, eps: float = 0.2, min_samples: int = 2):
    log_event("Training data page viewed")
    labeled_segments = context.store.get_labeled_segments()
    clustered = _cluster_segments(labeled_segments, eps=eps, min_samples=min_samples)
    # attach sample snippets for sparklines
    for seg in clustered:
        samples = context.store.get_samples_between(seg["start_ts"], seg["end_ts"])
        seg["samples"] = samples
    cluster_counts = {}
    for seg in clustered:
        cluster_counts[seg["cluster"]] = cluster_counts.get(seg["cluster"], 0) + 1
    appliances = context.store.list_appliances()
    return context.templates.TemplateResponse(
        "training_data.html",
        {
            "request": request,
            "config": context.config,
            "segments": clustered,
            "cluster_counts": cluster_counts,
            "appliances": appliances,
            "eps": eps,
            "min_samples": min_samples,
        },
    )


@router.post("/{segment_id}/update")
def update_label(segment_id: int, appliance: str = Form(...)):
    segment = context.store.get_segment(segment_id)
    if not segment:
        return RedirectResponse(url="/training-data", status_code=303)
    context.store.update_segment_label(segment_id, appliance, None)
    context.training_manager.trigger_training()
    log_event(f"Training data label changed for segment #{segment_id} -> {appliance}")
    return RedirectResponse(url="/training-data", status_code=303)


@router.post("/{segment_id}/delete")
def delete_segment(segment_id: int):
    context.store.delete_segment(segment_id)
    log_event(f"Training data segment deleted #{segment_id}")
    return RedirectResponse(url="/training-data", status_code=303)
