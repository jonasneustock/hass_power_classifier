from fastapi import APIRouter

from app import context
from app.logging_utils import log_event

router = APIRouter(prefix="/api")


@router.post("/retrain")
def api_retrain():
    context.training_manager.trigger_training()
    log_event("API retrain requested")
    return {"status": "ok", "message": "training triggered"}


@router.get("/metrics")
def api_metrics():
    log_event("API metrics requested")
    return {
        "training_state": context.training_manager.training_state,
        "current_metrics": context.training_manager.metrics_history[-1]
        if context.training_manager.metrics_history
        else None,
        "history": context.training_manager.metrics_history,
    }


@router.get("/segments")
def api_segments(limit: int = 100):
    log_event(f"API segments requested (limit={limit})")
    segments = context.store.list_segments(
        limit=limit, unlabeled_only=False, candidate_only=False
    )
    return {"segments": segments}


@router.get("/appliances")
def api_appliances():
    log_event("API appliances requested")
    return {"appliances": context.store.list_appliances()}


@router.post("/cleanup")
def api_cleanup():
    ts = int(time.time())
    deleted_segments = context.store.delete_unlabeled_before(
        ts - context.config.get("unlabeled_ttl", 7200)
    )
    pruned_samples = 0
    retention = context.config.get("sample_retention", 0)
    if retention and retention > 0:
        context.store.delete_samples_before(ts - retention)
        pruned_samples = 1
    log_event(
        f"API cleanup triggered: removed {deleted_segments} unlabeled segments; samples pruned={pruned_samples}"
    )
    return {
        "deleted_unlabeled_segments": deleted_segments,
        "samples_pruned": bool(pruned_samples),
    }
