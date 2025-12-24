from fastapi import APIRouter

from app import context

router = APIRouter(prefix="/api")


@router.post("/retrain")
def api_retrain():
    context.training_manager.trigger_training()
    return {"status": "ok", "message": "training triggered"}


@router.get("/metrics")
def api_metrics():
    return {
        "training_state": context.training_manager.training_state,
        "current_metrics": context.training_manager.metrics_history[-1]
        if context.training_manager.metrics_history
        else None,
        "history": context.training_manager.metrics_history,
    }


@router.get("/segments")
def api_segments(limit: int = 100):
    segments = context.store.list_segments(
        limit=limit, unlabeled_only=False, candidate_only=False
    )
    return {"segments": segments}


@router.get("/appliances")
def api_appliances():
    return {"appliances": context.store.list_appliances()}

