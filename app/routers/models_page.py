from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import context
from app.logging_utils import log_event

router = APIRouter(prefix="/models")


@router.get("", response_class=HTMLResponse)
def models_page(request: Request):
    labeled_segments = context.store.get_labeled_segments()
    counts = context.store.get_label_counts_by_appliance()
    appliances = context.store.list_appliances()
    current_metrics = (
        context.training_manager.metrics_history[-1]
        if context.training_manager.metrics_history
        else None
    )
    return context.templates.TemplateResponse(
        "models.html",
        {
            "request": request,
            "config": context.config,
            "training": context.classifier.last_metrics,
            "labeled_segments": labeled_segments,
            "counts": counts,
            "appliances": appliances,
            "training_state": context.training_manager.training_state,
            "metrics_history": context.training_manager.metrics_history,
            "current_metrics": current_metrics,
        },
    )


@router.post("/retrain")
def retrain_models():
    context.training_manager.trigger_training()
    log_event("Manual retrain requested")
    return RedirectResponse(url="/models", status_code=303)

