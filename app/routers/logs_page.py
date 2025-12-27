from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import context
from app.logging_utils import log_event

router = APIRouter()


@router.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request):
    log_event("Logs page viewed")
    return context.templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "config": context.config,
        },
    )


@router.get("/logs/feed")
def get_logs():
    return {"logs": list(context.recent_logs)[:100]}
