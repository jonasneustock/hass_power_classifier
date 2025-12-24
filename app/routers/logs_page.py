from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import context

router = APIRouter()


@router.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request):
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

