import sqlite3

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import context
from app.logging_utils import log_event
from app.utils import build_mqtt_topics

router = APIRouter(prefix="/appliances")


@router.get("", response_class=HTMLResponse)
def appliances_page(request: Request):
    appliances = context.store.list_appliances()
    if context.config.get("mqtt_enabled"):
        enriched = []
        for appliance in appliances:
            item = dict(appliance)
            item.update(build_mqtt_topics(appliance, context.config))
            enriched.append(item)
        appliances = enriched
    log_event("Appliances page viewed")
    return context.templates.TemplateResponse(
        "appliances.html",
        {"request": request, "appliances": appliances, "config": context.config},
    )


@router.post("")
def create_appliance(
    request: Request,
    name: str = Form(...),
    power_entity_id: str = Form(...),
    activity_sensors: str = Form(""),
    learning_appliance: int = Form(0),
    learning_sensor_id: str = Form(""),
):
    errors = []
    if not context.config.get("mqtt_enabled"):
        try:
            context.ha_client.get_state(power_entity_id)
        except Exception as exc:
            log_event(f"Power entity check failed: {exc}", level="warning")
            errors.append("Power entity ID not available in Home Assistant.")
    if learning_appliance:
        try:
            context.ha_client.get_state(learning_sensor_id)
        except Exception as exc:
            log_event(f"Learning sensor check failed: {exc}", level="warning")
            errors.append("Learning sensor ID not available in Home Assistant.")
    if errors:
        appliances = context.store.list_appliances()
        if context.config.get("mqtt_enabled"):
            enriched = []
            for appliance in appliances:
                item = dict(appliance)
                item.update(build_mqtt_topics(appliance, context.config))
                enriched.append(item)
            appliances = enriched
        return context.templates.TemplateResponse(
            "appliances.html",
            {
                "request": request,
                "appliances": appliances,
                "config": context.config,
                "error": " ".join(errors),
                "form": {
                    "name": name,
                    "power_entity_id": power_entity_id,
                    "activity_sensors": activity_sensors,
                    "learning_sensor_id": learning_sensor_id,
                    "learning_appliance": learning_appliance,
                },
            },
            status_code=400,
        )
    try:
        context.store.add_appliance(
            name,
            "",
            power_entity_id,
            activity_sensors,
            1 if learning_appliance else 0,
            learning_sensor_id,
        )
        log_event(f"Appliance created: {name}")
    except sqlite3.IntegrityError:
        appliances = context.store.list_appliances()
        if context.config.get("mqtt_enabled"):
            enriched = []
            for appliance in appliances:
                item = dict(appliance)
                item.update(build_mqtt_topics(appliance, context.config))
                enriched.append(item)
            appliances = enriched
        return context.templates.TemplateResponse(
            "appliances.html",
            {
                "request": request,
                "appliances": appliances,
                "config": context.config,
                "error": "Appliance name already exists.",
                "form": {
                    "name": name,
                    "power_entity_id": power_entity_id,
                    "activity_sensors": activity_sensors,
                    "learning_sensor_id": learning_sensor_id,
                    "learning_appliance": learning_appliance,
                },
            },
            status_code=400,
        )
    appliance_row = context.store.get_appliance(name)
    if appliance_row and context.mqtt_publisher:
        from app.main import publish_mqtt_discovery

        publish_mqtt_discovery(appliance_row, context.config, context.mqtt_publisher)
    return RedirectResponse(url="/appliances", status_code=303)


@router.post("/{name}/update")
def update_appliance(
    name: str,
    power_entity_id: str = Form(None),
    activity_sensors: str = Form(None),
    learning_appliance: int = Form(None),
    learning_sensor_id: str = Form(None),
):
    appliance = context.store.get_appliance(name)
    if not appliance:
        return RedirectResponse(url="/appliances", status_code=303)
    if (
        not context.config.get("mqtt_enabled")
        and power_entity_id
        and power_entity_id != appliance["power_entity_id"]
    ):
        try:
            context.ha_client.get_state(power_entity_id)
        except Exception as exc:
            log_event(f"Power entity check failed: {exc}", level="warning")
            return RedirectResponse(url="/appliances", status_code=303)
    context.store.update_appliance_config(
        name,
        power_entity_id=power_entity_id or appliance["power_entity_id"],
        activity_sensors=activity_sensors
        if activity_sensors is not None
        else appliance.get("activity_sensors", ""),
        learning_appliance=learning_appliance
        if learning_appliance is not None
        else appliance.get("learning_appliance", 0),
        learning_sensor_id=learning_sensor_id
        if learning_sensor_id is not None
        else appliance.get("learning_sensor_id", ""),
    )
    log_event(f"Appliance updated: {name}")
    return RedirectResponse(url="/appliances", status_code=303)


@router.post("/{name}/rename")
def rename_appliance(name: str, new_name: str = Form(...)):
    if context.store.get_appliance(new_name):
        return RedirectResponse(url="/appliances", status_code=303)
    try:
        context.store.rename_appliance(name, new_name)
        log_event(f"Appliance renamed from {name} to {new_name}")
    except Exception as exc:
        log_event(f"Failed to rename appliance: {exc}", level="warning")
    return RedirectResponse(url="/appliances", status_code=303)


@router.post("/{name}/delete")
def delete_appliance(name: str):
    context.store.delete_appliance(name)
    log_event(f"Appliance deleted: {name}")
    return RedirectResponse(url="/appliances", status_code=303)

