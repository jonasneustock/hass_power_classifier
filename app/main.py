import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import context
from app.logging_utils import log_event
from app.routers import api, appliances, dashboard, logs_page, models_page, segments
from app.utils import build_mqtt_topics


def publish_mqtt_discovery(appliance_row, config, mqtt_publisher):
    if not mqtt_publisher:
        return
    topics = build_mqtt_topics(appliance_row, config)
    device = {
        "identifiers": [config["mqtt_device_id"]],
        "name": config["app_title"],
        "manufacturer": "custom",
        "model": "ha_power_classifier",
    }
    power_payload = {
        "name": f"{appliance_row['name']} power",
        "state_topic": topics["power_state_topic"],
        "unique_id": f"{config['mqtt_device_id']}_{topics['power_object_id']}",
        "object_id": topics["power_object_id"],
        "device": device,
        "unit_of_measurement": "W",
        "device_class": "power",
        "state_class": "measurement",
    }
    mqtt_publisher.publish_json(topics["power_config_topic"], power_payload, retain=True)


def check_ha_connection():
    if not context.config["ha_token"] or not context.config["power_sensors"]:
        logging.warning("HA_TOKEN or HA_POWER_SENSORS not set")
        log_event("HA config missing token or sensors", level="warning")
        return False
    ok = True
    for sensor in context.config["power_sensors"]:
        try:
            state = context.ha_client.get_state(sensor)
            float(state["state"])
            logging.info("HA connection check succeeded for %s", sensor)
            log_event(f"HA check ok for {sensor}")
        except Exception as exc:
            logging.warning("HA connection check failed for %s: %s", sensor, exc)
            log_event(f"HA check failed for {sensor}: {exc}", level="warning")
            ok = False
    return ok


def create_app() -> FastAPI:
    app = FastAPI(title=context.config["app_title"])
    app.mount(
        "/static", StaticFiles(directory=str(context.base_dir / "static")), name="static"
    )

    @app.on_event("startup")
    def on_startup():
        context.training_manager.ensure_base_appliance()
        context.store.clear_phase_data()
        check_ha_connection()
        if context.mqtt_publisher:
            try:
                context.mqtt_publisher.connect()
            except Exception as exc:
                logging.warning("MQTT connection failed: %s", exc)
            else:
                for appliance in context.store.list_appliances():
                    publish_mqtt_discovery(appliance, context.config, context.mqtt_publisher)
                log_event("MQTT connected")
        log_event("Application started")
        context.training_manager.start_scheduler()
        context.poller.start()

    @app.on_event("shutdown")
    def on_shutdown():
        context.poller.stop()
        context.training_manager.stop_scheduler()
        if context.mqtt_publisher:
            context.mqtt_publisher.close()
        log_event("Application stopped")

    # Routers
    app.include_router(dashboard.router)
    app.include_router(appliances.router)
    app.include_router(segments.router)
    app.include_router(models_page.router)
    app.include_router(logs_page.router)
    app.include_router(api.router)

    # Redirect root to dashboard if needed
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/favicon.ico")
    def favicon():
        return RedirectResponse(url="/static/favicon.ico")

    return app


app = create_app()
