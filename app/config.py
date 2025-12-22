import os

from dotenv import load_dotenv


def load_config():
    load_dotenv()
    sensors_env = os.getenv("HA_POWER_SENSORS", "")
    sensors = [s.strip() for s in sensors_env.split(",") if s.strip()]
    if not sensors:
        fallback = os.getenv("HA_POWER_SENSOR_ENTITY", "")
        if fallback:
            sensors = [fallback]

    relative_env = os.getenv("RELATIVE_CHANGE_THRESHOLD", "0.2")
    absolute_env = os.getenv("ABSOLUTE_CHANGE_THRESHOLD", "")
    relative_change_threshold = float(relative_env) if relative_env != "" else None
    absolute_change_threshold = float(absolute_env) if absolute_env != "" else None

    return {
        "ha_base_url": os.getenv("HA_BASE_URL", "http://homeassistant.local:8123"),
        "ha_token": os.getenv("HA_TOKEN", ""),
        "power_sensors": sensors[:10],
        "poll_interval": float(os.getenv("POLL_INTERVAL_SECONDS", "5")),
        "relative_change_threshold": relative_change_threshold,
        "absolute_change_threshold": absolute_change_threshold,
        "segment_pre_samples": int(os.getenv("SEGMENT_PRE_SAMPLES", "15")),
        "segment_post_samples": int(os.getenv("SEGMENT_POST_SAMPLES", "15")),
        "min_labels": int(os.getenv("MIN_LABELS_PER_APPLIANCE", "5")),
        "status_ttl": int(os.getenv("STATUS_TTL_SECONDS", "300")),
        "unlabeled_ttl": int(os.getenv("UNLABELED_TTL_SECONDS", "7200")),
        "cleanup_interval": int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300")),
        "app_title": os.getenv("APP_TITLE", "HA Power Classifier"),
        "data_dir": os.getenv("DATA_DIR", "/data"),
        "mqtt_enabled": os.getenv("MQTT_ENABLED", "false").lower()
        in ("1", "true", "yes", "on"),
        "mqtt_host": os.getenv("MQTT_HOST", "localhost"),
        "mqtt_port": int(os.getenv("MQTT_PORT", "1883")),
        "mqtt_username": os.getenv("MQTT_USERNAME", ""),
        "mqtt_password": os.getenv("MQTT_PASSWORD", ""),
        "mqtt_base_topic": os.getenv("MQTT_BASE_TOPIC", "ha_power_classifier").rstrip(
            "/"
        ),
        "mqtt_discovery_prefix": os.getenv(
            "MQTT_DISCOVERY_PREFIX", "homeassistant"
        ).rstrip("/"),
        "mqtt_client_id": os.getenv("MQTT_CLIENT_ID", "ha-power-classifier"),
        "mqtt_device_id": os.getenv("MQTT_DEVICE_ID", "ha_power_classifier"),
    }

