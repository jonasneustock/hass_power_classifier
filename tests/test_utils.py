import math

import pytest

from app.utils import (
    build_mqtt_topics,
    compute_features,
    normalize_object_id,
    slugify,
)


def test_compute_features_relative_change():
    samples = [
        (0, 100.0),
        (1, 120.0),
        (2, 80.0),
    ]
    feats = compute_features(samples)
    assert math.isclose(feats["mean"], 0.0, abs_tol=1e-6)
    assert math.isclose(feats["max"], 0.2, abs_tol=1e-6)
    assert math.isclose(feats["min"], -0.2, abs_tol=1e-6)
    assert math.isclose(feats["change_score"], 0.4, abs_tol=1e-6)
    assert feats["duration"] == 2.0


def test_slugify_normalizes_and_falls_back():
    assert slugify(" My-Appliance 3000!") == "my_appliance_3000"
    assert slugify("   ") == "appliance"


def test_build_mqtt_topics_from_entity_ids():
    appliance_row = {
        "name": "Dryer",
        "status_entity_id": "sensor.dryer_status",
        "power_entity_id": "sensor.dryer_power",
    }
    config = {
        "mqtt_base_topic": "ha_power",
        "mqtt_discovery_prefix": "homeassistant",
    }
    topics = build_mqtt_topics(appliance_row, config)
    assert topics["status_state_topic"] == "ha_power/dryer/status"
    assert topics["power_state_topic"] == "ha_power/dryer/power"
    assert topics["status_config_topic"] == "homeassistant/sensor/dryer_status/config"
    assert topics["power_config_topic"] == "homeassistant/sensor/dryer_power/config"


def test_build_mqtt_topics_with_fallback_ids():
    appliance_row = {
        "name": "Cooler",
        "status_entity_id": "",
        "power_entity_id": "",
    }
    config = {
        "mqtt_base_topic": "ha_power",
        "mqtt_discovery_prefix": "homeassistant",
    }
    topics = build_mqtt_topics(appliance_row, config)
    assert topics["status_state_topic"] == "ha_power/cooler/status"
    assert topics["power_state_topic"] == "ha_power/cooler/power"
    assert topics["status_object_id"] == "cooler_status"
    assert topics["power_object_id"] == "cooler_power"


@pytest.mark.parametrize(
    "entity_id,expected",
    [
        ("sensor.test_value", "test_value"),
        ("justvalue", "justvalue"),
        ("", "fallback"),
    ],
)
def test_normalize_object_id(entity_id, expected):
    assert normalize_object_id(entity_id, "fallback") == expected
