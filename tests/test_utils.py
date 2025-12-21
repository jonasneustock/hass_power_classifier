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


def test_compute_features_flat():
    samples = [(0, 50.0), (1, 50.0), (2, 50.0)]
    feats = compute_features(samples)
    assert feats["change_score"] == 0.0
    assert feats["slope"] == 0.0


def test_compute_features_linear_increase():
    samples = [(0, 10.0), (1, 20.0), (2, 30.0)]
    feats = compute_features(samples)
    assert feats["max"] > feats["min"]
    assert feats["slope"] > 0


def test_slugify_normalizes_and_falls_back():
    assert slugify(" My-Appliance 3000!") == "my_appliance_3000"
    assert slugify("   ") == "appliance"


def test_slugify_collapses_separators():
    assert slugify("a--b__c   d") == "a_b_c_d"


def test_build_mqtt_topics_from_entity_ids():
    appliance_row = {
        "name": "Dryer",
        "power_entity_id": "sensor.dryer_power",
    }
    config = {
        "mqtt_base_topic": "ha_power",
        "mqtt_discovery_prefix": "homeassistant",
    }
    topics = build_mqtt_topics(appliance_row, config)
    assert topics["power_state_topic"] == "ha_power/dryer/power"
    assert topics["power_config_topic"] == "homeassistant/sensor/dryer_power/config"


def test_build_mqtt_topics_with_fallback_ids():
    appliance_row = {
        "name": "Cooler",
        "power_entity_id": "",
    }
    config = {
        "mqtt_base_topic": "ha_power",
        "mqtt_discovery_prefix": "homeassistant",
    }
    topics = build_mqtt_topics(appliance_row, config)
    assert topics["power_state_topic"] == "ha_power/cooler/power"
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


def test_build_mqtt_topics_custom_base_and_prefix():
    appliance_row = {"name": "Lamp", "power_entity_id": "sensor.lamp_power"}
    config = {"mqtt_base_topic": "custom/base", "mqtt_discovery_prefix": "hass"}
    topics = build_mqtt_topics(appliance_row, config)
    assert topics["power_state_topic"] == "custom/base/lamp/power"
    assert topics["power_config_topic"] == "hass/sensor/lamp_power/config"
