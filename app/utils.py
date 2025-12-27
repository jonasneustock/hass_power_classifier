import numpy as np

from app.logging_utils import log_event

def compute_features(samples):
    timestamps = np.array([s[0] for s in samples], dtype=np.float64)
    values = np.array([s[1] for s in samples], dtype=np.float64)
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    baseline = float(values[0]) if len(values) else 0.0
    denom = abs(baseline) if abs(baseline) > 1e-6 else 1.0
    relative_values = (values - baseline) / denom

    mean = float(np.mean(relative_values))
    std = float(np.std(relative_values))
    min_val = float(np.min(relative_values))
    max_val = float(np.max(relative_values))
    change_score = max_val - min_val

    if len(timestamps) > 1:
        t_centered = timestamps - np.mean(timestamps)
        v_centered = relative_values - np.mean(relative_values)
        denom = np.sum(t_centered ** 2)
        slope = float(np.sum(t_centered * v_centered) / denom) if denom else 0.0
    else:
        slope = 0.0

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "duration": duration,
        "slope": slope,
        "change_score": change_score,
    }


def slugify(value):
    value = (value or "").strip().lower()
    cleaned = []
    last_sep = False
    for ch in value:
        if ch.isalnum():
            cleaned.append(ch)
            last_sep = False
        elif ch in (" ", "-", "_"):
            if not last_sep:
                cleaned.append("_")
                last_sep = True
        else:
            if not last_sep:
                cleaned.append("_")
                last_sep = True
    slug = "".join(cleaned).strip("_")
    return slug or "appliance"


def normalize_object_id(entity_id, fallback):
    if not entity_id:
        return fallback
    if "." in entity_id:
        return entity_id.split(".", 1)[1]
    return entity_id


def build_mqtt_topics(appliance_row, config):
    slug = slugify(appliance_row["name"])
    base_topic = config["mqtt_base_topic"]
    discovery_prefix = config["mqtt_discovery_prefix"]

    power_object_id = normalize_object_id(
        appliance_row.get("power_entity_id"), f"{slug}_power"
    )

    return {
        "power_state_topic": f"{base_topic}/{slug}/power",
        "power_config_topic": f"{discovery_prefix}/sensor/{power_object_id}/config",
        "power_object_id": power_object_id,
        "slug": slug,
    }


def samples_to_diffs(samples):
    if not samples or len(samples) < 2:
        return []
    ordered = sorted(samples, key=lambda s: s["ts"])
    diffs = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        curr = ordered[i]
        diffs.append({"ts": curr["ts"], "value": curr["value"] - prev["value"]})
    return diffs


def compute_segment_delta(store, segment):
    samples = store.get_samples_between(segment["start_ts"], segment["end_ts"])
    diffs = samples_to_diffs(samples)
    if not diffs:
        return 0.0
    values = [d["value"] for d in diffs]
    delta = max(values) - min(values)
    return max(0.0, float(delta))


def push_power_value(appliance, watts, store, ha_client, mqtt_publisher, config):
    appliance_row = store.get_appliance(appliance)
    if not appliance_row:
        return
    watts = max(0.0, float(watts))
    latest_total = store.get_latest_sample()
    limit = abs(latest_total["value"]) if latest_total and "value" in latest_total else None

    appliances = store.list_appliances()
    current_total = sum((a.get("current_power") or 0) for a in appliances)
    proposed_total = current_total - (appliance_row.get("current_power") or 0) + watts
    if limit is not None and proposed_total > limit:
        # identify largest contributor in proposed state
        proposed_powers = {
            a["name"]: (appliance == a["name"] and watts) or (a.get("current_power") or 0)
            for a in appliances
        }
        largest = max(proposed_powers.items(), key=lambda kv: kv[1] if kv[1] is not None else 0)
        if largest[0] == appliance:
            log_event(
                f"Skip publishing for {appliance}: proposed total {round(proposed_total,2)} exceeds limit {round(limit,2)}",
                level="warning",
            )
            return

    store.update_appliance_current_power(appliance, watts)
    if config.get("mqtt_enabled") and mqtt_publisher:
        topics = build_mqtt_topics(appliance_row, config)
        if mqtt_publisher.publish_value(topics["power_state_topic"], round(watts, 2), retain=True):
            log_event(f"Power published for {appliance}: {round(watts,2)} W")
        else:
            log_event(f"Failed to publish MQTT power for {appliance}", level="warning")
        return
    try:
        ha_client.set_state(
            appliance_row["power_entity_id"],
            round(watts, 2),
            {"appliance": appliance, "source": "ha_power_classifier"},
        )
        log_event(f"Power set for {appliance}: {round(watts,2)} W")
    except Exception as exc:
        log_event(f"Failed to push power for {appliance}: {exc}", level="warning")
