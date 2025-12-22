import numpy as np


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
