import logging
import threading
import time
from collections import deque

from app.logging_utils import log_event
from app.utils import build_mqtt_topics, compute_features


class PowerPoller:
    def __init__(
        self,
        store,
        ha_client,
        classifier,
        regression_service,
        config,
        mqtt_publisher=None,
    ):
        self.store = store
        self.ha_client = ha_client
        self.classifier = classifier
        self.regression_service = regression_service
        self.config = config
        self.mqtt_publisher = mqtt_publisher
        self.sensors = config["power_sensors"]
        min_window = config["segment_pre_samples"] + config["segment_post_samples"] + 1
        max_samples = max(min_window + 20, 50)
        self.samples_diff = deque(maxlen=max_samples)
        self.pending_segment = None
        self.sample_count = 0
        self.last_cleanup_ts = 0
        self.activity_prev = {}
        self.activity_events = deque(maxlen=200)
        self.stop_event = threading.Event()
        self.thread = None
        self.active_sessions = {}
        self.prev_total = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def run(self):
        while not self.stop_event.is_set():
            ts = int(time.time())
            total_value = 0.0
            read_success = 0
            for sensor in self.sensors:
                try:
                    state = self.ha_client.get_state(sensor)
                    value = float(state["state"])
                    self.store.add_sensor_sample(ts, sensor, value)
                    total_value += value
                    read_success += 1
                except Exception as exc:
                    logging.warning("Failed to read HA sensor %s: %s", sensor, exc)
                    log_event(
                        f"Failed to read HA sensor {sensor}: {exc}",
                        level="warning",
                    )

            if read_success == 0:
                time.sleep(self.config["poll_interval"])
                continue

            self.store.add_sample(ts, total_value)
            if self.prev_total is None:
                self.prev_total = total_value
                time.sleep(self.config["poll_interval"])
                continue
            self._poll_activity_sensors(ts)

            diff_value = total_value - self.prev_total
            self.prev_total = total_value
            self.samples_diff.append((ts, diff_value))
            self.sample_count += 1

            if len(self.samples_diff) >= 2 and self.pending_segment is None:
                prev_value = self.samples_diff[-2][1]
                trigger = False
                relative_threshold = self.config.get("relative_change_threshold")
                absolute_threshold = self.config.get("absolute_change_threshold")
                if relative_threshold is not None:
                    denom = abs(prev_value) if abs(prev_value) > 1e-6 else 1.0
                    relative_change = abs(diff_value - prev_value) / denom
                    trigger = relative_change >= relative_threshold
                elif absolute_threshold is not None:
                    absolute_change = abs(diff_value - prev_value)
                    trigger = absolute_change >= absolute_threshold

                if (
                    trigger
                    and len(self.samples_diff) >= self.config["segment_pre_samples"] + 1
                ):
                    self.pending_segment = {
                        "trigger_count": self.sample_count,
                    }

            if self.pending_segment:
                post_samples = self.config["segment_post_samples"]
                if (
                    self.sample_count - self.pending_segment["trigger_count"]
                    >= post_samples
                ):
                    samples_list = list(self.samples_diff)
                    segment_length = (
                        self.config["segment_pre_samples"]
                        + self.config["segment_post_samples"]
                        + 1
                    )
                    if len(samples_list) >= segment_length:
                        segment_samples = samples_list[-segment_length:]
                        if len(segment_samples) >= 30:
                            features = compute_features(segment_samples)
                            delta = segment_samples[-1][1] - segment_samples[0][1]
                            if delta > 0:
                                flank = "positive"
                            elif delta < 0:
                                flank = "negative"
                            else:
                                flank = "flat"
                            segment = {
                                "start_ts": segment_samples[0][0],
                                "end_ts": segment_samples[-1][0],
                                "mean": features["mean"],
                                "std": features["std"],
                                "max": features["max"],
                                "min": features["min"],
                                "duration": features["duration"],
                                "slope": features["slope"],
                                "change_score": features["change_score"],
                                "candidate": True,
                                "flank": flank,
                                "created_ts": ts,
                            }
                            segment_id = self.store.add_segment(segment)
                            log_event(
                                f"Segment #{segment_id} created change={round(features['change_score']*100,1)}%"
                            )
                            hint = self._activity_hint(segment_samples[-1][0])
                            if hint:
                                segment["predicted_appliance"] = hint["appliance"]
                                segment["predicted_phase"] = hint["phase"]
                                self.store.update_segment_prediction(
                                    segment_id,
                                    hint["appliance"],
                                    hint["phase"],
                                )
                                log_event(
                                    f"Activity hint applied to segment #{segment_id}: {hint['appliance']}/{hint['phase']}"
                                )
                            prediction = self.classifier.predict(segment)
                            if prediction:
                                appliance, phase = prediction
                                self.store.update_segment_prediction(
                                    segment_id, appliance, phase
                                )
                                log_event(
                                    f"Prediction for segment #{segment_id}: {appliance}/{phase}"
                                )
                                self._record_phase(appliance, phase, ts)
                    self.pending_segment = None

            if ts - self.last_cleanup_ts >= self.config["cleanup_interval"]:
                cutoff = ts - self.config["unlabeled_ttl"]
                deleted = self.store.delete_unlabeled_before(cutoff)
                if deleted:
                    logging.info(
                        "Deleted %s unlabeled segments older than %s", deleted, cutoff
                    )
                    log_event(f"Cleanup removed {deleted} stale segments")
                self.last_cleanup_ts = ts

            self._push_power_allocations(ts, total_value)
            time.sleep(self.config["poll_interval"])

    def _poll_activity_sensors(self, ts):
        appliances = self.store.list_appliances()
        for appliance in appliances:
            sensors_raw = appliance.get("activity_sensors") or ""
            sensors = [s.strip() for s in sensors_raw.split(",") if s.strip()]
            if not sensors:
                continue
            for sensor in sensors:
                try:
                    state = self.ha_client.get_state(sensor)
                    value = str(state.get("state", "")).lower()
                except Exception as exc:
                    log_event(f"Failed to read activity sensor {sensor}: {exc}", level="warning")
                    continue
                prev = self.activity_prev.get(sensor)
                if prev == value:
                    continue
                self.activity_prev[sensor] = value
                if value == "on":
                    phase = "start"
                elif value == "off":
                    phase = "stop"
                else:
                    continue
                self.activity_events.append(
                    {"ts": ts, "appliance": appliance["name"], "phase": phase, "sensor": sensor}
                )
                log_event(f"Activity sensor {sensor} -> {phase} for {appliance['name']}")

    def _activity_hint(self, segment_ts):
        if not self.activity_events:
            return None
        window = max(self.config.get("poll_interval", 5) * 3, 10)
        for event in reversed(self.activity_events):
            if segment_ts - event["ts"] <= window:
                return event
        return None

    def _record_phase(self, appliance, phase, ts):
        appliance_row = self.store.get_appliance(appliance)
        if not appliance_row:
            return
        last_status = appliance_row.get("last_status")
        if last_status == phase:
            return
        if phase == "start":
            self.active_sessions[appliance] = ts
        if phase == "stop":
            self.active_sessions.pop(appliance, None)
        self.store.update_appliance_status(appliance, phase, ts)
        log_event(f"Phase update for {appliance}: {phase}")
        if last_status == "start" and phase == "stop":
            if self.config.get("mqtt_enabled") and self.mqtt_publisher:
                topics = build_mqtt_topics(appliance_row, self.config)
                if not self.mqtt_publisher.publish_value(
                    topics["power_state_topic"], 0, retain=True
                ):
                    logging.warning("Failed to publish MQTT power 0 for %s", appliance)
                else:
                    log_event(f"Power reset to 0 for {appliance}")
            else:
                try:
                    self.ha_client.set_state(
                        appliance_row["power_entity_id"],
                        0,
                        {
                            "appliance": appliance,
                            "source": "ha_power_classifier",
                        },
                    )
                    log_event(f"Power reset to 0 for {appliance}")
                except Exception as exc:
                    logging.warning("Failed to reset power for %s: %s", appliance, exc)

    def _push_power_allocations(self, ts, total_power):
        appliances = {a["name"]: a for a in self.store.list_appliances()}
        if not self.active_sessions:
            return

        for appliance_name, start_ts in list(self.active_sessions.items()):
            appliance = appliances.get(appliance_name)
            if not appliance:
                continue
            elapsed = max(0, ts - start_ts)
            predicted = self.regression_service.predict(appliance_name, elapsed)
            watts = predicted
            if watts is None or watts <= 0:
                watts = (
                    appliance.get("mean_power")
                    or appliance.get("running_watts")
                    or 0
                )
            if watts <= 0:
                continue
            if self.config.get("mqtt_enabled") and self.mqtt_publisher:
                topics = build_mqtt_topics(appliance, self.config)
                if not self.mqtt_publisher.publish_value(
                    topics["power_state_topic"], round(watts, 2), retain=True
                ):
                    logging.warning(
                        "Failed to publish MQTT power for %s", appliance_name
                    )
                else:
                    log_event(
                        f"Power published for {appliance_name}: {round(watts,2)} W"
                    )
                continue
            try:
                self.ha_client.set_state(
                    appliance["power_entity_id"],
                    round(watts, 2),
                    {
                        "appliance": appliance_name,
                        "source": "ha_power_classifier",
                    },
                )
                log_event(f"Power set for {appliance_name}: {round(watts,2)} W")
            except Exception as exc:
                logging.warning("Failed to push power for %s: %s", appliance_name, exc)
