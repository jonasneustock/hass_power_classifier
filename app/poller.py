import logging
import os
import threading
import time
from collections import deque
from statistics import pstdev

from app.logging_utils import log_event
from app.utils import build_mqtt_topics, compute_features, compute_segment_delta, push_power_value


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
        self.learning_prev = {}
        self.learning_events = deque(maxlen=200)
        self.stop_event = threading.Event()
        self.thread = None
        self.prev_total = None
        self.restart_attempts = 0
        self.recent_total_diffs = deque(maxlen=200)
        self.recent_sensor_diffs = {s: deque(maxlen=200) for s in self.sensors}
        self.last_ttl_check = 0

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.restart_attempts = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def run(self):
        while not self.stop_event.is_set() and self.restart_attempts < 5:
            try:
                self._loop()
                return
            except Exception as exc:
                self.restart_attempts += 1
                logging.exception(
                    "Poller crashed (attempt %s/5): %s", self.restart_attempts, exc
                )
                log_event(
                    f"Poller crashed (attempt {self.restart_attempts}/5): {exc}",
                    level="error",
                )
                time.sleep(1)
        if self.restart_attempts >= 5:
            logging.critical("Poller failed 5 times, shutting down application.")
            log_event("Poller failed 5 times, shutting down application.", level="error")
            os._exit(1)

    def _loop(self):
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
                log_event(f"Initial total set {total_value}", level="info")
                time.sleep(self.config["poll_interval"])
                continue
            self._poll_activity_sensors(ts)
            self._poll_learning_appliances(ts)

            diff_value = total_value - self.prev_total
            self.prev_total = total_value
            self.samples_diff.append((ts, diff_value))
            self.sample_count += 1
            log_event(
                f"Diff recorded ts={ts} value={round(diff_value,3)} sample_count={self.sample_count}",
                level="info",
            )
            self.recent_total_diffs.append({"ts": ts, "value": diff_value})
            for sensor in self.sensors:
                samples = self.store.get_recent_sensor_samples(sensor, limit=2)
                if len(samples) == 2:
                    dv = samples[1]["value"] - samples[0]["value"]
                    self.recent_sensor_diffs[sensor].append({"ts": samples[1]["ts"], "value": dv})

            if len(self.samples_diff) >= 2 and self.pending_segment is None:
                prev_value = self.samples_diff[-2][1]
                trigger = False
                relative_threshold = self.config.get("relative_change_threshold")
                absolute_threshold = self.config.get("absolute_change_threshold")
                threshold_used = None
                if (
                    self.config.get("adaptive_threshold_enabled")
                    and relative_threshold is not None
                ):
                    window = min(
                        len(self.samples_diff), self.config.get("adaptive_window", 50)
                    )
                    if window >= 5:
                        recent_vals = [v for _, v in list(self.samples_diff)[-window:]]
                        noise = pstdev(recent_vals) if len(recent_vals) > 1 else 0
                        adapt = noise * self.config.get("adaptive_multiplier", 3)
                        min_rel = self.config.get("adaptive_min_relative", 0.05)
                        max_rel = self.config.get("adaptive_max_relative", 1.0)
                        relative_threshold = max(min_rel, min(max_rel, adapt))
                        threshold_used = relative_threshold
                if relative_threshold is not None:
                    denom = abs(prev_value) if abs(prev_value) > 1e-6 else 1.0
                    relative_change = abs(diff_value - prev_value) / denom
                    trigger = relative_change >= relative_threshold
                    if trigger:
                        log_event(
                            f"Trigger detected (relative). change={round(relative_change,3)}, threshold={relative_threshold}",
                            level="info",
                        )
                elif absolute_threshold is not None:
                    absolute_change = abs(diff_value - prev_value)
                    trigger = absolute_change >= absolute_threshold
                    if trigger:
                        log_event(
                            f"Trigger detected (absolute). change={round(absolute_change,3)}, threshold={absolute_threshold}",
                            level="info",
                        )

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
                    log_event(
                        f"Segment window filled (pre={self.config['segment_pre_samples']}, post={post_samples}); building segment",
                        level="info",
                    )
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
                            learn_hint = self._learning_hint(segment_samples[-1][0])
                            if learn_hint:
                                self.store.update_segment_label(segment_id, learn_hint["appliance"], None)
                                log_event(
                                    f"Learning appliance auto-label for segment #{segment_id}: {learn_hint['appliance']}"
                                )
                                self._apply_power_for_segment(segment_id, segment, learn_hint["appliance"], learn_hint.get("flank", flank))
                            hint = self._activity_hint(segment_samples[-1][0])
                            if hint:
                                segment["predicted_appliance"] = hint["appliance"]
                                self.store.update_segment_prediction(segment_id, hint["appliance"])
                                log_event(
                                    f"Activity hint applied to segment #{segment_id}: {hint['appliance']}"
                                )
                            prediction = self.classifier.predict(segment)
                            if prediction:
                                appliance = prediction
                                self.store.update_segment_prediction(segment_id, appliance)
                                log_event(
                                    f"Prediction for segment #{segment_id}: {appliance}"
                                )
                                self._apply_power_for_segment(segment_id, segment, appliance, flank)
                    self.pending_segment = None

            if ts - self.last_cleanup_ts >= self.config["cleanup_interval"]:
                cutoff = ts - self.config["unlabeled_ttl"]
                deleted = self.store.delete_unlabeled_before(cutoff)
                if deleted:
                    logging.info(
                        "Deleted %s unlabeled segments older than %s", deleted, cutoff
                    )
                    log_event(f"Cleanup removed {deleted} stale segments")
                retention = self.config.get("sample_retention", 0)
                if retention and retention > 0:
                    self.store.delete_samples_before(ts - retention)
                    log_event("Old samples pruned for retention window")
                self.last_cleanup_ts = ts

            # TTL check for appliance presence
            if ts - self.last_ttl_check >= self.config.get("status_ttl", 300):
                self._verify_appliances(ts)
                self.last_ttl_check = ts

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

    def _poll_learning_appliances(self, ts):
        appliances = [
            a for a in self.store.list_appliances() if a.get("learning_appliance")
        ]
        for appliance in appliances:
            sensor = appliance.get("learning_sensor_id")
            if not sensor:
                continue
            try:
                state = self.ha_client.get_state(sensor)
                value = float(state.get("state", 0))
            except Exception as exc:
                log_event(f"Failed to read learning sensor {sensor}: {exc}", level="warning")
                continue
            prev = self.learning_prev.get(sensor)
            self.learning_prev[sensor] = value
            if prev is None:
                continue
            diff = value - prev
            if diff == 0:
                continue
            phase = "start" if diff > 0 else "stop"
            flank = "positive" if diff > 0 else "negative"
            self.learning_events.append(
                {
                    "ts": ts,
                    "appliance": appliance["name"],
                    "phase": phase,
                    "sensor": sensor,
                    "diff": diff,
                    "flank": flank,
                }
            )

    def _learning_hint(self, segment_ts):
        if not self.learning_events:
            return None
        window = max(self.config.get("poll_interval", 5) * 3, 10)
        for event in reversed(self.learning_events):
            if segment_ts - event["ts"] <= window:
                return event
        return None

    def _apply_power_for_segment(self, segment_id, segment, appliance, flank):
        delta = compute_segment_delta(self.store, segment)
        if delta <= 0:
            return
        appliance_row = self.store.get_appliance(appliance)
        if not appliance_row:
            log_event(f"Appliance {appliance} not found when applying power", level="warning")
            return
        current = appliance_row.get("current_power") or 0
        if flank == "negative":
            new_power = max(0.0, current - delta)
        else:
            new_power = current + delta
        log_event(
            f"Applying power delta for {appliance}: flank={flank}, delta={round(delta,3)}, new={round(new_power,3)}",
            level="info",
        )
        push_power_value(appliance, new_power, self.store, self.ha_client, self.mqtt_publisher, self.config)

    def _verify_appliances(self, ts):
        sensors = self.config.get("power_sensors") or []
        if not sensors:
            return
        try:
            total = 0.0
            for sensor in sensors:
                state = self.ha_client.get_state(sensor)
                total += float(state["state"])
        except Exception as exc:
            log_event(f"TTL check failed to read sensors: {exc}", level="warning")
            return
        appliances = self.store.list_appliances()
        sum_published = sum((a.get("current_power") or 0) for a in appliances)
        if sum_published > abs(total) + 1e-6:
            # reset the largest contributor
            largest = max(appliances, key=lambda a: a.get("current_power") or 0)
            log_event(
                f"TTL check: published sum {sum_published} exceeds total {total}. Resetting {largest['name']} to 0.",
                level="warning",
            )
            push_power_value(largest["name"], 0, self.store, self.ha_client, self.mqtt_publisher, self.config)
