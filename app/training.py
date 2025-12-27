import json
import threading
import time
from pathlib import Path

import numpy as np

from app.logging_utils import log_event
from app.utils import samples_to_diffs


class TrainingManager:
    def __init__(self, store, classifier, regression_service, config, data_dir: Path):
        self.store = store
        self.classifier = classifier
        self.regression_service = regression_service
        self.config = config
        self.metrics_file = Path(data_dir) / "model_metrics.json"
        self.training_state = {
            "running": False,
            "error": None,
            "last_started": None,
            "last_finished": None,
        }
        self.training_lock = threading.Lock()
        self.metrics_history = self._load_metrics_history()
        self._scheduler_thread = None
        self._stop_scheduler = threading.Event()

    def ensure_base_appliance(self):
        base = self.store.get_appliance("base")
        if not base:
            self.store.add_appliance("base", "", "", "")
            log_event("Base appliance created for baseline labeling")

    def trigger_training(self):
        log_event("Training trigger requested")
        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

    def _load_metrics_history(self):
        if self.metrics_file.exists():
            try:
                return json.loads(self.metrics_file.read_text())
            except Exception:
                return []
        return []

    def _save_metrics_entry(self, entry):
        history = self._load_metrics_history()
        history.append(entry)
        history = history[-100:]
        self.metrics_file.write_text(json.dumps(history, indent=2))
        self.metrics_history = history
        return history

    def start_scheduler(self):
        interval = self.config.get("retrain_interval", 0)
        if not interval or interval <= 0:
            return
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return

        def _loop():
            while not self._stop_scheduler.is_set():
                time.sleep(interval)
                if self._stop_scheduler.is_set():
                    break
                self.trigger_training()

        self._scheduler_thread = threading.Thread(target=_loop, daemon=True)
        self._scheduler_thread.start()

    def stop_scheduler(self):
        if self._scheduler_thread:
            self._stop_scheduler.set()

    def _compute_power_stats_by_appliance(self):
        appliances = self.store.list_appliances()
        labeled_segments = self.store.get_labeled_segments()
        stats = {}
        for appliance in appliances:
            name = appliance["name"]
            labeled = [
                seg
                for seg in labeled_segments
                if seg["label_appliance"] == name and seg["label_phase"] != "base"
            ]
            values = []
            for seg in labeled:
                samples = self.store.get_samples_between(seg["start_ts"], seg["end_ts"])
                diffs = samples_to_diffs(samples)
                if not diffs:
                    continue
                flank = seg.get("flank")
                if flank == "positive":
                    val = diffs[-1]["value"]
                elif flank == "negative":
                    val = diffs[0]["value"]
                else:
                    continue
                values.append(max(0.0, float(val)))
            if values:
                stats[name] = {
                    "min_power": float(np.min(values)),
                    "mean_power": float(np.mean(values)),
                    "max_power": float(np.max(values)),
                }
        return stats

    def _run_training(self):
        with self.training_lock:
            if self.training_state["running"]:
                log_event("Training already running, skipping", level="info")
                return
            self.training_state["running"] = True
        self.training_state["error"] = None
        self.training_state["last_started"] = int(time.time())
        log_event("Training started")
        try:
            appliances = self.store.list_appliances()
            log_event(f"Training: {len(appliances)} appliances loaded")
            if not appliances:
                log_event("Training skipped: no appliances", level="warning")
                self.training_state["last_finished"] = int(time.time())
                return
            counts = self.store.get_label_counts_by_appliance()
            min_required = max(self.config["min_labels"], 5)
            eligible = {
                appliance["name"]
                for appliance in appliances
                if counts.get(appliance["name"], 0) >= min_required
            }
            log_event(f"Training eligibility: {len(eligible)} / {len(appliances)} meet min_labels={min_required}")
            if not eligible:
                log_event(
                    "Training skipped: no appliance meets label threshold",
                    level="warning",
                )
                self.training_state["last_finished"] = int(time.time())
                return
            labeled_segments = self.store.get_labeled_segments()
            labeled_segments = [
                seg for seg in labeled_segments if seg["label_appliance"] in eligible
            ]
            log_event(f"Training: {len(labeled_segments)} labeled segments considered")
            if not labeled_segments:
                log_event("Training skipped: no labeled segments", level="warning")
                self.training_state["last_finished"] = int(time.time())
                return
            clf_metrics = self.classifier.train(
                labeled_segments,
                eligible_appliances=eligible,
                tune=self.config.get("hyperparam_tuning", False),
            )
            self.regression_service.train(
                labeled_segments,
                self.store,
                tune=self.config.get("hyperparam_tuning", False),
                sensors=self.config.get("power_sensors", []),
            )
            reg_metrics = self.regression_service.last_metrics
            power_stats = self._compute_power_stats_by_appliance()
            for name, stats in power_stats.items():
                self.store.update_appliance_power_stats(
                    name, stats["min_power"], stats["mean_power"], stats["max_power"]
                )
            log_event(
                f"Training finished: clf_metrics={clf_metrics}, reg_metrics={reg_metrics}, power_stats={list(power_stats.keys())}"
            )
            self.training_state["last_finished"] = int(time.time())
            metrics_entry = {
                "ts": self.training_state["last_finished"],
                "classifier": clf_metrics,
                "regression": reg_metrics,
            }
            self.metrics_history = self._save_metrics_entry(metrics_entry)
            log_event("Training finished")
        except Exception as exc:
            self.training_state["error"] = str(exc)
            self.training_state["last_finished"] = int(time.time())
            log_event(f"Training failed: {exc}", level="error")
        finally:
            self.training_state["running"] = False
