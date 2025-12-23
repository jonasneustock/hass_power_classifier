import os

from app import config
from app.logging_utils import log_event, recent_logs
from app.training import TrainingManager


def test_config_loader(monkeypatch):
    monkeypatch.setenv("RELATIVE_CHANGE_THRESHOLD", "")
    monkeypatch.setenv("ABSOLUTE_CHANGE_THRESHOLD", "0.5")
    monkeypatch.setenv("ADAPTIVE_THRESHOLD_ENABLED", "true")
    monkeypatch.setenv("RETRAIN_INTERVAL_SECONDS", "10")
    cfg = config.load_config()
    assert cfg["absolute_change_threshold"] == 0.5
    assert cfg["relative_change_threshold"] is None
    assert cfg["adaptive_threshold_enabled"] is True
    assert cfg["retrain_interval"] == 10


def test_logging_utils_captures():
    size_before = len(recent_logs)
    log_event("hello test")
    assert len(recent_logs) == size_before + 1
    assert recent_logs[0]["message"] == "hello test"


class _DummyStore:
    def get_appliance(self, name):
        return None

    def add_appliance(self, name, a, b, c):
        return None

    def list_appliances(self):
        return []

    def get_label_counts_by_appliance(self):
        return {}

    def get_labeled_segments(self):
        return []

    def update_appliance_power_stats(self, *args, **kwargs):
        return None


class _DummyModel:
    def train(self, *args, **kwargs):
        return None


def test_training_scheduler_disabled(tmp_path):
    cfg = {
        "min_labels": 5,
        "retrain_interval": 0,
        "hyperparam_tuning": False,
    }
    tm = TrainingManager(_DummyStore(), _DummyModel(), _DummyModel(), cfg, tmp_path)
    tm.start_scheduler()
    assert tm._scheduler_thread is None or not tm._scheduler_thread.is_alive()
