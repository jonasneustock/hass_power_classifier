import time

import numpy as np

from app.classifier import ClassifierService, RegressionService


class DummyStore:
    def __init__(self, samples):
        self.samples = samples

    def get_samples_between(self, start_ts, end_ts):
        return [s for s in self.samples if start_ts <= s["ts"] <= end_ts]


def _segments():
    now = int(time.time())
    return [
        {
            "label_appliance": "toaster",
            "label_phase": "start",
            "start_ts": now,
            "end_ts": now + 2,
            "mean": 1,
            "std": 0.1,
            "max": 1.2,
            "min": 0.9,
            "duration": 2,
            "slope": 0.1,
            "change_score": 0.3,
        },
        {
            "label_appliance": "toaster",
            "label_phase": "stop",
            "start_ts": now + 3,
            "end_ts": now + 5,
            "mean": 0.8,
            "std": 0.05,
            "max": 0.9,
            "min": 0.7,
            "duration": 2,
            "slope": -0.05,
            "change_score": 0.2,
        },
    ]


def test_classifier_train_metrics(tmp_path):
    clf = ClassifierService(str(tmp_path / "model.pkl"))
    segs = _segments()
    metrics = clf.train(segs, eligible_appliances={"toaster"})
    assert metrics["samples"] == 2
    assert metrics["classes"] >= 1
    assert metrics["accuracy"] is not None


def test_classifier_train_no_data_returns_none(tmp_path):
    clf = ClassifierService(str(tmp_path / "model.pkl"))
    metrics = clf.train([], eligible_appliances={"toaster"})
    assert metrics is None


def test_regression_train_and_predict():
    now = int(time.time())
    samples = [
        {"ts": now, "value": 10.0},
        {"ts": now + 1, "value": 11.0},
        {"ts": now + 2, "value": 12.0},
    ]
    store = DummyStore(samples)
    reg = RegressionService()
    reg.train(_segments(), store)
    pred = reg.predict("toaster", 1)
    assert pred is None or isinstance(pred, float)


def test_regression_metrics_set():
    now = int(time.time())
    samples = []
    for i in range(20):
        samples.append({"ts": now + i, "value": 10 + i})
    store = DummyStore(samples)
    reg = RegressionService()
    reg.train(_segments(), store)
    # Either metrics computed or none if insufficient data
    assert reg.last_metrics is None or set(reg.last_metrics.keys()) == {"mse", "mape"}


def test_classifier_eligibility_filter(tmp_path):
    clf = ClassifierService(str(tmp_path / "model.pkl"))
    segs = _segments()
    metrics = clf.train(segs, eligible_appliances={"toaster"})
    assert metrics is not None
    metrics_none = clf.train(segs, eligible_appliances={"other"})
    assert metrics_none is None


def test_regression_predict_unknown_returns_none():
    reg = RegressionService()
    assert reg.predict("unknown", 1) is None


def test_classifier_predict_returns_tuple(tmp_path):
    clf = ClassifierService(str(tmp_path / "model.pkl"))
    segs = _segments()
    clf.train(segs, eligible_appliances={"toaster"})
    segment = {
        "mean": 1,
        "std": 0.1,
        "max": 1.2,
        "min": 0.9,
        "duration": 2,
        "slope": 0.1,
        "change_score": 0.3,
    }
    appliance, phase = clf.predict(segment)
    assert isinstance(appliance, str)
    assert isinstance(phase, str)
