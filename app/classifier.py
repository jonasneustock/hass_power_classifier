import threading
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


class ClassifierService:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.lock = threading.Lock()
        self.model = None
        self.classes = None
        self.last_metrics = None
        self._load()

    def _load(self):
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data.get("model")
            self.classes = data.get("classes")
            self.last_metrics = data.get("metrics")

    def _save(self):
        data = {
            "model": self.model,
            "classes": self.classes,
            "metrics": self.last_metrics,
        }
        joblib.dump(data, self.model_path)

    def train(self, segments, eligible_appliances=None):
        if not segments:
            return None
        X = []
        y = []
        for segment in segments:
            appliance = segment["label_appliance"]
            if eligible_appliances and appliance not in eligible_appliances:
                continue
            label = f"{appliance}|{segment['label_phase']}"
            features = [
                segment["mean"],
                segment["std"],
                segment["max"],
                segment["min"],
                segment["duration"],
                segment["slope"],
                segment["change_score"],
            ]
            X.append(features)
            y.append(label)
        if not X:
            return None
        X = np.array(X)
        y = np.array(y)

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )
        model.fit(X, y)
        with self.lock:
            self.model = model
            self.classes = list(model.classes_)
            self.last_metrics = {
                "samples": len(y),
                "classes": len(model.classes_),
            }
            self._save()
        return self.last_metrics

    def predict(self, segment):
        with self.lock:
            if self.model is None:
                return None
            features = np.array(
                [
                    [
                        segment["mean"],
                        segment["std"],
                        segment["max"],
                        segment["min"],
                        segment["duration"],
                        segment["slope"],
                        segment["change_score"],
                    ]
                ]
            )
            prediction = self.model.predict(features)[0]
        if "|" in prediction:
            appliance, phase = prediction.split("|", 1)
        else:
            appliance, phase = prediction, "unknown"
        return appliance, phase


class RegressionService:
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def train(self, labeled_segments, store):
        models = {}
        by_appliance = {}
        for seg in labeled_segments:
            if seg["label_phase"] == "base":
                continue
            by_appliance.setdefault(seg["label_appliance"], []).append(seg)

        for appliance, segments in by_appliance.items():
            X = []
            y = []
            for seg in segments:
                samples = store.get_samples_between(seg["start_ts"], seg["end_ts"])
                for sample in samples:
                    t = sample["ts"] - seg["start_ts"]
                    X.append([t])
                    y.append(sample["value"])
            if len(X) < 5:
                continue
            model = LinearRegression()
            model.fit(np.array(X), np.array(y))
            models[appliance] = model

        with self.lock:
            self.models = models

    def predict(self, appliance, seconds_since_start):
        with self.lock:
            model = self.models.get(appliance)
        if not model:
            return None
        pred = model.predict(np.array([[seconds_since_start]]))[0]
        return max(0.0, float(pred))
