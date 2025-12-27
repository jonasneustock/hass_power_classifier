import logging
import threading
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from app.utils import samples_to_diffs
from app.logging_utils import log_event


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

    def train(self, segments, eligible_appliances=None, tune=False):
        if not segments:
            return None
        X = []
        y = []
        for segment in segments:
            appliance = segment["label_appliance"]
            if eligible_appliances and appliance not in eligible_appliances:
                continue
            label = appliance
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

        metrics = {
            "samples": len(y),
            "classes": len(set(y)),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

        if len(y) >= 5 and len(set(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        def train_eval(params):
            model_local = RandomForestClassifier(random_state=42, **params)
            model_local.fit(X_train, y_train)
            local_metrics = metrics.copy()
            local_metrics = {k: v for k, v in local_metrics.items()}
            if len(y_test) > 0:
                y_pred = model_local.predict(X_test)
                local_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="macro", zero_division=0
                )
                local_metrics["precision"] = float(precision)
                local_metrics["recall"] = float(recall)
                local_metrics["f1"] = float(f1)
            return model_local, local_metrics

        candidate_params = [
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
            {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
            {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 2},
        ]
        if not tune:
            candidate_params = [candidate_params[0]]

        best_model = None
        best_metrics = None
        best_acc = -1
        best_params = None
        for params in candidate_params:
            model_candidate, met = train_eval(params)
            acc = met.get("accuracy") if met else None
            if acc is None:
                acc = 0.0
            if acc >= best_acc:
                best_acc = acc
                best_model = model_candidate
                best_metrics = met
                best_params = params

        model = best_model
        metrics = best_metrics or metrics

        with self.lock:
            self.model = model
            self.classes = list(model.classes_)
            self.last_metrics = metrics
            self._save()
        log_event(
            f"Classifier trained: samples={metrics['samples']} classes={metrics['classes']} accuracy={metrics.get('accuracy')} params={best_params}",
            level="info",
        )
        return metrics

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
        return prediction

    def top_predictions(self, segment, top_n=3):
        with self.lock:
            if self.model is None or not hasattr(self.model, "predict_proba"):
                return []
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
            probs = self.model.predict_proba(features)[0]
            classes = self.model.classes_
        scored = []
        for cls, prob in zip(classes, probs):
            appliance = cls
            scored.append({"appliance": appliance, "prob": float(prob)})
        scored.sort(key=lambda x: x["prob"], reverse=True)
        return scored[:top_n]


class RegressionService:
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()
        self.last_metrics = None

    def train(self, labeled_segments, store, tune=False, sensors=None):
        models = {}
        all_y_true = []
        all_y_pred = []
        by_appliance = {}
        for seg in labeled_segments:
            if seg["label_phase"] == "base":
                continue
            by_appliance.setdefault(seg["label_appliance"], []).append(seg)

        for appliance, segments in by_appliance.items():
            X = []
            y = []
            for seg in segments:
                flank = seg.get("flank")
                if flank not in (None, "positive"):
                    continue
                if sensors:
                    for sensor in sensors:
                        samples = store.get_sensor_samples_between(
                            seg["start_ts"], seg["end_ts"], sensor=sensor
                        )
                        diffs = samples_to_diffs(samples)
                        for sample in diffs:
                            t = sample["ts"] - seg["start_ts"]
                            X.append([t])
                            y.append(sample["value"])
                else:
                    samples = store.get_samples_between(seg["start_ts"], seg["end_ts"])
                    diffs = samples_to_diffs(samples)
                    for sample in diffs:
                        t = sample["ts"] - seg["start_ts"]
                        X.append([t])
                        y.append(sample["value"])
            if len(X) < 5:
                log_event(f"Regression skipped for {appliance}: not enough samples ({len(X)})", level="warning")
                continue
            X_arr = np.array(X)
            y_arr = np.array(y)
            if len(y_arr) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_arr, y_arr, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X_arr, X_arr, y_arr, y_arr

            candidates = [
                {"max_depth": None, "min_samples_leaf": 5},
                {"max_depth": 8, "min_samples_leaf": 3},
                {"max_depth": 12, "min_samples_leaf": 5},
            ]
            if not tune:
                candidates = [candidates[0]]

            best_model = None
            best_mse = float("inf")
            for params in candidates:
                model_local = DecisionTreeRegressor(random_state=42, **params)
                model_local.fit(X_train, y_train)
                if len(y_test) > 0:
                    preds = model_local.predict(X_test)
                    mse_local = mean_squared_error(y_test, preds)
                else:
                    mse_local = 0
                if mse_local <= best_mse:
                    best_mse = mse_local
                    best_model = model_local
            if len(y_test) > 0 and best_model is not None:
                preds = best_model.predict(X_test)
                all_y_true.extend(list(y_test))
                all_y_pred.extend(list(preds))
            if best_model is not None:
                models[appliance] = best_model
                log_event(
                    f"Regression trained for {appliance}: samples={len(X_arr)}, mse_candidate={best_mse}",
                    level="info",
                )

        with self.lock:
            self.models = models
            if all_y_true and all_y_pred:
                mse = mean_squared_error(all_y_true, all_y_pred)
                mape = mean_absolute_percentage_error(all_y_true, all_y_pred)
                self.last_metrics = {"mse": float(mse), "mape": float(mape)}
            else:
                self.last_metrics = None
        log_event(
            f"Regression trained for appliances={len(models)} mse={self.last_metrics.get('mse') if self.last_metrics else 'n/a'}",
            level="info",
        )

    def predict(self, appliance, seconds_since_start):
        with self.lock:
            model = self.models.get(appliance)
        if not model:
            return None
        pred = model.predict(np.array([[seconds_since_start]]))[0]
        return max(0.0, float(pred))
