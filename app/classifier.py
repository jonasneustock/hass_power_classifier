import logging
import threading
from inspect import signature
from pathlib import Path

import joblib
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
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


FEATURE_COLUMNS = [
    "mean",
    "std",
    "max",
    "min",
    "duration",
    "slope",
    "change_score",
]


def _filtered_kwargs(func, **kwargs):
    params = signature(func).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def _feature_values(segment):
    return [segment[name] for name in FEATURE_COLUMNS]


def _feature_input(segment):
    values = [_feature_values(segment)]
    if pd is not None:
        return pd.DataFrame(values, columns=FEATURE_COLUMNS)
    return np.array(values)


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

    def clear(self):
        with self.lock:
            self.model = None
            self.classes = None
            self.last_metrics = None
            if self.model_path.exists():
                try:
                    self.model_path.unlink()
                except Exception:
                    pass

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
            features = _feature_values(segment)
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

        model = None
        best_params = None

        # Try PyCaret "turbo" (fast models only)
        try:
            from pycaret.classification import (
                setup as py_setup,
                compare_models,
                finalize_model,
            )
            if pd is None:
                raise ImportError("pandas not installed")

            df = pd.DataFrame(
                X,
                columns=FEATURE_COLUMNS,
            )
            df["label"] = y
            setup_kwargs = _filtered_kwargs(
                py_setup,
                data=df,
                target="label",
                session_id=42,
                silent=True,
                verbose=False,
                fold=3,
                use_gpu=False,
                log_experiment=False,
                html=False,
            )
            py_setup(**setup_kwargs)
            best = compare_models(turbo=True)
            model_candidate = finalize_model(best)
            # evaluate on full data quickly
            preds = model_candidate.predict(df.drop(columns=["label"]))
            metrics_local = metrics.copy()
            metrics_local = {k: v for k, v in metrics_local.items()}
            metrics_local["accuracy"] = float(accuracy_score(y, preds))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, preds, average="macro", zero_division=0
            )
            metrics_local["precision"] = float(precision)
            metrics_local["recall"] = float(recall)
            metrics_local["f1"] = float(f1)
            model = model_candidate
            metrics = metrics_local
            best_params = {"pycaret_model": str(type(best).__name__)}
            log_event("Classifier trained with PyCaret turbo models", level="info")
        except Exception as exc:
            log_event(f"PyCaret classification failed, falling back: {exc}", level="warning")
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
            self.classes = list(model.classes_) if hasattr(model, "classes_") else list(sorted(set(y)))
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
            features = _feature_input(segment)
            prediction = self.model.predict(features)[0]
            log_event(
                f"Classifier prediction: {prediction} for segment features mean={segment['mean']}, change={segment['change_score']}",
                level="info",
            )
        return prediction

    def top_predictions(self, segment, top_n=3):
        with self.lock:
            if self.model is None or not hasattr(self.model, "predict_proba"):
                return []
            features = _feature_input(segment)
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
            df = None
            if pd is not None:
                df = pd.DataFrame(X, columns=["t"])
                df["y"] = y
            model_for_appliance = None
            mse_local = None
            try:
                from pycaret.regression import (
                    setup as py_setup,
                    compare_models,
                    finalize_model,
                )
                if pd is None or df is None:
                    raise ImportError("pandas not installed")

                setup_kwargs = _filtered_kwargs(
                    py_setup,
                    data=df,
                    target="y",
                    session_id=42,
                    silent=True,
                    verbose=False,
                    fold=3,
                    use_gpu=False,
                    log_experiment=False,
                    html=False,
                )
                py_setup(**setup_kwargs)
                best = compare_models(turbo=True)
                model_candidate = finalize_model(best)
                preds = model_candidate.predict(df[["t"]])
                mse_local = mean_squared_error(df["y"], preds)
                model_for_appliance = model_candidate
                log_event(
                    f"Regression (PyCaret turbo) trained for {appliance}: samples={len(df)}, mse={mse_local}",
                    level="info",
                )
            except Exception as exc:
                log_event(f"PyCaret regression failed for {appliance}, fallback: {exc}", level="warning")
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
                        mse_candidate = mean_squared_error(y_test, preds)
                    else:
                        mse_candidate = 0
                    if mse_candidate <= best_mse:
                        best_mse = mse_candidate
                        best_model = model_local
                if len(y_test) > 0 and best_model is not None:
                    preds = best_model.predict(X_test)
                    all_y_true.extend(list(y_test))
                    all_y_pred.extend(list(preds))
                model_for_appliance = best_model
                mse_local = best_mse
                if best_model is not None:
                    log_event(
                        f"Regression (fallback) trained for {appliance}: samples={len(X_arr)}, mse_candidate={best_mse}",
                        level="info",
                    )

            if model_for_appliance is not None:
                models[appliance] = model_for_appliance
                if mse_local is not None:
                    if pd is not None:
                        all_y_true.extend(list(df["y"]))
                        all_y_pred.extend(list(model_for_appliance.predict(df[["t"]])))
                    else:
                        all_y_true.extend(y)
                        preds_all = model_for_appliance.predict(np.array(X))
                        all_y_pred.extend(list(preds_all))

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
        if pd is not None:
            features = pd.DataFrame({"t": [seconds_since_start]})
        else:
            features = np.array([[seconds_since_start]])
        pred = model.predict(features)[0]
        log_event(f"Regression prediction for {appliance} at t={seconds_since_start}s -> {pred}", level="info")
        return max(0.0, float(pred))

    def clear(self):
        with self.lock:
            self.models = {}
            self.last_metrics = None
