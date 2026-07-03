"""Calibration helpers for Phase 2C frozen-score adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

try:
    from .phase2c_metrics import EPS, clip_probabilities, logit, select_threshold, threshold_metrics
except ImportError:  # pragma: no cover - allows direct script execution
    from phase2c_metrics import EPS, clip_probabilities, logit, select_threshold, threshold_metrics


CALIBRATION_SIMPLICITY = {"intercept_only": 0, "platt": 1, "isotonic": 2}


class Calibrator(Protocol):
    method: str

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        ...


@dataclass
class InterceptOnlyCalibrator:
    intercept: float
    method: str = "intercept_only"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        z = logit(probabilities) + float(self.intercept)
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class PlattCalibrator:
    model: object
    method: str = "platt"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        x = logit(probabilities).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)


@dataclass
class IsotonicCalibrator:
    model: object
    method: str = "isotonic"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(clip_probabilities(probabilities)), dtype=float)


def available_calibration_methods(y_true: np.ndarray, probabilities: np.ndarray) -> list[str]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        return []
    methods = ["intercept_only"]
    if len(np.unique(p)) >= 2:
        methods.append("platt")
    class_counts = np.bincount(y, minlength=2)
    if len(y) >= 30 and int(np.min(class_counts)) >= 10 and len(np.unique(p)) >= 10:
        methods.append("isotonic")
    return methods


def fit_intercept_only(y_true: np.ndarray, probabilities: np.ndarray) -> InterceptOnlyCalibrator:
    y = np.asarray(y_true, dtype=float)
    base_logit = logit(probabilities)

    def loss(intercept: float) -> float:
        z = base_logit + intercept
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = clip_probabilities(pred)
        return float(-np.mean(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred)))

    left, right = -12.0, 12.0
    for _ in range(90):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        if loss(m1) < loss(m2):
            right = m2
        else:
            left = m1
    return InterceptOnlyCalibrator(intercept=(left + right) / 2.0)


def fit_calibrator(method: str, y_true: np.ndarray, probabilities: np.ndarray) -> Calibrator:
    methods = available_calibration_methods(y_true, probabilities)
    if method not in methods:
        raise ValueError(f"Calibration method {method} is not stable for this training split.")
    y = np.asarray(y_true, dtype=int)
    p = clip_probabilities(probabilities)
    if method == "intercept_only":
        return fit_intercept_only(y, p)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
        model.fit(logit(p).reshape(-1, 1), y)
        return PlattCalibrator(model=model)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return IsotonicCalibrator(model=model)
    raise ValueError(f"Unknown calibration method: {method}")


def select_calibration_method(
    folds: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    p = clip_probabilities(probabilities)
    fold_values = sorted(np.unique(folds))
    candidates = available_calibration_methods(y, p)
    if not candidates:
        raise ValueError("No stable calibration candidates are available.")
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    for method in candidates:
        briers: list[float] = []
        macro_f1s: list[float] = []
        valid_inner = 0
        for validation_fold in fold_values:
            train_mask = folds != validation_fold
            validation_mask = folds == validation_fold
            if method not in available_calibration_methods(y[train_mask], p[train_mask]):
                continue
            calibrator = fit_calibrator(method, y[train_mask], p[train_mask])
            pred = clip_probabilities(calibrator.predict(p[validation_mask]))
            brier = float(np.mean((pred - y[validation_mask]) ** 2))
            macro_f1 = float(threshold_metrics(y[validation_mask], pred, 0.5)["macro_f1"])
            briers.append(brier)
            macro_f1s.append(macro_f1)
            valid_inner += 1
        if not briers:
            continue
        row: dict[str, object] = {
            "method": method,
            "mean_brier": float(np.mean(briers)),
            "std_brier": float(np.std(briers, ddof=0)),
            "mean_macro_f1_at_0_5": float(np.mean(macro_f1s)),
            "valid_inner_folds": valid_inner,
        }
        rows.append(row)
        key = (
            -float(row["mean_brier"]),
            -float(row["std_brier"]),
            float(row["mean_macro_f1_at_0_5"]),
            -CALIBRATION_SIMPLICITY[method],
        )
        if best_row is None:
            best_row = row
        else:
            best_key = (
                -float(best_row["mean_brier"]),
                -float(best_row["std_brier"]),
                float(best_row["mean_macro_f1_at_0_5"]),
                -CALIBRATION_SIMPLICITY[str(best_row["method"])],
            )
            if key > best_key:
                best_row = row
    if best_row is None:
        raise ValueError("No calibration candidate completed inner validation.")
    calibrator = fit_calibrator(str(best_row["method"]), y, p)
    return {"selected": best_row, "rows": rows, "calibrator": calibrator}


def select_threshold_for_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return select_threshold(y_true, probabilities).threshold
