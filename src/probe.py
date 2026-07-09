"""Fit / evaluate / persist / apply a linear activation probe.

The functions here operate on plain numpy arrays, so they are fully testable
without a model (see tests/test_probe_synthetic.py). Training data quality is the
only thing that separates the synthetic test from the real run.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


@dataclass
class ProbeMetrics:
    accuracy: float
    auroc: float
    precision: float
    recall: float
    n_train: int
    n_test: int

    def pretty(self) -> str:
        return (f"acc={self.accuracy:.3f}  auroc={self.auroc:.3f}  "
                f"precision={self.precision:.3f}  recall={self.recall:.3f}  "
                f"(train={self.n_train}, test={self.n_test})")


def build_probe() -> "Pipeline":
    """A standard-scaler + L2 logistic regression. Cheap and linear by design."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )


def fit_and_eval(X: np.ndarray, y: np.ndarray, *, test_size: float = 0.3,
                 seed: int = 0):
    """Split, fit on train, evaluate on held-out test.

    Returns (fitted_probe, ProbeMetrics). The probe returned is refit on ALL
    data after evaluation, so the deployed probe uses every example while the
    reported metrics come from held-out data.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    probe = build_probe()
    probe.fit(X_tr, y_tr)

    proba = probe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = ProbeMetrics(
        accuracy=float(accuracy_score(y_te, pred)),
        auroc=float(roc_auc_score(y_te, proba)),
        precision=float(precision_score(y_te, pred, zero_division=0)),
        recall=float(recall_score(y_te, pred, zero_division=0)),
        n_train=int(len(y_tr)),
        n_test=int(len(y_te)),
    )

    # Refit on all data for the deployed artifact.
    deployed = build_probe()
    deployed.fit(X, y)
    return deployed, metrics


def score(probe, X: np.ndarray) -> np.ndarray:
    """Return P(injection) for each row of X."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    return probe.predict_proba(X)[:, 1]


def save(probe, metrics: ProbeMetrics, probe_path, metrics_path):
    import joblib
    joblib.dump(probe, probe_path)
    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)


def load(probe_path):
    import joblib
    return joblib.load(probe_path)


def main():
    from src.config import ACTIVATIONS_PATH, PROBE_PATH, METRICS_PATH
    if not ACTIVATIONS_PATH.exists():
        raise SystemExit(
            f"No activations at {ACTIVATIONS_PATH}. Run `python -m src.extract` first."
        )
    data = np.load(ACTIVATIONS_PATH)
    X, y = data["X"], data["y"]
    finite = np.isfinite(X).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        print(f"Dropping {n_bad} example(s) with non-finite activations")
        X, y = X[finite], y[finite]
    probe, metrics = fit_and_eval(X, y)
    save(probe, metrics, PROBE_PATH, METRICS_PATH)
    print("Probe trained.")
    print(metrics.pretty())
    print(f"Saved -> {PROBE_PATH}")


if __name__ == "__main__":
    main()
