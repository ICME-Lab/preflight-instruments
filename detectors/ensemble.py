"""Ensemble of detectors.

Combines per-detector injection scores into one. Two default strategies:
  - "max"  : P(injection) = max over detectors. Catches anything ANY detector
             flags (high recall, higher false positives). Good default for a
             security tripwire where a miss is worse than a false alarm.
  - "mean" : average of detector scores (smoother, fewer false positives).

The ensemble is the "cover the most" story: each detector catches a different
failure mode, so their union catches strictly more than any single one.

Detectors may need different inputs (text vs activations); the ensemble takes
both and lets each detector pick what it needs.
"""
from __future__ import annotations
from typing import Sequence, Optional
import numpy as np

from detectors.base import Detector, DetectorResult


class Ensemble:
    def __init__(self, detectors: Sequence[Detector], strategy: str = "max",
                 weights: Optional[Sequence[float]] = None):
        assert strategy in ("max", "mean", "weighted")
        self.detectors = list(detectors)
        self.strategy = strategy
        self.weights = np.array(weights, dtype=float) if weights is not None else None

    def score_matrix(self, *, texts=None, X=None) -> np.ndarray:
        """Return [n_examples, n_detectors] of per-detector scores."""
        cols = []
        for d in self.detectors:
            cols.append(d.score(texts=texts, X=X))
        return np.column_stack(cols)

    def score(self, *, texts=None, X=None) -> np.ndarray:
        M = self.score_matrix(texts=texts, X=X)
        if self.strategy == "max":
            return M.max(axis=1)
        if self.strategy == "mean":
            return M.mean(axis=1)
        # weighted
        w = self.weights if self.weights is not None else np.ones(M.shape[1])
        w = w / w.sum()
        return M @ w

    def result(self, *, texts=None, X=None, threshold: float = 0.5) -> DetectorResult:
        return DetectorResult(f"ensemble[{self.strategy}]",
                              self.score(texts=texts, X=X), threshold)
