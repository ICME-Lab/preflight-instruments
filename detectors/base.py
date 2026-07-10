"""The Detector interface every injection detector implements.

A detector maps an input to a score in [0,1] = P(injection). Detectors differ in
WHAT they read:
  - text detectors read the raw string (e.g. PromptGuard)
  - activation detectors read a model's hidden activations (e.g. probe, SAE)

To keep the harness uniform, every detector exposes:
  .name                       short id
  .modality                   "text" | "activation"
  .score_texts(texts)         -> np.ndarray[float]   (activation detectors may
                              require pre-extracted activations; see below)
  .score_activations(X)       -> np.ndarray[float]   (activation detectors only)

The eval harness already caches activations (data/activations.npz). So for
activation detectors we score from X directly; for text detectors we score from
the original strings. `score()` dispatches on what's available.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass
class DetectorResult:
    """Per-detector scores plus a threshold-applied verdict."""
    name: str
    scores: np.ndarray          # P(injection) per example, shape [n]
    threshold: float = 0.5

    @property
    def verdicts(self) -> np.ndarray:
        return (self.scores >= self.threshold).astype(int)


class Detector:
    """Base class. Subclasses implement at least one of score_texts /
    score_activations, and set `name` and `modality`."""

    name: str = "detector"
    modality: str = "text"      # "text" or "activation"

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} does not score raw text (modality={self.modality})")

    def score_activations(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} does not score activations (modality={self.modality})")

    def score(self, *, texts: Optional[Sequence[str]] = None,
              X: Optional[np.ndarray] = None) -> np.ndarray:
        """Dispatch to the right modality. Prefers the detector's native input."""
        if self.modality == "activation":
            if X is None:
                raise ValueError(f"{self.name} needs activations X")
            return self.score_activations(X)
        else:
            if texts is None:
                raise ValueError(f"{self.name} needs texts")
            return self.score_texts(texts)

    def result(self, *, texts=None, X=None, threshold: float = 0.5) -> DetectorResult:
        return DetectorResult(self.name, self.score(texts=texts, X=X), threshold)
