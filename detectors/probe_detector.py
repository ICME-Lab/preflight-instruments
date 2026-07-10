"""Activation-probe detector: wraps the existing linear probe.

Reads a frozen LLM's hidden activations and scores injection-compliance along a
single learned direction. This is the detector that already has a working ZK
proof (zkp/). Cheap, linear, fast.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from detectors.base import Detector


class ProbeDetector(Detector):
    name = "probe"
    modality = "activation"

    def __init__(self, probe=None, probe_path=None):
        if probe is None:
            from src.probe import load
            from src.config import PROBE_PATH
            probe = load(probe_path or PROBE_PATH)
        self.probe = probe

    def score_activations(self, X: np.ndarray) -> np.ndarray:
        from src.probe import score
        return score(self.probe, np.asarray(X))

    # Optional convenience: score raw texts by extracting activations first.
    # Requires torch + the model; the harness normally passes X directly.
    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        from src.extract import extract
        # extract() returns (X, y); we only need X. Labels are dummy here.
        X, _ = extract(list(texts), [0] * len(texts))
        return self.score_activations(X)
