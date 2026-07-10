"""SAE detector: a linear probe on sparse-autoencoder FEATURES of activations.

Pipeline: activation X -> SAE.encode(X) = sparse features H -> logistic
regression on H -> P(injection).

Rationale and honest caveat (repeated from sae.py): this frequently does NOT
beat the raw-activation probe on detection, because SAE reconstruction is lossy.
Its edge is interpretability (which features fire) and, later, intervention. The
harness reports whether it actually helps coverage.
"""
from __future__ import annotations
import numpy as np

from detectors.base import Detector
from detectors.sae import SAE


class SAEDetector(Detector):
    name = "sae"
    modality = "activation"

    def __init__(self, sae: SAE, clf=None):
        self.sae = sae
        self.clf = clf                # sklearn pipeline fit on SAE features

    @staticmethod
    def build_clf():
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        H = self.sae.features(X)
        self.clf = self.build_clf()
        self.clf.fit(H, np.asarray(y))
        return self

    def score_activations(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("SAEDetector not fitted; call .fit(X, y) first")
        H = self.sae.features(np.asarray(X))
        return self.clf.predict_proba(H)[:, 1]
