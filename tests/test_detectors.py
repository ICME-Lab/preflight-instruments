"""Model-free tests for the detector suite (SAE, detectors, ensemble).

All run on synthetic separable activations — no Qwen, no PromptGuard download,
no GPU. Proves the plumbing: SAE trains and reconstructs, detectors score in
[0,1], ensemble combines correctly.
"""
from __future__ import annotations
import numpy as np
import pytest

from detectors.base import Detector, DetectorResult
from detectors.sae import SAE, SAEConfig, train_sae
from detectors.sae_detector import SAEDetector
from detectors.probe_detector import ProbeDetector
from detectors.ensemble import Ensemble
from src.probe import build_probe

DIM = 64
N = 300
SEED = 0


def _synth(seed=SEED):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(DIM); d /= np.linalg.norm(d)
    y = (rng.random(N) > 0.5).astype(int)
    X = (rng.standard_normal((N, DIM)) + np.where(y[:, None] == 1, 1.5 * d, -1.5 * d)).astype(np.float32)
    return X, y


# ---- SAE ----

def test_sae_trains_and_reconstructs():
    X, _ = _synth()
    cfg = SAEConfig(d_in=DIM, d_hidden=128, l1=1e-3, epochs=20, batch_size=128)
    sae, hist = train_sae(X, cfg)
    # reconstruction loss should decrease from first to last epoch
    assert hist["recon"][-1] < hist["recon"][0]
    # features are non-negative (ReLU) and correct shape
    H = sae.features(X[:10])
    assert H.shape == (10, 128)
    assert (H >= 0).all()


def test_sae_save_load_roundtrip(tmp_path):
    X, _ = _synth()
    cfg = SAEConfig(d_in=DIM, d_hidden=96, epochs=10)
    sae, _ = train_sae(X, cfg)
    p = tmp_path / "sae.npz"
    sae.save(p)
    sae2 = SAE.load(p)
    assert np.allclose(sae.features(X[:5]), sae2.features(X[:5]))


# ---- detectors ----

def test_probe_detector_scores_in_range():
    X, y = _synth()
    probe = build_probe(); probe.fit(X, y)
    det = ProbeDetector(probe=probe)
    s = det.score_activations(X)
    assert s.shape == (N,)
    assert ((s >= 0) & (s <= 1)).all()
    assert det.modality == "activation"


def test_sae_detector_fits_and_scores():
    X, y = _synth()
    cfg = SAEConfig(d_in=DIM, d_hidden=128, epochs=20, batch_size=128)
    sae, _ = train_sae(X, cfg)
    det = SAEDetector(sae).fit(X, y)
    s = det.score_activations(X)
    assert ((s >= 0) & (s <= 1)).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, s) > 0.75   # separable data => useful


def test_sae_detector_requires_fit():
    X, y = _synth()
    cfg = SAEConfig(d_in=DIM, d_hidden=64, epochs=5)
    sae, _ = train_sae(X, cfg)
    det = SAEDetector(sae)
    with pytest.raises(RuntimeError):
        det.score_activations(X)


# ---- ensemble ----

def _two_detectors():
    X, y = _synth()
    probe = build_probe(); probe.fit(X, y)
    pd = ProbeDetector(probe=probe)
    cfg = SAEConfig(d_in=DIM, d_hidden=128, epochs=20, batch_size=128)
    sae, _ = train_sae(X, cfg)
    sd = SAEDetector(sae).fit(X, y)
    return X, y, pd, sd


def test_ensemble_max_is_elementwise_max():
    X, y, pd, sd = _two_detectors()
    ens = Ensemble([pd, sd], strategy="max")
    M = ens.score_matrix(X=X)
    s = ens.score(X=X)
    assert s.shape == (N,)
    assert np.allclose(s, M.max(axis=1))


def test_ensemble_mean():
    X, y, pd, sd = _two_detectors()
    ens = Ensemble([pd, sd], strategy="mean")
    s = ens.score(X=X)
    M = ens.score_matrix(X=X)
    assert np.allclose(s, M.mean(axis=1))


def test_ensemble_result_type():
    X, y, pd, sd = _two_detectors()
    ens = Ensemble([pd, sd], strategy="max")
    r = ens.result(X=X, threshold=0.5)
    assert isinstance(r, DetectorResult)
    assert r.verdicts.shape == (N,)
    assert set(np.unique(r.verdicts)).issubset({0, 1})


def test_detector_base_rejects_wrong_modality():
    """An activation detector should refuse text and vice versa."""
    X, y = _synth()
    probe = build_probe(); probe.fit(X, y)
    det = ProbeDetector(probe=probe)
    with pytest.raises(ValueError):
        det.score(texts=["hello"])   # activation detector needs X
