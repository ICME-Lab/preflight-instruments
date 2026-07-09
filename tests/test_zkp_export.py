"""Test the ONNX export produces a graph that matches the sklearn probe.

Trains a tiny probe on synthetic data, exports it, and checks parity — no Qwen
or GPU needed. Skips if skl2onnx/onnxruntime aren't installed.
"""
from __future__ import annotations
import numpy as np
import pytest

skl2onnx = pytest.importorskip("skl2onnx")
ort = pytest.importorskip("onnxruntime")

from src.probe import build_probe
from zkp import export_onnx_coreops as export_mod


def _train_tiny(dim=1536, n=200, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(dim); d /= np.linalg.norm(d)
    y = (rng.random(n) > 0.5).astype(int)
    X = rng.standard_normal((n, dim)) + np.where(y[:, None] == 1, 1.4 * d, -1.4 * d)
    p = build_probe(); p.fit(X, y)
    return p, dim


def test_export_and_parity(tmp_path, monkeypatch):
    probe, dim = _train_tiny()
    # Redirect export artifacts to tmp.
    monkeypatch.setattr(export_mod, "ONNX_PATH", tmp_path / "probe.onnx")
    monkeypatch.setattr(export_mod, "SAMPLE_PATH", tmp_path / "sample.npy")

    export_mod.export(probe)  # core-op exporter infers dim from the probe
    assert (tmp_path / "probe.onnx").exists()

    # Parity: ONNX vs sklearn on fresh inputs.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((32, dim)).astype(np.float32)
    sk = probe.predict_proba(X)[:, 1]

    sess = ort.InferenceSession(str(tmp_path / "probe.onnx"),
                                providers=["CPUExecutionProvider"])
    onnx_scores = np.array([
        float(np.ravel(sess.run(["logits"], {"input": row[None, :].astype(np.float32)})[0])[0])
        for row in X  # input [1,dim]; read logits[0]
    ])
    assert float(np.max(np.abs(sk - onnx_scores))) < 1e-4
