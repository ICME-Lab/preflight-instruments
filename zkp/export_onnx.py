"""Export the trained probe (StandardScaler + LogisticRegression) to ONNX so it
can be proven with JOLT Atlas (https://github.com/ICME-Lab/jolt-atlas).

JOLT Atlas proves ONNX *inference*. The probe is a tiny linear graph
(standardize -> linear -> sigmoid), so the proof is cheap relative to their
GPT-scale examples. What the proof establishes: the agreed classifier ran on the
given input vector and produced this score (execution integrity of the detection
step) — NOT that the vector is a genuine model activation, nor that the verdict
is semantically correct.

Usage:
    python -m zkp.export_onnx                 # uses artifacts/probe.joblib
    python -m zkp.export_onnx --check         # also verify ONNX==sklearn parity

Outputs:
    zkp/artifacts/probe.onnx      the graph JOLT Atlas consumes
    zkp/artifacts/sample_input.npy  one example input vector (for the prover)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from src.config import PROBE_PATH, ACTIVATIONS_PATH
from src.probe import load

ZKP_DIR = Path(__file__).resolve().parent
ART = ZKP_DIR / "artifacts"
ART.mkdir(exist_ok=True)
ONNX_PATH = ART / "probe.onnx"
SAMPLE_PATH = ART / "sample_input.npy"


def export(probe, dim: int) -> Path:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # Single float input row of length `dim`. Fixed batch size 1 keeps the graph
    # simple and matches a per-request prove call.
    initial_types = [("input", FloatTensorType([1, dim]))]
    onnx_model = convert_sklearn(
        probe,
        initial_types=initial_types,
        target_opset=17,
        options={type(probe.steps[-1][1]): {"zipmap": False}},  # raw prob array
    )
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())
    return ONNX_PATH


def _infer_dim(probe) -> int:
    # StandardScaler exposes n_features_in_.
    return int(probe.steps[0][1].n_features_in_)


def check_parity(probe, dim: int, n: int = 64, seed: int = 0):
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    sk = probe.predict_proba(X)[:, 1]

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[-1].name  # probabilities output
    onnx_scores = []
    for row in X:
        out = sess.run([out_name], {in_name: row[None, :]})[0]
        onnx_scores.append(float(np.ravel(out)[-1]))  # P(class=1)
    onnx_scores = np.asarray(onnx_scores)

    max_abs = float(np.max(np.abs(sk - onnx_scores)))
    print(f"parity check on {n} samples: max|sklearn - onnx| = {max_abs:.2e}")
    # float32 vs float64 path: expect tiny differences only.
    assert max_abs < 1e-4, f"ONNX diverges from sklearn (max {max_abs})"
    print("parity OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify ONNX==sklearn")
    args = ap.parse_args()

    if not PROBE_PATH.exists():
        raise SystemExit(f"No probe at {PROBE_PATH}. Train it: python -m src.probe")
    probe = load(PROBE_PATH)
    dim = _infer_dim(probe)

    export(probe, dim)
    print(f"exported ONNX ({dim}-dim input) -> {ONNX_PATH}")

    # Save a representative sample input for the prover: prefer a real activation
    # row if available, else a random one.
    if ACTIVATIONS_PATH.exists():
        d = np.load(ACTIVATIONS_PATH, allow_pickle=True)
        row = np.asarray(d["X"][0], dtype=np.float32)
    else:
        row = np.random.default_rng(0).standard_normal(dim).astype(np.float32)
    np.save(SAMPLE_PATH, row)
    print(f"saved sample input -> {SAMPLE_PATH}")

    if args.check:
        check_parity(probe, dim)


if __name__ == "__main__":
    main()
