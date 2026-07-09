"""Export the probe as a MINIMAL CORE-OP ONNX graph for JOLT Atlas.

Why not skl2onnx: it emits ai.onnx.ml ops (Scaler, LinearClassifier) that the
Atlas tracer doesn't support. Atlas supports core ops (MatMul, Add, Sigmoid...).
So we fold StandardScaler + LogisticRegression into a single linear op + sigmoid,
built directly from core ONNX ops in the default domain.

Fold:
    x' = (x - mean) / scale             (StandardScaler)
    z  = w . x' + b                     (LogisticRegression)
    =>  W_eff = w / scale ,  b_eff = b - sum(W_eff * mean)
        prob  = sigmoid(W_eff . x + b_eff)

Graph:  input[1,dim] --MatMul(W[dim,1])--> Add(b[1]) --> Sigmoid --> prob[1,1]

Usage:
    python -m zkp.export_onnx_coreops --check
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from src.config import PROBE_PATH, ACTIVATIONS_PATH
from src.probe import load

ZKP_DIR = Path(__file__).resolve().parent
ART = ZKP_DIR / "artifacts"; ART.mkdir(exist_ok=True)
ONNX_PATH = ART / "probe.onnx"
SAMPLE_PATH = ART / "sample_input.npy"


def _fold(probe):
    scaler = probe.steps[0][1]
    clf = probe.steps[-1][1]
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    w = clf.coef_.reshape(-1).astype(np.float64)      # [dim]
    b = float(clf.intercept_.reshape(-1)[0])
    W_eff = w / scale                                  # [dim]
    b_eff = b - float(np.sum(W_eff * mean))
    return W_eff, b_eff


def export(probe, opset: int = 14) -> Path:
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    W_eff, b_eff = _fold(probe)
    dim = W_eff.shape[0]

    # Registry aliases (dims.rs) that map to the canonical `k,nk->n` handler
    # include `mk,nk->n`, which requires BOTH operands rank-2:
    #   input : [m, k] = [1, dim]
    #   weight: [n, k] = [2, dim]   (n padded to 2 so it can't be squeezed)
    #   output: [n]    = [2]
    # A rank-1 input instead yields `mk,k->m` (unregistered), and n==1 yields
    # `k,k->` (unregistered) — so we need m-dim present AND n>=2.
    W = np.zeros((2, dim), dtype=np.float32)           # [n=2, k=dim]
    W[0, :] = W_eff.astype(np.float32)
    B = np.array([b_eff, 0.0], dtype=np.float32)       # [n=2] bias per row

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, dim])  # [m=1,k]
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [2])      # [n=2]

    W_init = numpy_helper.from_array(W, name="W")
    B_init = numpy_helper.from_array(B, name="B")

    # einsum mk,nk->n : contract k between input[1,k] and weight[2,k] -> [2].
    n_es = helper.make_node(
        "Einsum", ["input", "W"], ["z0"], name="matvec", equation="mk,nk->n",
    )
    n_add = helper.make_node("Add", ["z0", "B"], ["z1"], name="add_bias")
    n_sig = helper.make_node("Sigmoid", ["z1"], ["logits"], name="sigmoid")

    graph = helper.make_graph(
        [n_es, n_add, n_sig], "probe",
        [inp], [out], initializer=[W_init, B_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)],
        producer_name="injection-probe",
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    ONNX_PATH.write_bytes(model.SerializeToString())
    return ONNX_PATH


def check_parity(probe, n: int = 64, seed: int = 0):
    import onnxruntime as ort
    dim = probe.steps[0][1].n_features_in_
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    sk = probe.predict_proba(X)[:, 1]
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_scores = np.array([
        float(np.ravel(sess.run(["logits"], {"input": r[None, :].astype(np.float32)})[0])[0])
        for r in X  # input [1,dim]; read logits[0], logits[1] is dummy
    ])
    mx = float(np.max(np.abs(sk - ort_scores)))
    print(f"core-op parity on {n} samples: max|sklearn - onnx| = {mx:.2e}")
    assert mx < 1e-4, f"diverges: {mx}"
    print("parity OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()
    if not PROBE_PATH.exists():
        raise SystemExit(f"No probe at {PROBE_PATH}. Train: python -m src.probe")
    probe = load(PROBE_PATH)
    export(probe, opset=args.opset)
    dim = probe.steps[0][1].n_features_in_
    print(f"exported CORE-OP ONNX (input [1,{dim}], opset {args.opset}) -> {ONNX_PATH}")
    if ACTIVATIONS_PATH.exists():
        d = np.load(ACTIVATIONS_PATH, allow_pickle=True)
        np.save(SAMPLE_PATH, np.asarray(d["X"][0], dtype=np.float32))
    else:
        np.save(SAMPLE_PATH, np.random.default_rng(0).standard_normal(dim).astype(np.float32))
    if args.check:
        check_parity(probe)


if __name__ == "__main__":
    main()
