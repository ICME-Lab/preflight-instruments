# zkp — proving the probe with JOLT Atlas

This directory adds a **zero-knowledge proof of the probe's inference** using
[JOLT Atlas](https://github.com/ICME-Lab/jolt-atlas) (ICME Labs' zkML framework),
and a benchmark comparing latency **with and without** the proof.

## What is (and isn't) proven

The probe is a tiny linear graph: `standardize -> linear -> sigmoid`. JOLT Atlas
proves that **this exact classifier ran on this input vector and produced this
score** — execution integrity of the detection step, verifiable by anyone,
without trusting the party that ran it.

It does **not** prove:
- that the input vector is a genuine Qwen activation (feature extraction happens
  outside the proof; proving the full Qwen forward pass is the GPT-scale job,
  not this one),
- that the score is *semantically correct* (the probe is still a probabilistic
  classifier; a wrong verdict gets a valid proof of the wrong verdict).

So this turns "trust me, the guardrail ran" into "here's a receipt that the
guardrail ran." That's the honest, useful claim — process integrity, not
prevention.

## Pipeline

```
artifacts/probe.joblib   (trained probe, from `python -m src.probe`)
        │  python -m zkp.export_onnx
        ▼
zkp/artifacts/probe.onnx   (standardize → linear → sigmoid, opset 17)
        │  copied into the jolt-atlas repo as an example model
        ▼
JOLT Atlas: setup → prove → verify   (Rust, on your machine)
```

## Full walkthrough: clone to verified proof

This reproduces the prove/verify benchmark from scratch. Assumes you've already
trained the probe (`python -m src.probe`, which writes `artifacts/probe.joblib`).

### 0. Prerequisites

- Python env with this repo's deps plus the ONNX tooling:
  ```bash
  pip install -r requirements.txt
  pip install skl2onnx onnx onnxruntime
  ```
- Rust toolchain (stable) for JOLT Atlas: https://rustup.rs

### 1. Export the probe to a JOLT-Atlas-compatible ONNX graph

```bash
python -m zkp.export_onnx_coreops --check
```

Use `export_onnx_coreops`, NOT `export_onnx`. Details and the reasons for the
specific graph shape are in the "Graph shape" section below — they matter if you
ever change the probe. `--check` confirms the ONNX matches scikit-learn
(max abs diff ~1e-7) and writes `zkp/artifacts/probe.onnx`.

### 2. Get JOLT Atlas and confirm it builds

Clone it next to or inside this repo (these instructions assume inside, at
`./jolt-atlas`; adjust paths if you put it elsewhere):

```bash
git clone https://github.com/ICME-Lab/jolt-atlas.git
cd jolt-atlas
# Sanity-check the toolchain on one of their own examples first (first build
# pulls many crates and takes several minutes):
cargo run --release --package jolt-atlas-core --example nanoGPT -- --trace-terminal
cd ..
```

You want `Proof verified successfully!` from nanoGPT before continuing. If that
fails, it's a Rust/toolchain issue, not a probe issue.

### 3. Wire the probe into JOLT Atlas

```bash
# model directory + the exported graph (renamed to their convention):
mkdir -p jolt-atlas/atlas-onnx-tracer/models/probe
cp zkp/artifacts/probe.onnx jolt-atlas/atlas-onnx-tracer/models/probe/network.onnx

# the example that loads it and runs prove -> verify:
cp zkp/probe_example.rs jolt-atlas/jolt-atlas-core/examples/probe.rs
```

**IMPORTANT — re-copy after every re-export.** The tracer reads the *copied*
`network.onnx`, not `zkp/artifacts/probe.onnx`. If you re-run the exporter you
MUST re-copy, or you'll prove a stale graph. Verify they match:

```bash
md5sum zkp/artifacts/probe.onnx jolt-atlas/atlas-onnx-tracer/models/probe/network.onnx
# the two hashes must be identical
```

### 4. Run the proof

```bash
cd jolt-atlas
cargo run --release --package jolt-atlas-core --example probe -- --trace-terminal
cd ..
```

Expected: a computation-graph summary whose `Einsum` node reads `mk,nk->n`, then
`Proof generation took ...`, `Proof verification took ...`, and
`Proof verified successfully!`. Reference numbers on a single CPU core: prove
~66 ms, verify ~10 ms for the 1536-feature probe.

### 5. Full with/without benchmark

```bash
python -m zkp.benchmark --runs 200 --jolt-atlas jolt-atlas --jolt-runs 5 --json bench_results.json
```

Prints plain-inference latency (WITHOUT zk) next to prove+verify (WITH zk), plus
the overhead factor, and writes `bench_results.json`.

> **Reading the benchmark honestly:** the harness's `end-to-end` wall time
> includes process spawn + model load + setup, so the raw "overhead factor" is
> inflated. The steady-state cryptographic cost is the ~66 ms prove + ~10 ms
> verify from step 4's trace. Quote those for a fair comparison; the end-to-end
> number is what you'd pay per cold `cargo run`, not per proof in a warm server.

---

## Reference: individual steps

### Step 1 — export the probe to ONNX (tested, runs here)

```bash
python -m zkp.export_onnx_coreops --check
```

Use **`export_onnx_coreops`**, not `export_onnx`. The JOLT Atlas tracer supports
core ONNX ops (MatMul, Add, Sigmoid, ...) but NOT the `ai.onnx.ml` ops that
skl2onnx emits (`Scaler`, `LinearClassifier`). So `export_onnx_coreops` folds the
StandardScaler + LogisticRegression into a single core-op graph:

    input[1,dim] --MatMul(W)--> Add(b) --> Sigmoid --> prob[1,1]

using `W_eff = w/scale`, `b_eff = b - sum(W_eff*mean)`. `--check` confirms it
matches scikit-learn (max abs diff ~1e-7). Produces `zkp/artifacts/probe.onnx`
(opset 14, to match their download scripts) and `sample_input.npy`.

(`export_onnx.py` is kept only for reference and is marked deprecated — its
ML-domain ops will fail in the tracer.)

### On quantization

Their download scripts (`scripts/download_*.py`) export **float** ONNX via
optimum and then *normalize the graph* to tracer-supported ops — they do NOT
pre-quantize; the tracer quantizes to fixed-point i32 internally. Their examples
feed `Vec<i32>` because the inputs are integer **token IDs**, not because the
model is quantized. The probe differs: its input is a continuous feature vector.
For a **timing** benchmark the input values don't matter (cost depends on graph
size). For a **correct** proof of a specific activation, encode that activation
in the tracer's fixed-point convention (see `RunArgs` / the loader's scale) —
check `probe.rs` comments.

### Graph shape: the n=2 weight trick

The Atlas Einsum registry (`jolt-atlas-core/src/utils/dims.rs`) implements
`mk,kn->mn` and `k,nk->n` (plus aliases), but NOT a dot product (`k,k->`) or
`mk,k->mn`. The probe's linear layer is a matrix-vector product; if the weight's
output dimension `n == 1`, tract squeezes it to a bare vector and the op
collapses to the unregistered `k,k->m`. To keep it in the registered `k,nk->n`
family, the exporter pads the weight to **`n = 2`** (real probe weights in row 0,
a zero row in row 1) so the shape can't be squeezed. The proof reads `logits[0]`
as the score; `logits[1]` is discarded. This is why the input is rank-1 `[dim]`
and the output is `[2]`.

## Step 2 — benchmark WITHOUT zk (baseline)

```bash
python -m zkp.benchmark --runs 200
```

Reports plain inference latency (ONNX runtime and sklearn), in milliseconds.
This is what the probe adds to each request today.

## Step 3 — wire the probe into JOLT Atlas

1. Clone and build JOLT Atlas per its README (Rust + Cargo).
2. Copy the exported model into the repo:
   ```bash
   mkdir -p <jolt-atlas>/atlas-onnx-tracer/models/probe
   cp zkp/artifacts/probe.onnx <jolt-atlas>/atlas-onnx-tracer/models/probe/network.onnx
   ```
3. Copy the example stub in and adjust imports to match the repo's current API
   (use the existing `examples/nanoGPT.rs` / `examples/gpt2.rs` as the source of
   truth — the JOLT Atlas API may have moved since this stub was written):
   ```bash
   cp zkp/probe_example.rs <jolt-atlas>/jolt-atlas-core/examples/probe.rs
   ```
4. Smoke-test it directly:
   ```bash
   cd <jolt-atlas>
   cargo run --release --package jolt-atlas-core --example probe -- --trace-terminal
   # expect: "Proof verified successfully!"
   ```

## Step 4 — benchmark WITH zk (prove + verify)

```bash
python -m zkp.benchmark --runs 200 --jolt-atlas /path/to/jolt-atlas --jolt-runs 3
```

This runs the plain baseline AND the JOLT Atlas example (3x), parses its
`setup / prove / verify / end-to-end` timings, and prints a comparison plus the
overhead factor (prove+verify wall time vs. plain inference).

## Reading the result

Expect the two halves to differ by **orders of magnitude**: plain inference is
sub-millisecond; a zk prove is seconds. For reference, JOLT Atlas's own numbers
on an M3 are ~2.3 s to prove nanoGPT (~250k params) and ~0.13 s to verify; the
probe is far smaller than nanoGPT (a single linear op over a 1536-vector), so its
prove time should be well under those figures — but measure it, don't assume.

The point of the benchmark isn't that zk is "slow" — it's to quantify, honestly,
the cost of a verifiable receipt so you can decide where it's worth paying:
- **verify** is cheap (fractions of a second) and is what a third party runs, so
  the asymmetry matters — you prove once, anyone verifies cheaply;
- **prove** is the cost the serving side pays per request (or per batch), and is
  the number that decides whether per-request proving is viable vs. proving
  sampled/audited requests only.

## Notes / caveats

- **Quantization.** zkML operates over fixed-point/integer arithmetic. If JOLT
  Atlas quantizes the ONNX graph, re-check probe scores post-quantization against
  the float probe (extend `export_onnx.py --check` to compare against the
  quantized path) so the proven model still matches the one you evaluated.
- **This is the classifier proof only.** It composes with, and does not replace,
  the deterministic action/policy gate. A verifiable probe score is a stronger
  input to that gate, not a substitute for bounding what the agent can do.
- The Rust stub is a scaffold; the flow (load → setup → prove → verify, timed) is
  correct, but mirror the repo's current example code for exact API names.
```
