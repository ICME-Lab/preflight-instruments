"""Benchmark: probe inference WITHOUT zk vs. WITH JOLT Atlas prove+verify.

Two halves:

  A. WITHOUT zk  — plain ONNX/sklearn inference latency (runs here, no Rust).
                   This is the baseline the probe adds to every request today.

  B. WITH zk     — JOLT Atlas proves the probe's ONNX inference and verifies it.
                   This runs the Rust example in a checked-out jolt-atlas repo
                   and parses its timing. Point --jolt-atlas at that repo.

The comparison you care about is: baseline inference (ms) vs. prove+verify
(seconds). They differ by orders of magnitude — that gap is the cost of a
verifiable receipt, and quantifying it honestly is the whole point.

Usage:
    # A only (no Rust needed):
    python -m zkp.benchmark --runs 200

    # A + B (needs the jolt-atlas repo built once):
    python -m zkp.benchmark --runs 200 --jolt-atlas /path/to/jolt-atlas
"""
from __future__ import annotations
import argparse
import json
import re
import statistics as stats
import subprocess
import time
from pathlib import Path

import numpy as np

from zkp.export_onnx_coreops import ONNX_PATH, SAMPLE_PATH


# ---------- A. plain inference (no zk) --------------------------------------

def bench_plain_onnx(runs: int, warmup: int = 20):
    import onnxruntime as ort
    if not ONNX_PATH.exists():
        raise SystemExit("Run `python -m zkp.export_onnx` first.")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[-1].name
    x = np.load(SAMPLE_PATH).astype(np.float32)[None, :]

    for _ in range(warmup):
        sess.run([out_name], {in_name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run([out_name], {in_name: x})
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    return times


def bench_plain_sklearn(runs: int, warmup: int = 20):
    from src.config import PROBE_PATH
    from src.probe import load
    probe = load(PROBE_PATH)
    x = np.load(SAMPLE_PATH).astype(np.float64)[None, :]
    for _ in range(warmup):
        probe.predict_proba(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        probe.predict_proba(x)
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    return times


def _summ(ms):
    ms_sorted = sorted(ms)
    p50 = ms_sorted[len(ms) // 2]
    p95 = ms_sorted[int(len(ms) * 0.95)]
    return {
        "n": len(ms),
        "mean_ms": stats.mean(ms),
        "std_ms": stats.pstdev(ms),
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": min(ms),
    }


# ---------- B. JOLT Atlas prove + verify ------------------------------------

# Regexes matching the timing lines JOLT Atlas prints with --trace-terminal.
_PATTERNS = {
    "setup_prover_s": re.compile(r"setup_prover.*?([\d.]+)\s*s", re.I),
    "prove_s": re.compile(r"(?:ONNXProof::)?prove\b.*?([\d.]+)\s*s", re.I),
    "verify_s": re.compile(r"(?:ONNXProof::)?verify\b.*?([\d.]+)\s*s", re.I),
    "e2e_s": re.compile(r"(?:end-to-end|total).*?([\d.]+)\s*s", re.I),
}


def bench_jolt_atlas(repo: Path, example: str = "probe", runs: int = 1):
    """Run the JOLT Atlas Rust example `runs` times and parse timings.

    Assumes you've added an example named `probe` to jolt-atlas-core that loads
    zkp/artifacts/probe.onnx and runs prove->verify (see zkp/probe_example.rs and
    zkp/README.md). Each run shells out to cargo and parses --trace-terminal.
    """
    repo = Path(repo).expanduser().resolve()
    if not (repo / "Cargo.toml").exists():
        raise SystemExit(f"{repo} doesn't look like the jolt-atlas repo.")

    all_runs = []
    for i in range(runs):
        cmd = [
            "cargo", "run", "--release",
            "--package", "jolt-atlas-core",
            "--example", example, "--", "--trace-terminal",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        out = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            print(out[-2000:])
            raise SystemExit(f"jolt-atlas example failed (run {i+1}).")

        parsed = {"wall_s": wall}
        for k, pat in _PATTERNS.items():
            m = pat.search(out)
            if m:
                parsed[k] = float(m.group(1))
        all_runs.append(parsed)
        print(f"  jolt run {i+1}/{runs}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in parsed.items()))
    return all_runs


# ---------- CLI -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200, help="plain-inference runs")
    ap.add_argument("--jolt-atlas", type=str, default=None,
                    help="path to checked-out jolt-atlas repo (enables zk half)")
    ap.add_argument("--jolt-runs", type=int, default=3, help="prove+verify runs")
    ap.add_argument("--example", type=str, default="probe",
                    help="jolt-atlas-core example name")
    ap.add_argument("--json", type=str, default=None, help="write results JSON")
    args = ap.parse_args()

    results = {}

    print("=" * 60)
    print("A. WITHOUT zk  — plain probe inference")
    print("=" * 60)
    onnx_ms = bench_plain_onnx(args.runs)
    skl_ms = bench_plain_sklearn(args.runs)
    results["plain_onnx"] = _summ(onnx_ms)
    results["plain_sklearn"] = _summ(skl_ms)
    print(f"  ONNX runtime : mean={results['plain_onnx']['mean_ms']:.3f} ms  "
          f"p50={results['plain_onnx']['p50_ms']:.3f}  "
          f"p95={results['plain_onnx']['p95_ms']:.3f}")
    print(f"  sklearn      : mean={results['plain_sklearn']['mean_ms']:.3f} ms  "
          f"p50={results['plain_sklearn']['p50_ms']:.3f}  "
          f"p95={results['plain_sklearn']['p95_ms']:.3f}")

    if args.jolt_atlas:
        print("\n" + "=" * 60)
        print("B. WITH zk  — JOLT Atlas prove + verify")
        print("=" * 60)
        jolt = bench_jolt_atlas(Path(args.jolt_atlas), args.example, args.jolt_runs)
        results["jolt_atlas"] = jolt

        # Summary comparison.
        prove = [r.get("prove_s") for r in jolt if "prove_s" in r]
        verify = [r.get("verify_s") for r in jolt if "verify_s" in r]
        e2e = [r.get("e2e_s") or r.get("wall_s") for r in jolt]
        base_ms = results["plain_onnx"]["mean_ms"]
        print("\n" + "-" * 60)
        print("COMPARISON")
        print("-" * 60)
        print(f"  without zk (inference)  : {base_ms:.3f} ms")
        if prove:
            print(f"  with zk    (prove)      : {stats.mean(prove):.3f} s")
        if verify:
            print(f"  with zk    (verify)     : {stats.mean(verify):.3f} s")
        if e2e:
            overhead = (stats.mean(e2e) * 1000) / base_ms
            print(f"  with zk    (end-to-end) : {stats.mean(e2e):.3f} s")
            print(f"  overhead factor         : ~{overhead:,.0f}x "
                  f"(prove+verify vs. plain inference)")
    else:
        print("\n(no --jolt-atlas path given; ran the WITHOUT-zk half only)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
