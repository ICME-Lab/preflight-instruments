"""Injection-detection suite: multiple detectors behind one interface.

Each detector scores a text (or its activation) for injection-compliance and
exposes a uniform API so the eval harness can benchmark them individually and as
an ensemble.

Detectors:
  probe        - linear probe on frozen-LLM activations (reads internal state)
  promptguard  - Meta Llama Prompt Guard 2 (reads raw text)
  sae          - probe on sparse-autoencoder features of activations

ZK note: only the linear `probe` currently has a working JOLT Atlas proof
(see zkp/). PromptGuard and SAE are transformer/large-net detectors; proving
them is deferred. The Detector interface is proof-agnostic so a `proof()` hook
can be added per detector later without changing callers.
"""
from detectors.base import Detector, DetectorResult

__all__ = ["Detector", "DetectorResult"]
