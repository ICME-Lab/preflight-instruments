"""Model-free tests for the runtime scanner's voting and output.

No Qwen / PromptGuard needed — we stub detector scores and check the vote,
verdict thresholds, and JSON/pretty output.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.scan import (_verdict, InjectionScanner, ScanResult,
                      LIKELY, PROBABLE, NONE, DEFAULT_THRESHOLD)


def test_verdict_thresholds():
    assert _verdict(3) == LIKELY
    assert _verdict(2) == LIKELY
    assert _verdict(1) == PROBABLE
    assert _verdict(0) == NONE


class _FakeDet:
    def __init__(self, score, modality):
        self.score = score
        self.modality = modality

    def score_activations(self, X):
        return np.array([self.score])

    def score_texts(self, t):
        return np.array([self.score])


def _scanner(scores):
    """Build a scanner with stubbed detectors (bypassing model load)."""
    s = InjectionScanner.__new__(InjectionScanner)
    s.detectors = {
        "probe": _FakeDet(scores[0], "activation"),
        "sae": _FakeDet(scores[1], "activation"),
        "promptguard": _FakeDet(scores[2], "text"),
    }
    s._needs_activation = True
    s._activation_for = lambda text: np.zeros((1, 4))
    return s


@pytest.mark.parametrize("scores,thr,expected", [
    ((0.95, 0.92, 0.10), 0.70, LIKELY),     # 2 over
    ((0.99, 0.99, 0.99), 0.70, LIKELY),     # 3 over
    ((0.95, 0.10, 0.10), 0.70, PROBABLE),   # 1 over
    ((0.10, 0.10, 0.10), 0.70, NONE),       # 0 over
    ((0.70, 0.70, 0.10), 0.70, LIKELY),     # exactly-at-threshold counts as over
])
def test_scan_verdicts(scores, thr, expected):
    r = _scanner(scores).scan("x", thr)
    assert r.verdict == expected


def test_scan_result_fields_and_json():
    r = _scanner((0.95, 0.20, 0.20)).scan("hello", 0.70)
    assert r.n_over == 1
    assert r.verdict == PROBABLE
    assert r.over_threshold == {"probe": True, "sae": False, "promptguard": False}
    import json
    parsed = json.loads(r.to_json())
    assert parsed["verdict"] == PROBABLE
    assert parsed["scores"]["probe"] == 0.95


def test_default_threshold_is_90pct():
    assert DEFAULT_THRESHOLD == 0.70
    r = _scanner((0.69, 0.69, 0.69)).scan("x")   # all just under default
    assert r.verdict == NONE
    r2 = _scanner((0.71, 0.71, 0.10)).scan("x")   # two just over default
    assert r2.verdict == LIKELY
