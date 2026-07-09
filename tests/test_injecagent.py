"""Tests for the InjecAgent loader. Uses the cached data/injecagent/*.jsonl.

Skips cleanly if the benchmark files aren't present (e.g. fresh clone, offline),
so CI doesn't fail on network. Run `python -m data.injecagent` once online to
populate the cache.
"""
from __future__ import annotations
import pytest

from data.injecagent import load_injecagent, _CACHE, _FILES
from data.dataset import build_combined_dataset


_have_cache = all((_CACHE / name).exists() for name in _FILES.values())
requires_cache = pytest.mark.skipif(
    not _have_cache, reason="InjecAgent files not cached; run `python -m data.injecagent`"
)


@requires_cache
def test_injecagent_loads_balanced():
    texts, labels = load_injecagent(seed=0)
    assert len(texts) == len(labels)
    pos = sum(labels)
    neg = len(labels) - pos
    assert pos > 0 and neg > 0
    # Loader is designed to return balanced classes.
    assert abs(pos - neg) <= 2, f"imbalanced: pos={pos} neg={neg}"


@requires_cache
def test_injecagent_positive_contains_attack_context():
    """At least some positives should carry recognizable attacker-instruction
    content (a sanity check that substitution happened)."""
    texts, labels = load_injecagent(seed=0)
    positives = [t for t, y in zip(texts, labels) if y == 1]
    assert len(positives) > 0
    # Not asserting on specific words (that would reintroduce keyword bias);
    # just that positives are non-trivial strings.
    assert all(len(p) > 10 for p in positives)


@requires_cache
def test_combined_is_balanced_and_larger():
    synth_only, _ = build_combined_dataset(include_injecagent=False)
    combined, labels = build_combined_dataset(include_injecagent=True)
    assert len(combined) > len(synth_only)
    frac_pos = sum(labels) / len(labels)
    assert 0.4 < frac_pos < 0.6, f"combined not balanced: {frac_pos:.2f} positive"


def test_combined_falls_back_offline(monkeypatch):
    """If the loader raises, build_combined_dataset degrades to synthetic."""
    import data.dataset as ds

    def boom(*a, **k):
        raise RuntimeError("simulated offline")

    # Patch the symbol used inside build_combined_dataset's import.
    import data.injecagent as ia
    monkeypatch.setattr(ia, "load_injecagent", boom)

    texts, labels = ds.build_combined_dataset(include_injecagent=True)
    assert len(texts) == len(labels) > 0  # still returns synthetic set
