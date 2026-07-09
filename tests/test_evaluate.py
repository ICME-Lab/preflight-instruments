"""Model-free tests for the evaluation harness (src/evaluate.py).

Uses synthetic, separable activations with realistic group tags so we can verify
each evaluation returns sane, correctly-shaped results without a model.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.evaluate import cross_validate, per_group_report, by_attack_type


DIM = 64
SEED = 0


def _synth_with_groups(seed=SEED):
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(DIM); direction /= np.linalg.norm(direction)

    specs = [
        ("injecagent_dh", 1, 20),
        ("injecagent_ds", 1, 20),
        ("synthetic_direct", 1, 15),
        ("benign_action", 0, 25),
        ("benign_doc", 0, 15),
        ("injecagent_benign", 0, 25),
    ]
    Xs, ys, gs = [], [], []
    for group, label, n in specs:
        base = rng.standard_normal((n, DIM))
        base += (1.6 if label == 1 else -1.6) * direction
        Xs.append(base)
        ys.extend([label] * n)
        gs.extend([group] * n)
    X = np.vstack(Xs)
    y = np.asarray(ys)
    g = np.asarray(gs, dtype=object)
    return X, y, g


def test_cross_validate_shape_and_range():
    X, y, _ = _synth_with_groups()
    cv = cross_validate(X, y, k=5, seed=SEED)
    assert cv["k"] == 5
    assert 0.0 <= cv["auroc_mean"] <= 1.0
    assert cv["auroc_std"] >= 0.0
    # Separable synthetic data -> should be clearly better than chance.
    assert cv["auroc_mean"] > 0.75


def test_per_group_report_covers_all_groups():
    X, y, g = _synth_with_groups()
    rep = per_group_report(X, y, g, k=5, seed=SEED)
    assert set(rep.keys()) == set(np.unique(g).tolist())
    for name, r in rep.items():
        if r["type"] == "attack":
            assert 0.0 <= r["recall"] <= 1.0
        else:
            assert 0.0 <= r["fpr"] <= 1.0


def test_by_attack_type_generalizes():
    X, y, g = _synth_with_groups()
    res = by_attack_type(X, y, g, seed=SEED)
    # Both attack intents should be evaluable.
    assert "injecagent_dh" in res and "injecagent_ds" in res
    for r in res.values():
        assert 0.0 <= r["recall_on_held"] <= 1.0
        # On separable synthetic data the probe should transfer to the unseen
        # attack type well above chance.
        assert r["auroc"] > 0.7


def test_by_attack_type_has_both_classes_in_train():
    """Regression guard for the bug where holding out an attack type left the
    training set with only positives."""
    X, y, g = _synth_with_groups()
    # Should not raise and should return results (fit succeeded => 2 classes).
    res = by_attack_type(X, y, g, seed=SEED)
    assert len(res) == 2


def test_threshold_sweep_monotonic_and_calibrated():
    """Recall should not decrease as the FPR budget grows, and achieved FPR
    should stay near the target."""
    from src.evaluate import threshold_sweep
    X, y, g = _synth_with_groups()
    rows = threshold_sweep(X, y, g, fprs=(0.01, 0.05, 0.10), seed=SEED)
    assert len(rows) == 3
    recalls = [r["recall_in_dist"] for r in rows]
    # Monotonic non-decreasing recall as budget loosens.
    assert recalls[0] <= recalls[1] <= recalls[2] + 1e-9
    # Achieved FPR should respect the target (with small-sample slack).
    for r in rows:
        assert r["achieved_fpr"] <= r["target_fpr"] + 0.05
        assert 0.0 <= r["recall_in_dist"] <= 1.0
