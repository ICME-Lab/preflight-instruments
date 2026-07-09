"""Model-free validation of the probe pipeline.

We synthesize two clusters of 'activations' separated along a hidden linear
direction (mimicking the empirical finding that injection-compliance lives on a
roughly linear direction in the residual stream), then assert the probe:
  1. recovers high held-out AUROC/accuracy,
  2. round-trips through save/load,
  3. scores a held-out injected vector higher than a clean one,
  4. actually 'catches' injections at the 0.5 threshold.

If this passes, the plumbing is correct and only representation quality is
unknown when you swap in real Qwen activations.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.probe import fit_and_eval, score, save, load, ProbeMetrics


HIDDEN_DIM = 128
N_PER_CLASS = 200
SEED = 0


def _synth(seed: int = SEED):
    """Two Gaussian blobs offset along a random unit direction + noise."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(HIDDEN_DIM)
    direction /= np.linalg.norm(direction)

    # Shared background covariance (isotropic-ish) so the classes overlap a bit
    # and the problem isn't trivially separable.
    base_clean = rng.standard_normal((N_PER_CLASS, HIDDEN_DIM))
    base_inj = rng.standard_normal((N_PER_CLASS, HIDDEN_DIM))

    # Push injections along +direction; clean along -direction.
    # Margin chosen so the classes are separable but not trivially so.
    margin = 1.8
    X_clean = base_clean - margin * direction
    X_inj = base_inj + margin * direction

    X = np.vstack([X_clean, X_inj])
    y = np.hstack([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)]).astype(int)
    return X, y, direction


def test_probe_learns_direction():
    X, y, _ = _synth()
    probe, metrics = fit_and_eval(X, y, seed=SEED)
    assert isinstance(metrics, ProbeMetrics)
    # With a real (if modest) margin the linear probe should be clearly useful.
    assert metrics.auroc > 0.85, f"AUROC too low: {metrics.auroc}"
    assert metrics.accuracy > 0.80, f"accuracy too low: {metrics.accuracy}"


def test_scores_are_probabilities():
    X, y, _ = _synth()
    probe, _ = fit_and_eval(X, y, seed=SEED)
    p = score(probe, X[:10])
    assert p.shape == (10,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_injection_scores_higher_than_clean():
    X, y, direction = _synth()
    probe, _ = fit_and_eval(X, y, seed=SEED)

    # Construct fresh held-out exemplars far along each pole.
    clean_vec = -2.0 * direction
    inj_vec = +2.0 * direction
    p_clean = float(score(probe, clean_vec)[0])
    p_inj = float(score(probe, inj_vec)[0])
    assert p_inj > p_clean
    assert p_inj > 0.5 > p_clean, f"p_inj={p_inj}, p_clean={p_clean}"


def test_catches_synthetic_injection_batch():
    """The 'does it catch it' test: recall on a held-out injected batch.

    Note we assert a *reasonable* recall floor, not perfection. A probe is
    probabilistic: at a fixed threshold some injections slip through. This is the
    honest property to test.
    """
    X, y, direction = _synth()
    probe, _ = fit_and_eval(X, y, seed=SEED)

    rng = np.random.default_rng(SEED + 99)
    held_out_injections = rng.standard_normal((100, HIDDEN_DIM)) + 1.8 * direction
    p = score(probe, held_out_injections)
    caught = (p >= 0.5).mean()
    assert caught >= 0.80, f"only caught {caught:.0%} of injections at thr=0.5"


def test_threshold_tradeoff():
    """Lowering the threshold raises recall (catches more) at the cost of more
    false positives. This is the core operating-point decision for a real probe.
    """
    X, y, direction = _synth()
    probe, _ = fit_and_eval(X, y, seed=SEED)

    rng = np.random.default_rng(SEED + 7)
    injections = rng.standard_normal((200, HIDDEN_DIM)) + 1.8 * direction
    clean = rng.standard_normal((200, HIDDEN_DIM)) - 1.8 * direction

    p_inj = score(probe, injections)
    p_clean = score(probe, clean)

    # A more sensitive threshold catches strictly more injections.
    recall_strict = (p_inj >= 0.5).mean()
    recall_sensitive = (p_inj >= 0.3).mean()
    assert recall_sensitive >= recall_strict

    # Choose the threshold that achieves a target 5% FPR on clean data, then
    # report the recall we get there. This is how you actually pick an operating
    # point: fix the tolerable false-alarm rate, measure what you catch.
    target_fpr = 0.05
    thr = float(np.quantile(p_clean, 1 - target_fpr))
    recall_at_thr = float((p_inj >= thr).mean())
    achieved_fpr = float((p_clean >= thr).mean())

    assert achieved_fpr <= target_fpr + 1e-9
    # At a strict 5% FPR the linear probe should still catch a solid majority.
    assert recall_at_thr > 0.60, f"recall {recall_at_thr:.0%} at {target_fpr:.0%} FPR"


def test_save_load_roundtrip(tmp_path):
    X, y, _ = _synth()
    probe, metrics = fit_and_eval(X, y, seed=SEED)
    ppath = tmp_path / "probe.joblib"
    mpath = tmp_path / "metrics.json"
    save(probe, metrics, ppath, mpath)
    assert ppath.exists() and mpath.exists()

    reloaded = load(ppath)
    p1 = score(probe, X[:20])
    p2 = score(reloaded, X[:20])
    assert np.allclose(p1, p2), "reloaded probe scores differ"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
