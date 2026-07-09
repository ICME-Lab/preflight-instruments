"""Sanity checks on the dataset builder."""
from data.dataset import build_dataset


def test_balanced_and_labelled():
    texts, labels = build_dataset(seed=0)
    assert len(texts) == len(labels)
    pos = sum(labels)
    neg = len(labels) - pos
    # Should have both classes, reasonably balanced.
    assert pos > 0 and neg > 0
    assert 0.3 < pos / len(labels) < 0.7


def test_deterministic_with_seed():
    a = build_dataset(seed=42)
    b = build_dataset(seed=42)
    assert a == b


def test_labels_are_binary():
    _, labels = build_dataset()
    assert set(labels) <= {0, 1}


def test_contains_benign_action_negatives():
    """Regression guard for the set-5 false-positive failure: the dataset MUST
    contain legitimate action requests labelled benign, so the probe can't learn
    'action-verb ==> attack'."""
    from data.dataset import _BENIGN_ACTIONS
    texts, labels = build_dataset()
    label_by_text = {t: y for t, y in zip(texts, labels)}
    for action in _BENIGN_ACTIONS:
        assert label_by_text.get(action) == 0, f"benign action mislabelled: {action!r}"
    assert len(_BENIGN_ACTIONS) >= 30, "need a meaningful number of benign actions"


def test_has_diverse_attack_styles():
    """Generalization depends on attack-style diversity beyond InjecAgent's one
    format; guard that the extra attack styles are present and labelled 1."""
    from data.dataset import _ATTACK_STYLES
    texts, labels = build_dataset()
    label_by_text = {t: y for t, y in zip(texts, labels)}
    for atk in _ATTACK_STYLES:
        assert label_by_text.get(atk) == 1, f"attack style mislabelled: {atk!r}"
    assert len(_ATTACK_STYLES) >= 12


def test_tagged_groups_present():
    from data.dataset import build_dataset_tagged
    _, _, groups = build_dataset_tagged()
    g = set(groups)
    for expected in {"benign_action", "benign_qa", "attack_style",
                     "synthetic_direct", "synthetic_indirect"}:
        assert expected in g, f"missing group {expected}"
