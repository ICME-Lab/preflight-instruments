"""Load the deepset/prompt-injections dataset for attack-STYLE diversity.

Why this source: InjecAgent is all tool-response *indirect* injection in one
house style. deepset/prompt-injections is direct user-input injections and
jailbreaks, multilingual, from a different distribution. Adding it is aimed
squarely at the generalization gap — teaching the probe attack *styles* it
otherwise never sees, rather than more of the same.

Schema (662 rows): columns `text` (str) and `label` (int; 1=injection, 0=benign),
pre-split into train/test. We merge both splits and tag every example
`deepset_attack` / `deepset_benign` so the harness can slice on it.

Requires the `datasets` library:  pip install datasets
Loads from the HF hub on first call; `datasets` caches locally afterwards.
"""
from __future__ import annotations


def load_deepset_tagged(max_examples: int | None = None, seed: int = 0):
    """Return (texts, labels, groups). Raises RuntimeError with guidance if the
    datasets library or network isn't available."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "deepset loader needs the `datasets` library: pip install datasets"
        ) from e

    try:
        ds = load_dataset("deepset/prompt-injections")
    except Exception as e:
        raise RuntimeError(
            f"Could not load deepset/prompt-injections ({e}). Needs network on "
            "first run; afterwards it's cached by `datasets`."
        ) from e

    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    for split in ds.keys():                      # 'train' and 'test'
        for row in ds[split]:
            txt = (row.get("text") or "").strip()
            if not txt:
                continue
            lab = int(row["label"])
            texts.append(txt)
            labels.append(lab)
            groups.append("deepset_attack" if lab == 1 else "deepset_benign")

    if max_examples is not None and len(texts) > max_examples:
        import random
        rng = random.Random(seed)
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        idx = idx[:max_examples]
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
        groups = [groups[i] for i in idx]

    return texts, labels, groups


def load_deepset(max_examples: int | None = None, seed: int = 0):
    t, y, _ = load_deepset_tagged(max_examples=max_examples, seed=seed)
    return t, y


if __name__ == "__main__":
    try:
        t, y, g = load_deepset_tagged()
        from collections import Counter
        print(f"deepset: {len(t)} examples | pos={sum(y)} neg={len(y)-sum(y)}")
        print("groups:", dict(Counter(g)))
        for txt, lab in list(zip(t, y))[:4]:
            print(f"[{lab}] {txt[:80]!r}")
    except RuntimeError as e:
        print("Loader not runnable in this environment:", e)
