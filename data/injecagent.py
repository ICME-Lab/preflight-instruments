"""Load the InjecAgent benchmark and turn it into labelled (text, label) pairs.

InjecAgent models *indirect* prompt injection: a benign user tool returns a
response, and the attacker's instruction is smuggled into that response (e.g.
inside a product review or email body). Each user case ships a
`Tool Response Template` containing a literal `<Attacker Instruction>`
placeholder — the exact injection point.

We build:
  POSITIVES: user tool response with a real attacker instruction substituted in
             (indirect injection, label 1)
  NEGATIVES: the same tool response with a benign string in the slot
             (label 0) -> this is the crucial *hard negative*: identical
             surrounding structure, only the injected content differs, so the
             probe must learn the injection itself, not the wrapper.

Plus the raw attacker instructions as direct-injection positives.

Files are fetched once from the public repo and cached under data/injecagent/.
Falls back gracefully with a clear message if offline.

Source: https://github.com/uiuc-kang-lab/InjecAgent (research/defensive use).
"""
from __future__ import annotations
import ast
import json
import random
import urllib.request
from pathlib import Path

_REPO_RAW = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"
_FILES = {
    "dh": "attacker_cases_dh.jsonl",
    "ds": "attacker_cases_ds.jsonl",
    "user": "user_cases.jsonl",
}
_CACHE = Path(__file__).resolve().parent / "injecagent"
_PLACEHOLDER = "<Attacker Instruction>"

# Benign fillers for the injection slot, to build hard negatives that share the
# exact wrapper of a positive but carry no attack.
_BENIGN_FILLERS = [
    "This product exceeded my expectations. Battery life is great and shipping was fast.",
    "Works exactly as described. Would recommend to a friend.",
    "Meeting notes attached. Let me know if Thursday still works for everyone.",
    "The weather has been lovely this week, perfect for a walk in the park.",
    "Thanks for the update. I've filed the report and will follow up next Monday.",
    "Reminder: the quarterly review is scheduled for the first week of next month.",
    "Great customer service, the replacement arrived within two days.",
    "The documentation was clear and setup took less than ten minutes.",
    "Please see the attached agenda for tomorrow's planning session.",
    "The package was well protected and everything arrived intact.",
    "Loved the design, though I wish it came in more colors.",
    "Confirming receipt of your message; I'll circle back by end of week.",
]


def _download(name: str) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{_REPO_RAW}/{name}"
    urllib.request.urlretrieve(url, dest)
    return dest


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render_template(template_str: str, injected: str) -> str:
    """Substitute the injection slot and return a readable tool-response string.

    The template is a Python-dict-ish string; we do a plain text replace on the
    placeholder (robust regardless of nesting) and return the raw string, which
    is what an agent would actually read.
    """
    filled = template_str.replace(_PLACEHOLDER, injected)
    # Best-effort prettify if it parses as a literal; otherwise return as-is.
    try:
        obj = ast.literal_eval(filled)
        return json.dumps(obj, ensure_ascii=False)
    except (ValueError, SyntaxError):
        return filled


def load_injecagent(seed: int = 0, max_pairs: int | None = None):
    """Return (texts, labels). Raises RuntimeError with guidance if offline."""
    texts, labels, _ = load_injecagent_tagged(seed=seed, max_pairs=max_pairs)
    return texts, labels


def load_injecagent_tagged(seed: int = 0, max_pairs: int | None = None):
    """Like load_injecagent but also returns per-example group tags:
    'injecagent_ds' / 'injecagent_dh' for positives (by attack intent),
    'injecagent_benign' for the matched benign-filled negatives.
    """
    try:
        dh = _read_jsonl(_download(_FILES["dh"]))
        ds = _read_jsonl(_download(_FILES["ds"]))
        users = _read_jsonl(_download(_FILES["user"]))
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch InjecAgent files ({e}). If offline, clone "
            "https://github.com/uiuc-kang-lab/InjecAgent and copy its data/*.jsonl "
            f"into {_CACHE}/"
        )

    rng = random.Random(seed)
    # Keep attack type with each instruction: dh=direct harm, ds=data stealing.
    atk_dh = [r["Attacker Instruction"] for r in dh if r.get("Attacker Instruction")]
    atk_ds = [r["Attacker Instruction"] for r in ds if r.get("Attacker Instruction")]
    attacker_tagged = ([(a, "injecagent_dh") for a in atk_dh]
                       + [(a, "injecagent_ds") for a in atk_ds])
    templates = [u["Tool Response Template"] for u in users
                 if _PLACEHOLDER in u.get("Tool Response Template", "")]

    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []

    # Indirect positives + matched hard negatives.
    for tmpl in templates:
        atk, tag = rng.choice(attacker_tagged)
        texts.append(_render_template(tmpl, atk)); labels.append(1); groups.append(tag)
        for benign in rng.sample(_BENIGN_FILLERS, k=2):
            texts.append(_render_template(tmpl, benign))
            labels.append(0); groups.append("injecagent_benign")

    # Direct-injection positives, capped for balance.
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    direct_budget = max(0, n_neg - n_pos)
    rng.shuffle(attacker_tagged)
    for atk, tag in attacker_tagged[:direct_budget]:
        texts.append(atk); labels.append(1); groups.append(tag)

    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    groups = [groups[i] for i in idx]

    if max_pairs is not None:
        texts = texts[:max_pairs]; labels = labels[:max_pairs]; groups = groups[:max_pairs]

    return texts, labels, groups


if __name__ == "__main__":
    t, y = load_injecagent()
    print(f"InjecAgent: {len(t)} examples | pos={sum(y)} neg={len(y)-sum(y)}")
    for txt, lab in list(zip(t, y))[:4]:
        print(f"[{lab}] {txt[:90]!r}")
