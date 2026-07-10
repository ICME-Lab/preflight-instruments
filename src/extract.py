"""Extract residual-stream activations from Qwen for each labelled prompt.

Run on a machine with `torch` + `transformers` installed (GPU recommended).
Writes an .npz with X (n_examples, hidden_dim) and y (n_examples,).

Everything is kept deliberately simple: one forward pass per prompt with
output_hidden_states=True, then pool the chosen layer over tokens.
"""
from __future__ import annotations
import sys
import numpy as np

from src.config import (
    MODEL_NAME, LAYER_INDEX, POOLING, ACTIVATIONS_PATH,
)


def _pool(hidden, attention_mask, mode: str):
    """hidden: (seq, dim) tensor for one example; returns (dim,) numpy vector."""
    import torch
    n_tokens = int(attention_mask.sum().item())
    if n_tokens == 0:
        # Degenerate/empty input: return a zero vector rather than NaN/inf.
        return np.zeros(hidden.shape[-1], dtype=np.float32)
    if mode == "last":
        last = max(0, n_tokens - 1)
        vec = hidden[last]
    else:  # mean over non-pad tokens
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (seq,1)
        summed = (hidden * mask).sum(dim=0)
        count = mask.sum(dim=0).clamp(min=1)
        vec = summed / count
    out = vec.float().cpu().numpy()
    # Guard against fp16 overflow / stray non-finite values.
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def extract(texts, labels, groups=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    vecs = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            # Guard against empty/whitespace inputs from scraped data.
            safe_text = text if (text and text.strip()) else "[empty]"
            enc = tok(safe_text, return_tensors="pt", truncation=True,
                      max_length=512).to(device)
            out = model(**enc)
            # hidden_states is a tuple len = n_layers+1; pick our layer.
            hs = out.hidden_states[LAYER_INDEX][0]          # (seq, dim)
            am = enc["attention_mask"][0]                    # (seq,)
            vecs.append(_pool(hs, am, POOLING))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(texts)}", flush=True)

    X = np.vstack(vecs)
    y = np.asarray(labels, dtype=np.int64)
    if groups is None:
        groups = ["unknown"] * len(labels)
    g = np.asarray(groups, dtype=object)
    # Save the source texts too, so text-based detectors (PromptGuard) score the
    # EXACT strings that produced each activation — no index-alignment guessing.
    txt = np.asarray(list(texts), dtype=object)
    np.savez_compressed(ACTIVATIONS_PATH, X=X, y=y, groups=g, texts=txt)
    print(f"Saved {X.shape} activations -> {ACTIVATIONS_PATH}")
    return X, y



# ---- runtime single-text extraction (model cached across calls) ----
_RT = {"tok": None, "model": None, "device": None}


def _ensure_model():
    """Load the model once and cache it for repeated single-text extraction
    (used by the runtime scanner / server, not the batch pipeline)."""
    if _RT["model"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
    ).to(device)
    model.eval()
    _RT.update(tok=tok, model=model, device=device)


def extract_single(text: str) -> "np.ndarray":
    """Return a [1, dim] activation vector for one text, reusing a cached model.

    Same layer / pooling as the batch extractor, so scores are consistent with
    the trained probe and SAE.
    """
    import torch
    _ensure_model()
    tok, model, device = _RT["tok"], _RT["model"], _RT["device"]
    safe_text = text if (text and text.strip()) else "[empty]"
    with torch.no_grad():
        enc = tok(safe_text, return_tensors="pt", truncation=True,
                  max_length=512).to(device)
        out = model(**enc)
        hs = out.hidden_states[LAYER_INDEX][0]
        am = enc["attention_mask"][0]
        vec = _pool(hs, am, POOLING)
    return np.asarray(vec, dtype=np.float64)[None, :]


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("ERROR: install torch + transformers first:\n"
              "  pip install torch transformers", file=sys.stderr)
        sys.exit(1)

    from data.dataset import build_combined_dataset_tagged
    texts, labels, groups = build_combined_dataset_tagged(include_injecagent=True)
    print(f"Extracting {len(texts)} examples "
          f"(pos={sum(labels)}, neg={len(labels)-sum(labels)})")
    extract(texts, labels, groups)


if __name__ == "__main__":
    main()
