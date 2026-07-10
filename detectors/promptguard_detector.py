"""PromptGuard detector: Meta Llama Prompt Guard 2 (86M, multilingual).

Reads the RAW TEXT and classifies benign vs. malicious. This is an INDEPENDENT
signal from the activation probe: it sees the string before the model processes
it, so it catches attacks at a different stage and fails differently. That
independence is the point — an ensemble of text + activation detectors covers
strictly more than either alone.

Model: meta-llama/Llama-Prompt-Guard-2-86M
  - gated on HF (Llama license); request access and `huggingface-cli login`
  - binary benign/malicious, 512-token context, multilingual
  - label 1 (malicious) probability is our injection score

ZK note: this is an 86M transformer. Proving it is deferred (transformer-scale,
unlike the linear probe). The detector is proof-agnostic; a proof hook can be
added later.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

from detectors.base import Detector

DEFAULT_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"


class PromptGuardDetector(Detector):
    name = "promptguard"
    modality = "text"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_length: int = 512, batch_size: int = 16):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self._tok = None
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self._device).eval()
        # Identify which logit index is "malicious". PromptGuard 2 uses
        # LABEL_1 = malicious; confirm via id2label if present.
        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._malicious_idx = 1
        for idx, lab in id2label.items():
            if str(lab).lower() in ("malicious", "label_1", "injection", "jailbreak"):
                self._malicious_idx = int(idx)
                break

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._lazy_load()
        import torch
        texts = list(texts)
        out = np.zeros(len(texts), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self._tok(batch, return_tensors="pt", truncation=True,
                                padding=True, max_length=self.max_length).to(self._device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self._malicious_idx]
                out[i:i + len(batch)] = probs.float().cpu().numpy()
        return out
