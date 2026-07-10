"""Runtime injection scanner: score one text through the ensemble, with voting.

Console usage:
    python -m src.scan "text to check"
    python -m src.scan "text to check" 0.9          # custom threshold (default 0.90)

Voting over the three detectors (probe, sae, promptguard), each compared to the
threshold:
    >= 2 detectors over threshold  -> "prompt injection detected"
    exactly 1 over threshold       -> "possible injection"
    0 over threshold               -> "safe"

Output is human-readable plus a JSON line (so a server can parse it). The
individual detector scores are always shown, because the three detectors do NOT
have equal coverage — PromptGuard is largely blind to INDIRECT injection (hidden
in tool output / documents), while the activation probe/SAE catch it. So "1 over
threshold" can still be a real attack that only one detector can see. Downstream
logic should look at WHICH detector fired, not just the count.

Server use: import InjectionScanner, construct once (loads models), call
.scan(text, threshold) per request. The CLI constructs one per invocation.
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


DEFAULT_THRESHOLD = 0.70

# Verdict tiers, by how many detectors score at/above threshold:
#   0 over  -> none        ("no prompt injection detected")
#   1 over  -> probable    ("probable prompt injection")
#   2-3 over -> likely     ("likely prompt injection")
# This favors recall: a single detector's signal is surfaced (as the lower
# "probable" tier) rather than discarded, because the detectors have very
# different coverage and a lone strong signal is often a real attack the others
# structurally cannot see. Cost: benign inputs that trip one detector will show
# as "probable" — an intentional recall/precision trade for a security tripwire.
LIKELY = "likely prompt injection"
PROBABLE = "probable prompt injection"
NONE = "no prompt injection detected"


@dataclass
class ScanResult:
    text: str
    threshold: float
    scores: dict            # detector name -> P(injection)
    over_threshold: dict    # detector name -> bool
    n_over: int
    verdict: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def pretty(self) -> str:
        lines = [f'verdict: {self.verdict.upper()}',
                 f'threshold: {self.threshold:.2f}   ({self.n_over}/{len(self.scores)} detectors over)']
        for name, s in self.scores.items():
            mark = "OVER " if self.over_threshold[name] else "under"
            lines.append(f'  {name:12s} {s:.3f}  [{mark}]')
        return "\n".join(lines)


def _verdict(n_over: int) -> str:
    if n_over >= 2:
        return LIKELY
    if n_over == 1:
        return PROBABLE
    return NONE


class InjectionScanner:
    """Holds the loaded detectors so a server can reuse them across requests.

    Constructing this loads Qwen (for probe + SAE activations) and PromptGuard.
    Detectors that fail to load (e.g. no PromptGuard access) are dropped, and the
    voting adapts to however many are available — but the intended config is all
    three.
    """

    def __init__(self, require_all: bool = False):
        self.detectors = {}
        self._load(require_all)

    def _load(self, require_all: bool):
        errors = {}

        # Probe (activation) ------------------------------------------------
        try:
            from detectors.probe_detector import ProbeDetector
            self.detectors["probe"] = ProbeDetector()          # loads probe.joblib
        except Exception as e:
            errors["probe"] = str(e)

        # SAE (activation) --------------------------------------------------
        try:
            from detectors.sae import SAE
            from detectors.sae_detector import SAEDetector
            from src.train_sae import SAE_PATH
            from src.config import ACTIVATIONS_PATH
            sae = SAE.load(SAE_PATH)
            # The SAE detector needs a fitted probe-on-features. Fit it once on
            # the cached activations at construction time.
            d = np.load(ACTIVATIONS_PATH, allow_pickle=True)
            X = np.asarray(d["X"], dtype=np.float64)
            yv = np.asarray(d["y"])
            finite = np.isfinite(X).all(axis=1)
            self.detectors["sae"] = SAEDetector(sae).fit(X[finite], yv[finite])
        except Exception as e:
            errors["sae"] = str(e)

        # PromptGuard (text) ------------------------------------------------
        try:
            from detectors.promptguard_detector import PromptGuardDetector
            pg = PromptGuardDetector()
            pg._lazy_load()                                    # force the download/load now
            self.detectors["promptguard"] = pg
        except Exception as e:
            errors["promptguard"] = str(e)

        if errors:
            for name, err in errors.items():
                print(f"[warn] detector '{name}' unavailable: {err.splitlines()[0]}",
                      file=sys.stderr)
        if require_all and len(self.detectors) < 3:
            raise RuntimeError(f"only {len(self.detectors)}/3 detectors loaded; "
                               f"missing: {sorted(errors)}")
        if not self.detectors:
            raise RuntimeError("no detectors could be loaded")

        # Shared activation extraction for probe + SAE (extract once per text).
        self._needs_activation = any(
            getattr(d, "modality", "text") == "activation"
            for d in self.detectors.values()
        )

    def _activation_for(self, text: str) -> Optional[np.ndarray]:
        """Extract one activation vector for the text (probe + SAE share it)."""
        if not self._needs_activation:
            return None
        from src.extract import extract_single
        return extract_single(text)                            # shape [1, dim]

    def scan(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> ScanResult:
        X = self._activation_for(text)
        scores = {}
        for name, det in self.detectors.items():
            if getattr(det, "modality", "text") == "activation":
                scores[name] = float(det.score_activations(X)[0])
            else:
                scores[name] = float(det.score_texts([text])[0])
        over = {n: (s >= threshold) for n, s in scores.items()}
        n_over = int(sum(over.values()))
        return ScanResult(text=text, threshold=threshold, scores=scores,
                          over_threshold=over, n_over=n_over,
                          verdict=_verdict(n_over))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m src.scan "text to check" [threshold]', file=sys.stderr)
        raise SystemExit(2)
    text = argv[0]
    threshold = float(argv[1]) if len(argv) > 1 else DEFAULT_THRESHOLD
    if not (0.0 <= threshold <= 1.0):
        print(f"threshold must be in [0,1], got {threshold}", file=sys.stderr)
        raise SystemExit(2)

    scanner = InjectionScanner()
    result = scanner.scan(text, threshold)
    print(result.pretty())
    print(result.to_json())


if __name__ == "__main__":
    main()
