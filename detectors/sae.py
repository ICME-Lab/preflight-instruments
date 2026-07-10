"""A small sparse autoencoder (SAE) over LLM activations, plus training.

An SAE learns an overcomplete, sparse dictionary: it encodes each activation
vector into many mostly-zero feature activations, then decodes back. The features
are (hopefully) more interpretable/monosemantic than raw dimensions.

We train it on the SAME activations used for the probe (data/activations.npz),
unsupervised (labels unused for training). A detector then fits a linear probe on
the SAE feature activations.

Honest note: a probe on SAE features often does NOT beat a probe on raw
activations for detection — SAE reconstruction is lossy and the injection signal
may live partly in what the SAE drops. We build it properly and let the harness
report whether it helps. Its stronger value is interpretability/intervention.

This is a compact NumPy/torch implementation (ReLU SAE with an L1 sparsity
penalty and tied-ish init). It is intentionally small and dependency-light so it
trains on CPU for modest activation sets; scale up for real use.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass
class SAEConfig:
    d_in: int                       # activation dim (e.g. 1536)
    d_hidden: int                   # dictionary size (e.g. 4*d_in)
    l1: float = 1e-3                # sparsity penalty
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    seed: int = 0


class SAE:
    """ReLU sparse autoencoder: h = relu(We (x - b_dec) + b_enc); x_hat = Wd h + b_dec.

    Stored as numpy arrays so it can be applied without torch at inference.
    Training uses torch if available (faster), else a numpy Adam fallback.
    """

    def __init__(self, cfg: SAEConfig, We=None, Wd=None, b_enc=None, b_dec=None):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        scale = 1.0 / np.sqrt(cfg.d_in)
        self.Wd = Wd if Wd is not None else (rng.standard_normal((cfg.d_hidden, cfg.d_in)) * scale)
        # normalize decoder rows (standard for SAEs)
        self.Wd = self.Wd / (np.linalg.norm(self.Wd, axis=1, keepdims=True) + 1e-8)
        self.We = We if We is not None else self.Wd.T.copy()          # [d_in, d_hidden]
        self.b_enc = b_enc if b_enc is not None else np.zeros(cfg.d_hidden)
        self.b_dec = b_dec if b_dec is not None else np.zeros(cfg.d_in)

    # ---- inference (numpy, no torch needed) ----
    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        pre = (X - self.b_dec) @ self.We + self.b_enc
        return np.maximum(pre, 0.0)                                   # ReLU features

    def decode(self, H: np.ndarray) -> np.ndarray:
        return H @ self.Wd + self.b_dec

    def features(self, X: np.ndarray) -> np.ndarray:
        return self.encode(X)

    # ---- persistence ----
    def save(self, path):
        path = Path(path)
        np.savez_compressed(path, We=self.We, Wd=self.Wd, b_enc=self.b_enc,
                            b_dec=self.b_dec, cfg=json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        cfg = SAEConfig(**json.loads(str(d["cfg"])))
        return cls(cfg, We=d["We"], Wd=d["Wd"], b_enc=d["b_enc"], b_dec=d["b_dec"])


def train_sae(X: np.ndarray, cfg: SAEConfig) -> tuple[SAE, dict]:
    """Train an SAE on activations X. Returns (sae, history).

    Uses torch if importable (autograd + Adam); otherwise a small numpy Adam on
    the closed-form gradients. Loss = MSE reconstruction + l1 * mean(|h|).
    """
    X = np.asarray(X, dtype=np.float32)
    try:
        import torch
        return _train_torch(X, cfg)
    except ImportError:
        return _train_numpy(X, cfg)


def _train_torch(X, cfg: SAEConfig):
    import torch
    g = torch.Generator().manual_seed(cfg.seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    n, d = Xt.shape

    Wd = torch.randn(cfg.d_hidden, d, generator=g) / (d ** 0.5)
    Wd = Wd / (Wd.norm(dim=1, keepdim=True) + 1e-8)
    We = Wd.t().clone()
    b_enc = torch.zeros(cfg.d_hidden)
    b_dec = Xt.mean(0).clone()
    for p in (Wd, We, b_enc, b_dec):
        p.requires_grad_(True)

    opt = torch.optim.Adam([We, Wd, b_enc, b_dec], lr=cfg.lr)
    hist = {"loss": [], "recon": [], "l1": []}
    idx = torch.arange(n)
    for ep in range(cfg.epochs):
        perm = idx[torch.randperm(n, generator=g)]
        for i in range(0, n, cfg.batch_size):
            b = Xt[perm[i:i + cfg.batch_size]]
            h = torch.relu((b - b_dec) @ We + b_enc)
            xhat = h @ Wd + b_dec
            recon = ((xhat - b) ** 2).mean()
            l1 = cfg.l1 * h.abs().mean()
            loss = recon + l1
            opt.zero_grad(); loss.backward()
            # keep decoder rows unit-norm (project grad + renorm)
            opt.step()
            with torch.no_grad():
                Wd.div_(Wd.norm(dim=1, keepdim=True) + 1e-8)
        hist["loss"].append(float(loss)); hist["recon"].append(float(recon)); hist["l1"].append(float(l1))

    sae = SAE(cfg, We=We.detach().numpy(), Wd=Wd.detach().numpy(),
              b_enc=b_enc.detach().numpy(), b_dec=b_dec.detach().numpy())
    return sae, hist


def _train_numpy(X, cfg: SAEConfig):
    rng = np.random.default_rng(cfg.seed)
    n, d = X.shape
    sae = SAE(cfg)
    We, Wd = sae.We.copy(), sae.Wd.copy()
    b_enc, b_dec = sae.b_enc.copy(), X.mean(0).astype(np.float64)
    params = [We, Wd, b_enc, b_dec]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = {"loss": [], "recon": [], "l1": []}
    t = 0
    for ep in range(cfg.epochs):
        perm = rng.permutation(n)
        for i in range(0, n, cfg.batch_size):
            t += 1
            xb = X[perm[i:i + cfg.batch_size]].astype(np.float64)
            pre = (xb - b_dec) @ We + b_enc
            h = np.maximum(pre, 0.0)
            xhat = h @ Wd + b_dec
            err = xhat - xb
            bs = xb.shape[0]
            # grads
            gWd = h.T @ err / bs
            gh = err @ Wd.T + cfg.l1 * (pre > 0)   # l1 subgrad through relu mask
            gpre = gh * (pre > 0)
            gWe = (xb - b_dec).T @ gpre / bs
            gb_enc = gpre.mean(0)
            gb_dec = (-gpre @ We.T - err).mean(0) * 0 + (-err).mean(0)
            grads = [gWe, gWd, gb_enc, gb_dec]
            for j, (p, gr) in enumerate(zip(params, grads)):
                m[j] = b1 * m[j] + (1 - b1) * gr
                v[j] = b2 * v[j] + (1 - b2) * gr * gr
                mhat = m[j] / (1 - b1 ** t); vhat = v[j] / (1 - b2 ** t)
                p -= cfg.lr * mhat / (np.sqrt(vhat) + eps)
            Wd /= (np.linalg.norm(Wd, axis=1, keepdims=True) + 1e-8)
        recon = float((err ** 2).mean()); l1 = float(cfg.l1 * np.abs(h).mean())
        hist["loss"].append(recon + l1); hist["recon"].append(recon); hist["l1"].append(l1)
    sae.We, sae.Wd, sae.b_enc, sae.b_dec = We, Wd, b_enc, b_dec
    return sae, hist
