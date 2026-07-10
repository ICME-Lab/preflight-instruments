"""Train a sparse autoencoder on the extracted activations.

    python -m src.train_sae                    # defaults: d_hidden = 4*d_in
    python -m src.train_sae --hidden 8192 --epochs 300 --l1 2e-3

Reads data/activations.npz (from `python -m src.extract`), trains an SAE, and
saves it to artifacts/sae.npz. The SAE detector then fits a probe on its features
inside the eval harness.
"""
from __future__ import annotations
import argparse
import numpy as np

from src.config import ACTIVATIONS_PATH, ARTIFACT_DIR
from detectors.sae import SAEConfig, train_sae

SAE_PATH = ARTIFACT_DIR / "sae.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=None, help="dictionary size (default 4*d_in)")
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not ACTIVATIONS_PATH.exists():
        raise SystemExit(f"No activations at {ACTIVATIONS_PATH}. Run `python -m src.extract`.")
    d = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    finite = np.isfinite(X).all(axis=1)
    if (~finite).any():
        print(f"dropping {int((~finite).sum())} non-finite rows")
        X = X[finite]
    d_in = X.shape[1]
    d_hidden = args.hidden or (4 * d_in)

    cfg = SAEConfig(d_in=d_in, d_hidden=d_hidden, l1=args.l1, lr=args.lr,
                    epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    print(f"training SAE: d_in={d_in} d_hidden={d_hidden} l1={args.l1} "
          f"epochs={args.epochs} on {X.shape[0]} activations")
    sae, hist = train_sae(X, cfg)
    sae.save(SAE_PATH)

    # quick diagnostics: reconstruction + sparsity (mean active features/example)
    H = sae.features(X)
    active = (H > 1e-6).sum(axis=1).mean()
    print(f"done. final recon={hist['recon'][-1]:.4f}  "
          f"mean active features/example={active:.1f}/{d_hidden}")
    print(f"saved -> {SAE_PATH}")


if __name__ == "__main__":
    main()
