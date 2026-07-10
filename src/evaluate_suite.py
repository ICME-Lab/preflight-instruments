"""Benchmark the whole detector suite: probe, PromptGuard, SAE, and the ensemble.

Reports, per detector AND for the ensemble, the same honest numbers the probe
harness uses: cross-val-style AUROC, plus per-source recall and per-benign-group
false-positive rate so you can see WHERE each detector's coverage comes from.

The "cover the most" claim is evidenced here: the ensemble row should catch
attacks that individual detectors miss (higher union recall), and the per-group
table shows each detector's complementary strengths.

Inputs:
  - activations + tags: data/activations.npz  (probe, SAE)
  - original texts:     rebuilt from the dataset builder, aligned to activations
                        (PromptGuard needs raw text)

Because activations.npz stores X/y/groups but not the source strings, we rebuild
the texts from the same seeded dataset builder and align by index. The builder is
deterministic given the seed, so order matches the extraction order.

    python -m src.evaluate_suite                     # all available detectors
    python -m src.evaluate_suite --no-promptguard    # skip the gated model
    python -m src.evaluate_suite --no-sae

Detectors that can't load (gated PromptGuard without access, missing SAE) are
skipped with a note, so the suite still runs on whatever is available.
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import ACTIVATIONS_PATH, PROBE_PATH
from src.train_sae import SAE_PATH


# ---------- data ----------

def _load_activations():
    d = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    X, y = np.asarray(d["X"], dtype=np.float64), np.asarray(d["y"])
    groups = d["groups"] if "groups" in d else np.array(["?"] * len(y), dtype=object)
    texts = d["texts"] if "texts" in d else None    # saved by newer extract.py
    finite = np.isfinite(X).all(axis=1)
    texts = texts[finite] if texts is not None else None
    return X[finite], y[finite], groups[finite], finite, texts


def _rebuild_texts(finite_mask):
    """Fallback for older activation files that didn't store texts: rebuild via
    the same seeded builder and align by index."""
    from data.dataset import build_combined_dataset_tagged
    texts, labels, groups = build_combined_dataset_tagged(include_injecagent=True,
                                                          include_deepset=True)
    texts = np.array(texts, dtype=object)
    if len(texts) != len(finite_mask):
        return None
    return texts[finite_mask]


# ---------- per-detector scoring (out-of-fold for the trainable ones) ----------

def _oof_probe_like(build_and_fit, X, y, k=5, seed=0):
    """Generic out-of-fold scorer for detectors that FIT on activations
    (probe, SAE-probe). `build_and_fit(Xtr, ytr)` returns a scorer(Xte)->scores."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        scorer = build_and_fit(X[tr], y[tr])
        oof[te] = scorer(X[te])
    return oof


def _static_scores(score_fn, n):
    """For detectors that don't fit on our data (PromptGuard): score once."""
    return score_fn()


# ---------- report ----------

def _auroc(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m]) if len(np.unique(y[m])) == 2 else float("nan")


def _per_group(y, s, groups, thr=0.5):
    rep = {}
    pred = (s >= thr).astype(int)
    for g in sorted(set(groups.tolist())):
        mask = (groups == g) & ~np.isnan(s)
        n = int(mask.sum())
        if n == 0:
            continue
        if y[mask].mean() >= 0.5:
            rep[g] = ("attack", n, float(pred[mask].mean()))          # recall
        else:
            rep[g] = ("benign", n, float(pred[mask].mean()))          # FPR
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-promptguard", action="store_true")
    ap.add_argument("--no-sae", action="store_true")
    # SAE options. Default is the honest per-fold retrain (leak-free). Pass
    # --sae-pretrained to use a single artifacts/sae.npz instead (faster, leaky).
    ap.add_argument("--sae-pretrained", action="store_true",
                    help="use artifacts/sae.npz (fast, LEAKY); default retrains per fold")
    ap.add_argument("--sae-hidden", type=int, default=None,
                    help="SAE dictionary size for per-fold training (default 4*d_in)")
    ap.add_argument("--sae-l1", type=float, default=1e-3)
    ap.add_argument("--sae-epochs", type=int, default=120)
    args = ap.parse_args()

    if not ACTIVATIONS_PATH.exists():
        raise SystemExit("Run `python -m src.extract` first.")
    X, y, groups, finite, saved_texts = _load_activations()
    print(f"Loaded {len(y)} examples (pos={int(y.sum())}, neg={int((y==0).sum())})")

    scores = {}   # name -> oof/static scores aligned to X

    # --- probe (out-of-fold) ---
    if not args.no_probe:
        from src.probe import build_probe, score as probe_score
        def fit_probe(Xtr, ytr):
            p = build_probe(); p.fit(Xtr, ytr)
            return lambda Xte: probe_score(p, Xte)
        scores["probe"] = _oof_probe_like(fit_probe, X, y, args.k, args.seed)
        print("scored: probe")

    # --- SAE-probe (out-of-fold) ---
    if not args.no_sae:
        from detectors.sae import SAE, SAEConfig, train_sae
        from detectors.sae_detector import SAEDetector

        if args.sae_pretrained:
            # FAST but LEAKY: SAE trained once on all data (dictionary has seen
            # eval examples). Only the probe-on-features is cross-validated.
            # Use for quick iteration, NOT for reported numbers.
            try:
                sae = SAE.load(SAE_PATH)
                def fit_sae(Xtr, ytr):
                    det = SAEDetector(sae).fit(Xtr, ytr)
                    return lambda Xte: det.score_activations(Xte)
                scores["sae"] = _oof_probe_like(fit_sae, X, y, args.k, args.seed)
                print("scored: sae (PRETRAINED sae — leaky, for iteration only)")
            except FileNotFoundError:
                print("skip sae: no artifacts/sae.npz (run `python -m src.train_sae`)")
        else:
            # HONEST: retrain the SAE inside each fold on the training split only,
            # so the dictionary never sees eval examples. Slower but leak-free.
            d_in = X.shape[1]
            cfg = SAEConfig(d_in=d_in, d_hidden=args.sae_hidden or (4 * d_in),
                            l1=args.sae_l1, epochs=args.sae_epochs,
                            batch_size=256, seed=args.seed)
            def fit_sae_fold(Xtr, ytr):
                sae_f, _ = train_sae(Xtr.astype("float32"), cfg)
                det = SAEDetector(sae_f).fit(Xtr, ytr)
                return lambda Xte: det.score_activations(Xte)
            print(f"scoring sae per-fold (d_hidden={cfg.d_hidden}, "
                  f"epochs={cfg.epochs}) — leak-free, slower ...")
            scores["sae"] = _oof_probe_like(fit_sae_fold, X, y, args.k, args.seed)
            print("scored: sae (per-fold, honest)")

    # --- PromptGuard (static text scores; no fitting on our data) ---
    if not args.no_promptguard:
        texts = saved_texts if saved_texts is not None else _rebuild_texts(finite)
        if texts is None:
            print("skip promptguard: no saved texts in activations.npz and could "
                  "not rebuild/align. Re-run `python -m src.extract` to store texts.")
        else:
            try:
                from detectors.promptguard_detector import PromptGuardDetector
                pg = PromptGuardDetector()
                scores["promptguard"] = pg.score_texts(list(texts))
                print("scored: promptguard")
            except Exception as e:
                print(f"skip promptguard: {e}")

    if not scores:
        raise SystemExit("No detectors available to score.")

    # --- ensemble (max over available detector scores) ---
    stacked = np.column_stack([scores[k] for k in scores])
    scores["ENSEMBLE(max)"] = np.nanmax(stacked, axis=1)

    # --- report ---
    print("\n" + "=" * 66)
    print("Per-detector AUROC")
    print("=" * 66)
    for name, s in scores.items():
        print(f"  {name:16s} AUROC = {_auroc(y, s):.3f}")

    print("\n" + "=" * 66)
    print("Per-group recall (attack) / FPR (benign)")
    print("=" * 66)
    names = list(scores.keys())
    header = "  group".ljust(24) + "".join(f"{n[:11]:>13s}" for n in names)
    print(header)
    all_groups = sorted(set(groups.tolist()))
    for g in all_groups:
        row = f"  {g[:22]:22s}"
        for n in names:
            rep = _per_group(y, scores[n], groups)
            if g in rep:
                _, cnt, val = rep[g]
                row += f"{val:>13.3f}"
            else:
                row += f"{'-':>13s}"
        print(row)

    print("\nReading it: for [attack] groups higher is better (recall); for "
          "[benign] groups lower is better (false-positive rate). The ensemble "
          "row should match or beat the best single detector on attack recall.")


if __name__ == "__main__":
    main()
