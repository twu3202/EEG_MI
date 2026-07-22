#!/usr/bin/env python
"""Which mental tasks differ in embedding space? — the core of the personal-MI scheme.

For ONE subject (your constraint), embeds every trial, measures **cross-session** pairwise
separability between task classes, picks the best mutually-separable k-set (max-min clique),
and runs a **permutation test** so a separability value is only trusted if it beats shuffled
labels. Cross-session (train day-A / test day-B) is the honest test that separability is
neural, not cap-placement (research/MI_4class_personal_design.md).

Embedding spaces (--embedding):
  tangent   Riemannian tangent space of the covariance   (8-30 Hz, no training)  [default]
  cbramod   frozen CBraMod embedding                      (0.5-45 Hz, 200 Hz, MPS)
  labram    frozen LaBraM embedding                       (0.5-45 Hz, 200 Hz, MPS)

Demo on public 4-class MI (BCI IV-2a, one subject). Point it at your own multi-task
screening recording later (same MOABB/MNE epoch format).

  python src/exploration/task_separability.py --embedding tangent --permutations 200
  python src/exploration/task_separability.py --embedding cbramod --permutations 100
"""
from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MNE_DATA", str(ROOT / "mne_data"))
os.environ.setdefault("MOABB_RESULTS", str(ROOT / "mne_data"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
(ROOT / "mne_data").mkdir(parents=True, exist_ok=True)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from moabb.paradigms import MotorImagery

DATASETS = {"2a": "BNCI2014_001", "2b": "BNCI2014_004"}

# per-embedding default preprocessing (band, resample)
PREP = {
    "tangent": dict(fmin=8.0, fmax=32.0, resample=None),
    "cbramod": dict(fmin=0.5, fmax=45.0, resample=200.0),
    "labram": dict(fmin=0.5, fmax=45.0, resample=200.0),
}


def load_subject(key, subject, n_classes, fmin, fmax, resample):
    ds = getattr(import_module("moabb.datasets"), DATASETS[key])()
    ds.subject_list = [subject]
    kw = dict(n_classes=n_classes, fmin=fmin, fmax=fmax)
    if resample:
        kw["resample"] = resample
    X, y, meta = MotorImagery(**kw).get_data(dataset=ds)
    return np.asarray(X), np.asarray(y), meta


def build_features(X, embedding):
    """Return (feat, make_clf). tangent keeps raw epochs (cov+TS fit per fold);
    FM precomputes frozen embeddings (fit only a linear head per fold)."""
    if embedding == "tangent":
        make_clf = lambda: make_pipeline(Covariances("oas"), TangentSpace(),
                                         LogisticRegression(max_iter=1000))
        return X, make_clf
    # frozen foundation-model embedding
    sys.path.insert(0, str(ROOT / "src" / "foundation"))
    from encoders import CBraModEncoder, LaBraMEncoder
    enc = CBraModEncoder() if embedding == "cbramod" else LaBraMEncoder()
    print(f"  encoding {len(X)} trials with {enc.name} …")
    feat = enc.encode(X)
    make_clf = lambda: make_pipeline(StandardScaler(),
                                     PCA(n_components=min(50, feat.shape[1])),
                                     LogisticRegression(max_iter=1000))
    return feat, make_clf


def _split(sessions, y):
    sess = np.unique(sessions)
    if len(sess) >= 2:
        return sessions == sess[0], sessions != sess[0], "cross-session"
    cut = len(y) // 2
    tr = np.zeros(len(y), bool); tr[:cut] = True
    return tr, ~tr, "within-session (only one session found)"


def pairwise_separability(feat, y, sessions, classes, make_clf):
    tr_mask, te_mask, split = _split(sessions, y)
    n = len(classes)
    M = np.full((n, n), np.nan)
    for a, b in combinations(range(n), 2):
        sel = np.isin(y, [classes[a], classes[b]])
        tr, te = sel & tr_mask, sel & te_mask
        if tr.sum() < 4 or te.sum() < 4:
            continue
        clf = make_clf()
        clf.fit(feat[tr], y[tr])
        M[a, b] = M[b, a] = balanced_accuracy_score(y[te], clf.predict(feat[te]))
    return M, split


def mean_sep(M):
    iu = np.triu_indices_from(M, k=1)
    return float(np.nanmean(M[iu]))


def best_kset(M, classes, k):
    best, best_score = None, -1.0
    for combo in combinations(range(len(classes)), k):
        pairs = [M[i, j] for i, j in combinations(combo, 2)]
        if any(np.isnan(pairs)):
            continue
        if min(pairs) > best_score:
            best_score, best = min(pairs), combo
    return best, best_score


def permutation_test(feat, y, sessions, classes, make_clf, n_perm, obs, seed=7):
    """Shuffle labels within each session, recompute mean pairwise separability -> null."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        yp = y.copy()
        for s in np.unique(sessions):
            m = sessions == s
            yp[m] = rng.permutation(yp[m])
        Mp, _ = pairwise_separability(feat, yp, sessions, classes, make_clf)
        null[p] = mean_sep(Mp)
    pval = (1 + int(np.sum(null >= obs))) / (n_perm + 1)
    return null, pval


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="2a", choices=list(DATASETS))
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--embedding", default="tangent", choices=list(PREP))
    ap.add_argument("--permutations", type=int, default=0, help="0 = skip permutation test")
    args = ap.parse_args()

    p = PREP[args.embedding]
    X, y, meta = load_subject(args.dataset, args.subject, args.n_classes,
                              p["fmin"], p["fmax"], p["resample"])
    sessions = meta["session"].to_numpy()
    classes = sorted(set(y))
    print(f"subject {args.subject} | embedding={args.embedding} | X={X.shape} | "
          f"classes={classes} | sessions={sorted(set(sessions))}\n")

    feat, make_clf = build_features(X, args.embedding)
    M, split = pairwise_separability(feat, y, sessions, classes, make_clf)

    print(f"pairwise separability ({split}, balanced acc; 0.5=indistinguishable):")
    print("        " + "".join(f"{c[:6]:>8}" for c in classes))
    for i, c in enumerate(classes):
        row = "".join(f"{M[i,j]:>8.2f}" if not np.isnan(M[i, j]) else f"{'·':>8}"
                      for j in range(len(classes)))
        print(f"{c[:7]:<8}{row}")

    pairs = sorted([(classes[i], classes[j], M[i, j])
                    for i, j in combinations(range(len(classes)), 2) if not np.isnan(M[i, j])],
                   key=lambda t: t[2], reverse=True)
    print("\neasiest → hardest pairs:")
    for a, b, s in pairs:
        print(f"  {s:.2f}  {a} vs {b}")

    obs = mean_sep(M)
    print(f"\nmean pairwise separability = {obs:.3f}")
    if args.k <= len(classes):
        best, score = best_kset(M, classes, args.k)
        if best:
            print(f"best {args.k}-set (max-min = {score:.2f}): {[classes[i] for i in best]}")

    if args.permutations:
        print(f"\npermutation test ({args.permutations} shuffles, labels shuffled within session)…")
        null, pval = permutation_test(feat, y, sessions, classes, make_clf,
                                      args.permutations, obs)
        print(f"  observed mean-sep = {obs:.3f} | null mean = {null.mean():.3f} "
              f"(95th pct {np.percentile(null,95):.3f}) | p = {pval:.4f}")
        print("  → " + ("REAL structure (p<0.05)" if pval < 0.05 else
                        "NOT distinguishable from chance — suspect artifact / too little data"))


if __name__ == "__main__":
    main()
