#!/usr/bin/env python
"""Cross-subject MI with Euclidean Alignment (EA) — the biggest calibration-free win.

Euclidean Alignment (He & Wu, 2020): for each subject, whiten every trial by that
subject's own reference matrix R^{-1/2} (R = mean spatial covariance over the
subject's trials). R needs no labels, so it can be computed on the *test* subject's
trials too — i.e. this stays calibration-free. It pulls each subject's covariance
cloud to a common center, cutting inter-subject shift before any classifier.

We run a manual leave-one-subject-out loop (so alignment is per-subject) and compare
`none` vs `ea` for the strong Riemannian pipelines.

  python src/baselines/riemann_alignment.py --dataset 2a --subjects 9
"""
from __future__ import annotations

import argparse
import os
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[2] / "mne_data"
_DATA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_DATA", str(_DATA))
os.environ.setdefault("MOABB_RESULTS", str(_DATA))

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from moabb.paradigms import LeftRightImagery

warnings.filterwarnings("ignore")

DATASETS = {"2a": "BNCI2014_001", "2b": "BNCI2014_004"}


def load_mi(key: str, n_subjects: int):
    ds = getattr(import_module("moabb.datasets"), DATASETS[key])()
    if n_subjects:
        ds.subject_list = ds.subject_list[:n_subjects]
    X, y, meta = LeftRightImagery(fmin=8, fmax=32).get_data(dataset=ds)
    return np.asarray(X), np.asarray(y), meta["subject"].to_numpy()


def inv_sqrt(R: np.ndarray) -> np.ndarray:
    """Real symmetric matrix inverse square root via eigendecomposition."""
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """EA on one subject's trials. X: (n, ch, t) -> aligned (n, ch, t)."""
    covs = np.matmul(X, X.transpose(0, 2, 1))          # (n, ch, ch)
    W = inv_sqrt(covs.mean(0))                          # R^{-1/2}, label-free
    return np.matmul(W[None], X)


def make_clf(kind: str):
    if kind == "TS+LR":
        return make_pipeline(Covariances("oas"), TangentSpace(),
                             LogisticRegression(max_iter=1000))
    if kind == "MDM":
        return make_pipeline(Covariances("oas"), MDM())
    raise ValueError(kind)


def loso(X, y, subj, clf_kind: str, align: str):
    aucs = {}
    for te in np.unique(subj):
        tr_mask = subj != te
        # align each subject independently (train subjects + the held-out subject)
        Xtr_parts, ytr_parts = [], []
        for s in np.unique(subj[tr_mask]):
            m = subj == s
            Xs = euclidean_align(X[m]) if align == "ea" else X[m]
            Xtr_parts.append(Xs)
            ytr_parts.append(y[m])
        Xtr = np.concatenate(Xtr_parts)
        ytr = np.concatenate(ytr_parts)
        Xte = euclidean_align(X[subj == te]) if align == "ea" else X[subj == te]
        yte = y[subj == te]

        clf = make_clf(clf_kind)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        y_bin = (yte == clf.classes_[1]).astype(int)
        aucs[te] = roc_auc_score(y_bin, proba)
    return aucs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="2a", choices=list(DATASETS))
    ap.add_argument("--subjects", type=int, default=9)
    args = ap.parse_args()

    X, y, subj = load_mi(args.dataset, args.subjects)
    print(f"loaded X={X.shape}, {len(np.unique(subj))} subjects, classes={sorted(set(y))}\n")

    print(f"{'pipeline':<10}{'align':<8}{'mean AUC':>10}{'std':>8}")
    print("-" * 36)
    results = {}
    for clf_kind in ["TS+LR", "MDM"]:
        for align in ["none", "ea"]:
            aucs = loso(X, y, subj, clf_kind, align)
            vals = np.array(list(aucs.values()))
            results[(clf_kind, align)] = vals
            print(f"{clf_kind:<10}{align:<8}{vals.mean():>10.3f}{vals.std():>8.3f}")

    print("\nEA improvement (mean AUC):")
    for clf_kind in ["TS+LR", "MDM"]:
        d = results[(clf_kind, "ea")].mean() - results[(clf_kind, "none")].mean()
        print(f"  {clf_kind}: {d:+.3f}")


if __name__ == "__main__":
    main()
