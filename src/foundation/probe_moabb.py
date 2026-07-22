#!/usr/bin/env python
"""Frozen-backbone linear-probe harness for EEG foundation models (model-agnostic).

Loads MI epochs from MOABB (same `LeftRightImagery` paradigm as the pyRiemann
baseline), runs them through a *frozen* encoder to get per-trial embeddings, and
fits a linear probe under leave-one-subject-out (LOSO). Swap the encoder for
EEGPT / CBraMod / LaBraM once their repos are cloned; the data + eval stay
identical, so the numbers are directly comparable to the baseline.

  python src/foundation/probe_moabb.py --dataset 2b --subjects 4 --encoder raw
"""
from __future__ import annotations

import argparse
import os
from importlib import import_module
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "mne_data"
_DATA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_DATA", str(_DATA))
os.environ.setdefault("MOABB_RESULTS", str(_DATA))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from moabb.paradigms import LeftRightImagery

DATASETS = {"2a": "BNCI2014_001", "2b": "BNCI2014_004"}


def load_mi(dataset_key: str, n_subjects: int, sfreq: float, fmin: float, fmax: float):
    ds = getattr(import_module("moabb.datasets"), DATASETS[dataset_key])()
    if n_subjects:
        ds.subject_list = ds.subject_list[:n_subjects]
    paradigm = LeftRightImagery(fmin=fmin, fmax=fmax, resample=sfreq)
    X, y, meta = paradigm.get_data(dataset=ds)
    return X, np.asarray(y), meta


# ---- pluggable encoders --------------------------------------------------
class RawFlattenEncoder:
    """Trivial 'floor' encoder: flatten bandpassed epochs. Foundation-model
    embeddings should beat this — it's the sanity reference."""

    name = "raw-flatten"

    def encode(self, X):  # X: (n_trials, n_ch, n_times)
        return X.reshape(len(X), -1)


def get_encoder(kind: str):
    if kind == "raw":
        return RawFlattenEncoder()
    if kind == "cbramod":
        from encoders import CBraModEncoder
        return CBraModEncoder()
    if kind == "labram":
        from encoders import LaBraMEncoder
        return LaBraMEncoder()
    # TODO — EEGPT needs its figshare checkpoint (256 Hz, 58ch native); download manually
    #   into checkpoints/, then add an EEGPTEncoder mirroring EEGPTClassifier.
    raise SystemExit(f"encoder '{kind}' not wired yet — see TODO in this file")


def loso_probe(X, y, subjects, encoder):
    emb = encoder.encode(X)
    aucs = []
    for s in np.unique(subjects):
        te = subjects == s
        tr = ~te
        n_comp = int(min(64, emb.shape[1], tr.sum() - 1))
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp),
            LogisticRegression(max_iter=1000),
        )
        clf.fit(emb[tr], y[tr])
        proba = clf.predict_proba(emb[te])[:, 1]
        y_bin = (y[te] == clf.classes_[1]).astype(int)
        auc = roc_auc_score(y_bin, proba)
        aucs.append(auc)
        print(f"  held-out subject {s}: AUC={auc:.3f}")
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="2b", choices=list(DATASETS))
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--encoder", default="raw")
    ap.add_argument("--sfreq", type=float, default=200.0)
    ap.add_argument("--fmin", type=float, default=0.5, help="broadband for FMs; use 8 for classic MI band")
    ap.add_argument("--fmax", type=float, default=45.0)
    args = ap.parse_args()

    X, y, meta = load_mi(args.dataset, args.subjects, args.sfreq, args.fmin, args.fmax)
    subjects = meta["subject"].to_numpy()
    print(f"loaded X={X.shape}, {len(np.unique(subjects))} subjects, classes={sorted(set(y))}")

    enc = get_encoder(args.encoder)
    mean, std = loso_probe(X, y, subjects, enc)
    print(f"\n{enc.name}: LOSO AUC = {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
