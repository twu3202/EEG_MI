#!/usr/bin/env python
"""Cross-subject Motor-Imagery baseline: MOABB + pyRiemann.

Runs classic, well-validated MI decoders under within-session and
cross-subject (leave-one-subject-out) evaluation on public datasets, so we can
(1) sanity-check the whole pipeline end-to-end and (2) get a realistic
"no per-subject retraining" performance reference before we ever touch our own
32-ch dry cap.

Pipelines
  CSP+LDA   classic Common Spatial Patterns + LDA
  TS+LR     Riemannian tangent space + logistic regression  (MOABB's strongest avg MI decoder)
  MDM       Minimum Distance to Mean  (training-light, robust to noise)

Examples
  python moabb_riemann_baseline.py --dataset 2b --subjects 4 --eval cross
  python moabb_riemann_baseline.py --dataset 2a --subjects 9 --eval cross
  python moabb_riemann_baseline.py --dataset 2a --subjects 3 --eval within
"""
from __future__ import annotations

import argparse
import os
import warnings
from importlib import import_module
from pathlib import Path

# Keep all MOABB/MNE downloads + results inside the repo (before moabb import).
_DATA = Path(__file__).resolve().parents[2] / "mne_data"
_DATA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_DATA", str(_DATA))
os.environ.setdefault("MOABB_RESULTS", str(_DATA))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

import moabb
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import CrossSubjectEvaluation, WithinSessionEvaluation

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# key -> (MOABB dataset class name, n_channels)
DATASETS = {
    "2a": ("BNCI2014_001", 22),  # BCI IV 2a: 9 subj, 22 ch, 250 Hz, 4-class -> LR keeps 2
    "2b": ("BNCI2014_004", 3),   # BCI IV 2b: 9 subj, 3 ch,  250 Hz, 2-class
}


def get_dataset(key: str, n_subjects: int):
    name, _ = DATASETS[key]
    ds = getattr(import_module("moabb.datasets"), name)()
    if n_subjects:
        ds.subject_list = ds.subject_list[:n_subjects]
    return ds


def build_pipelines(n_ch: int):
    csp_comp = min(8, n_ch)  # CSP components must be <= n_channels
    return {
        "CSP+LDA": make_pipeline(CSP(n_components=csp_comp), LDA()),
        "TS+LR": make_pipeline(
            Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=1000)
        ),
        "MDM": make_pipeline(Covariances("oas"), MDM()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="2b", choices=list(DATASETS))
    ap.add_argument("--subjects", type=int, default=4, help="limit #subjects (0=all)")
    ap.add_argument("--eval", default="cross", choices=["cross", "within"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    name, n_ch = DATASETS[args.dataset]
    ds = get_dataset(args.dataset, args.subjects)
    paradigm = LeftRightImagery()  # 2-class left vs right hand, 8-32 Hz bandpass
    pipelines = build_pipelines(n_ch)

    print(f"\n=== {name} | {args.eval}-subject | {len(ds.subject_list)} subjects ===")
    Eval = CrossSubjectEvaluation if args.eval == "cross" else WithinSessionEvaluation
    evaluation = Eval(
        paradigm=paradigm, datasets=[ds], overwrite=True, suffix="mi_baseline"
    )
    results = evaluation.process(pipelines)

    summary = results.groupby("pipeline")["score"].agg(["mean", "std", "min", "max"])
    print("\n----- summary (score = ROC-AUC for 2-class) -----")
    print(summary.to_string())

    csv = out / f"results_{args.dataset}_{args.eval}.csv"
    results.to_csv(csv, index=False)

    order = list(summary.index)
    plt.figure(figsize=(7, 4))
    for i, pipe in enumerate(order):
        s = results.loc[results.pipeline == pipe, "score"].to_numpy()
        plt.bar(i, s.mean(), yerr=s.std(), width=0.6, color="#4C78A8",
                alpha=0.85, capsize=4)
        plt.scatter(np.full_like(s, i, dtype=float), s, color="black",
                    s=18, alpha=0.6, zorder=3)
    plt.xticks(range(len(order)), order)
    plt.axhline(0.5, ls="--", c="gray", label="chance")
    plt.ylim(0.4, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title(f"{name} — {args.eval}-subject MI (left vs right hand)")
    plt.legend()
    plt.tight_layout()
    png = out / f"results_{args.dataset}_{args.eval}.png"
    plt.savefig(png, dpi=130)
    print(f"\nsaved: {csv}\nsaved: {png}")


if __name__ == "__main__":
    main()
