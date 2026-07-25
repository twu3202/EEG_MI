#!/usr/bin/env python
"""Broad offline exploration: is there ANY separable structure in these recordings?

The default pipeline (Riemannian tangent space on all channels, one hand-picked band)
found hands-vs-rest but nothing for hands/feet/math. Before concluding "no signal", search
much wider — different feature spaces (including foundation-model embeddings), different
channel subsets (some electrodes on this cap are dead or actively noisy), and different
time windows — and visualise the embedding space directly.

Two methodological rules, because it is easy to fool yourself with 19–75 trials:
  * anything data-driven (channel ranking, band choice) is fitted INSIDE the CV fold;
  * the headline number is a permutation test on ONE pre-declared pipeline, and every
    other number is reported as exploratory.

Also tests the INHIBITION HYPOTHESIS. The subject reports that during "motor imagery" the
dominant mental act was suppressing the urge to actually move, not imagining movement.
That predicts a specific pattern: an inhibition/effort component shared by every motor
class (so hands vs feet stays at chance), absent during rest (so hands vs rest separates),
and different from a purely cognitive task (so motor-pooled vs math should separate).
Motor inhibition also tends to RAISE beta, whereas imagery lowers it, so the two partly
cancel — which would explain weak, inconsistent beta ERD.

  python src/analysis/explore_embeddings.py --dataset both
  python src/analysis/explore_embeddings.py --dataset 3class --no-fm     # skip FM encoders
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # src/
sys.path.insert(0, str(HERE))                                 # src/analysis/
import load as L                                               # noqa: E402
from common.montage import SENSORIMOTOR                        # noqa: E402

RESULTS = HERE.parents[1] / "results"
REC = HERE.parents[1] / "recordings"
DATASETS = {
    "hands-rest": REC / "cap32_20260725_143756_hands-rest.npz",
    "3class": REC / "cap32_20260725_163251_hands-feet-math.npz",
}


# ------------------------------------------------------------------ data
def build_epochs(path, tmin=-2.0, tmax=4.0):
    """Recording → (Epochs, labels, names). Keeps ALL channels; drops only trials whose
    window runs past the end of the recording (a stalled session leaves fake trials)."""
    import mne
    mne.set_log_level("CRITICAL")
    z = np.load(path, allow_pickle=True)
    fs, N = float(z["fs"]), z["data"].shape[1]
    on, nm = z["trial_onset"], np.array([str(s) for s in z["trial_name"]])
    ok = np.where((on >= int(-tmin * fs)) & (on + int((tmax + 0.2) * fs) < N))[0]
    raw, _, _ = L.read_recording(str(path))
    raw, bad = L.clean_raw(raw, 1.0, 40.0, 50.0, car=True, interpolate=False)  # keep bads FLAGGED
    names = sorted(set(nm[ok]))
    code = {n: i + 1 for i, n in enumerate(names)}
    ev = np.column_stack([on[ok], np.zeros(len(ok), int), [code[n] for n in nm[ok]]])
    ep = mne.Epochs(raw, ev, code, tmin=tmin, tmax=tmax, baseline=None, preload=True,
                    verbose=False)
    return ep, ep.events[:, 2] - 1, names, bad


CH_SETS = {
    "all32": lambda ep: ep.ch_names,
    "drop_bad": lambda ep: [c for c in ep.ch_names if c not in ep.info["bads"]],
    "sensorimotor": lambda ep: [c for c in SENSORIMOTOR if c in ep.ch_names],
    "central+frontal": lambda ep: [c for c in ep.ch_names if c not in ep.info["bads"]
                                   and c[0] in "FC" or c in ("CZ", "FZ")],
}
BANDS = {"mu 8-13": (8, 13), "beta 13-30": (13, 30), "mu+beta 8-30": (8, 30),
         "wide 4-40": (4, 40), "theta 4-8": (4, 8)}


# --------------------------------------------------------------- pipelines
def make_pipelines(n_classes):
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.svm import SVC
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
    from mne.decoding import CSP
    lr = lambda: LogisticRegression(max_iter=2000, C=1.0)
    return {
        "TS+LR (oas)": make_pipeline(Covariances("oas"), TangentSpace(), StandardScaler(), lr()),
        "TS+LR (lwf)": make_pipeline(Covariances("lwf"), TangentSpace(), StandardScaler(), lr()),
        "TS+SVM": make_pipeline(Covariances("oas"), TangentSpace(), StandardScaler(),
                                SVC(kernel="rbf", C=1.0)),
        "MDM": make_pipeline(Covariances("oas"), MDM()),
        "CSP+LDA": make_pipeline(CSP(n_components=4, log=True), LDA()),
        "logvar+LR": make_pipeline(LogVar(), StandardScaler(), lr()),
    }


class LogVar:
    """Simple, robust baseline feature: log band-power per channel."""
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log(np.var(X, axis=-1) + 1e-12)
    def fit_transform(self, X, y=None): return self.transform(X)
    def get_params(self, deep=True): return {}
    def set_params(self, **k): return self


# --------------------------------------------------------- FM embeddings
def fm_embeddings(ep, chans, which="cbramod", tw=(0.5, 3.5)):
    """(n_trials, d) frozen foundation-model embedding per trial."""
    import torch
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from foundation.encoders import CBraModEncoder, LaBraMEncoder
    # Both encoders want the model's native rate (CBraMod/LaBraM: 200 Hz) and µV/100 —
    # the ÷100 is the encoder's own `scale`, so hand it plain µV at 200 Hz.
    e = ep.copy().pick(chans).filter(0.3, 45.0).resample(200.0).crop(*tw)
    X = e.get_data(copy=False) * 1e6                       # (n, ch, t) µV
    n_patch = (X.shape[-1] // 200) * 200                   # whole 1 s patches only
    X = X[..., :n_patch]
    enc = (CBraModEncoder() if which == "cbramod" else LaBraMEncoder())
    with torch.no_grad():
        Z = enc.encode(X)
    return np.asarray(Z).reshape(len(X), -1)


# --------------------------------------------------------------- evaluation
def evaluate(X, y, pipes, n_rep=5, seed=0):
    """Repeated stratified CV. Returns {name: (mean, std)} of accuracy."""
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    out = {}
    for name, clf in pipes.items():
        try:
            s = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            out[name] = (float(s.mean()), float(s.std()))
        except Exception as ex:
            out[name] = (float("nan"), 0.0)
    return out


def topk_channels_cv(X, y, k=12):
    """Rank channels by univariate log-variance discriminability. Used INSIDE folds only."""
    from sklearn.feature_selection import f_classif
    f = np.log(np.var(X, axis=-1) + 1e-12)
    F, _ = f_classif(f, y)
    return np.argsort(np.nan_to_num(F))[::-1][:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="both", choices=["both", "hands-rest", "3class"])
    ap.add_argument("--no-fm", action="store_true", help="skip foundation-model embeddings")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    keys = list(DATASETS) if args.dataset == "both" else [args.dataset]

    for key in keys:
        path = DATASETS[key]
        if not path.exists():
            print(f"missing {path}"); continue
        ep, y, names, bad = build_epochs(path)
        chance = 1.0 / len(names)
        print("\n" + "=" * 78)
        print(f"  {key}   {len(ep)} trials · classes {names} · chance {chance:.3f}")
        print(f"  flagged bad channels: {bad}")
        print("=" * 78)

        # ---- feature space × channel set × band sweep (EXPLORATORY) ----
        pipes = make_pipelines(len(names))
        print(f"\n[A] 特征空间 × 通道集 × 频段  (准确率, 5×5-fold; 随机={chance:.3f})")
        rows = []
        for cs_name, cs_fn in CH_SETS.items():
            chans = [c for c in cs_fn(ep) if c in ep.ch_names]
            if len(chans) < 4:
                continue
            for b_name, (lo, hi) in BANDS.items():
                e = ep.copy().pick(chans).filter(lo, hi).crop(0.5, 3.5)
                X = e.get_data(copy=False) * 1e6
                res = evaluate(X, y, pipes, n_rep=args.reps)
                for p_name, (m, s) in res.items():
                    rows.append((m, s, cs_name, b_name, p_name, len(chans)))
        rows = [r for r in rows if np.isfinite(r[0])]     # CSP goes rank-deficient on 32ch
        rows.sort(key=lambda r: -r[0])
        print(f"    {'acc':>7} {'std':>6}  {'channels':<16}{'band':<14}{'pipeline':<14}")
        for m, s, cs, b, p, n in rows[:12]:
            print(f"    {m:7.3f} {s:6.3f}  {cs:<11}({n:2d})  {b:<14}{p:<14}")
        print(f"    … {len(rows)} 组合中最好 {rows[0][0]:.3f} / 最差 {rows[-1][0]:.3f}")

        # ---- FM embeddings ----
        if not args.no_fm:
            print(f"\n[B] Foundation-model embedding (frozen) + 线性分类")
            for which in ("cbramod", "labram"):
                try:
                    chans = [c for c in ep.ch_names if c not in bad]
                    Z = fm_embeddings(ep, chans, which)
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.linear_model import LogisticRegression
                    clf = {"linear probe": make_pipeline(
                        StandardScaler(), LogisticRegression(max_iter=3000))}
                    r = evaluate(Z, y, clf, n_rep=args.reps)
                    m, s = r["linear probe"]
                    print(f"    {which:9s} dim={Z.shape[1]:5d} → {m:.3f} ± {s:.3f}")
                    np.save(f"/tmp/emb_{key}_{which}.npy", Z)
                except Exception as ex:
                    print(f"    {which:9s} 失败: {type(ex).__name__}: {ex}")
        np.save(f"/tmp/y_{key}.npy", y)
        with open(f"/tmp/names_{key}.txt", "w") as fh:
            fh.write("\n".join(names))


if __name__ == "__main__":
    main()
