#!/usr/bin/env python
"""Representation health check — a pre-flight gate to run BEFORE trusting any probe score.

A frozen foundation model can produce a representation that carries almost no trial-specific
information and still post a respectable cross-validated accuracy on a small dataset, purely
by fitting noise. That is not hypothetical: in this project BENDR was ranked the best of
seven models at 0.733 while its 512-dimensional embedding effectively spanned ~1.6
dimensions; once the input was fixed the same model scored 0.303, below chance. The score
alone never revealed the problem — these four numbers did.

Checks, in the order they catch things:

  1. PADDING       what fraction of the model's input is zero-filled. Feeding a 3 s trial
                   to a checkpoint built for 15 s made 80 % of LaBraM's input zeros, and the
                   probe read the padding instead of the EEG (it scored exactly chance).
  2. COVERAGE      how many of our electrodes actually reach the model after montage mapping.
  3. VARIATION     between-trial variation divided by representation magnitude, measured
                   AFTER centring. Raw trial-to-trial correlation is useless here: a large
                   shared constant drives it to 1.0 regardless of the real content (two of
                   our models sat at exactly 1.0000).
  4. EFFECTIVE RANK  participation ratio of the centred embedding — how many dimensions the
                   representation really uses across trials, not how many it declares.

The thresholds are heuristics calibrated on this cap and these session sizes; treat the
result as a smell test that decides whether a score is worth interpreting, not as a verdict
on the model.

  python src/foundation/embed_health.py                    # gate every wired model
  python src/foundation/embed_health.py --dataset 3class
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

# thresholds: (fail below/above, warn below/above)
PAD_FAIL, PAD_WARN = 0.05, 0.01          # fraction of input that is padding
COV_FAIL, COV_WARN = 0.30, 0.60          # fraction of our channels reaching the model
VAR_FAIL, VAR_WARN = 0.05, 0.20          # between-trial variation / magnitude
RANK_FAIL, RANK_WARN = 3.0, 6.0          # effective rank of the centred embedding

G, Y, R, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"


def health(Z, X_in=None, n_our_channels=None, n_matched=None):
    """Z (n_trials, d) frozen embedding → dict of health metrics."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(0)
    mag = np.abs(Z).mean() + 1e-12
    var_ratio = float(Zc.std(0).mean() / mag)
    s = np.linalg.svd(Zc, compute_uv=False)
    eff_rank = float((s.sum() ** 2) / ((s ** 2).sum() + 1e-30))
    dead = float((Zc.std(0) < 1e-9).mean())
    out = dict(dim=int(Z.shape[1]), n=int(Z.shape[0]), var_ratio=var_ratio,
               eff_rank=eff_rank, dead_dims=dead, max_rank=float(min(Z.shape[0] - 1, Z.shape[1])))
    if X_in is not None:
        nz = (np.abs(X_in).sum(axis=tuple(range(X_in.ndim - 1))) > 0)
        out["padding"] = float(1.0 - nz.mean())
        # Coverage = how many of OUR electrodes found a home in the model's montage.
        # Counting non-zero MODEL inputs instead overcounts: an alias (BENDR's T5/T6 are
        # our P7/P8) lets one electrode fill two slots, which produced a nonsense 106%.
        if n_matched is not None and n_our_channels:
            out["coverage"] = min(1.0, float(n_matched) / float(n_our_channels))
    return out


def verdict(h):
    """(status, [reasons]) — FAIL means the probe score should not be interpreted."""
    bad, warn = [], []
    if h.get("padding") is not None:
        if h["padding"] > PAD_FAIL:
            bad.append(f"输入 {h['padding']*100:.0f}% 是补零")
        elif h["padding"] > PAD_WARN:
            warn.append(f"补零 {h['padding']*100:.0f}%")
    if h.get("coverage") is not None:
        if h["coverage"] < COV_FAIL:
            bad.append(f"仅 {h['coverage']*100:.0f}% 通道进入模型")
        elif h["coverage"] < COV_WARN:
            warn.append(f"通道覆盖 {h['coverage']*100:.0f}%")
    if h["var_ratio"] < VAR_FAIL:
        bad.append(f"表征近乎常数 (变异比 {h['var_ratio']:.3f})")
    elif h["var_ratio"] < VAR_WARN:
        warn.append(f"变异比偏低 {h['var_ratio']:.3f}")
    if h["eff_rank"] < RANK_FAIL:
        bad.append(f"有效秩仅 {h['eff_rank']:.1f}/{h['max_rank']:.0f}")
    elif h["eff_rank"] < RANK_WARN:
        warn.append(f"有效秩 {h['eff_rank']:.1f}/{h['max_rank']:.0f}")
    if h["dead_dims"] > 0.5:
        warn.append(f"{h['dead_dims']*100:.0f}% 维度恒定")
    return ("FAIL" if bad else "WARN" if warn else "PASS"), (bad + warn)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hands-rest", choices=["hands-rest", "3class"])
    ap.add_argument("--window", type=float, default=3.0)
    args = ap.parse_args()

    import mne
    mne.set_log_level("CRITICAL")
    from explore_embeddings import build_epochs, DATASETS
    from foundation.bd_encoders import load_all

    ep, y, names, bad_ch = build_epochs(DATASETS[args.dataset])
    chans = [c for c in ep.ch_names if c not in bad_ch]
    print(f"\n{B}表征健康度门槛{X}  数据集 {args.dataset} · {len(ep)} trials · {len(chans)} 导入模")
    print(f"{'模型':12s} {'dim':>7} {'补零':>7} {'覆盖':>7} {'变异比':>8} {'有效秩':>12}  判定")
    print("-" * 78)
    encs, fail = load_all(chans, window_s=args.window)
    for nm, enc in encs.items():
        Xin = enc._prepare(ep, chans, (0.5, 0.5 + args.window))
        Z = enc.encode(ep, chans, tw=(0.5, 0.5 + args.window))
        m = enc.matched()
        h = health(Z, Xin, len(chans), n_matched=(m if isinstance(m, int) else len(chans)))
        st, why = verdict(h)
        col = {"PASS": G, "WARN": Y, "FAIL": R}[st]
        print(f"{nm:12s} {h['dim']:7d} {h.get('padding',0)*100:6.0f}% "
              f"{h.get('coverage',float('nan'))*100:6.0f}% {h['var_ratio']:8.3f} "
              f"{h['eff_rank']:6.1f}/{h['max_rank']:<5.0f} {col}{st}{X}"
              + (f"  {'; '.join(why)}" if why else ""))
    for nm, m in fail.items():
        print(f"{nm:12s} {R}加载失败{X} {m[:50]}")
    print(f"\n{B}判读{X}: FAIL 的模型其准确率数字\033[1m不应解读\033[0m —— 近乎常数或严重降维的表征"
          f"即使拿到高分也只是小样本上的噪声拟合。")
    print("阈值是在本帽子/本样本量上的启发式,作参考门槛而非对模型的定论。")


if __name__ == "__main__":
    main()
