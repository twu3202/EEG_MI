#!/usr/bin/env python
"""ERD/ERS validation — the make-or-break sanity check for MI on this cap.

Motor imagery shows up as **ERD** (event-related desynchronisation = mu/beta power DROP)
over the hemisphere *contralateral* to the imagined hand, followed by a beta **ERS**
(rebound = power increase) after the movement. Left-hand imagery → ERD over the RIGHT
sensorimotor cortex (C4); right-hand imagery → ERD over the LEFT (C3). If we can see that
crossover in your own recording, the whole MI pipeline is real; if not, decoding won't work.

This produces three figures from a recording made by cap_gui's MI paradigm:
  1. erd_ers_maps.png       — time-frequency ERDS maps at C3 & C4, per class
  2. erd_ers_timecourse.png — mu-band (8–13 Hz) ERD% over time, C3 vs C4 (the crossover)
  3. erd_ers_topo.png       — scalp topography of mu-ERD (the "separate ERS validation" fig)

  # analyse a real recording (left/right MI):
  python src/analysis/erd_ers.py recordings/cap32_YYYYMMDD_HHMMSS.npz
  # no hardware — generate a synthetic MI recording with known contralateral ERD, then analyse:
  python src/analysis/erd_ers.py --synth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # src/
from common.montage import CAP32_CHANNELS                      # noqa: E402
from common.mi_events import MI_TASK_CODES                      # noqa: E402
import load as loadmod                                          # noqa: E402  (src/analysis/load.py)

FREQS = np.arange(4.0, 35.0, 1.0)
MU = (8.0, 13.0)
BETA = (13.0, 30.0)
IMAGERY = (0.5, 3.5)          # analysis window inside the 0–4 s imagery phase
RESULTS = HERE.parents[1] / "results"


# ============================================================ synthetic recording
def _hemisphere(name):
    d = name[-1]
    if d.isdigit():
        return "L" if int(d) % 2 == 1 else "R"
    return "M"                                                 # z-line = midline


# frontal-dominant weights for synthetic eye-blink topography (anterior gradient)
_FRONT_W = {"FP1": 1.0, "FP2": 1.0, "AF3": 0.85, "AF4": 0.85, "F7": 0.7, "F8": 0.7,
            "F3": 0.62, "F4": 0.62, "FZ": 0.6, "FC5": 0.42, "FC6": 0.42, "FC1": 0.36,
            "FC2": 0.36, "T7": 0.3, "T8": 0.3}


def synth_mi_recording(fs=250, reps=15, seed=1, out=None, blinks=False):
    """Continuous (32,N) µV recording + marker track with realistic contralateral mu ERD.
    Saved in cap_gui's .npz format so load.py reads it exactly like a real one.
    With blinks=True, adds frontal-dominant eye-blink artifacts (to test ICA removal)."""
    sys.path.insert(0, str(HERE.parents[1] / "src" / "experiment"))
    from mi_paradigm import make_sequence, Timing
    rng = np.random.default_rng(seed)
    ch = CAP32_CHANNELS
    nch = len(ch)
    T = Timing()
    seq = make_sequence(["left", "right"], reps, seed)
    hemi = np.array([_hemisphere(c) for c in ch])
    # per-channel resting mu/beta amplitude — strong over sensorimotor, weak elsewhere
    sm = np.array([c in ("C3", "C4", "CZ", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6")
                   for c in ch])
    a_mu = np.where(sm, 9.0, 3.0)
    a_beta = np.where(sm, 4.0, 1.5)

    chunks, marks = [], []
    n_fix, n_cue = int(T.fixation * fs), int(T.cue * fs)
    n_img, n_rest = int(T.imagery * fs), int(T.rest * fs)
    for name in seq:
        code = MI_TASK_CODES[name]
        contra = "R" if name == "left" else "L"               # ERD hemisphere
        n = n_fix + n_cue + n_img + n_rest
        t = np.arange(n) / fs
        # mu-power envelope per channel: 1.0 everywhere, drop to 0.35 on contralateral during imagery
        env = np.ones((nch, n))
        img0 = n_fix + n_cue
        ramp = np.clip((np.arange(n) - img0) / (0.3 * fs), 0, 1) * \
               np.clip((img0 + n_img - np.arange(n)) / (0.3 * fs), 0, 1)
        for c in range(nch):
            if hemi[c] == contra and sm[c]:
                env[c] = 1.0 - 0.65 * ramp                      # ERD: up to -65 % mu power
        sig = np.empty((nch, n))
        for c in range(nch):
            ph_m, ph_b = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            pink = np.cumsum(rng.normal(0, 1, n)); pink -= pink.mean(); pink *= 6.0 / (pink.std() + 1e-9)
            sig[c] = (env[c] * a_mu[c] * np.sin(2 * np.pi * 10.0 * t + ph_m)
                      + env[c] * a_beta[c] * np.sin(2 * np.pi * 20.0 * t + ph_b)
                      + rng.normal(0, 4.0, n) + pink)
        mk = np.zeros(n, int); mk[img0:img0 + n_img] = code
        chunks.append(sig.astype(np.float32)); marks.append(mk)
    data = np.concatenate(chunks, axis=1)
    marker = np.concatenate(marks)
    trigger = np.zeros_like(marker)
    if blinks:
        N = data.shape[1]
        w = np.array([_FRONT_W.get(c, 0.08) for c in ch])
        tall = np.arange(N) / fs
        for _ in range(int(N / fs / 3.5)):                     # a blink every ~3.5 s
            tc = rng.uniform(0.5, N / fs - 0.5)
            amp = rng.uniform(150, 320)
            shape = np.exp(-0.5 * ((tall - tc) / 0.09) ** 2)   # ~0.2 s deflection
            data += (w[:, None] * amp * shape[None, :]).astype(np.float32)
    out = out or (HERE.parents[1] / "recordings" / "synth_mi.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=data, trigger=trigger, marker=marker, fs=float(fs),
                        ch_names=np.array(ch))
    print(f"synthetic recording: {data.shape[1]} samples ({data.shape[1]/fs:.0f}s), "
          f"{len(seq)} trials → {out}")
    return str(out)


# ============================================================ ERDS computation
def _tfr(epochs, decim=2):
    """Per-condition baseline-normalised ERDS (percent). Returns {label: AverageTFR}."""
    out = {}
    for label in epochs.event_id:
        if not len(epochs[label]):
            continue
        tfr = epochs[label].compute_tfr("morlet", FREQS, n_cycles=FREQS / 2.0,
                                        average=True, return_itc=False, decim=decim,
                                        verbose="ERROR")
        tfr.apply_baseline(loadmod.DEFAULT_BASELINE, mode="percent", verbose="ERROR")
        out[label] = tfr                                       # data in fraction; ×100 = ERD/ERS %
    return out


def _band_timecourse(tfr, ch, band):
    """mu- or beta-band mean ERD% time-course for one channel from an ERDS TFR."""
    ci = tfr.ch_names.index(ch)
    fmask = (tfr.freqs >= band[0]) & (tfr.freqs <= band[1])
    return tfr.times, tfr.data[ci][fmask].mean(0) * 100.0


# ============================================================ figures
def fig_maps(tfrs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    labels = [l for l in ("left", "right") if l in tfrs] or list(tfrs)
    chans = ["C3", "C4"]
    fig, axes = plt.subplots(len(labels), 2, figsize=(10, 3.1 * len(labels)), squeeze=False)
    fig.patch.set_facecolor("white")
    norm = TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)
    im = None
    for r, lab in enumerate(labels):
        tfr = tfrs[lab]
        for c, chn in enumerate(chans):
            ax = axes[r][c]
            ci = tfr.ch_names.index(chn)
            im = ax.pcolormesh(tfr.times, tfr.freqs, tfr.data[ci] * 100.0,
                               cmap="RdBu_r", norm=norm, shading="gouraud")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            for b in (MU, BETA):
                ax.axhline(b[0], color="0.5", lw=0.4); ax.axhline(b[1], color="0.5", lw=0.4)
            contra = (lab == "left" and chn == "C4") or (lab == "right" and chn == "C3")
            ax.set_title(f"{lab}-hand  ·  {chn}" + ("   ← contralateral (expect ERD)" if contra else ""),
                         fontsize=10, color="#b23b3b" if contra else "#333")
            if c == 0:
                ax.set_ylabel("Hz")
            if r == len(labels) - 1:
                ax.set_xlabel("time from imagery onset (s)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("power change vs baseline (%)   —   blue = ERD (↓) · red = ERS (↑)")
    fig.suptitle("ERD/ERS time-frequency maps at C3 / C4", fontsize=13, y=0.99)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


def fig_timecourse(tfrs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [l for l in ("left", "right") if l in tfrs] or list(tfrs)
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4), squeeze=False)
    fig.patch.set_facecolor("white")
    for i, lab in enumerate(labels):
        ax = axes[0][i]; tfr = tfrs[lab]
        for chn, col in (("C3", "#2b6cb0"), ("C4", "#c0392b")):
            t, y = _band_timecourse(tfr, chn, MU)
            ax.plot(t, y, color=col, lw=2, label=chn)
        ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvspan(*IMAGERY, color="0.9", zorder=0)
        contra = "C4" if lab == "left" else "C3"
        ax.set_title(f"{lab}-hand imagery  ·  mu(8–13Hz)\nexpect deeper ERD at {contra}", fontsize=10)
        ax.set_xlabel("time from imagery onset (s)")
        if i == 0:
            ax.set_ylabel("mu power change vs baseline (%)")
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Mu-ERD time course — the contralateral crossover", fontsize=12, y=1.02)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


def fig_topo(tfrs, info, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    labels = [l for l in ("left", "right") if l in tfrs] or list(tfrs)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.2), squeeze=False)
    fig.patch.set_facecolor("white")
    im = None
    for i, lab in enumerate(labels):
        tfr = tfrs[lab]
        fmask = (tfr.freqs >= MU[0]) & (tfr.freqs <= MU[1])
        tmask = (tfr.times >= IMAGERY[0]) & (tfr.times <= IMAGERY[1])
        val = tfr.data[:, fmask][:, :, tmask].mean(axis=(1, 2)) * 100.0     # per-channel mu ERD%
        picks = [info.ch_names.index(c) for c in tfr.ch_names]
        im, _ = mne.viz.plot_topomap(val, mne.pick_info(info, picks), axes=axes[0][i],
                                     cmap="RdBu_r", vlim=(-60, 60), show=False, contours=4)
        axes[0][i].set_title(f"{lab}-hand  ·  mu ERD", fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.set_label("mu power change (%)  ·  blue = ERD")
    fig.suptitle("Scalp topography of mu-ERD during imagery", fontsize=12, y=1.02)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


# ============================================================ driver
def analyse(path, outdir=RESULTS):
    ep = loadmod.make_epochs(path, tmin=loadmod.DEFAULT_TMIN, tmax=loadmod.DEFAULT_TMAX,
                             baseline=None, l_freq=1.0, h_freq=40.0, notch=50.0)
    ep = ep.pick("eeg")
    tfrs = _tfr(ep)
    if not tfrs:
        raise SystemExit("no conditions with epochs to analyse")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig_maps(tfrs, outdir / "erd_ers_maps.png")
    fig_timecourse(tfrs, outdir / "erd_ers_timecourse.png")
    fig_topo(tfrs, ep.info, outdir / "erd_ers_topo.png")
    # one-line quantitative check
    for lab in tfrs:
        contra = "C4" if lab == "left" else "C3"
        ipsi = "C3" if lab == "left" else "C4"
        _, yc = _band_timecourse(tfrs[lab], contra, MU)
        _, yi = _band_timecourse(tfrs[lab], ipsi, MU)
        tt = tfrs[lab].times
        w = (tt >= IMAGERY[0]) & (tt <= IMAGERY[1])
        print(f"  {lab}-hand: mu ERD  contra {contra} {yc[w].mean():+.0f}%   "
              f"ipsi {ipsi} {yi[w].mean():+.0f}%   (contra should be more negative)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="recording .npz/.fif (omit with --synth)")
    ap.add_argument("--synth", action="store_true", help="generate + analyse a synthetic MI recording")
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    path = synth_mi_recording(reps=args.reps) if args.synth else args.path
    if not path:
        ap.error("give a recording path or use --synth")
    analyse(path)


if __name__ == "__main__":
    main()
