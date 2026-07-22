# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Artifact handling — ICA blink removal, ICLabel auto-labelling, autoreject epoch repair.

A flag-driven pipeline so the same code powers batch analysis, the interactive review UI
(clean_ui.py), and the live de-blink in cap_gui. Every step is optional and reports what it
did, so you can compare "raw vs cleaned" and see exactly what each module removed.

Steps (in order):
    notch → band-pass → bad-channel detect+interpolate → CAR → ICA(remove eye/muscle/…)
then, at the epoch stage:  autoreject (repair/drop bad epochs)

Constraints for THIS cap: no dedicated EOG/EMG electrodes, so eye components are found
either by ICLabel (a trained classifier) or by correlation with the frontal channels
FP1/FP2 used as an EOG proxy.

  python src/analysis/artifacts.py recordings/xxx.npz            # full clean + report
  python src/analysis/artifacts.py --synth-blinks                # demo: inject+remove blinks
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # src/
sys.path.insert(0, str(HERE))                                 # src/analysis/
import load as loadmod                                         # noqa: E402
from load import detect_bad_channels                           # noqa: E402

EOG_PROXY = ("FP1", "FP2")
ICLABEL_KEEP = {"brain", "other"}                             # remove everything else (over prob)


@dataclass
class CleanFlags:
    notch: float = 50.0
    l_freq: float = 1.0
    h_freq: float = 40.0
    interp: bool = True         # bad-channel detect + spherical interpolation
    car: bool = True            # common average reference
    ica: bool = True            # ICA artifact removal
    ica_method: str = "iclabel" # "iclabel" | "eog"  (eog = FP1/FP2 correlation proxy)
    ica_prob: float = 0.80      # ICLabel confidence needed to drop a component


# ------------------------------------------------------------------- ICA core
def fit_ica(raw, n_components=None, seed=42):
    """Fit ICA suitable for ICLabel: extended-infomax on a 1 Hz-highpassed copy."""
    import mne
    fit_raw = raw.copy().filter(1.0, None, verbose="ERROR")   # highpass 1 Hz for stable ICA
    picks = mne.pick_types(fit_raw.info, eeg=True, exclude="bads")
    if n_components is None:
        n_components = min(20, len(picks) - 1)
    ica = mne.preprocessing.ICA(n_components=n_components, method="infomax",
                                fit_params=dict(extended=True), max_iter="auto",
                                random_state=seed, verbose="ERROR")
    ica.fit(fit_raw, picks=picks, verbose="ERROR")
    return ica


def label_ica(ica, raw, method="iclabel", eog_ch=EOG_PROXY, prob=0.80):
    """Return (labels, probs, exclude). ICLabel classifies every component; the EOG proxy
    only flags eye components by correlating with FP1/FP2."""
    n = ica.n_components_
    if method == "iclabel":
        from mne_icalabel import label_components
        raw_car = raw.copy().set_eeg_reference("average", verbose="ERROR")
        res = label_components(raw_car, ica, method="iclabel")
        labels = list(res["labels"])
        probs = np.asarray(res["y_pred_proba"]).ravel()
        exclude = [i for i in range(n)
                   if labels[i] not in ICLABEL_KEEP and probs[i] >= prob]
        return labels, probs, exclude
    # EOG proxy: correlate components with the frontal channels
    exclude, scoremap = [], np.zeros(n)
    for ch in eog_ch:
        if ch not in raw.ch_names:
            continue
        idx, scores = ica.find_bads_eog(raw, ch_name=ch, verbose="ERROR")
        exclude += idx
        scoremap = np.maximum(scoremap, np.abs(np.asarray(scores)[:n]))
    exclude = sorted(set(exclude))
    labels = ["eye (EOG proxy)" if i in exclude else "kept" for i in range(n)]
    return labels, scoremap, exclude


# ------------------------------------------------------------- raw-level clean
def preprocess(raw, flags: CleanFlags, verbose=False):
    """Apply the flag-driven pipeline. Returns (clean_raw, report)."""
    import mne
    report = {"interpolated": [], "ica": None}
    raw = raw.copy()
    if flags.notch:
        raw.notch_filter(flags.notch, verbose="ERROR")
    if flags.l_freq or flags.h_freq:
        raw.filter(flags.l_freq or None, flags.h_freq or None, verbose="ERROR")
    if flags.interp:
        bad = detect_bad_channels(raw)
        raw.info["bads"] = bad
        if bad and raw.get_montage() is not None:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        report["interpolated"] = bad
    if flags.car:
        raw.set_eeg_reference("average", verbose="ERROR")
    if flags.ica:
        try:
            ica = fit_ica(raw)
            labels, probs, exclude = label_ica(ica, raw, flags.ica_method,
                                               prob=flags.ica_prob)
            ica.exclude = exclude
            ica.apply(raw, verbose="ERROR")
            report["ica"] = dict(method=flags.ica_method, n_components=ica.n_components_,
                                 labels=labels, probs=[float(p) for p in np.ravel(probs)],
                                 exclude=exclude, removed=len(exclude))
            report["_ica_obj"] = ica
        except Exception as e:
            report["ica"] = dict(error=f"{type(e).__name__}: {e}")
    if verbose:
        print("  interpolated:", report["interpolated"])
        print("  ICA:", {k: v for k, v in (report["ica"] or {}).items() if k != "probs"})
    return raw, report


# --------------------------------------------------------- epoch-level (autoreject)
def clean_epochs(epochs, seed=11, verbose=False):
    """autoreject: cross-validated per-channel thresholds → repair or drop bad epochs.
    Returns (epochs_clean, reject_log). Epoch-based & offline — not a live operation."""
    from autoreject import AutoReject
    ar = AutoReject(random_state=seed, n_jobs=1, verbose=False)
    clean = ar.fit_transform(epochs, return_log=False)
    log = ar.get_reject_log(epochs)
    if verbose:
        print(f"  autoreject: {len(epochs)} → {len(clean)} epochs "
              f"({int(log.bad_epochs.sum())} dropped)")
    return clean, log


# --------------------------------------------------- live de-blink linear operator
def build_deblink_operator(ica, info):
    """Precompute the 32×32 sensor→sensor cleaning matrix M (and bias b) for a fitted ICA,
    so the live scope can de-blink a chunk with one matmul: x_clean ≈ M @ x + b.
    Derived by probing the (affine) ica.apply with impulses — exact & version-independent."""
    import mne
    nch = info["nchan"]
    probe = np.hstack([np.zeros((nch, 1)), np.eye(nch)]) * 1e-6   # col0 = zero (bias)
    r = mne.io.RawArray(probe, info, verbose="ERROR")
    ica.apply(r, verbose="ERROR")
    out = r.get_data()                                            # (nch, nch+1) volts
    b = out[:, 0]
    M = (out[:, 1:] - b[:, None]) / 1e-6                          # unit response (dimensionless)
    return M.astype(np.float32), (b / 1e-6).astype(np.float32)


class LiveDeblink:
    """Fit ICA+labels once on a calibration buffer, then apply M @ chunk online."""

    def __init__(self, M, b, info, report):
        self.M, self.b, self.info, self.report = M, b, info, report

    @classmethod
    def calibrate(cls, buf_uv, fs, ch_names, method="iclabel", prob=0.80):
        import mne
        info = mne.create_info(list(ch_names), fs, "eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         match_case=False, on_missing="ignore", verbose="ERROR")
        raw = mne.io.RawArray(buf_uv * 1e-6, info, verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")
        ica = fit_ica(raw)
        labels, probs, exclude = label_ica(ica, raw, method, prob=prob)
        ica.exclude = exclude
        M, b = build_deblink_operator(ica, info)
        rep = dict(method=method, n_components=ica.n_components_, labels=labels,
                   exclude=exclude, removed=len(exclude))
        return cls(M, b, info, rep)

    def apply(self, chunk_uv):                                    # (nch, m) µV -> cleaned
        return self.M @ chunk_uv + self.b[:, None]


# --------------------------------------------------------------------- CLI / demo
def _synth_blinks_demo():
    """Generate a synthetic MI recording WITH blink artifacts, clean it, and report that
    ICA removed the blinks while the C3/C4 mu-ERD survived."""
    sys.path.insert(0, str(HERE))
    from erd_ers import synth_mi_recording, _band_timecourse, _tfr, MU, IMAGERY
    path = synth_mi_recording(reps=12, out=HERE.parents[1] / "recordings" / "synth_mi_blinks.npz",
                              blinks=True)
    raw, events, event_id = loadmod.read_recording(path)
    flags = CleanFlags(ica=True, ica_method="iclabel")
    clean, report = preprocess(raw, flags, verbose=True)
    fp1 = raw.ch_names.index("FP1")
    print(f"  FP1 std: raw {raw.get_data()[fp1].std()*1e6:.1f} µV → clean "
          f"{clean.get_data()[fp1].std()*1e6:.1f} µV  (blink power should drop)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--synth-blinks", action="store_true", help="inject+remove blinks demo")
    ap.add_argument("--method", default="iclabel", choices=["iclabel", "eog"])
    ap.add_argument("--epochs", action="store_true", help="also run autoreject on epochs")
    args = ap.parse_args()
    if args.synth_blinks:
        _synth_blinks_demo(); return
    if not args.path:
        ap.error("give a recording path or use --synth-blinks")
    raw, events, event_id = loadmod.read_recording(args.path)
    clean, report = preprocess(raw, CleanFlags(ica_method=args.method), verbose=True)
    if args.epochs and len(events):
        import mne
        ep = mne.Epochs(clean, events, event_id, tmin=loadmod.DEFAULT_TMIN,
                        tmax=loadmod.DEFAULT_TMAX, baseline=None, preload=True, verbose="ERROR")
        clean_epochs(ep, verbose=True)


if __name__ == "__main__":
    main()
