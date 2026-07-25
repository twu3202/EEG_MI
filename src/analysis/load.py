#!/usr/bin/env python
"""Load a cap recording and cut it into labelled MI epochs.

This is the "录制 → 分析加载, 按 trigger 切 epoch" link. A recording saved by cap_gui is
either a `.npz` (raw µV + a per-sample label track) or an MNE `.fif` (with the labels as
annotations). Here we:

    recording ──► MNE Raw (µV→V, montage) ──► filter (notch + band-pass)
              ──► bad-channel detect + interpolate     (simple artifact handling)
              ──► events from the label track's rising edges
              ──► Epochs, one per imagery onset, baseline-corrected

The label track is the software `marker` (small MI codes 1–5, see common.mi_events) when
present, else the hardware `trigger` column. Heavier artifact handling (ICA / autoreject /
EOG-EMG regression) is deliberately NOT here yet — see analysis/artifacts.py / the plan.

CLI:
  python src/analysis/load.py recordings/cap32_*.npz              # summary + event counts
  python src/analysis/load.py recordings/cap32_*.npz --epochs     # build epochs, print shape
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # src/
from common.montage import CAP32_CHANNELS                      # noqa: E402
from common.mi_events import CODE_TO_LABEL                      # noqa: E402

DEFAULT_TMIN, DEFAULT_TMAX = -2.0, 4.0
DEFAULT_BASELINE = (-1.5, -0.5)      # pre-imagery window used as the ERD/ERS reference


# --------------------------------------------------------------------- raw npz
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    data = z["data"].astype(np.float64)                        # (n_ch, N) µV
    fs = float(z["fs"])
    ch = [str(c) for c in z["ch_names"]] if "ch_names" in z else list(CAP32_CHANNELS)
    trigger = z["trigger"].astype(np.int64) if "trigger" in z else np.zeros(data.shape[1], int)
    marker = z["marker"].astype(np.int64) if "marker" in z else None
    gap = z["gap"].astype(np.int8) if "gap" in z else None      # 1 = reconstructed sample
    rec = dict(data=data, fs=fs, ch_names=ch, trigger=trigger, marker=marker, gap=gap)
    # format v2: explicit trial table + metadata written by the paradigm
    if "trial_onset" in z:
        rec["trials"] = dict(onset=z["trial_onset"].astype(np.int64),
                             code=z["trial_code"].astype(np.int64),
                             name=[str(s) for s in z["trial_name"]],
                             cue=z["trial_cue_onset"].astype(np.int64)
                             if "trial_cue_onset" in z else None)
    if "meta_json" in z:
        import json
        try:
            rec["meta"] = json.loads(str(z["meta_json"]))
        except Exception:
            pass
    return rec


def events_from_trials(trials):
    """Exact events from the paradigm's trial table (preferred over edge detection)."""
    on = np.asarray(trials["onset"]); code = np.asarray(trials["code"])
    keep = on >= 0
    on, code = on[keep], code[keep]
    events = np.column_stack([on, np.zeros_like(on), code]).astype(int)
    event_id = {}
    for c, n in zip(code.tolist(), np.asarray(trials["name"])[keep].tolist()):
        event_id.setdefault(str(n), int(c))
    return events, event_id


def label_track(rec):
    """Pick the label track: software `marker` if it carries any event, else hardware."""
    m = rec.get("marker")
    if m is not None and np.any(m != 0):
        return m, "marker"
    return rec["trigger"], "trigger"


# --------------------------------------------------------------- events / raw
def events_from_track(track, fs=None):
    """Rising edges of a per-sample integer label -> MNE events (n,3) + event_id.
    An event starts where the label changes to a nonzero value."""
    track = np.asarray(track)
    onsets = np.where((track[1:] != track[:-1]) & (track[1:] != 0))[0] + 1
    codes = track[onsets].astype(int)
    events = np.column_stack([onsets, np.zeros_like(onsets), codes]).astype(int)
    present = sorted(set(codes.tolist()))
    event_id = {CODE_TO_LABEL.get(c, f"code{c}"): int(c) for c in present}
    return events, event_id


def to_raw(rec, montage=True):
    import mne
    info = mne.create_info(list(rec["ch_names"]), rec["fs"], "eeg")
    raw = mne.io.RawArray(rec["data"] * 1e-6, info, verbose="ERROR")   # µV -> V
    if montage:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        match_case=False, on_missing="ignore", verbose="ERROR")
    return raw


def read_recording(path):
    """Unified reader → (raw, events, event_id). Accepts .npz or .fif."""
    import mne
    path = str(path)
    if path.endswith(".fif"):
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        try:
            events, ann_id = mne.events_from_annotations(raw, verbose="ERROR")
            # map annotation descriptions ("left"/"right"/"T2"/…) back to MI codes
            from common.mi_events import MI_TASK_CODES
            event_id = {k: v for k, v in ann_id.items()}
            return raw, events, event_id
        except Exception:
            return raw, np.empty((0, 3), int), {}
    rec = load_npz(path)
    raw = to_raw(rec)
    if rec.get("trials") is not None and np.any(np.asarray(rec["trials"]["onset"]) >= 0):
        events, event_id = events_from_trials(rec["trials"])   # v2: exact, from the paradigm
    else:
        track, _src = label_track(rec)
        events, event_id = events_from_track(track, rec["fs"])
    return raw, events, event_id


# ------------------------------------------------------- simple artifact steps
def detect_bad_channels(raw, flat_uv=0.5, noisy_uv=150.0, rail_uv=1.5e5, hf_z=6.0):
    """Conservative bad-channel flags (no ICA) using ABSOLUTE thresholds so real signal is
    never touched — a channel with more mu/beta is NOT bad. Flags only:
      • flat / dead  (std < flat_uv),
      • railed       (peak-to-peak > rail_uv ≈ 150 mV, ADS1299 saturated),
      • grossly noisy (std > noisy_uv µV — non-physiological for scalp EEG),
      • high-frequency junk (sample-to-sample jitter a strong outlier vs the array).
    Returns a list of channel names. Deliberately misses the "slightly noisy posterior"
    case — that is left to ICA/autoreject, which we add separately."""
    x = raw.get_data() * 1e6                                   # V -> µV
    sd = x.std(axis=1)
    ptp = np.ptp(x, axis=1)
    hf = np.std(np.diff(x, axis=1), axis=1)                    # high-freq content proxy
    bad = set()
    for i, name in enumerate(raw.ch_names):
        if sd[i] < flat_uv or ptp[i] > rail_uv or sd[i] > noisy_uv:
            bad.add(name)
    # high-freq outlier: broken electrodes jitter far more than the median channel
    keep = np.array([raw.ch_names[i] not in bad for i in range(len(hf))])
    if keep.sum() >= 6:
        med = np.median(hf[keep]); mad = np.median(np.abs(hf[keep] - med)) + 1e-9
        for i, name in enumerate(raw.ch_names):
            if name not in bad and (hf[i] - med) / (1.4826 * mad) > hf_z and hf[i] > 40.0:
                bad.add(name)
    return sorted(bad)


def clean_raw(raw, l_freq=1.0, h_freq=40.0, notch=50.0, car=False,
              interpolate=True, verbose=False):
    """Filter + (optional) notch + bad-channel interpolation. Returns a *copy*.
    This is the light, dependency-free cleaning; ICA/autoreject live elsewhere."""
    raw = raw.copy()
    if notch:
        raw.notch_filter(notch, verbose="ERROR")
    raw.filter(l_freq, h_freq, verbose="ERROR")
    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    if verbose and bad:
        print(f"  bad channels: {bad}")
    if interpolate and bad and raw.get_montage() is not None:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    if car:
        raw.set_eeg_reference("average", verbose="ERROR")
    return raw, bad


# --------------------------------------------------------------------- epochs
def make_epochs(path, tmin=DEFAULT_TMIN, tmax=DEFAULT_TMAX, baseline=DEFAULT_BASELINE,
                picks=None, l_freq=1.0, h_freq=40.0, notch=50.0, car=False,
                interpolate=True, reject_uv=None, verbose=True, drop_filled=False):
    """Recording path → cleaned, baseline-corrected MI Epochs (labelled by task name)."""
    import mne
    raw, events, event_id = read_recording(path)
    if len(events) == 0:
        raise SystemExit(f"no events/markers found in {path} — was a paradigm run recorded?")
    if drop_filled and str(path).endswith(".npz"):
        gap = load_npz(path).get("gap")
        if gap is not None and gap.any():        # drop trials overlapping reconstructed data
            fs0 = raw.info["sfreq"]
            lo, hi = int(tmin * fs0), int(tmax * fs0)
            keep = [not gap[max(0, o + lo): o + hi].any() for o in events[:, 0]]
            n_drop = len(keep) - sum(keep)
            events = events[np.array(keep, bool)]
            if verbose and n_drop:
                print(f"  dropped {n_drop} epoch(s) containing UDP-gap-filled samples")
    raw, bad = clean_raw(raw, l_freq, h_freq, notch, car, interpolate, verbose)
    reject = dict(eeg=reject_uv * 1e-6) if reject_uv else None
    ep = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=baseline, picks=picks, preload=True, reject=reject,
                    reject_by_annotation=False, verbose="ERROR")
    if verbose:
        counts = {k: int((ep.events[:, 2] == v).sum()) for k, v in ep.event_id.items()}
        print(f"epochs: {len(ep)} × {len(ep.ch_names)}ch × {ep.times.size} samp "
              f"[{tmin}, {tmax}]s @ {raw.info['sfreq']:.0f}Hz  ·  {counts}"
              + (f"  ·  interpolated {bad}" if bad else ""))
    return ep


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--epochs", action="store_true", help="build epochs (not just summarise)")
    ap.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    ap.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    ap.add_argument("--no-interp", action="store_true")
    args = ap.parse_args()

    raw, events, event_id = read_recording(args.path)
    print(f"{args.path}\n  {len(raw.ch_names)} ch @ {raw.info['sfreq']:.0f} Hz, "
          f"{raw.n_times} samples ({raw.times[-1]:.1f}s)")
    print(f"  events: {len(events)}  event_id: {event_id}")
    if events.size:
        for name, code in event_id.items():
            print(f"    {name:8s} (code {code}): {(events[:,2]==code).sum()} trials")
    if args.epochs:
        make_epochs(args.path, tmin=args.tmin, tmax=args.tmax,
                    interpolate=not args.no_interp)


if __name__ == "__main__":
    main()
