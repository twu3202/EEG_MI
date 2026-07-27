#!/usr/bin/env python
"""Frozen-embedding wrappers for the pretrained EEG foundation models in braindecode 1.6.

Each checkpoint was pretrained on its own montage and sampling rate, so a 32-channel dry
cap cannot be fed in directly. This module handles the two mismatches uniformly:

  * CHANNELS. Models whose checkpoint carries a named montage (EEGPT 62 ch, LaBraM 128 ch,
    SignalJEPA 62 ch) get a zero-filled array of THEIR width with our electrodes dropped
    into the matching names — no interpolation, no invented signal, and unmatched inputs
    stay at zero. Models that only fix a channel COUNT (BIOT 18, BENDR 20) go through
    braindecode's Interpolated* wrapper, which spline-interpolates our montage onto theirs.
  * RATE / LENGTH. Data is resampled to the checkpoint's sfreq and cropped/padded to its
    n_times.

The classification head is replaced by Identity, so `encode()` returns the frozen
penultimate representation — the thing a linear probe is supposed to read.

  python src/foundation/bd_encoders.py --list          # which checkpoints load
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
HF_HOME = ROOT / "checkpoints" / "hf"

# name -> (hf repo, braindecode class, interpolated class or None, n_times, sfreq)
SPECS = {
    "EEGPT":      ("braindecode/eegpt-pretrained", "EEGPT", None, 1000, 250.0),
    "LaBraM-bd":  ("braindecode/labram-pretrained", "Labram", None, 3000, 200.0),
    "SignalJEPA": ("braindecode/signal-jepa", "SignalJEPA", None, 1000, 128.0),
    "BIOT":       ("braindecode/biot-pretrained-six-datasets-18chs", "BIOT",
                   "InterpolatedBIOT", 1000, 200.0),
    "BENDR":      ("braindecode/braindecode-bendr", "BENDR", "InterpolatedBENDR", 1000, 250.0),
}


def _set_hf_home():
    import os
    os.environ.setdefault("HF_HOME", str(HF_HOME))


def _ckpt_channels(repo):
    """Channel names the checkpoint was pretrained on (None if it only fixes a count)."""
    import json
    for cfg in (HF_HOME / "hub").glob(f"models--{repo.replace('/', '--')}/snapshots/*/config.json"):
        c = json.load(open(cfg))
        ci = c.get("chs_info")
        if isinstance(ci, list) and ci and isinstance(ci[0], dict):
            return [str(x.get("ch_name", "")) for x in ci], c   # keep ORIGINAL case:
            # EEGPT looks channel names up in an internal vocabulary; upper-casing them
            # silently drops most matches and the chans_id buffer shrinks 62 -> 19.
        return None, c
    return None, {}


class BDEncoder:
    """Load one pretrained braindecode model and expose frozen embeddings."""

    def __init__(self, name, our_channels, device=None):
        import torch, mne, braindecode.models as M
        _set_hf_home()
        mne.set_log_level("CRITICAL")
        repo, cls_name, interp_name, n_times, sfreq = SPECS[name]
        self.name, self.repo = name, repo
        self.n_times, self.sfreq = n_times, sfreq
        self.our = [c.upper() for c in our_channels]
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        ck_ch, cfg = _ckpt_channels(repo)
        if ck_ch:                                   # named montage → place by name
            self.target = ck_ch
            self.idx = [(self.target.index(c), i) for i, c in enumerate(self.our)
                        if c in self.target]
            info = mne.create_info(list(self.target), sfreq, "eeg")
            info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                             match_case=False, on_missing="ignore")
            if cls_name == "EEGPT":
                # EEGPT's __init__ falls back to a default 19-channel list instead of using
                # the chs_info we pass, so `chans_id` comes out (1,19) and the 62-channel
                # checkpoint refuses to load. Recompute the buffer from the checkpoint's own
                # channel names (prepare_chan_ids maps them through EEGPT's CHANNEL_DICT)
                # and then load the weights.
                import glob as _g
                from safetensors.torch import load_file
                # chan_proj_type defaults to a mode that FORCES EEGPT's standard 19-channel
                # projection and ignores chs_info; the 62-channel checkpoint was saved with
                # "none", so ask for that explicitly.
                model = M.EEGPT(chs_info=info["chs"], n_times=n_times, sfreq=sfreq,
                                n_outputs=2, chan_proj_type="none")
                sd = load_file(_g.glob(str(HF_HOME / "hub" /
                    f"models--{repo.replace('/', '--')}" / "snapshots/*/model.safetensors"))[0])
                missing, unexpected = model.load_state_dict(sd, strict=False)
                self.load_note = f"missing={len(missing)} unexpected={len(unexpected)}"
            else:
                model = getattr(M, cls_name).from_pretrained(
                    repo, chs_info=info["chs"], n_times=n_times, sfreq=sfreq, n_outputs=2)
            self.mode = "name-match"
        elif cls_name == "BENDR":                   # fixed 20-channel order incl. "SCALE"
            from braindecode.models.bendr import BENDR_CHANNEL_ORDER as ORDER
            self.target = list(ORDER)
            alias = {"T5": "P7", "T6": "P8"}         # BENDR uses the pre-1991 names
            self.idx = []
            for k, c in enumerate(self.target):
                src = alias.get(c, c)
                if src in self.our:
                    self.idx.append((k, self.our.index(src)))
            info = mne.create_info(list(self.target), sfreq, "eeg")
            model = getattr(M, cls_name).from_pretrained(repo)
            self.mode = "name-match"
        else:                                       # count-only → braindecode interpolation
            info = mne.create_info(list(our_channels), sfreq, "eeg")
            info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                             match_case=False, on_missing="ignore")
            model = getattr(M, interp_name).from_pretrained(
                repo, chs_info=info["chs"], n_times=n_times, sfreq=sfreq, n_outputs=2)
            self.target, self.idx, self.mode = None, None, "interpolated"

        # strip the classification head so we read the frozen representation
        for attr in ("final_layer", "classifier", "head", "fc"):
            if hasattr(model, attr):
                setattr(model, attr, torch.nn.Identity()); break
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def _prepare(self, ep, chans, tw):
        """Epochs → (n, n_ch_model, n_times) at the checkpoint's rate."""
        e = ep.copy().pick(chans).filter(0.3, min(45.0, self.sfreq / 2 * 0.9))
        e = e.resample(self.sfreq).crop(*tw)
        X = e.get_data(copy=False) * 1e6                       # µV
        n = self.n_times
        X = X[..., :n] if X.shape[-1] >= n else np.pad(X, ((0, 0), (0, 0), (0, n - X.shape[-1])))
        if self.mode == "interpolated":
            return X.astype(np.float32)
        Y = np.zeros((X.shape[0], len(self.target), n), np.float32)   # unmatched stay 0
        up = [c.upper() for c in e.ch_names]
        tgt_up = [c.upper() for c in self.target]
        alias = {"T5": "P7", "T6": "P8"}
        for k, c in enumerate(tgt_up):
            src = alias.get(c, c)
            if src in up:
                Y[:, k] = X[:, up.index(src)]
        return Y

    def encode(self, ep, chans, tw=(0.5, 3.5), batch=16, scale=0.01):
        import torch
        X = self._prepare(ep, chans, tw) * scale               # models expect µV/100
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                t = torch.from_numpy(X[i:i + batch]).float().to(self.device)
                z = self.model(t)
                out.append(z.reshape(len(t), -1).cpu().numpy())
        return np.concatenate(out, 0)

    def matched(self):
        return len(self.idx) if self.idx is not None else "interp"


def load_all(our_channels, only=None):
    ok, fail = {}, {}
    for name in (only or SPECS):
        try:
            ok[name] = BDEncoder(name, our_channels)
        except Exception as e:
            fail[name] = f"{type(e).__name__}: {str(e)[:80]}"
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from common.montage import CAP32_CHANNELS
    ok, fail = load_all(CAP32_CHANNELS)
    print(f"{'model':12s} {'params':>9}  {'mode':<13}{'matched ch':>11}  {'input':>16}")
    for n, e in ok.items():
        print(f"{n:12s} {e.n_params/1e6:8.1f}M  {e.mode:<13}{str(e.matched()):>11}  "
              f"(32,{e.n_times})@{e.sfreq:.0f}Hz")
    for n, m in fail.items():
        print(f"{n:12s} ❌ {m}")


if __name__ == "__main__":
    main()
