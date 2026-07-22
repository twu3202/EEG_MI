# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
"""Synthetic 32-ch EEG source — lets us test the viewer and pipeline with no hardware.

Produces plausible-looking µV data: posterior alpha (~10 Hz), central mu, pink-ish
background, frontal eye-blinks, a bit of 50 Hz line noise, and one deliberately bad
channel so the quality panel has something to show.
"""
from __future__ import annotations

import numpy as np

from montage import CAP32_CHANNELS

_POSTERIOR = {"O1", "O2", "OZ", "PO3", "PO4", "P3", "P4", "P7", "P8", "PZ"}
_CENTRAL = {"C3", "C4", "CZ", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6"}
_FRONTAL = {"FP1", "FP2", "AF3", "AF4"}
_BAD_CH = None  # set to a channel name (e.g. "T7") to simulate a poor-contact electrode


class SynthCap:
    def __init__(self, sfreq: float = 250.0, seed: int = 7):
        self.sfreq = sfreq
        self.ch = CAP32_CHANNELS
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        # per-channel pink-noise state
        self._b = np.zeros(len(self.ch))

    def get_chunk(self, n: int) -> np.ndarray:
        """Return (32, n) µV."""
        fs = self.sfreq
        idx = np.arange(n)
        t = self.t + idx / fs
        out = np.zeros((len(self.ch), n))
        for i, name in enumerate(self.ch):
            # pink-ish background via leaky-integrated white noise (~8 µV)
            w = self.rng.standard_normal(n)
            b = np.empty(n)
            prev = self._b[i]
            for k in range(n):
                prev = 0.97 * prev + 0.3 * w[k]
                b[k] = prev
            self._b[i] = prev
            sig = 8.0 * b
            # rhythms
            if name in _POSTERIOR:
                sig += 18.0 * np.sin(2 * np.pi * 10.0 * t + i)      # strong alpha
            elif name in _CENTRAL:
                sig += 9.0 * np.sin(2 * np.pi * 11.0 * t + i * 0.5)  # mu
            elif name in _FRONTAL:
                sig += 5.0 * np.sin(2 * np.pi * 9.5 * t + i)
            # 50 Hz line noise (small)
            sig += 1.5 * np.sin(2 * np.pi * 50.0 * t)
            # occasional frontal blink
            if name in _FRONTAL and self.rng.random() < 0.02:
                sig += 80.0 * np.exp(-((idx - self.rng.integers(0, n)) ** 2) / (2 * (0.05 * fs) ** 2))
            # bad channel: railed / very noisy
            if name == _BAD_CH:
                sig = 60.0 * self.rng.standard_normal(n) + 30.0 * np.sin(2 * np.pi * 50.0 * t)
            out[i] = sig
        self.t += n / fs
        return out.astype(np.float32)
