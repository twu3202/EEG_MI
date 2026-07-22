"""Real-time streaming EEG filter — mirrors the vendor app's RealTimeEEGFilter.

Butterworth band-pass/low-pass (order 4) + optional 50 Hz iir-notch, applied with
stateful `lfilter` (per-channel `zi`) so it works causally on live chunks. Plus
per-channel baseline (DC/drift) removal via a slow running mean.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi


class RealTimeEEGFilter:
    def __init__(self, fs, n_ch, lowcut=0.0, highcut=40.0,
                 notch_freq=50.0, notch_q=30.0, order=4, baseline=True):
        self.n_ch = n_ch
        self.baseline = baseline
        self._mean = np.zeros(n_ch)          # running mean per channel (drift removal)
        self._mean_a = 0.001                 # running-mean adaptation rate
        self.update(fs, lowcut, highcut, notch_freq, notch_q, order)

    def update(self, fs, lowcut, highcut, notch_freq=50.0, notch_q=30.0, order=4):
        self.fs = fs
        nyq = fs / 2.0
        highcut = min(highcut, nyq * 0.99)
        if lowcut and lowcut > 0:
            self.b, self.a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
        else:
            self.b, self.a = butter(order, highcut / nyq, btype="lowpass")
        self._zi = [lfilter_zi(self.b, self.a) for _ in range(self.n_ch)]
        self.notch_on = bool(notch_freq)
        if self.notch_on:
            self.bn, self.an = iirnotch(notch_freq, notch_q, fs)
            self._zin = [lfilter_zi(self.bn, self.an) for _ in range(self.n_ch)]

    def process(self, chunk):
        """chunk: (n_ch, n) -> filtered (n_ch, n)."""
        out = np.empty_like(chunk, dtype=np.float64)
        for c in range(self.n_ch):
            x = chunk[c].astype(np.float64)
            if self.baseline:                                  # drift/DC removal
                self._mean[c] += self._mean_a * (x.mean() - self._mean[c])
                x = x - self._mean[c]
            y, self._zi[c] = lfilter(self.b, self.a, x, zi=self._zi[c])
            if self.notch_on:
                y, self._zin[c] = lfilter(self.bn, self.an, y, zi=self._zin[c])
            out[c] = y
        return out
