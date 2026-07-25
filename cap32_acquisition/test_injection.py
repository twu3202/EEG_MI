# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Prove (or disprove) the 31.2 Hz impedance-mode current injection — empirically.

We've been *claiming* the board injects a 31.2 Hz AC current when it gets `b'%'` (from
reverse-reading the vendor code). This script actually measures it with an A/B/A protocol:

    phase A   EEG mode (`b'*'`)          → expect NO 31.2 Hz line
    phase B   impedance mode (`b'%'`)    → expect a STRONG narrow 31.2 Hz line on every ch
    phase C   EEG mode (`b'*'`) again    → the line must DISAPPEAR

If the 31.2 Hz peak is present only in B (and gone again in C), the injection is real and
mode-controlled. 50 Hz mains is present in all three phases → it's the built-in control:
it shows the spectrum is genuine and that only the 31.2 Hz component is mode-dependent.

Outputs a 4-panel figure (results/injection_test.png) + a printed verdict with numbers.

  # real cap (join ESPBCI, IP 192.168.4.2):
  python src/acquisition/test_injection.py --seconds 6
  # no hardware — validate the script's logic + plots on synthetic data:
  python src/acquisition/test_injection.py --demo
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # src/
sys.path.insert(0, str(HERE))                                 # src/acquisition/
from common.montage import CAP32_CHANNELS as CH               # noqa: E402

F0_NOMINAL = 31.2                                             # vendor's stated injection freq
NCH = len(CH)
RESULTS = HERE / "results"


# ----------------------------------------------------------------- spectra / stats
def amp_spectrum(x, fs):
    """x (nch, N) µV → (freqs, A) where A (nch, nf) = single-sided Hann amplitude in µV."""
    N = x.shape[1]
    w = np.hanning(N)
    xw = (x - x.mean(1, keepdims=True)) * w
    A = np.abs(np.fft.rfft(xw, axis=1)) * (2.0 / w.sum())
    f = np.fft.rfftfreq(N, 1 / fs)
    return f, A


def peak_at(f, Amean, f0, half=0.6):
    """Amplitude at the bin closest to f0 (µV)."""
    k = int(np.argmin(np.abs(f - f0)))
    return float(Amean[k]), k


def snr_db(f, Amean, f0):
    """Amplitude SNR (dB) of the f0 line vs the local background (±1.5–6 Hz, line excluded)."""
    peak, _ = peak_at(f, Amean, f0)
    bg_mask = (((f >= f0 - 6) & (f <= f0 - 1.5)) | ((f >= f0 + 1.5) & (f <= f0 + 6)))
    bg = float(np.median(Amean[bg_mask])) if bg_mask.any() else float(np.median(Amean))
    return 20 * np.log10(peak / max(bg, 1e-9)), peak, bg


def analyse(x, fs, f0):
    f, A = amp_spectrum(x, fs)
    Amean = A.mean(0)
    snr, peak, bg = snr_db(f, Amean, f0)
    _, k = peak_at(f, Amean, f0)
    return dict(f=f, A=A, Amean=Amean, snr=snr, peak=peak, bg=bg,
                per_ch=A[:, k], median_amp=float(np.median(A[:, k])))


def detect_injection_freq(x, fs, lo=28.0, hi=35.0):
    """Find the dominant narrow peak in [lo,hi] Hz (robust to 31.2 vs 31.25=fs/8 etc.)."""
    f, A = amp_spectrum(x, fs)
    Amean = A.mean(0)
    band = (f >= lo) & (f <= hi)
    return float(f[band][np.argmax(Amean[band])])


# ----------------------------------------------------------------- data collection
def collect(src, parse_packet, fs, seconds, settle):
    """Read frames for `settle`+`seconds`, keep the last `seconds` (drops the switch transient)."""
    src.sock.settimeout(2.0)
    buf, deadline = [], time.time() + settle + seconds
    try:
        for pkt in src.frames():
            p = parse_packet(pkt)
            if p is not None:
                buf.append(p[0])
            if time.time() >= deadline:
                break
    except socket.timeout:
        pass
    if not buf:
        return None
    x = np.array(buf, dtype=np.float64).T                     # (nch, M) µV
    keep = int(seconds * fs)
    return x[:, -keep:] if x.shape[1] > keep else x


def run_hardware(host, port, fs, seconds, settle):
    from udp_lsl_bridge import (UdpSource, parse_packet, board_init,
                                IMPEDANCE_MODE, EEG_MODE)
    src = UdpSource(host, port)
    board_init(src, fs)                                       # b → rate → * (EEG mode)
    time.sleep(0.4)
    try:
        print("phase A — EEG mode (*)  … collecting");
        A = collect(src, parse_packet, fs, seconds, settle=0.5)
        print("phase B — impedance mode (%)  … collecting")
        src.send(IMPEDANCE_MODE)
        B = collect(src, parse_packet, fs, seconds, settle=settle)
        print("phase C — EEG mode (*) restored … collecting")
        src.send(EEG_MODE)
        C = collect(src, parse_packet, fs, seconds, settle=settle)
    finally:
        try:
            src.send(EEG_MODE)                                # ALWAYS leave it in EEG mode
        except OSError:
            pass
    return A, B, C


# ----------------------------------------------------------------- synthetic demo
def demo_phases(fs, seconds, f_inj=31.25):
    """EEG-like phases with NO injection, and an impedance phase WITH a 31.25 Hz line —
    only to validate this script's detection + plots without hardware."""
    N = int(seconds * fs)
    t = np.arange(N) / fs
    rng = np.random.default_rng(0)

    def eeg():
        x = np.empty((NCH, N))
        for c in range(NCH):
            pink = np.cumsum(rng.normal(0, 1, N)); pink -= pink.mean(); pink *= 8.0 / (pink.std() + 1e-9)
            x[c] = (pink + 6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 7))
                    + 5 * np.sin(2 * np.pi * 50 * t) + rng.normal(0, 3, N))   # +alpha +50Hz mains
        return x

    A = eeg(); C = eeg()
    B = eeg()
    for c in range(NCH):                                      # add the injection to phase B only
        B[c] += rng.uniform(1500, 3000) * np.sin(2 * np.pi * f_inj * t + rng.uniform(0, 7))
    return A, B, C


# ----------------------------------------------------------------- figure
def make_figure(A, B, C, fs, f0, out, demo=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rA, rB, rC = analyse(A, fs, f0), analyse(B, fs, f0), analyse(C, fs, f0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("white")
    cols = {"A": "#6b7480", "B": "#c0392b", "C": "#2b6cb0"}

    # (1) full mean amplitude spectrum, log-y — the money plot
    a = ax[0][0]
    for name, r in (("A EEG", rA), ("B impedance", rB), ("C EEG again", rC)):
        a.semilogy(r["f"], np.clip(r["Amean"], 1e-2, None), lw=1.4,
                   color=cols[name[0]], label=name)
    a.axvline(f0, color="#c0392b", ls="--", lw=1, alpha=0.7)
    a.axvline(50, color="#888", ls=":", lw=1, alpha=0.7)
    a.text(f0, a.get_ylim()[1], f" {f0:.2f}Hz", color="#c0392b", va="top", fontsize=9)
    a.text(50, a.get_ylim()[1], " 50Hz mains", color="#888", va="top", fontsize=8)
    a.set_xlim(0, min(60, fs / 2)); a.set_xlabel("Hz"); a.set_ylabel("amplitude (µV)")
    a.set_title("Mean spectrum (all ch) — 31.2 Hz only in impedance mode?"); a.legend(fontsize=9)

    # (2) zoom around the injection line, linear-y
    a = ax[0][1]
    for name, r in (("A EEG", rA), ("B impedance", rB), ("C EEG again", rC)):
        a.plot(r["f"], r["Amean"], lw=1.6, color=cols[name[0]], label=name)
    a.axvline(f0, color="#c0392b", ls="--", lw=1, alpha=0.7)
    a.set_xlim(f0 - 6, f0 + 6); a.set_xlabel("Hz"); a.set_ylabel("amplitude (µV)")
    a.set_title(f"Zoom on {f0:.2f} Hz"); a.legend(fontsize=9)

    # (3) per-channel amplitude at the injection freq
    a = ax[1][0]
    xpos = np.arange(NCH)
    a.bar(xpos - 0.27, rA["per_ch"], 0.27, color=cols["A"], label="A EEG")
    a.bar(xpos + 0.00, rB["per_ch"], 0.27, color=cols["B"], label="B impedance")
    a.bar(xpos + 0.27, rC["per_ch"], 0.27, color=cols["C"], label="C EEG again")
    a.set_xticks(xpos); a.set_xticklabels(CH, rotation=90, fontsize=6)
    a.set_ylabel(f"amp @ {f0:.2f} Hz (µV)"); a.set_title("Per-channel injection amplitude")
    a.legend(fontsize=9)

    # (4) time-domain snippet on the median-amplitude channel
    a = ax[1][1]
    cbest = int(np.argsort(rB["per_ch"])[NCH // 2])           # a typical channel
    n = int(min(0.5 * fs, A.shape[1], B.shape[1]))
    t = np.arange(n) / fs * 1000
    a.plot(t, B[cbest, :n] - B[cbest, :n].mean(), color=cols["B"], lw=1.0, label="B impedance")
    a.plot(t, A[cbest, :n] - A[cbest, :n].mean(), color=cols["A"], lw=1.0, label="A EEG")
    a.set_xlabel("ms"); a.set_ylabel("µV"); a.set_title(f"Raw {CH[cbest]} — {1000/f0:.1f} ms period?")
    a.legend(fontsize=9)

    sub = ("  ⚠ SYNTHETIC DEMO — validates the script only; run on the cap for the real answer"
           if demo else "  (measured on the cap)")
    fig.suptitle("31.2 Hz impedance-injection test  (A/B/A)" + sub,
                 fontsize=14, y=1.0, color=("#b23b3b" if demo else "#1f2733"))
    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rA, rB, rC


# ----------------------------------------------------------------- verdict
def verdict(rA, rB, rC, f0):
    ratio = rB["median_amp"] / max(rA["median_amp"], 1e-6)
    present_B = (rB["snr"] > 10.0) and (ratio > 3.0)
    gone_C = rC["median_amp"] < 3.0 * max(rA["median_amp"], 1e-6)
    quiet_A = rA["snr"] < 8.0
    print("\n" + "=" * 66)
    print(f"  {'phase':<16}{'med amp@f0 (µV)':>18}{'SNR@f0 (dB)':>14}")
    for nm, r in (("A EEG", rA), ("B impedance", rB), ("C EEG again", rC)):
        print(f"  {nm:<16}{r['median_amp']:>18.1f}{r['snr']:>14.1f}")
    print("-" * 66)
    print(f"  B/A amplitude ratio: {ratio:>8.1f}×   (want ≫1)")
    if present_B and gone_C and quiet_A:
        print(f"\n  ✅ CONFIRMED: a strong {f0:.2f} Hz line appears ONLY in impedance mode")
        print(f"     (B) and disappears again in EEG mode (C). The 31.2 Hz current")
        print(f"     injection is REAL and mode-controlled by '%' / '*'.")
    elif present_B and not gone_C:
        print(f"\n  ⚠ {f0:.2f} Hz is strong in B but did NOT clear in C — the board may not")
        print("     have switched back, or C was collected too soon after '*'.")
    elif not present_B:
        print(f"\n  ❌ NO {f0:.2f} Hz injection detected in impedance mode.")
        print("     Either the firmware ignores '%', or no data/contact. Check the WiFi/IP,")
        print("     and that phase B actually received frames.")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=int, default=250)
    ap.add_argument("--seconds", type=float, default=6.0, help="collection per phase")
    ap.add_argument("--settle", type=float, default=1.0, help="discard after each mode switch")
    ap.add_argument("--demo", action="store_true", help="synthetic data (no hardware)")
    ap.add_argument("--out", default=str(RESULTS / "injection_test.png"))
    args = ap.parse_args()

    if args.demo:
        A, B, C = demo_phases(args.sfreq, args.seconds)
        if args.out == str(RESULTS / "injection_test.png"):
            args.out = str(RESULTS / "injection_test_demo.png")
    else:
        A, B, C = run_hardware(args.host, args.port, args.sfreq, args.seconds, args.settle)
        if A is None or B is None or C is None:
            print("no data in one or more phases — on 'ESPBCI' with IP 192.168.4.2? board streaming?")
            return

    f0 = detect_injection_freq(B, args.sfreq)                 # measured peak (≈31.2/31.25)
    print(f"\ncollected {A.shape[1]}/{B.shape[1]}/{C.shape[1]} samples per phase @ {args.sfreq} Hz")
    print(f"detected candidate injection peak in impedance mode: {f0:.3f} Hz "
          f"(nominal {F0_NOMINAL})")
    rA, rB, rC = make_figure(A, B, C, args.sfreq, f0, args.out, demo=args.demo)
    verdict(rA, rB, rC, f0)
    print(f"\nfigure → {args.out}")


if __name__ == "__main__":
    main()
