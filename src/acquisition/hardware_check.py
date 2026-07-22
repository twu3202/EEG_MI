#!/usr/bin/env python
"""Hardware self-check for the 32-ch ADS1299 cap — the objective way to judge board quality
(instead of guessing from off-head stray coupling). Each test collects RAW data (NO CAR, NO
filter) and produces a plot + a printed verdict.

    noise      input-referred NOISE FLOOR.     Setup: short all electrodes together / to REF
               (or bundle+wrap in foil). Good ADS1299 @gain24 ≈ 1 µV RMS input-referred.
               Plot: per-channel RMS + amplitude spectral density (µV/√Hz).

    dc         DC OFFSET & RAILING.             Setup: on the head, normal wear.
               How big are the per-channel DC offsets, and how often do channels hit the
               ±187.5 mV rail? (dry-electrode polarisation — the thing that railed a recording.)
               Plot: per-channel DC (mV) vs the rail, and % of samples clipped.

    crosstalk  CHANNEL-TO-CHANNEL LEAKAGE.      Setup: drive ONE electrode with a clean tone
               (function generator / phone tone through a wire), leave the rest. Measures how
               much of that tone leaks into the OTHER channels. Good design < 1 % (−40 dB).
               Plot: per-channel leakage (dB) relative to the source channel.

    mains      50 Hz MAINS PICKUP (CMRR proxy). Setup: normal wear.
               How much 50 Hz each channel picks up — a practical proxy for interference
               rejection. (True CMRR needs a tied-input common-mode source; noted below.)
               Plot: per-channel 50 Hz amplitude + share of total power.

  python src/acquisition/hardware_check.py --test noise
  python src/acquisition/hardware_check.py --test dc --seconds 20
  python src/acquisition/hardware_check.py --test crosstalk --source-ch C3 --probe-hz 10
  python src/acquisition/hardware_check.py --test mains
  python src/acquisition/hardware_check.py --test noise --demo      # no hardware: validate plots
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
from common.montage import CAP32_CHANNELS as CH, ADC_MICROVOLTS_PER_COUNT  # noqa: E402

NCH = len(CH)
FULLSCALE_UV = (2 ** 23) * ADC_MICROVOLTS_PER_COUNT           # ≈187.5 mV = ±rail
RESULTS = HERE.parents[1] / "results"

SETUP = {
    "noise": "把所有电极短接在一起(或接到 REF/地),最好用锡纸包住。测放大器本底噪声。",
    "dc": "正常戴在头上。测各通道直流偏置和顶轨(rail)频率。",
    "crosstalk": "只在一个电极上加一个干净的正弦(信号发生器/手机播放音调经导线接入),其余不动。",
    "mains": "正常戴在头上(或拿在手里)。测各通道 50Hz 拾取。",
}


# ----------------------------------------------------------------- data
def collect_raw(host, port, fs, seconds):
    """RAW µV per channel — NO CAR, NO filter (parse_packet output directly)."""
    from udp_lsl_bridge import UdpSource, parse_packet, board_init, EEG_MODE
    src = UdpSource(host, port)
    board_init(src, fs); time.sleep(0.4)                      # b -> rate -> * (EEG mode)
    src.sock.settimeout(2.0)
    buf, end = [], time.time() + seconds
    try:
        for pkt in src.frames():
            p = parse_packet(pkt)
            if p is not None:
                buf.append(p[0])
            if time.time() >= end:
                break
    except socket.timeout:
        pass
    finally:
        try:
            src.send(EEG_MODE)
        except OSError:
            pass
    return np.array(buf, dtype=np.float64).T if buf else None


def amp_spectrum(x, fs):
    """(freqs, A) — single-sided Hann amplitude (µV) per channel."""
    n = x.shape[1]; w = np.hanning(n)
    A = np.abs(np.fft.rfft((x - x.mean(1, keepdims=True)) * w, axis=1)) * (2.0 / w.sum())
    return np.fft.rfftfreq(n, 1 / fs), A


def bandpass(x, fs, lo, hi):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [lo / (fs / 2), min(hi, fs / 2 * 0.99) / (fs / 2)], "bandpass")
    return filtfilt(b, a, x, axis=1)


def _figpre():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _barcolors(vals, good, warn, reverse=False):
    out = []
    for v in vals:
        ok = v <= good if not reverse else v >= good
        mid = v <= warn if not reverse else v >= warn
        out.append("#2e9e5b" if ok else ("#c58a00" if mid else "#d1495b"))
    return out


# ----------------------------------------------------------------- tests
def test_noise(x, fs, out, demo=False):
    from scipy.signal import welch
    xb = bandpass(x, fs, 1.0, min(45.0, fs / 2 * 0.98))
    rms = xb.std(axis=1)
    f, P = welch(x - x.mean(1, keepdims=True), fs=fs, nperseg=min(x.shape[1], int(fs * 4)), axis=1)
    asd = np.sqrt(P)                                          # µV/√Hz
    med = float(np.median(rms))
    plt = _figpre()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); fig.patch.set_facecolor("white")
    ax[0].bar(range(NCH), rms, color=_barcolors(rms, 3, 8))
    ax[0].axhline(1.0, color="#2e9e5b", ls="--", lw=1, label="ADS1299 ~1 µV (ideal)")
    ax[0].set_xticks(range(NCH)); ax[0].set_xticklabels(CH, rotation=90, fontsize=6)
    ax[0].set_ylabel("RMS 1–45 Hz (µV)"); ax[0].set_title(f"Noise floor per channel  (median {med:.1f} µV)")
    ax[0].legend(fontsize=8)
    m = f >= 0.5
    ax[1].loglog(f[m], asd[:, m].T, color="#9aa3b2", lw=0.5, alpha=0.5)
    ax[1].loglog(f[m], np.median(asd[:, m], 0), color="#2b6cb0", lw=2, label="median")
    ax[1].axhline(1.0 / np.sqrt(fs / 2), color="#2e9e5b", ls="--", lw=1)
    ax[1].set_xlabel("Hz"); ax[1].set_ylabel("ASD (µV/√Hz)"); ax[1].set_title("Amplitude spectral density")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2, which="both")
    fig.suptitle("Hardware check — NOISE FLOOR (short inputs)" + ("  [DEMO]" if demo else ""),
                 fontweight="bold")
    _save(fig, out)
    print(f"  median noise {med:.1f} µV RMS;  noisy (>8µV): {[CH[i] for i in range(NCH) if rms[i]>8]}")
    print(f"  → good boards land ~1–3 µV RMS with shorted inputs; >8 µV = noisy channel/contact.")


def test_dc(x, fs, out, demo=False):
    dc = x.mean(axis=1) / 1000.0                              # mV
    railfrac = (np.abs(x) > 0.97 * FULLSCALE_UV).mean(axis=1) * 100.0
    plt = _figpre()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); fig.patch.set_facecolor("white")
    ax[0].axhspan(187.5, 260, color="#f3d6d6"); ax[0].axhspan(-260, -187.5, color="#f3d6d6")
    ax[0].bar(range(NCH), dc, color=_barcolors(np.abs(dc), 50, 150))
    ax[0].axhline(187.5, color="#d1495b", lw=1); ax[0].axhline(-187.5, color="#d1495b", lw=1)
    ax[0].set_ylim(-260, 260); ax[0].set_xticks(range(NCH)); ax[0].set_xticklabels(CH, rotation=90, fontsize=6)
    ax[0].set_ylabel("DC offset (mV)"); ax[0].set_title("Per-channel DC offset vs ±187.5 mV rail")
    ax[1].bar(range(NCH), railfrac, color=_barcolors(railfrac, 0.1, 2))
    ax[1].set_xticks(range(NCH)); ax[1].set_xticklabels(CH, rotation=90, fontsize=6)
    ax[1].set_ylabel("% samples clipped"); ax[1].set_title("Railing / saturation")
    fig.suptitle("Hardware check — DC OFFSET & RAILING (on head)" + ("  [DEMO]" if demo else ""),
                 fontweight="bold")
    _save(fig, out)
    railed = [CH[i] for i in range(NCH) if railfrac[i] > 1]
    print(f"  DC offset |median| {np.median(np.abs(dc)):.0f} mV;  railing channels: {railed or 'none'}")
    print("  → big DC offsets are mostly dry-electrode polarisation, not necessarily a board fault;")
    print("    but a railed channel clips in recordings (use 1 Hz high-pass / re-seat that electrode).")


def test_crosstalk(x, fs, out, source=None, probe_hz=None, demo=False):
    f, A = amp_spectrum(x, fs)
    band = (f >= 3) & (f <= 45) & (np.abs(f - 50) > 1.5)      # exclude DC & mains
    if probe_hz is None:
        k = np.arange(len(f))[band][np.argmax(A[:, band].max(0))]
    else:
        k = int(np.argmin(np.abs(f - probe_hz)))
    src_i = CH.index(source) if source in CH else int(np.argmax(A[:, k]))
    leak = A[:, k] / max(A[src_i, k], 1e-9)
    leak_db = 20 * np.log10(np.clip(leak, 1e-6, None))
    plt = _figpre()
    fig, ax = plt.subplots(figsize=(12, 4.6)); fig.patch.set_facecolor("white")
    cols = ["#2b6cb0" if i == src_i else
            ("#2e9e5b" if leak[i] < 0.01 else "#c58a00" if leak[i] < 0.05 else "#d1495b")
            for i in range(NCH)]
    ax.bar(range(NCH), leak_db, color=cols)
    ax.axhline(-40, color="#2e9e5b", ls="--", lw=1, label="−40 dB (1% leakage)")
    ax.set_xticks(range(NCH)); ax.set_xticklabels(CH, rotation=90, fontsize=7)
    ax.set_ylabel("leakage vs source (dB)")
    ax.set_title(f"Crosstalk from {CH[src_i]} @ {f[k]:.1f} Hz  (blue = source)")
    ax.legend(fontsize=9)
    fig.suptitle("Hardware check — CROSSTALK (drive one electrode)" + ("  [DEMO]" if demo else ""),
                 fontweight="bold")
    _save(fig, out)
    others = np.delete(leak, src_i)
    print(f"  source {CH[src_i]} @ {f[k]:.1f} Hz;  worst other-channel leakage "
          f"{20*np.log10(others.max()):.0f} dB ({others.max()*100:.1f}%), median "
          f"{20*np.log10(np.median(others)):.0f} dB")
    print("  → < −40 dB (1%) is good isolation; > −26 dB (5%) suggests poor layout/shielding.")


def test_mains(x, fs, out, demo=False):
    f, A = amp_spectrum(x, fs)
    k50 = int(np.argmin(np.abs(f - 50)))
    a50 = A[:, k50]
    rms = x.std(axis=1)
    share = a50 / (rms + 1e-9) * 100
    plt = _figpre()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); fig.patch.set_facecolor("white")
    ax[0].bar(range(NCH), a50, color=_barcolors(a50, 5, 20))
    ax[0].set_xticks(range(NCH)); ax[0].set_xticklabels(CH, rotation=90, fontsize=6)
    ax[0].set_ylabel("50 Hz amplitude (µV)"); ax[0].set_title(f"Mains pickup  (median {np.median(a50):.1f} µV)")
    ax[1].bar(range(NCH), share, color=_barcolors(share, 10, 30))
    ax[1].set_xticks(range(NCH)); ax[1].set_xticklabels(CH, rotation=90, fontsize=6)
    ax[1].set_ylabel("50 Hz share of RMS (%)"); ax[1].set_title("How much of the signal is mains")
    fig.suptitle("Hardware check — 50 Hz MAINS PICKUP" + ("  [DEMO]" if demo else ""),
                 fontweight="bold")
    _save(fig, out)
    print(f"  50 Hz median {np.median(a50):.1f} µV;  worst {CH[int(np.argmax(a50))]} {a50.max():.0f} µV")
    print("  → high, uniform 50 Hz across channels = common-mode (helped by CAR + notch).")
    print("    True CMRR needs a tied-input common-mode source; this is the practical proxy.")


def _save(fig, out):
    import matplotlib.pyplot as plt
    fig.tight_layout(); RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------- demo data
def demo_data(fs, seconds, test):
    n = int(seconds * fs); t = np.arange(n) / fs; rng = np.random.default_rng(1)
    x = rng.normal(0, 2.0, (NCH, n))                          # ~2 µV white noise floor
    x[7] += rng.normal(0, 14, n); x[22] += rng.normal(0, 11, n)   # a couple noisy channels
    if test == "dc":
        x *= 1000                                            # → mV-scale swings
        x += (rng.uniform(-120000, 120000, NCH))[:, None]    # per-channel DC offsets (µV)
        x[14] += 240000                                      # T7 pushed past the rail
    if test == "mains" or test == "dc":
        for c in range(NCH):
            x[c] += rng.uniform(3, 40) * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 6))
    if test == "dc":
        x = np.clip(x, -FULLSCALE_UV, FULLSCALE_UV)          # ADC saturates -> T7 rails
    if test == "crosstalk":
        src = CH.index("C3")
        tone = 500 * np.sin(2 * np.pi * 10 * t)
        x[src] += tone
        for c in range(NCH):                                 # 0.3–3 % leakage to others
            if c != src:
                x[c] += rng.uniform(0.003, 0.03) * tone
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, choices=["noise", "dc", "crosstalk", "mains"])
    ap.add_argument("--host", default="192.168.4.1"); ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=int, default=250); ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--source-ch", default=None); ap.add_argument("--probe-hz", type=float, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(RESULTS / f"hw_{args.test}.png")

    if args.demo:
        x = demo_data(args.sfreq, args.seconds, args.test)
    else:
        print(f"\n【{args.test}】设置要求: {SETUP[args.test]}")
        try:
            input("设置好后按回车开始采集(Ctrl-C 取消)…")
        except (KeyboardInterrupt, EOFError):
            return
        x = collect_raw(args.host, args.port, args.sfreq, args.seconds)
        if x is None:
            print("无数据 — 连到 ESPBCI 且 IP=192.168.4.2 了吗?"); return
        print(f"collected {x.shape[1]} samples ({x.shape[1]/args.sfreq:.1f}s) @ {args.sfreq} Hz\n")

    fn = {"noise": test_noise, "dc": test_dc, "mains": test_mains}
    if args.test == "crosstalk":
        test_crosstalk(x, args.sfreq, out, args.source_ch, args.probe_hz, demo=args.demo)
    else:
        fn[args.test](x, args.sfreq, out, demo=args.demo)


if __name__ == "__main__":
    main()
