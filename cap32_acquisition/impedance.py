#!/usr/bin/env python
"""Impedance probe — try the board's (undocumented) impedance mode.

The vendor GUI implements it even though the provider didn't expose/document it:
it sends `b'%'` to make the board inject a **31.2 Hz, 24 µA** current on every channel,
reads the voltage amplitude at 31.2 Hz, and maps it to impedance via a linear fit
(reverse-read from `WidgetImpedancePlot.computeZ1`):

    V1 = 2·|rfft(hann·v)[k@31.2Hz]| / (N · mean(hann))      # µV amplitude at 31.2 Hz
    Z(Ω) = 32.1073 · V1 − 3982.9797

This sends `%`, measures for a few seconds, prints per-channel impedance, then sends
`*` to restore normal EEG mode. If the board doesn't support it, the 31.2 Hz amplitude
stays ~0 (no injection) — that tells us the firmware lacks impedance mode.

  python src/acquisition/impedance.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from montage import CAP32_CHANNELS as CH  # noqa: E402
from udp_lsl_bridge import UdpSource, parse_packet, IMPEDANCE_MODE, EEG_MODE, RATE_CMD, START  # noqa: E402

F0 = 31.2  # injection frequency (Hz) — vendor constant


def band_amp(v, fs, f):
    """Windowed single-sided amplitude (µV) at frequency f."""
    v = v - v.mean()
    n = len(v)
    w = np.hanning(n)
    V = np.fft.rfft(v * w)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    k = int(np.argmin(np.abs(freqs - f)))
    return 2 * np.abs(V[k]) / (n * (w.sum() / n))


def impedance(v, fs):
    """Vendor computeZ1: µV @31.2Hz -> Ω. Returns (Z_ohms, amp_uV_at_31.2)."""
    v1 = band_amp(v, fs, F0)
    z = 32.1073 * v1 - 3982.9797                  # vendor linear calibration -> Ω
    return z, v1


def monitor(host, port, fs, seconds):
    """Log per-channel amp@31.2Hz (injection), amp@50Hz (mains pickup), and RMS over
    time so we can see what actually tracks electrode contact. Watch the numbers while
    you (1) hold it in air, (2) put it on, (3) press one electrode, (4) lift one."""
    import csv
    from udp_lsl_bridge import UdpSource, parse_packet, START, RATE_CMD, IMPEDANCE_MODE, EEG_MODE
    from cap_gui import Ring
    src = UdpSource(host, port)
    time.sleep(0.3); src.send(START); time.sleep(0.2)
    src.send(RATE_CMD.get(fs, b"1")); time.sleep(0.1)
    src.send(IMPEDANCE_MODE); time.sleep(0.3)
    print("impedance mode ('%'). Logging amp@31.2Hz every 0.5 s. Ctrl-C to stop.\n")
    ring = Ring(32, int(1.5 * fs))
    os.makedirs("recordings", exist_ok=True)
    f = open("recordings/imp_monitor.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["t"] + [f"{c}_a31" for c in CH] + [f"{c}_a50" for c in CH])
    t0 = last = time.time()
    try:
        for pkt in src.frames():
            p = parse_packet(pkt)
            if p:
                ring.append(p[0].reshape(32, 1))
            now = time.time()
            if now - last >= 0.5 and now - t0 > 1.0:
                last = now
                b = ring.snapshot()
                a31 = np.array([band_amp(b[c], fs, F0) for c in range(32)])
                a50 = np.array([band_amp(b[c], fs, 50.0) for c in range(32)])
                w.writerow([f"{now-t0:.1f}"] + [f"{x:.0f}" for x in a31] + [f"{x:.0f}" for x in a50])
                f.flush()
                print(f"[t={now-t0:5.1f}s]  amp@31.2Hz median={np.median(a31):7.0f}  "
                      f"amp@50Hz median={np.median(a50):7.0f} µV")
                for r in range(8):
                    print("   " + "  ".join(f"{CH[r*4+c]:>4}:{a31[r*4+c]:7.0f}" for c in range(4)))
                print()
            if now - t0 > seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.send(EEG_MODE)
        f.close()
        print("→ restored EEG mode (*). log: recordings/imp_monitor.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=int, default=250)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--monitor", action="store_true",
                    help="live per-channel amp@31.2Hz + amp@50Hz table (to diagnose contact)")
    args = ap.parse_args()

    if args.monitor:
        monitor(args.host, args.port, args.sfreq, seconds=max(args.seconds, 60))
        return

    src = UdpSource(args.host, args.port)
    time.sleep(0.3)
    src.send(START); time.sleep(0.2)
    src.send(RATE_CMD.get(args.sfreq, b"1")); time.sleep(0.1)
    print("→ sending '%' (impedance mode) …")
    src.send(IMPEDANCE_MODE); time.sleep(0.6)

    rows, t0 = [], time.time()
    try:
        for pkt in src.frames():
            p = parse_packet(pkt)
            if p:
                rows.append(p[0])
            if time.time() - t0 > args.seconds:
                break
    finally:
        src.send(EEG_MODE)   # ALWAYS restore normal EEG mode
        print("→ sent '*' to restore EEG mode.")

    if not rows:
        print("no data — is the Mac on 'ESPBCI' and IP 192.168.4.2?")
        return
    M = np.array(rows).T                          # (32, N) µV
    print(f"collected {M.shape[1]} samples @ {args.sfreq} Hz\n")
    print("  ch     Z (kΩ)   amp@31.2Hz(µV)")
    amps = []
    for c in range(32):
        z, v1 = impedance(M[c], args.sfreq)
        amps.append(v1)
        tag = "" if v1 < 5 else ("  low" if z < 20000 else "  HIGH" if z < 200000 else "")
        zt = "–" if v1 < 5 else f"{z/1000:8.1f}"
        print(f"  {CH[c]:<4} {zt:>9}  {v1:12.1f}{tag}")
    if np.median(amps) < 5:
        print("\n⚠ 31.2 Hz amplitude ≈ 0 on all channels → the board did NOT inject a current.")
        print("  This firmware likely doesn't implement impedance mode (the '%' command is ignored).")
    else:
        print(f"\n✅ clear 31.2 Hz injection (median amp {np.median(amps):.0f} µV) → impedance mode WORKS.")
        print("  Higher amp@31.2Hz = higher electrode impedance. (Absolute kΩ uses the vendor's")
        print("  calibration — trust the relative ranking; recalibrate against known resistors for accuracy.)")


if __name__ == "__main__":
    main()
