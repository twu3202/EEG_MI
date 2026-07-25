# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Eyes-open / eyes-closed ALPHA test — the single best full-chain acceptance test for an
EEG cap, needing NO extra equipment. It uses the Berger effect as a biological reference:

    close your eyes  ->  posterior (O1/O2/Oz/PO/P) 8-12 Hz ALPHA power shoots up
    open  your eyes  ->  it drops again

This is universal and reproducible, so it IS the standard. If this cap shows a strong,
POSTERIOR-dominant alpha increase on eye-closure, then the whole acquisition chain works:
electrodes make contact, the amplifier and referencing are sane, the montage/wiring maps
occipital electrodes to the back of the head, and the µV/Hz scaling is in the right ballpark.

The script records alternating eyes-open / eyes-closed blocks (it prompts you), then:
  * posterior mean spectrum, open vs closed  (an alpha bump should appear when closed)
  * scalp topomap of the closed/open alpha ratio  (should be posterior-dominant)
  * per-channel closed/open alpha ratio bar
and prints a PASS/FAIL verdict.

  python src/acquisition/alpha_check.py                 # real cap (prompts you)
  python src/acquisition/alpha_check.py --blocks 3 --secs 12
  python src/acquisition/alpha_check.py --demo          # no hardware: validate plots
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
from montage import CAP32_CHANNELS as CH               # noqa: E402

NCH = len(CH)
RESULTS = HERE / "results"
ALPHA = (8.0, 12.0)
POSTERIOR = ["O1", "O2", "OZ", "PO3", "PO4", "PZ", "P3", "P4", "P7", "P8"]
FRONTAL = ["FP1", "FP2", "AF3", "AF4", "F3", "F4", "FZ"]


# ----------------------------------------------------------------- acquisition
def _grab(src, parse_packet, fs, secs):
    buf, end = [], time.time() + secs
    for pkt in src.frames():
        p = parse_packet(pkt)
        if p is not None:
            buf.append(p[0])
        if time.time() >= end:
            break
    return np.array(buf, dtype=np.float64).T if buf else None


def collect_blocks(host, port, fs, n_blocks, secs, settle=1.5):
    """Alternate eyes-open / eyes-closed blocks, prompting the user. RAW µV (no CAR/filter)."""
    from udp_lsl_bridge import UdpSource, parse_packet, board_init, EEG_MODE
    src = UdpSource(host, port); board_init(src, fs); time.sleep(0.4)
    src.sock.settimeout(2.0)
    out = {"open": [], "closed": []}
    schedule = []
    for _ in range(n_blocks):
        schedule += [("open", secs), ("closed", secs)]
    try:
        for cond, dur in schedule:
            word = "闭眼" if cond == "closed" else "睁眼(看屏幕十字)"
            print(f"\n>>> 请【{word}】保持 {dur:.0f}s", flush=True)
            for k in (3, 2, 1):
                print(f"    {k}…", end="", flush=True); time.sleep(0.6)
            print(" 开始")
            _grab(src, parse_packet, fs, settle)              # discard transient
            x = _grab(src, parse_packet, fs, dur)
            if x is not None:
                out[cond].append(x)
    except (KeyboardInterrupt, socket.timeout):
        pass
    finally:
        try:
            src.send(EEG_MODE)
        except OSError:
            pass
    return out


# ----------------------------------------------------------------- analysis
def band_power(x, fs, lo, hi):
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=min(x.shape[1], int(fs * 2)), axis=1)
    return f, P, P[:, (f >= lo) & (f <= hi)].mean(1)          # (freqs, PSD, per-ch band power)


def analyse(blocks, fs, out, demo=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    if not blocks["open"] or not blocks["closed"]:
        print("缺少 open 或 closed 数据段"); return
    xo = np.concatenate(blocks["open"], axis=1)
    xc = np.concatenate(blocks["closed"], axis=1)
    f, Po, ao = band_power(xo, fs, *ALPHA)
    _, Pc, ac = band_power(xc, fs, *ALPHA)
    ratio = ac / np.clip(ao, 1e-9, None)                      # closed/open alpha, per channel
    post = [CH.index(c) for c in POSTERIOR if c in CH]
    front = [CH.index(c) for c in FRONTAL if c in CH]
    r_post = float(np.median(ratio[post]))
    r_front = float(np.median(ratio[front]))

    info = mne.create_info(list(CH), fs, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False, on_missing="ignore", verbose="ERROR")

    fig = plt.figure(figsize=(14, 4.6)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.4])

    ax0 = fig.add_subplot(gs[0])                              # posterior spectrum
    m = (f >= 2) & (f <= 30)
    ax0.plot(f[m], Po[post].mean(0)[m], color="#6b7480", lw=2, label="eyes OPEN")
    ax0.plot(f[m], Pc[post].mean(0)[m], color="#2b6cb0", lw=2, label="eyes CLOSED")
    ax0.axvspan(*ALPHA, color="#2e9e5b", alpha=0.12)
    ax0.set_xlabel("Hz"); ax0.set_ylabel("PSD (µV²/Hz)")
    ax0.set_title("Posterior spectrum — alpha bump on closing?"); ax0.legend(fontsize=9)

    ax1 = fig.add_subplot(gs[1])                              # topomap of alpha ratio
    rr = np.clip(ratio, 0.3, 5.0)
    im, _ = mne.viz.plot_topomap(rr, info, axes=ax1, cmap="RdBu_r",
                                 vlim=(0.3, 3.0), show=False, contours=4)
    ax1.set_title("closed/open alpha ratio\n(posterior-dominant?)", fontsize=10)
    fig.colorbar(im, ax=ax1, shrink=0.7)

    ax2 = fig.add_subplot(gs[2])                              # per-channel ratio
    cols = ["#2b6cb0" if CH[i] in POSTERIOR else ("#c58a00" if CH[i] in FRONTAL else "#9aa3b2")
            for i in range(NCH)]
    ax2.bar(range(NCH), ratio, color=cols)
    ax2.axhline(1, color="#333", lw=0.8); ax2.axhline(2, color="#2e9e5b", ls="--", lw=1)
    ax2.set_xticks(range(NCH)); ax2.set_xticklabels(CH, rotation=90, fontsize=6)
    ax2.set_ylabel("closed/open alpha"); ax2.set_title("Per-channel (blue=posterior, amber=frontal)")

    fig.suptitle("Acceptance — EYES-OPEN/CLOSED ALPHA (Berger effect)" + ("  [DEMO]" if demo else ""),
                 fontweight="bold")
    fig.tight_layout(); RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)

    print(f"\n  posterior alpha closed/open = {r_post:.2f}×   frontal = {r_front:.2f}×")
    if r_post >= 2.0 and r_post > 1.5 * max(r_front, 1e-6):
        print("  ✅ PASS: 明显的枕区 α 反应且以后脑为主 —— 整条采集链工作正常。")
    elif r_post >= 1.3:
        print("  ⚠ 弱 α 反应 —— 接触/放松不足,或后部电极座不好。多测几组、闭眼放松再试。")
    else:
        print("  ❌ 无 α 反应 —— 后部电极没接触好 / 接线映射可疑 / 参考异常。先查通道映射与接触。")


# ----------------------------------------------------------------- demo
def demo_blocks(fs, n_blocks, secs):
    rng = np.random.default_rng(2)
    post = set(CH.index(c) for c in POSTERIOR if c in CH)

    def block(closed):
        n = int(secs * fs); t = np.arange(n) / fs
        x = np.empty((NCH, n))
        for c in range(NCH):
            pink = np.cumsum(rng.normal(0, 1, n)); pink -= pink.mean(); pink *= 6 / (pink.std() + 1e-9)
            a = (18 if closed else 4) if c in post else 3      # posterior alpha up when closed
            x[c] = a * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6)) + rng.normal(0, 4, n) + pink
        return x

    out = {"open": [], "closed": []}
    for _ in range(n_blocks):
        out["open"].append(block(False)); out["closed"].append(block(True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.4.1"); ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=int, default=250)
    ap.add_argument("--blocks", type=int, default=3, help="open/closed pairs")
    ap.add_argument("--secs", type=float, default=12.0, help="seconds per block")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "acceptance_alpha.png"))
    args = ap.parse_args()

    if args.demo:
        blocks = demo_blocks(args.sfreq, args.blocks, args.secs)
    else:
        print("即将交替【睁眼/闭眼】各若干段。闭眼时请放松、别用力眯眼(会有肌电)。")
        blocks = collect_blocks(args.host, args.port, args.sfreq, args.blocks, args.secs)
        no = sum(b.shape[1] for b in blocks["open"]) if blocks["open"] else 0
        nc = sum(b.shape[1] for b in blocks["closed"]) if blocks["closed"] else 0
        if not no or not nc:
            print("无数据 — 连到 ESPBCI 且 IP=192.168.4.2 了吗?"); return
        print(f"\ncollected open {no} / closed {nc} samples @ {args.sfreq} Hz")
    analyse(blocks, args.sfreq, args.out, demo=args.demo)


if __name__ == "__main__":
    main()
