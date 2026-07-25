# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""UDP packet-loss test for the cap's WiFi stream.

Each 105-byte frame carries a 1-byte sequence counter (byte[1], 0–255, wraps). We count how
many frames actually arrive and, from jumps in that counter, how many were LOST — so you get
received / lost / loss-rate, plus effective sample rate, over a long run. This is the "WiFi
reliability" acceptance test (DIY UDP streams can drop under interference / distance).

Reports:
  * total received, total lost, expected, LOSS RATE (%), effective Hz
  * per-second loss % and effective Hz over time (is it steady or bursty? does it degrade?)
  * histogram of loss-burst sizes (mostly 1-frame blips, or big bursts?)

  python src/acquisition/udp_loss_test.py --seconds 120          # 2-min run on the cap
  python src/acquisition/udp_loss_test.py --minutes 10           # long soak test
  python src/acquisition/udp_loss_test.py --demo                 # no hardware: validate the math

Note: the seq counter is 1 byte, so a single burst of EXACTLY ≥256 lost frames (>~1 s at
250 Hz) can alias and be undercounted — fine for typical WiFi blips; watch the effective-Hz
plot for sustained drops.
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
RESULTS = HERE / "results"
G, B, X = "\033[32m", "\033[1m", "\033[0m"


class LossAccount:
    """Counts received / lost frames from the 1-byte sequence counter, binned per window.

    Reports STARTUP and STEADY-STATE separately: `board_init` sends b → rate → '*' with
    sleeps in between, and the board restarts its stream on each mode switch, so the first
    second routinely shows a burst of "lost" frames that says nothing about link quality.
    Averaging that into the headline number badly understates a good link (measured: a
    200-frame startup burst turned a genuinely 0.000% link into a reported 0.663%)."""

    def __init__(self, window=1.0, settle=3.0):
        self.window = window
        self.settle = settle          # frames before this (s) are counted as STARTUP
        self.last = None
        self.recv = 0
        self.lost = 0
        self.recv_ss = 0              # steady-state only (t >= settle)
        self.lost_ss = 0
        self.gaps = []          # (t_rel, burst_size) for each detected loss
        self.windows = []       # (t_rel, recv_in_window, lost_in_window)
        self._wstart = None
        self._wrecv = 0
        self._wlost = 0

    def add(self, t, seq):
        if self._wstart is None:
            self._wstart = t
        steady = t >= self.settle
        if self.last is not None:
            g = (seq - self.last - 1) % 256          # frames missing between last and this
            if g:
                self.lost += g
                self._wlost += g
                self.gaps.append((t, g))
                if steady:
                    self.lost_ss += g
        self.last = seq
        self.recv += 1
        self._wrecv += 1
        if steady:
            self.recv_ss += 1
        if t - self._wstart >= self.window:
            dur = t - self._wstart
            row = (t, self._wrecv, self._wlost, dur)
            self.windows.append((t, self._wrecv, self._wlost))
            self._wstart, self._wrecv, self._wlost = t, 0, 0
            return row
        return None

    @property
    def total(self):
        return self.recv + self.lost

    @property
    def loss_pct(self):
        return 100.0 * self.lost / max(1, self.total)

    @property
    def loss_pct_ss(self):
        """The number that actually characterises the link (startup burst excluded)."""
        return 100.0 * self.lost_ss / max(1, self.recv_ss + self.lost_ss)


# ----------------------------------------------------------------- live capture
def run_live(host, port, fs, seconds, window, settle=3.0):
    from udp_lsl_bridge import UdpSource, parse_packet, board_init, drain
    src = UdpSource(host, port)
    board_init(src, fs); time.sleep(0.4)
    drain(src)                      # discard the board_init transient
    src.sock.settimeout(2.0)
    acc = LossAccount(window, settle)
    print(f"measuring UDP loss for {seconds:.0f}s @ nominal {fs} Hz …  (Ctrl-C to stop early)\n")
    t0 = time.time()
    try:
        for pkt in src.frames():
            p = parse_packet(pkt)
            if p is None:
                continue
            _, seq, _ = p
            now = time.time() - t0
            row = acc.add(now, seq)
            if row is not None:
                t, wr, wl, dur = row
                print(f"[{t:6.1f}s] recv {acc.recv:>7} lost {acc.lost:>5} "
                      f"({acc.loss_pct:5.2f}%)  ·  this {window:.0f}s: {wl} lost / {wr+wl}  "
                      f"·  eff {wr/dur:5.0f} Hz")
            if now >= seconds:
                break
    except KeyboardInterrupt:
        print("\n(stopped early)")
    except socket.timeout:
        print("\n⚠ 2 秒没有数据 — 掉线了?(连到 ESPBCI 且 IP=192.168.4.2 了吗)")
    return acc, time.time() - t0


# ----------------------------------------------------------------- demo
def run_demo(fs, seconds, window, loss=0.03, seed=0, settle=3.0):
    rng = np.random.default_rng(seed)
    acc = LossAccount(window, settle)
    n, dt, idx = int(fs * seconds), 1.0 / fs, 0
    while idx < n:
        r = rng.random()
        if r < 0.0015:                               # occasional multi-frame burst
            idx += int(rng.integers(3, 10)); continue
        if r < loss:                                 # single-frame drop
            idx += 1; continue
        acc.add(idx * dt, idx % 256)
        idx += 1
    return acc, seconds


# ----------------------------------------------------------------- report + plot
def report(acc, dur, fs, out, demo=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    expected = fs * dur
    eff = acc.recv / max(dur, 1e-9)
    startup_lost = acc.lost - acc.lost_ss
    ss_dur = max(dur - acc.settle, 1e-9)
    pct = acc.loss_pct_ss                       # judge the link on STEADY STATE
    print("\n" + "=" * 66)
    print(f"  时长 duration        : {dur:.1f} s @ nominal {fs} Hz")
    print(f"  收到 received        : {acc.recv}")
    print(f"  丢失 lost (总)       : {acc.lost}   (from seq-counter jumps)")
    print(f"  应收 expected        : ~{expected:.0f}   (recv+lost = {acc.total})")
    print("  " + "-" * 62)
    print(f"  启动瞬态 (<{acc.settle:.0f}s)     : 丢 {startup_lost}   "
          f"← board_init 切换模式导致,不代表链路质量")
    print(f"  {B}稳态 STEADY-STATE   : 丢 {acc.lost_ss} / {acc.recv_ss + acc.lost_ss}  "
          f"= {pct:.3f} %{X}   ({ss_dur:.0f}s)")
    print(f"  总体(含瞬态)         : {acc.loss_pct:.3f} %   ← 会被开头稀释,仅供参考")
    print("  " + "-" * 62)
    print(f"  有效采样率 eff. Hz   : {eff:.1f}  (标称 {fs})")
    verdict = ("✅ excellent (<0.5%)" if pct < 0.5 else
               "🟡 acceptable (0.5–2%)" if pct < 2 else
               "🟠 marginal (2–5%) — 换独立热点/靠近/减少干扰" if pct < 5 else
               "🔴 bad (>5%) — WiFi 链路有问题")
    print(f"  评价 verdict (稳态)  : {verdict}")
    if startup_lost and not acc.lost_ss:
        print(f"  {G}→ 链路本身是干净的:所有丢包都发生在启动瞬态{X}")
    print("=" * 66)

    win = np.array(acc.windows) if acc.windows else np.zeros((1, 3))
    t = win[:, 0]; wr = win[:, 1]; wl = win[:, 2]
    wpct = 100 * wl / np.clip(wr + wl, 1, None)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7)); fig.patch.set_facecolor("white")

    ax[0][0].plot(t, wpct, color="#c0392b", lw=1.5)
    ax[0][0].axhline(acc.loss_pct, color="#6b7480", ls="--", lw=1, label=f"overall {acc.loss_pct:.2f}%")
    ax[0][0].set_xlabel("time (s)"); ax[0][0].set_ylabel("loss (%)")
    ax[0][0].set_title("Loss rate over time"); ax[0][0].legend(fontsize=8)

    ax[0][1].plot(t, wr / np.maximum(np.diff(np.concatenate([[0], t])), 1e-9), color="#2b6cb0", lw=1.5)
    ax[0][1].axhline(fs, color="#2e9e5b", ls="--", lw=1, label=f"nominal {fs} Hz")
    ax[0][1].set_xlabel("time (s)"); ax[0][1].set_ylabel("effective Hz")
    ax[0][1].set_title("Effective sample rate"); ax[0][1].legend(fontsize=8)

    if acc.gaps:
        sizes = np.array([g for _, g in acc.gaps])
        bins = np.arange(1, min(sizes.max(), 20) + 2)
        ax[1][0].hist(sizes, bins=bins, color="#c58a00", align="left", rwidth=0.85)
        ax[1][0].set_xlabel("consecutive frames lost (burst size)")
        ax[1][0].set_ylabel("count"); ax[1][0].set_title(f"Loss-burst sizes ({len(sizes)} events)")
    else:
        ax[1][0].text(0.5, 0.5, "no losses 🎉", ha="center", va="center", transform=ax[1][0].transAxes)
        ax[1][0].set_title("Loss-burst sizes")

    cum_recv = np.cumsum(wr); cum_exp = fs * t
    ax[1][1].plot(t, cum_exp, color="#2e9e5b", ls="--", lw=1.5, label="expected (fs·t)")
    ax[1][1].plot(t, cum_recv, color="#2b6cb0", lw=1.5, label="received")
    ax[1][1].set_xlabel("time (s)"); ax[1][1].set_ylabel("cumulative frames")
    ax[1][1].set_title("Cumulative received vs expected"); ax[1][1].legend(fontsize=8)

    fig.suptitle(f"UDP packet-loss test — {acc.loss_pct:.2f}% loss over {dur:.0f}s"
                 + ("  [DEMO]" if demo else "  (measured on the cap)"), fontweight="bold")
    fig.tight_layout(); RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.4.1"); ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=int, default=250)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--minutes", type=float, default=None, help="overrides --seconds")
    ap.add_argument("--window", type=float, default=1.0, help="stats bin size (s)")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="首 N 秒算作启动瞬态,单独统计(默认 3)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--loss", type=float, default=0.03, help="demo synthetic loss fraction")
    ap.add_argument("--out", default=str(RESULTS / "udp_loss_test.png"))
    args = ap.parse_args()
    seconds = args.minutes * 60 if args.minutes else args.seconds

    if args.demo:
        out = args.out if args.out != str(RESULTS / "udp_loss_test.png") else str(RESULTS / "udp_loss_test_demo.png")
        acc, dur = run_demo(args.sfreq, seconds, args.window, loss=args.loss, settle=args.settle)
    else:
        out = args.out
        acc, dur = run_live(args.host, args.port, args.sfreq, seconds, args.window, args.settle)
        if acc.recv == 0:
            print("无数据 — 连到 ESPBCI 且 IP=192.168.4.2 了吗?"); return
    report(acc, dur, args.sfreq, out, demo=args.demo)


if __name__ == "__main__":
    main()
