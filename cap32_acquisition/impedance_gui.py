#!/usr/bin/env python
"""Impedance / electrode-contact UI — live head topomap.

Sends the board into impedance mode (`b'%'`, 31.2 Hz injection — CONFIRMED real) and
shows per-electrode contact on a scalp map. Coloured by the **amplitude at 31.2 Hz**
(lower = lower impedance = better contact), with:
  - saturation/rail detection: a floating or drifted (polarised) dry electrode rails
    the amplifier; a clipped signal's 31.2 Hz amplitude is meaningless, so railed
    channels are flagged RAIL (dark red) instead of flickering green/red;
  - temporal smoothing (EMA) to stop the flicker;
  - value labels placed off the dots.
The vendor's absolute-kΩ calibration is for wet electrodes — treat kΩ as rough; judge
by colour / relative amplitude. Restores EEG mode (`b'*'`) on exit.

  python src/acquisition/impedance_gui.py --host 192.168.4.1 --port 8086
  python src/acquisition/impedance_gui.py --screenshot results/impedance_preview.png
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE))
from montage import CAP32_CHANNELS, ADC_MICROVOLTS_PER_COUNT  # noqa: E402
from cap_gui import Ring  # noqa: E402
from impedance import impedance, F0  # noqa: E402

NCH = len(CAP32_CHANNELS)
WIN_S = 2.0
# EMPIRICAL (from a controlled test): on this board, better contact => HIGHER amp@31.2Hz
# (the board drives the body and each electrode measures pickup). So higher = better.
AMP_GOOD, AMP_OK = 2500.0, 1200.0         # µV @31.2Hz: >good green, >ok amber, else red
INJ_MIN = 50.0                            # median amp above this => board is injecting
RAIL_UV = 0.80 * (2 ** 23) * ADC_MICROVOLTS_PER_COUNT   # ≈150 mV: near the ADS1299 rail
EMA = 0.8                                  # heavy smoothing — reading is noisy (CV≈2)


def head_positions():
    from montage import CAP32_XY   # precomputed 10-20 positions (no MNE needed)
    return np.array([CAP32_XY[c] for c in CAP32_CHANNELS])


def acolor(amp, railed, injected):
    if railed:
        return "#8b1e2d"                   # dark red — railed/saturated (no valid contact)
    if not injected:
        return "#5a6472"                   # gray — no injection detected
    if amp > AMP_GOOD:
        return "#2e9e5b"                   # high amp = good contact
    if amp > AMP_OK:
        return "#e0a800"
    return "#d1495b"                        # low amp = poor contact


class Reader(threading.Thread):
    def __init__(self, host, port, fs, ring, report):
        super().__init__(daemon=True)
        self.host, self.port, self.fs, self.ring, self.report = host, port, fs, ring, report
        self.running = True
        self.src = None
        self.n = 0

    def run(self):
        try:
            from udp_lsl_bridge import UdpSource, parse_packet, START, RATE_CMD, IMPEDANCE_MODE
            self.src = UdpSource(self.host, self.port)
            time.sleep(0.3)
            self.src.send(START); time.sleep(0.2)
            self.src.send(RATE_CMD.get(int(self.fs), b"1")); time.sleep(0.1)
            self.src.send(IMPEDANCE_MODE); time.sleep(0.3)
            self.report("impedance mode ('%') sent — measuring…")
            for pkt in self.src.frames():
                if not self.running:
                    break
                p = parse_packet(pkt)
                if p:
                    self.ring.append(p[0].reshape(NCH, 1))
                    self.n += 1
        except Exception as e:
            self.report(f"⚠ {type(e).__name__}: {e} — 连到 ESPBCI 且 IP=192.168.4.2 了吗？")

    def stop(self):
        self.running = False
        try:
            from udp_lsl_bridge import EEG_MODE
            if self.src:
                self.src.send(EEG_MODE)
        except Exception:
            pass


def measure(ring, fs):
    b = ring.snapshot()                                   # (32, W) µV
    amp = np.empty(NCH)
    railed = np.max(np.abs(b), axis=1) > RAIL_UV          # saturated -> invalid
    for c in range(NCH):
        amp[c] = impedance(b[c], fs)[1]
    valid = amp[~railed]
    injected = valid.size > 0 and np.median(valid) >= INJ_MIN
    return amp, railed, injected


def build(fs):
    import pyqtgraph as pg
    from PyQt6 import QtWidgets
    pg.setConfigOption("background", "#0e1116"); pg.setConfigOption("foreground", "#c8ced8")
    pg.setConfigOptions(antialias=True)

    root = QtWidgets.QWidget(); root.resize(760, 820)
    root.setStyleSheet("background:#0e1116;color:#c8ced8;font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;")
    v = QtWidgets.QVBoxLayout(root); v.setContentsMargins(10, 8, 10, 8)
    title = QtWidgets.QLabel("Electrode CONTACT strength  ·  press to improve (higher % = better; "
                             "signal pickup, NOT a calibrated impedance)")
    title.setStyleSheet("font-size:16px;font-weight:600;color:#e8edf4;"); v.addWidget(title)
    stat = QtWidgets.QLabel("connecting…"); stat.setStyleSheet("color:#8b95a5;font-size:12px;"); v.addWidget(stat)

    plot = pg.PlotWidget(); plot.setMenuEnabled(False); plot.hideAxis("left"); plot.hideAxis("bottom")
    plot.setAspectLocked(True); plot.setXRange(-1.4, 1.4); plot.setYRange(-1.4, 1.5)
    v.addWidget(plot, 1)
    th = np.linspace(0, 2 * np.pi, 120)
    plot.plot(np.cos(th) * 1.18, np.sin(th) * 1.18, pen=pg.mkPen("#3a4150", width=2))
    plot.plot([-0.12, 0, 0.12], [1.16, 1.35, 1.16], pen=pg.mkPen("#3a4150", width=2))

    xy = head_positions()
    scatter = pg.ScatterPlotItem(size=22, pen=pg.mkPen("#0e1116", width=2)); plot.addItem(scatter)
    labels = []
    for i, c in enumerate(CAP32_CHANNELS):
        nm = pg.TextItem(c, color="#9aa3b2", anchor=(0.5, 0.5)); nm.setPos(xy[i, 0], xy[i, 1] + 0.135)
        nm.setScale(0.72); plot.addItem(nm)
        vt = pg.TextItem("–", color="#8b95a5", anchor=(0.5, 0.5)); vt.setPos(xy[i, 0], xy[i, 1] - 0.135)
        vt.setScale(0.78); plot.addItem(vt); labels.append(vt)

    legend = QtWidgets.QLabel(
        "<span style='color:#2e9e5b'>■</span> good contact   "
        "<span style='color:#e0a800'>■</span> ok   "
        "<span style='color:#d1495b'>■</span> poor (press it)   "
        "<span style='color:#8b1e2d'>■</span> RAIL   "
        "<span style='color:#5a6472'>■</span> no injection")
    legend.setStyleSheet("font-size:12px;"); v.addWidget(legend)
    return dict(root=root, stat=stat, scatter=scatter, labels=labels, xy=xy, fs=fs, ema=None)


def refresh(ctx, ring, reader):
    import pyqtgraph as pg
    if reader is not None and reader.n < ctx["fs"]:
        ctx["stat"].setText(f"waiting for frames… ({reader.n} samples received)")
        return
    amp, railed, injected = measure(ring, ctx["fs"])
    ctx["ema"] = amp if ctx["ema"] is None else EMA * ctx["ema"] + (1 - EMA) * amp
    a = ctx["ema"]
    spots = []
    for i in range(NCH):
        col = acolor(a[i], railed[i], injected)
        spots.append({"pos": ctx["xy"][i], "brush": pg.mkBrush(col)})
        pct = int(np.clip((a[i] - 1000.0) / 3000.0, 0, 1) * 100)   # ~contact strength %
        ctx["labels"][i].setText("RAIL" if railed[i] else (f"{pct}%" if injected else "?"))
        ctx["labels"][i].setColor(col)
    ctx["scatter"].setData(spots)
    if injected:
        ng = int(((a > AMP_GOOD) & ~railed).sum()); nr = int(railed.sum())
        ctx["stat"].setText(f"✅ 注入正常 · {ng}/{NCH} 接触好(绿) · {nr} railed · "
                            f"% = 接触强度(注入拾取量),越高越好,不是真实阻抗 · 压一压会升高")
    else:
        ctx["stat"].setText(f"⚠ 未检测到 31.2Hz 注入 (median {np.median(amp[~railed]) if (~railed).any() else 0:.0f} µV)"
                            " — 没连上/没在阻抗模式,或全部 railed")


def run_live(host, port, fs):
    from PyQt6 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(fs); ctx["root"].setWindowTitle("Cap32 impedance")
    ring = Ring(NCH, int(WIN_S * fs))
    reader = Reader(host, port, fs, ring,
                    lambda m: QtCore.QMetaObject.invokeMethod(
                        ctx["stat"], "setText", QtCore.Qt.ConnectionType.QueuedConnection,
                        QtCore.Q_ARG(str, m)))
    reader.start(); app.aboutToQuit.connect(reader.stop)
    ctx["root"].show()
    t = QtCore.QTimer(); t.timeout.connect(lambda: refresh(ctx, ring, reader)); t.start(500)
    app.exec()


def _fill_demo(ring, fs):
    xy = head_positions()
    r = np.hypot(xy[:, 0], xy[:, 1])
    N = int(WIN_S * fs); t = np.arange(N) / fs
    rng = np.random.default_rng(3)
    for _ in range(3):
        chunk = np.empty((NCH, N))
        for c in range(NCH):
            v1 = 5500 - r[c] * 4200 + rng.normal(0, 300)   # center high(good) -> edge low(poor)
            chunk[c] = v1 * np.sin(2 * np.pi * F0 * t) + rng.normal(0, 40, N)
        chunk[14] = RAIL_UV * 1.2 * np.sign(np.sin(2 * np.pi * 3 * t))   # T7 = railed demo
        ring.append(chunk)


def screenshot(out, fs):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(fs)
    ring = Ring(NCH, int(WIN_S * fs)); _fill_demo(ring, fs)
    ctx["root"].show(); app.processEvents(); refresh(ctx, ring, None); app.processEvents()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ctx["root"].grab().save(str(out)); print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.4.1"); ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=int, default=250)
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()
    if args.screenshot:
        screenshot(args.screenshot, args.sfreq)
    else:
        run_live(args.host, args.port, args.sfreq)


if __name__ == "__main__":
    main()
