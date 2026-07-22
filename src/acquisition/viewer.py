#!/usr/bin/env python
"""Simple 32-ch live EEG viewer (PyQtGraph) for the cap — scrolling traces + a
per-channel signal-quality panel (our software substitute for the missing impedance
hardware) + a marker button for MI cues.

Data source is pluggable:
  --source synth   built-in synthetic 32-ch EEG (no hardware) [default]
  --source lsl     read the 'Cap32' LSL stream from udp_lsl_bridge.py

Render a still preview (no window/display needed):
  python viewer.py --screenshot results/ui_preview.png
Live:
  python viewer.py --source synth
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.montage import CAP32_CHANNELS  # noqa: E402

WINDOW_S = 4.0          # seconds shown
SPACING_UV = 130.0      # vertical gap between channel baselines (µV)


# ---------------------------------------------------------------- quality proxy
def channel_quality(x: np.ndarray, fs: float):
    """Return (status, rms_uv). status in {'good','warn','bad'}."""
    x0 = x - x.mean()
    rms = float(np.std(x0))
    ptp = float(np.ptp(x0))
    # 50 Hz line ratio via Hann-windowed FFT
    w = np.hanning(len(x0))
    X = np.abs(np.fft.rfft(x0 * w))
    freqs = np.fft.rfftfreq(len(x0), 1 / fs)
    line = X[np.argmin(np.abs(freqs - 50.0))]
    line_ratio = float(line / (X.sum() + 1e-9))
    if rms < 1.0 or rms > 120.0 or ptp > 500.0:
        return "bad", rms
    if rms > 40.0 or line_ratio > 0.25:
        return "warn", rms
    return "good", rms


_COLORS = {"good": "#2e9e5b", "warn": "#e0a800", "bad": "#d1495b"}


# --------------------------------------------------------------------- sources
class SynthSource:
    kind = "synth"

    def __init__(self, fs):
        from synth import SynthCap
        self.cap = SynthCap(sfreq=fs)
        self.fs = fs

    def get_chunk(self, n):
        return self.cap.get_chunk(n)


class LslSource:
    kind = "lsl"

    def __init__(self, fs):
        from pylsl import StreamInlet, resolve_byprop
        streams = resolve_byprop("name", "Cap32", timeout=5)
        if not streams:
            raise SystemExit("no 'Cap32' LSL stream found — start udp_lsl_bridge.py first")
        self.inlet = StreamInlet(streams[0], max_buflen=4)
        self.fs = fs

    def get_chunk(self, n):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=n)
        if not chunk:
            return np.zeros((len(CAP32_CHANNELS), 0), dtype=np.float32)
        return np.asarray(chunk, dtype=np.float32).T  # (ch, n)


# ------------------------------------------------------------------ the widget
def build_app(source, fs, title_suffix=""):
    import pyqtgraph as pg
    from PyQt6 import QtWidgets, QtCore, QtGui

    pg.setConfigOption("background", "#0e1116")
    pg.setConfigOption("foreground", "#c8ced8")
    pg.setConfigOptions(antialias=True)

    W = int(WINDOW_S * fs)
    ch = CAP32_CHANNELS
    nch = len(ch)
    buf = np.zeros((nch, W), dtype=np.float32)
    baselines = [(nch - 1 - i) * SPACING_UV for i in range(nch)]

    root = QtWidgets.QWidget()
    root.setStyleSheet("background:#0e1116; color:#c8ced8; font-family:-apple-system,Helvetica,Arial;")
    root.resize(1240, 780)
    outer = QtWidgets.QVBoxLayout(root)
    outer.setContentsMargins(10, 8, 10, 8)

    # header
    header = QtWidgets.QHBoxLayout()
    title = QtWidgets.QLabel(f"Cap32  ·  32-ch ADS1299 EEG{title_suffix}")
    title.setStyleSheet("font-size:17px; font-weight:600; color:#e8edf4;")
    info = QtWidgets.QLabel(f"{fs:.0f} Hz   ·   source: {source.kind}   ·   window {WINDOW_S:.0f}s")
    info.setStyleSheet("color:#8b95a5;")
    btn_mark = QtWidgets.QPushButton("● Marker (T)")
    btn_mark.setStyleSheet("background:#2b6cb0; color:white; border:none; padding:6px 12px; border-radius:6px; font-weight:600;")
    btn_stream = QtWidgets.QPushButton("■ Stop")
    btn_stream.setStyleSheet("background:#3a3f4b; color:#dfe4ec; border:none; padding:6px 12px; border-radius:6px;")
    header.addWidget(title)
    header.addStretch(1)
    header.addWidget(info)
    header.addSpacing(14)
    header.addWidget(btn_stream)
    header.addWidget(btn_mark)
    outer.addLayout(header)

    body = QtWidgets.QHBoxLayout()
    outer.addLayout(body, 1)

    # ---- left: scrolling traces ----
    yticks = [[(baselines[i], ch[i]) for i in range(nch)]]
    axis = pg.AxisItem(orientation="left")
    axis.setTicks(yticks)
    axis.setStyle(tickTextOffset=6)
    plot = pg.PlotWidget(axisItems={"left": axis})
    plot.setMenuEnabled(False)
    plot.showGrid(x=True, y=False, alpha=0.15)
    plot.setXRange(0, WINDOW_S, padding=0)
    plot.setYRange(-SPACING_UV, nch * SPACING_UV, padding=0)
    plot.setLabel("bottom", "time", units="s")
    plot.getAxis("left").setWidth(56)
    tvec = np.linspace(0, WINDOW_S, W)
    curves = []
    for i in range(nch):
        color = "#d1495b" if ch[i] == "T7" else "#6fd3b8"  # bad channel red, rest teal
        c = plot.plot(tvec, buf[i] + baselines[i], pen=pg.mkPen(color, width=1))
        curves.append(c)
    # scale bar
    sb = pg.TextItem(f"scale: {SPACING_UV:.0f} µV / division", color="#8b95a5", anchor=(0, 1))
    sb.setPos(0.05, nch * SPACING_UV)
    plot.addItem(sb)
    body.addWidget(plot, 1)

    # ---- right: signal-quality panel ----
    qpanel = QtWidgets.QWidget()
    qpanel.setFixedWidth(300)
    qv = QtWidgets.QVBoxLayout(qpanel)
    qv.setContentsMargins(6, 2, 2, 2)
    qtitle = QtWidgets.QLabel("Signal quality  (impedance proxy)")
    qtitle.setStyleSheet("font-size:13px; font-weight:600; color:#e8edf4; padding-bottom:4px;")
    qv.addWidget(qtitle)
    grid = QtWidgets.QGridLayout()
    grid.setSpacing(3)
    qv.addLayout(grid)
    qv.addStretch(1)
    legend = QtWidgets.QLabel("<span style='color:#2e9e5b'>■ good</span>   "
                              "<span style='color:#e0a800'>■ noisy</span>   "
                              "<span style='color:#d1495b'>■ bad</span>")
    legend.setStyleSheet("font-size:11px;")
    qv.addWidget(legend)

    cells = []  # (name_label, value_label, dot)
    for i in range(nch):
        r, col = i % 16, (i // 16) * 3
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet("color:#2e9e5b; font-size:14px;")
        name = QtWidgets.QLabel(ch[i])
        name.setStyleSheet("color:#c8ced8; font-size:11px;")
        name.setFixedWidth(40)
        val = QtWidgets.QLabel("–")
        val.setStyleSheet("color:#8b95a5; font-size:11px;")
        val.setFixedWidth(56)
        grid.addWidget(dot, r, col)
        grid.addWidget(name, r, col + 1)
        grid.addWidget(val, r, col + 2)
        cells.append((dot, val))
    body.addWidget(qpanel)

    # ---- update logic ----
    state = {"streaming": True}

    def refresh_quality():
        for i in range(nch):
            status, rms = channel_quality(buf[i], fs)
            dot, val = cells[i]
            dot.setStyleSheet(f"color:{_COLORS[status]}; font-size:14px;")
            val.setText(f"{rms:5.1f} µV")
            val.setStyleSheet(f"color:{_COLORS[status]}; font-size:11px;")

    def pull_and_draw(n):
        nonlocal buf
        new = source.get_chunk(n)
        if new.shape[1] == 0:
            return
        m = new.shape[1]
        buf = np.roll(buf, -m, axis=1)
        buf[:, -m:] = new
        for i in range(nch):
            curves[i].setData(tvec, np.clip(buf[i], -SPACING_UV, SPACING_UV) + baselines[i])

    def on_marker():
        print("MARKER pushed (T)")  # live: push to Markers outlet / send TX to board

    btn_mark.clicked.connect(on_marker)

    def toggle():
        state["streaming"] = not state["streaming"]
        btn_stream.setText("▶ Start" if not state["streaming"] else "■ Stop")
    btn_stream.clicked.connect(toggle)

    return root, pull_and_draw, refresh_quality, state, int(fs)


def run_live(source, fs):
    from PyQt6 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    root, pull_and_draw, refresh_quality, state, fsi = build_app(source, fs)
    root.setWindowTitle("Cap32 EEG viewer")
    root.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: state["streaming"] and pull_and_draw(max(1, fsi // 30)))
    timer.start(33)
    qtimer = QtCore.QTimer()
    qtimer.timeout.connect(refresh_quality)
    qtimer.start(500)
    app.exec()


def screenshot(source, fs, out_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    root, pull_and_draw, refresh_quality, state, fsi = build_app(source, fs, title_suffix="   (synthetic preview)")
    root.show()
    app.processEvents()
    # prefill the whole window with data
    filled = 0
    W = int(WINDOW_S * fs)
    while filled < W + fsi:  # a little extra so blinks/rhythms develop
        pull_and_draw(fsi // 5)
        filled += fsi // 5
    refresh_quality()
    app.processEvents()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    root.grab().save(str(out_path))
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="synth", choices=["synth", "lsl"])
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--screenshot", metavar="PATH", default=None)
    args = ap.parse_args()

    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    src = SynthSource(args.sfreq) if args.source == "synth" else LslSource(args.sfreq)
    if args.screenshot:
        screenshot(src, args.sfreq, args.screenshot)
    else:
        run_live(src, args.sfreq)


if __name__ == "__main__":
    main()
