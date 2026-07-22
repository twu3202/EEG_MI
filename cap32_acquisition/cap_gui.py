#!/usr/bin/env python
"""Cap32 acquisition GUI — our clean reimplementation of the vendor main_ui.

Direct acquisition (no LSL): a background receiver reads the cap over UDP/TCP, parses
the `0xA0 [seq] …0xC0` frames (FRAME_LEN = n_ch*3+9) and applies the vendor-style
real-time filter (Butterworth band/low-pass + 50 Hz iir-notch + baseline removal).
Mirrors the vendor architecture (IO thread → parse → filtered ring buffer → Qt view)
but in our own code, reusing src/common + src/acquisition.

  python src/acquisition/cap_gui.py                          # synthetic source (no hardware)
  python src/acquisition/cap_gui.py --source udp --host 192.168.4.1 --port 8086
  python src/acquisition/cap_gui.py --source tcp --host 192.168.4.1 --port <tcp_port>
  python src/acquisition/cap_gui.py --screenshot results/cap_gui_preview.png
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
sys.path.insert(0, str(HERE))            # src/
sys.path.insert(0, str(HERE))                   # src/acquisition/
from montage import CAP32_CHANNELS       # noqa: E402
from rt_filter import RealTimeEEGFilter          # noqa: E402

WINDOW_S = 5.0
SPACING_UV = 130.0


# ------------------------------------------------------------------ ring buffer
class Ring:
    def __init__(self, n_ch, n):
        self.buf = np.zeros((n_ch, n), dtype=np.float32)
        self.lock = threading.Lock()

    def append(self, chunk):                     # chunk (n_ch, m)
        m = chunk.shape[1]
        if m == 0:
            return
        with self.lock:
            self.buf = np.roll(self.buf, -m, axis=1)
            self.buf[:, -m:] = chunk

    def snapshot(self):
        with self.lock:
            return self.buf.copy()


# --------------------------------------------------------------------- receiver
class Receiver(threading.Thread):
    """Reads a source, filters chunks, writes filtered→disp ring, raw→raw ring."""

    def __init__(self, source_kind, host, port, fs, n_ch, disp, raw, filt, report):
        super().__init__(daemon=True)
        self.kind, self.host, self.port = source_kind, host, port
        self.fs, self.n_ch = fs, n_ch
        self.disp, self.raw, self.filt, self.report = disp, raw, filt, report
        self.running = True
        self.lost = self.n = 0
        self._last_seq = None
        self.src = None                  # set once connected; lets the GUI send %/*
        self.recording = False
        self._rec, self._rec_trig = [], []

    def start_rec(self):
        self._rec, self._rec_trig = [], []
        self.recording = True

    def stop_rec(self):
        self.recording = False
        if not self._rec:
            return None
        data = np.concatenate(self._rec, axis=1)          # (n_ch, N) RAW µV (pre-CAR)
        trig = np.concatenate(self._rec_trig)
        return data, trig

    def _record(self, raw_chunk, trigger):
        if self.recording:
            self._rec.append(raw_chunk.astype(np.float32))
            self._rec_trig.append(np.full(raw_chunk.shape[1], trigger, dtype=np.int32))

    def send(self, data: bytes):
        if self.src is not None:
            self.src.send(data)

    def _emit(self, raw_chunk):
        # Common Average Reference (robust, median) — removes common-mode / floating-REF
        # drift shared across channels. The vendor GUI does this (data -= data.mean(axis=0))
        # before filtering; without it a drifting reference rails every channel.
        car = raw_chunk - np.median(raw_chunk, axis=0, keepdims=True)
        self.raw.append(car.astype(np.float32))
        self.disp.append(self.filt.process(car).astype(np.float32))
        self.n += raw_chunk.shape[1]

    def _stat(self):
        self.report(f"{self.kind} · {self.n} samples · dropped {self.lost} "
                    f"({100*self.lost/max(1,self.n+self.lost):.2f}%)")

    def run(self):
        try:
            if self.kind == "synth":
                from synth import SynthCap
                cap = SynthCap(sfreq=self.fs)
                step = max(1, int(self.fs / 60))
                while self.running:
                    chunk = cap.get_chunk(step)
                    self._emit(chunk)
                    self._record(chunk, 0)
                    self._stat()
                    time.sleep(step / self.fs)
                return
            # hardware: udp/tcp via the shared parser
            from udp_lsl_bridge import UdpSource, TcpSource, parse_packet, board_init
            self.report(f"connecting {self.kind}://{self.host}:{self.port} …")
            self.src = (UdpSource(self.host, self.port) if self.kind == "udp"
                        else TcpSource(self.host, self.port))
            board_init(self.src, self.fs)   # 'b' start -> rate -> '*' EEG mode (leaves impedance)
            self.report(f"connected · sent init (b / rate / *) · waiting for frames …")
            for pkt in self.src.frames():
                if not self.running:
                    break
                parsed = parse_packet(pkt)
                if parsed is None:
                    continue
                sample, seq, trigger = parsed
                if self._last_seq is not None:
                    self.lost += (seq - self._last_seq - 1) % 256
                self._last_seq = seq
                s = sample.reshape(self.n_ch, 1)
                self._emit(s)
                self._record(s, trigger)
                if self.n % (int(self.fs) * 2) == 0:
                    self._stat()
        except Exception as e:
            self.report(f"⚠ {type(e).__name__}: {e}  —  连到帽子的 WiFi 热点(ESPBCI)了吗？")
            self.running = False

    def stop(self):
        self.running = False


# --------------------------------------------------------------- quality proxy
def quality(x, fs):
    x0 = x - x.mean()
    rms = float(np.std(x0))
    if rms < 1.0 or rms > 120.0 or np.ptp(x0) > 500.0:
        return "bad", rms
    w = np.hanning(len(x0))
    X = np.abs(np.fft.rfft(x0 * w))
    f = np.fft.rfftfreq(len(x0), 1 / fs)
    line = X[np.argmin(np.abs(f - 50.0))] / (X.sum() + 1e-9)
    return ("warn", rms) if (rms > 40 or line > 0.25) else ("good", rms)


_COL = {"good": "#2e9e5b", "warn": "#e0a800", "bad": "#d1495b"}


def _hdr(QtWidgets, text):
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet("font-size:13px;font-weight:600;color:#e8edf4;")
    return lbl


def save_recording(data, trig, fs, ch_names, outdir="recordings"):
    """Save RAW µV data (n_ch, N) + per-sample trigger to .npz, and an MNE .fif with
    trigger onsets as annotations (markers) for the MI pipeline."""
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(outdir, f"cap32_{stamp}")
    np.savez_compressed(base + ".npz", data=data.astype(np.float32),
                        trigger=trig.astype(np.int32), fs=float(fs),
                        ch_names=np.array(list(ch_names)))
    try:
        import mne
        info = mne.create_info(list(ch_names), fs, "eeg")
        raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")   # µV -> V
        info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         match_case=False, on_missing="ignore")
        onsets = np.where((trig[1:] != trig[:-1]) & (trig[1:] != 0))[0] + 1   # trigger edges
        if len(onsets):
            raw.set_annotations(mne.Annotations(
                onset=onsets / fs, duration=0.0,
                description=[f"T{int(trig[o])}" for o in onsets]))
        raw.save(base + "_raw.fif", overwrite=True, verbose="ERROR")
    except Exception as e:
        print("(.fif save skipped:", e, ")")
    return base


# ------------------------------------------------------------------ the widget
def build(fs, source_kind, host, port, note=""):
    import pyqtgraph as pg
    from PyQt6 import QtWidgets, QtCore

    pg.setConfigOption("background", "#0e1116")
    pg.setConfigOption("foreground", "#c8ced8")
    pg.setConfigOptions(antialias=True)

    ch = CAP32_CHANNELS
    nch = len(ch)
    W = int(WINDOW_S * fs)
    disp, raw = Ring(nch, W), Ring(nch, W)
    filt = RealTimeEEGFilter(fs, nch, lowcut=0.0, highcut=40.0, notch_freq=50.0, baseline=False)
    baselines = [(nch - 1 - i) * SPACING_UV for i in range(nch)]

    root = QtWidgets.QWidget()
    root.setStyleSheet("background:#0e1116;color:#c8ced8;font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;")
    root.resize(1320, 820)
    outer = QtWidgets.QVBoxLayout(root); outer.setContentsMargins(10, 8, 10, 8)

    # ---- control bar (mirrors vendor: connect / rate / filters / record) ----
    bar = QtWidgets.QHBoxLayout()
    def chip(txt, col="#3a3f4b"):
        b = QtWidgets.QPushButton(txt)
        b.setStyleSheet(f"background:{col};color:#eef2f8;border:none;padding:6px 11px;border-radius:6px;")
        return b
    title = QtWidgets.QLabel("Cap32  ·  32-ch ADS1299"); title.setStyleSheet("font-size:16px;font-weight:600;color:#e8edf4;")
    src_cb = QtWidgets.QComboBox(); src_cb.addItems(["synth", "udp", "tcp"]); src_cb.setCurrentText(source_kind)
    host_e = QtWidgets.QLineEdit(host); host_e.setFixedWidth(110)
    port_e = QtWidgets.QLineEdit(str(port)); port_e.setFixedWidth(56)
    rate_cb = QtWidgets.QComboBox(); rate_cb.addItems(["250", "500", "1000"]); rate_cb.setCurrentText(str(int(fs)))
    low_e = QtWidgets.QLineEdit("0"); low_e.setFixedWidth(38)
    high_e = QtWidgets.QLineEdit("40"); high_e.setFixedWidth(38)
    notch_cb = QtWidgets.QCheckBox("50Hz notch"); notch_cb.setChecked(True)
    btn_conn = chip("● Connect", "#2b6cb0")
    btn_rec = chip("● Record", "#3a3f4b")
    for w in (title,):
        bar.addWidget(w)
    bar.addStretch(1)
    for lbl, w in [("src", src_cb), ("host", host_e), ("port", port_e), ("Hz", rate_cb),
                   ("low", low_e), ("high", high_e)]:
        l = QtWidgets.QLabel(lbl); l.setStyleSheet("color:#8b95a5;"); bar.addWidget(l); bar.addWidget(w)
    bar.addWidget(notch_cb); bar.addWidget(btn_conn); bar.addWidget(btn_rec)
    outer.addLayout(bar)

    stat = QtWidgets.QLabel("idle"); stat.setStyleSheet("color:#8b95a5;font-size:11px;"); outer.addWidget(stat)

    body = QtWidgets.QHBoxLayout(); outer.addLayout(body, 1)

    # ---- scope ----
    axis = pg.AxisItem("left"); axis.setTicks([[(baselines[i], ch[i]) for i in range(nch)]]); axis.setWidth(52)
    plot = pg.PlotWidget(axisItems={"left": axis}); plot.setMenuEnabled(False)
    plot.showGrid(x=True, y=False, alpha=0.15); plot.setXRange(0, WINDOW_S, padding=0)
    plot.setYRange(-SPACING_UV, nch * SPACING_UV, padding=0); plot.setLabel("bottom", "time", units="s")
    tvec = np.linspace(0, WINDOW_S, W)
    curves = [plot.plot(tvec, disp.buf[i] + baselines[i],
                        pen=pg.mkPen("#d1495b" if ch[i] == "T7" else "#6fd3b8", width=1)) for i in range(nch)]
    body.addWidget(plot, 1)

    # ---- right column: live spectrum (top) + signal quality (bottom) ----
    right = QtWidgets.QWidget(); right.setFixedWidth(360)
    rv = QtWidgets.QVBoxLayout(right); rv.setContentsMargins(6, 0, 0, 0); rv.setSpacing(4)

    rv.addWidget(_hdr(QtWidgets, "Spectrum  ·  µV vs Hz (live, all-ch mean)"))
    fft_plot = pg.PlotWidget(); fft_plot.setMenuEnabled(False)
    fft_plot.setLogMode(False, True)                     # log amplitude (µV)
    fft_plot.setXRange(0, 60, padding=0); fft_plot.setLimits(xMin=0, xMax=fs / 2)
    fft_plot.showGrid(x=True, y=True, alpha=0.15)
    fft_plot.setLabel("bottom", "frequency", units="Hz")
    fft_plot.setMinimumHeight(230)
    for lo, hi, col in [(8, 13, (90, 200, 130, 45)), (13, 30, (90, 140, 210, 35))]:
        reg = pg.LinearRegionItem([lo, hi], movable=False, brush=col)
        reg.setZValue(-10); fft_plot.addItem(reg)       # μ/α (8–13) & β (13–30) bands
    fft_plot.addItem(pg.InfiniteLine(50, angle=90,
                     pen=pg.mkPen("#d1495b", style=QtCore.Qt.PenStyle.DashLine)))  # 50 Hz line
    freqs = np.fft.rfftfreq(W, 1 / fs)
    fft_all = fft_plot.plot([], [], pen=pg.mkPen("#6fd3b8", width=2))
    fft_post = fft_plot.plot([], [], pen=pg.mkPen("#e0a800", width=1))            # posterior (α)
    post_idx = [i for i, c in enumerate(ch) if c in
                {"O1", "O2", "OZ", "PO3", "PO4", "P3", "P4", "PZ"}]
    rv.addWidget(fft_plot, 3)

    rv.addWidget(_hdr(QtWidgets, "Signal quality  ·  impedance proxy   "
                      "<span style='font-weight:400;font-size:11px'>"
                      "<span style='color:#2e9e5b'>■</span> good "
                      "<span style='color:#e0a800'>■</span> noisy "
                      "<span style='color:#d1495b'>■</span> bad</span>"))
    grid = QtWidgets.QGridLayout(); grid.setSpacing(3); rv.addLayout(grid, 4)
    cells = []
    for i in range(nch):
        r, c0 = i % 16, (i // 16) * 3
        dot = QtWidgets.QLabel("●"); dot.setStyleSheet("color:#2e9e5b;font-size:13px;")
        nm = QtWidgets.QLabel(ch[i]); nm.setStyleSheet("color:#c8ced8;font-size:11px;"); nm.setFixedWidth(38)
        val = QtWidgets.QLabel("–"); val.setStyleSheet("color:#8b95a5;font-size:11px;"); val.setFixedWidth(54)
        grid.addWidget(dot, r, c0); grid.addWidget(nm, r, c0 + 1); grid.addWidget(val, r, c0 + 2)
        cells.append((dot, val))
    body.addWidget(right)

    ctx = dict(root=root, plot=plot, curves=curves, cells=cells, disp=disp, raw=raw, filt=filt,
               tvec=tvec, baselines=baselines, fs=fs, nch=nch, stat=stat, note=note, W=W,
               fft_all=fft_all, fft_post=fft_post, freqs=freqs, post_idx=post_idx,
               ctrls=dict(src=src_cb, host=host_e, port=port_e, rate=rate_cb, low=low_e,
                          high=high_e, notch=notch_cb, conn=btn_conn, rec=btn_rec))
    return ctx


def refresh_scope(ctx):
    b = ctx["disp"].snapshot()
    for i, c in enumerate(ctx["curves"]):
        y = b[i] - b[i].mean()   # center each channel now (don't wait for a slow baseline)
        c.setData(ctx["tvec"], np.clip(y, -SPACING_UV, SPACING_UV) + ctx["baselines"][i])


def refresh_quality(ctx):
    b = ctx["raw"].snapshot()
    for i, (dot, val) in enumerate(ctx["cells"]):
        s, rms = quality(b[i], ctx["fs"])
        dot.setStyleSheet(f"color:{_COL[s]};font-size:13px;")
        val.setText(f"{rms:5.1f} µV"); val.setStyleSheet(f"color:{_COL[s]};font-size:11px;")


def refresh_fft(ctx):
    b = ctx["raw"].snapshot()                          # (nch, W) CAR'd µV
    win = np.hanning(b.shape[1])
    xw = (b - b.mean(1, keepdims=True)) * win
    amp = np.abs(np.fft.rfft(xw, axis=1)) * (2.0 / win.sum())   # single-sided µV amplitude
    f = ctx["freqs"]
    m = f >= 0.5                                        # skip DC
    allm = np.clip(amp.mean(0)[m], 1e-3, None)
    ctx["fft_all"].setData(f[m], allm)
    if ctx["post_idx"]:
        postm = np.clip(amp[ctx["post_idx"]].mean(0)[m], 1e-3, None)
        ctx["fft_post"].setData(f[m], postm)


def run_live(fs, source_kind, host, port):
    from PyQt6 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(fs, source_kind, host, port)
    ctx["root"].setWindowTitle("Cap32 acquisition")
    rec = {"thread": None}

    def report(msg):  # thread-safe status update from the receiver thread
        QtCore.QMetaObject.invokeMethod(
            ctx["stat"], "setText", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, msg))

    def apply_filter():
        try:
            lo, hi = float(ctx["ctrls"]["low"].text()), float(ctx["ctrls"]["high"].text())
        except ValueError:
            return
        ctx["filt"].update(ctx["fs"], lo, hi, 50.0 if ctx["ctrls"]["notch"].isChecked() else 0.0)
    for w in ("low", "high"):
        ctx["ctrls"][w].editingFinished.connect(apply_filter)
    ctx["ctrls"]["notch"].stateChanged.connect(apply_filter)

    def connect():
        if rec["thread"] and rec["thread"].is_alive():
            rec["thread"].stop(); ctx["ctrls"]["conn"].setText("● Connect")
            ctx["stat"].setText("disconnected"); return
        c = ctx["ctrls"]
        r = Receiver(c["src"].currentText(), c["host"].text(), int(c["port"].text()),
                     ctx["fs"], ctx["nch"], ctx["disp"], ctx["raw"], ctx["filt"], report)
        r.start(); rec["thread"] = r; c["conn"].setText("■ Disconnect")
    ctx["ctrls"]["conn"].clicked.connect(connect)

    def toggle_rec():
        r = rec["thread"]
        if not (r and r.is_alive()):
            ctx["stat"].setText("连接后再录制"); return
        if not r.recording:
            r.start_rec(); ctx["ctrls"]["rec"].setText("■ Stop rec")
        else:
            out = r.stop_rec(); ctx["ctrls"]["rec"].setText("● Record")
            if out is None:
                ctx["stat"].setText("no data recorded"); return
            data, trig = out
            base = save_recording(data, trig, ctx["fs"], CAP32_CHANNELS)
            ntrig = int((np.diff(trig) != 0).sum())
            ctx["stat"].setText(f"saved {base}.npz  ({data.shape[1]} samples, {ntrig} trigger edges)")
    ctx["ctrls"]["rec"].clicked.connect(toggle_rec)

    ctx["root"].show()
    t1 = QtCore.QTimer(); t1.timeout.connect(lambda: refresh_scope(ctx)); t1.start(33)
    t2 = QtCore.QTimer(); t2.timeout.connect(lambda: refresh_quality(ctx)); t2.start(500)
    t3 = QtCore.QTimer(); t3.timeout.connect(lambda: refresh_fft(ctx)); t3.start(300)
    QtCore.QTimer.singleShot(200, connect)   # auto-connect on launch (uses --source)
    app.exec()


def screenshot(fs, out):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    from synth import SynthCap
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(fs, "synth", "192.168.4.1", 8086, note="synthetic preview")
    ctx["ctrls"]["conn"].setText("■ Disconnect")
    # synchronously fill the window with filtered synthetic data
    cap = SynthCap(sfreq=fs)
    step = max(1, int(fs // 5))
    for _ in range(int((WINDOW_S + 1) * fs / step)):
        chunk = cap.get_chunk(step)
        ctx["raw"].append(chunk)
        ctx["disp"].append(ctx["filt"].process(chunk).astype(np.float32))
    ctx["stat"].setText("synthetic preview · filtered 0–40 Hz + 50 Hz notch + baseline removal")
    ctx["root"].show(); app.processEvents()
    refresh_scope(ctx); refresh_quality(ctx); refresh_fft(ctx); app.processEvents()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ctx["root"].grab().save(str(out)); print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="synth", choices=["synth", "udp", "tcp"])
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--screenshot", metavar="PATH", default=None)
    args = ap.parse_args()
    if args.screenshot:
        screenshot(args.sfreq, args.screenshot)
    else:
        run_live(args.sfreq, args.source, args.host, args.port)


if __name__ == "__main__":
    main()
