# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Cap32 acquisition GUI — clean reimplementation of the vendor main_ui (light theme).

Direct acquisition (no LSL): a background receiver reads the cap over UDP/TCP, parses
the `0xA0 [seq] …0xC0` frames (FRAME_LEN = n_ch*3+9) and applies the vendor-style
real-time filter (Butterworth band/low-pass + 50 Hz iir-notch + baseline removal).

Adds the MI experiment loop: **▶ MI Task** launches the left/right(/feet/rest) paradigm
(src/experiment/mi_paradigm.py) full-screen, auto-starts recording, stamps a per-sample
marker at each imagery onset (software track — guaranteed) AND sends the hardware `TXXXX`
to the board, then auto-saves. Feed the recording to src/analysis (load.py, erd_ers.py).

  python src/acquisition/cap_gui.py                          # synthetic source (no hardware)
  python src/acquisition/cap_gui.py --source udp --host 192.168.4.1 --port 8086
  python src/acquisition/cap_gui.py --source tcp --host 192.168.4.1 --port <tcp_port>
  python src/acquisition/cap_gui.py --screenshot results/cap_gui_preview.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # src/
sys.path.insert(0, str(HERE))                   # src/acquisition/
sys.path.insert(0, str(HERE.parent / "experiment"))   # src/experiment/
sys.path.insert(0, str(HERE.parent / "analysis"))     # src/analysis/ (LiveDeblink)
from montage import CAP32_CHANNELS       # noqa: E402
from rt_filter import RealTimeEEGFilter          # noqa: E402

WINDOW_S = 5.0
SPACING_UV = 130.0

# ---- light theme palette (OpenBCI-style cards on a soft-white page) ----
BG, CARD, LINE = "#eef1f5", "#ffffff", "#dfe3ea"
TXT, SUB = "#1f2733", "#6b7480"
ACC, TRACE, HILITE = "#2b6cb0", "#3a6ea5", "#c0392b"
GOOD, WARN, BAD = "#2e9e5b", "#c58a00", "#d1495b"
MI_HILITE = {"C3", "C4"}                         # emphasise the MI channels in the scope

# band-power widget (OpenBCI-style) + live head-map bands
BANDS = [("δ", 1, 4), ("θ", 4, 8), ("α/μ", 8, 13), ("β", 13, 30), ("γ", 30, 45)]
BAND_COLS = ["#6b7a8f", "#4a90d9", "#2e9e5b", "#c58a00", "#c0392b"]
HEAD_BANDS = {"μ 8–13 Hz": (8, 13), "β 13–30 Hz": (13, 30),
              "broadband 1–40": (1, 40), "α post 8–12": (8, 12)}
SCALE_UV = [50, 100, 150, 250, 500]              # vertical scale options (µV full-swing)


def head_xy():
    """2-D scalp positions for CAP32 channels (standard_1020), normalised to a unit circle."""
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    nm = {k.upper(): k for k in pos}
    xy = np.array([[pos[nm[c.upper()]][0], pos[nm[c.upper()]][1]] for c in CAP32_CHANNELS])
    xy = xy - xy.mean(0)
    return (xy / (np.abs(xy).max() * 1.15)).astype(float)


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
    """Reads a source, filters chunks, writes filtered→disp ring, raw→raw ring.
    Holds a software `marker` the paradigm sets at imagery onset — recorded per-sample
    alongside the hardware trigger, and used as the primary epoching label."""

    def __init__(self, source_kind, host, port, fs, n_ch, disp, raw, filt, report):
        super().__init__(daemon=True)
        self.kind, self.host, self.port = source_kind, host, port
        self.fs, self.n_ch = fs, n_ch
        self.disp, self.raw, self.filt, self.report = disp, raw, filt, report
        self.running = True
        self.lost = self.n = 0
        self._last_seq = None
        self.src = None                  # set once connected; lets the GUI send %/*/TXXXX
        self.recording = False
        self.marker = 0                  # software marker (set by the paradigm)
        self._rec, self._rec_trig, self._rec_marker, self._rec_gap = [], [], [], []
        self._rec_n = 0                  # samples recorded so far (for trial sample indices)
        self._last_sample = None
        self.filled = 0                  # samples reconstructed to cover dropped frames
        self.car = True                  # live toggle: common average reference
        self.deblink = None              # live toggle: a calibrated LiveDeblink operator or None
        self.calib = Ring(n_ch, int(30 * fs))   # rolling buffer for ICA calibration

    # ---- recording ----
    MAX_FILL = 2500                      # ≥10 s at 250 Hz; beyond this don't reconstruct

    def start_rec(self):
        self._rec, self._rec_trig, self._rec_marker, self._rec_gap = [], [], [], []
        self._rec_n = 0
        self.filled = 0
        self.recording = True

    def rec_len(self):
        """Samples recorded so far — lets the paradigm stamp exact trial sample indices."""
        return self._rec_n

    def stop_rec(self):
        self.recording = False
        if not self._rec:
            return None
        data = np.concatenate(self._rec, axis=1)          # (n_ch, N) RAW µV (pre-CAR)
        trig = np.concatenate(self._rec_trig)
        marker = np.concatenate(self._rec_marker)
        gap = np.concatenate(self._rec_gap)
        return data, trig, marker, gap

    def _fill_gap(self, n, nxt, trigger):
        """A dropped frame is a MISSING SAMPLE. Skipping it silently compresses the
        recording's time axis — every downstream latency and frequency estimate then
        drifts (an ERD window would land early, mu/beta would read slightly high). So we
        insert `n` linearly-interpolated samples to keep sample-index ↔ wall-clock exact,
        and flag them in a `gap` track so analysis can exclude affected epochs."""
        n = int(min(n, self.MAX_FILL))
        a, b = self._last_sample, nxt
        w = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[None, :]
        fill = (a + (b - a) * w).astype(np.float32)       # (n_ch, n)
        self._emit(fill)
        self._record(fill, trigger, filled=True)
        self.filled += n

    def set_marker(self, code):
        self.marker = int(code)

    def _record(self, raw_chunk, trigger, filled=False):
        if self.recording:
            m = raw_chunk.shape[1]
            self._rec.append(raw_chunk.astype(np.float32))
            self._rec_trig.append(np.full(m, trigger, dtype=np.int32))
            self._rec_marker.append(np.full(m, self.marker, dtype=np.int32))
            self._rec_gap.append(np.full(m, 1 if filled else 0, dtype=np.int8))
            self._rec_n += m

    def send(self, data: bytes):
        if self.src is not None:
            self.src.send(data)

    def _emit(self, raw_chunk):
        # Common Average Reference (robust, median) — removes common-mode / floating-REF
        # drift shared across channels (the vendor GUI does this before filtering). Toggleable
        # so you can see the untouched (floating-reference) data.
        base = raw_chunk - np.median(raw_chunk, axis=0, keepdims=True) if self.car else raw_chunk
        self.calib.append(base.astype(np.float32))            # pre-deblink, for ICA calibration
        if self.deblink is not None:                          # live ICA eye-artifact removal
            base = self.deblink.apply(base).astype(np.float32)
        self.raw.append(base.astype(np.float32))
        self.disp.append(self.filt.process(base).astype(np.float32))
        self.n += raw_chunk.shape[1]

    def _stat(self):
        rec = "  ● REC" if self.recording else ""
        self.report(f"{self.kind} · {self.n} samples · dropped {self.lost} "
                    f"({100*self.lost/max(1,self.n+self.lost):.2f}%){rec}")

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
            self.report("connected · sent init (b / rate / *) · waiting for frames …")
            for pkt in self.src.frames():
                if not self.running:
                    break
                parsed = parse_packet(pkt)
                if parsed is None:
                    continue
                sample, seq, trigger = parsed
                s = sample.reshape(self.n_ch, 1)
                gap = 0
                if self._last_seq is not None:
                    gap = (seq - self._last_seq - 1) % 256
                    self.lost += gap
                self._last_seq = seq
                if gap and self._last_sample is not None:
                    self._fill_gap(gap, s, trigger)   # keep the time axis honest
                self._emit(s)
                self._record(s, trigger)
                self._last_sample = s
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


_COL = {"good": GOOD, "warn": WARN, "bad": BAD}


def _hdr(QtWidgets, text):
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet(f"font-size:12px;font-weight:600;color:{TXT};padding:2px 2px 4px 2px;")
    return lbl


def _card(QtWidgets):
    f = QtWidgets.QFrame()
    f.setStyleSheet(f"QFrame{{background:{CARD};border:1px solid {LINE};border-radius:8px;}}")
    lay = QtWidgets.QVBoxLayout(f); lay.setContentsMargins(10, 8, 10, 10); lay.setSpacing(4)
    return f, lay


FORMAT_VERSION = 2


def save_recording(data, trig, marker, fs, ch_names, outdir="recordings",
                   trials=None, meta=None, tag=None, gap=None):
    """Save a self-describing recording.

    `.npz` (format v2) contains:
      data      (n_ch, N) float32  RAW µV, pre-CAR, unfiltered  — the ground truth
      trigger   (N,)      int32    hardware trigger bytes from the board
      marker    (N,)      int32    software marker: MI task code during imagery, else 0
      gap       (N,)      int8     1 = sample RECONSTRUCTED to cover a dropped UDP frame
                                   (interpolated; exclude these epochs for strict analysis)
      fs, ch_names
      trial_*   (T,)               explicit trial table (onset SAMPLE indices + labels),
                                   written by the paradigm — more precise & unambiguous
                                   than re-deriving edges from `marker`
      meta_json str                JSON: paradigm (tasks/timing/sequence/imagery mode),
                                   acquisition (source, fs, CAR/filter/de-blink state),
                                   link quality (frames received/lost), session notes
    Also writes an MNE `.fif` with imagery onsets as task-labelled annotations."""
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(outdir, f"cap32_{stamp}" + (f"_{tag}" if tag else ""))

    fields = dict(data=data.astype(np.float32), trigger=trig.astype(np.int32),
                  marker=marker.astype(np.int32), fs=float(fs),
                  ch_names=np.array(list(ch_names)), format_version=FORMAT_VERSION)
    if gap is not None:
        fields["gap"] = gap.astype(np.int8)
    trials = trials or []
    if trials:
        fields.update(
            trial_index=np.array([t["trial"] for t in trials], dtype=np.int32),
            trial_code=np.array([t["code"] for t in trials], dtype=np.int32),
            trial_name=np.array([t["task"] for t in trials]),
            trial_onset=np.array([t.get("imagery", -1) for t in trials], dtype=np.int64),
            trial_cue_onset=np.array([t.get("cue", -1) for t in trials], dtype=np.int64),
            trial_end=np.array([t.get("rest", -1) for t in trials], dtype=np.int64))
    meta = dict(meta or {})
    if gap is not None:
        meta["n_filled_samples"] = int(gap.sum())
        meta["filled_pct"] = round(100.0 * float(gap.sum()) / max(1, gap.size), 4)
    meta.update(format_version=FORMAT_VERSION, saved=stamp, n_samples=int(data.shape[1]),
                n_channels=int(data.shape[0]), sfreq=float(fs), n_trials=len(trials),
                duration_s=round(data.shape[1] / float(fs), 2), units="microvolts",
                notes=meta.get("notes", "RAW µV, pre-CAR, unfiltered"))
    fields["meta_json"] = json.dumps(meta, ensure_ascii=False, indent=1)
    np.savez_compressed(base + ".npz", **fields)
    with open(base + ".json", "w") as fh:                 # human-readable sidecar
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    try:
        import mne
        from mi_events import label_of
        info = mne.create_info(list(ch_names), fs, "eeg")
        raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")   # µV -> V
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        match_case=False, on_missing="ignore", verbose="ERROR")
        track = marker if np.any(marker != 0) else trig               # prefer software marker
        onsets = np.where((track[1:] != track[:-1]) & (track[1:] != 0))[0] + 1
        if len(onsets):
            desc = [label_of(int(track[o])) if np.any(marker != 0) else f"T{int(track[o])}"
                    for o in onsets]
            raw.set_annotations(mne.Annotations(onset=onsets / fs, duration=0.0, description=desc))
        raw.save(base + "_raw.fif", overwrite=True, verbose="ERROR")
    except Exception as e:
        print("(.fif save skipped:", e, ")")
    return base


# ------------------------------------------------------------------ the widget
def build(fs, source_kind, host, port, note=""):
    import pyqtgraph as pg
    from PyQt6 import QtWidgets, QtCore

    pg.setConfigOption("background", CARD)
    pg.setConfigOption("foreground", "#414852")
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    ch = CAP32_CHANNELS
    nch = len(ch)
    W = int(WINDOW_S * fs)
    disp, raw = Ring(nch, W), Ring(nch, W)
    filt = RealTimeEEGFilter(fs, nch, lowcut=1.0, highcut=40.0, notch_freq=50.0, baseline=False)
    baselines = [(nch - 1 - i) * SPACING_UV for i in range(nch)]

    root = QtWidgets.QWidget()
    root.setStyleSheet(f"background:{BG};color:{TXT};font-family:'Helvetica Neue',Helvetica,Arial;")
    root.resize(1660, 880)
    outer = QtWidgets.QVBoxLayout(root); outer.setContentsMargins(12, 10, 12, 10); outer.setSpacing(8)

    # ---- control bar (card) ----
    barcard, barrow = _card(QtWidgets)
    barrow.setContentsMargins(12, 8, 12, 8)
    bar = QtWidgets.QHBoxLayout(); barrow.addLayout(bar)

    def chip(txt, col=ACC, fg="#ffffff"):
        b = QtWidgets.QPushButton(txt)
        b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet(f"QPushButton{{background:{col};color:{fg};border:none;padding:7px 13px;"
                        f"border-radius:6px;font-weight:600;}}QPushButton:hover{{background:{col};}}")
        return b

    def field(w, width):
        w.setFixedWidth(width)
        w.setStyleSheet(f"QLineEdit,QComboBox,QSpinBox{{background:#f7f9fc;color:{TXT};"
                        f"border:1px solid {LINE};border-radius:5px;padding:4px 6px;}}")
        return w

    title = QtWidgets.QLabel("Cap32  ·  32-ch ADS1299")
    title.setStyleSheet(f"font-size:17px;font-weight:700;color:{TXT};")
    src_cb = field(QtWidgets.QComboBox(), 78); src_cb.addItems(["synth", "udp", "tcp"]); src_cb.setCurrentText(source_kind)
    host_e = field(QtWidgets.QLineEdit(host), 108)
    port_e = field(QtWidgets.QLineEdit(str(port)), 56)
    rate_cb = field(QtWidgets.QComboBox(), 66); rate_cb.addItems(["250", "500", "1000"]); rate_cb.setCurrentText(str(int(fs)))
    low_e = field(QtWidgets.QLineEdit("1"), 38)
    high_e = field(QtWidgets.QLineEdit("40"), 38)
    notch_cb = QtWidgets.QCheckBox("50Hz"); notch_cb.setChecked(True); notch_cb.setStyleSheet(f"color:{SUB};")
    btn_conn = chip("● Connect", ACC)
    btn_rec = chip("● Record", "#eef1f5", TXT)

    # MI paradigm controls — task sets incl. cognitive tasks (减7 / 想词 / 放歌 / 走房间)
    from mi_paradigm import TASK_SETS
    task_cb = field(QtWidgets.QComboBox(), 240)
    task_cb.addItems(list(TASK_SETS))
    reps_sp = field(QtWidgets.QSpinBox(), 54); reps_sp.setRange(2, 60); reps_sp.setValue(15)
    btn_task = chip("▶ MI Task", "#2f855a")

    def vsep():
        s = QtWidgets.QFrame(); s.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        s.setStyleSheet(f"color:{LINE};background:{LINE};max-width:1px;"); return s

    def tag(t):
        l = QtWidgets.QLabel(t); l.setStyleSheet(f"color:{SUB};font-size:12px;"); return l

    bar.setSpacing(7)
    bar.addWidget(title)
    bar.addStretch(1)
    for lbl, w in [("src", src_cb), ("host", host_e), ("port", port_e), ("Hz", rate_cb)]:
        bar.addWidget(tag(lbl)); bar.addWidget(w)
    bar.addSpacing(4); bar.addWidget(vsep()); bar.addSpacing(4)
    bar.addWidget(tag("band")); bar.addWidget(low_e); bar.addWidget(tag("–")); bar.addWidget(high_e)
    bar.addWidget(notch_cb)
    bar.addSpacing(4); bar.addWidget(vsep()); bar.addSpacing(4)
    bar.addWidget(btn_conn); bar.addWidget(btn_rec)
    bar.addSpacing(4); bar.addWidget(vsep()); bar.addSpacing(4)
    bar.addWidget(tag("task")); bar.addWidget(task_cb)
    bar.addWidget(tag("×")); bar.addWidget(reps_sp); bar.addWidget(btn_task)

    # ---- second row: live clean toggles + view controls ----
    crow = QtWidgets.QHBoxLayout(); crow.setSpacing(9); barrow.addSpacing(2); barrow.addLayout(crow)
    clean_lbl = QtWidgets.QLabel("live clean:"); clean_lbl.setStyleSheet(f"color:{SUB};font-weight:600;font-size:12px;")
    car_cb = QtWidgets.QCheckBox("CAR"); car_cb.setChecked(True); car_cb.setStyleSheet(f"color:{TXT};")
    deblink_cb = QtWidgets.QCheckBox("ICA de-blink"); deblink_cb.setEnabled(False)
    deblink_cb.setStyleSheet(f"color:{SUB};")
    btn_cal = chip("Calibrate de-blink", "#eef1f5", TXT)
    crow.addWidget(clean_lbl); crow.addSpacing(2); crow.addWidget(car_cb)
    crow.addSpacing(10); crow.addWidget(vsep()); crow.addSpacing(10)
    crow.addWidget(btn_cal); crow.addWidget(deblink_cb)
    cal_hint = QtWidgets.QLabel("先采≥15s再校准"); cal_hint.setStyleSheet(f"color:{SUB};font-size:11px;")
    crow.addWidget(cal_hint)
    crow.addStretch(1)
    # imagery mode: kinesthetic (feel it) vs visual (see it) — big individual difference
    mode_cb = field(QtWidgets.QComboBox(), 132)
    mode_cb.addItems(["KMI 动觉(感觉)", "VMI 视觉(看到)"])
    crow.addWidget(tag("imagery")); crow.addWidget(mode_cb)
    crow.addSpacing(10); crow.addWidget(vsep()); crow.addSpacing(10)
    # view controls: vertical scale
    scale_cb = field(QtWidgets.QComboBox(), 78); scale_cb.addItems([f"±{s} µV" for s in SCALE_UV])
    scale_cb.setCurrentText("±100 µV")
    crow.addWidget(tag("scale")); crow.addWidget(scale_cb)
    crow.addSpacing(10); crow.addWidget(vsep()); crow.addSpacing(10)
    note_lbl = QtWidgets.QLabel("坏道插值 / autoreject / 完整ICA → clean_ui.py 回看")
    note_lbl.setStyleSheet(f"color:{SUB};font-size:11px;"); crow.addWidget(note_lbl)
    outer.addWidget(barcard)

    stat = QtWidgets.QLabel("idle"); stat.setStyleSheet(f"color:{SUB};font-size:11px;padding-left:4px;")
    outer.addWidget(stat)

    body = QtWidgets.QHBoxLayout(); body.setSpacing(8); outer.addLayout(body, 1)

    # ---- scope card ----
    scard, slay = _card(QtWidgets)
    slay.addWidget(_hdr(QtWidgets, "Time series  ·  32 ch  (CAR + filter)   "
                        f"<span style='font-weight:400;color:{SUB};font-size:11px'>"
                        f"C3 / C4 highlighted</span>"))
    axis = pg.AxisItem("left"); axis.setTicks([[(baselines[i], ch[i]) for i in range(nch)]]); axis.setWidth(52)
    plot = pg.PlotWidget(axisItems={"left": axis}); plot.setMenuEnabled(False)
    plot.setBackground(CARD)
    plot.showGrid(x=True, y=False, alpha=0.12); plot.setXRange(0, WINDOW_S, padding=0)
    plot.setYRange(-SPACING_UV, nch * SPACING_UV, padding=0); plot.setLabel("bottom", "time", units="s")
    tvec = np.linspace(0, WINDOW_S, W)
    curves = [plot.plot(tvec, disp.buf[i] + baselines[i],
                        pen=pg.mkPen(HILITE if ch[i] in MI_HILITE else TRACE,
                                     width=1.2 if ch[i] in MI_HILITE else 0.9)) for i in range(nch)]
    slay.addWidget(plot, 1)
    body.addWidget(scard, 1)

    # ---- middle column: live head-map (topomap) + band-power bars (OpenBCI-style) ----
    mid = QtWidgets.QVBoxLayout(); mid.setSpacing(8)
    mw = QtWidgets.QWidget(); mw.setFixedWidth(300); mw.setLayout(mid)

    hcard, hlay = _card(QtWidgets)
    hhdr = QtWidgets.QHBoxLayout()
    hhdr.addWidget(_hdr(QtWidgets, "Head map  ·  band power (live)"))
    head_band = QtWidgets.QComboBox(); head_band.addItems(list(HEAD_BANDS)); head_band.setCurrentText("μ 8–13 Hz")
    head_band.setStyleSheet(f"QComboBox{{background:#f7f9fc;color:{TXT};border:1px solid {LINE};"
                            f"border-radius:5px;padding:2px 6px;font-size:11px;}}"); head_band.setFixedWidth(118)
    hhdr.addStretch(1); hhdr.addWidget(head_band); hlay.addLayout(hhdr)
    hplot = pg.PlotWidget(); hplot.setMenuEnabled(False); hplot.setBackground(CARD)
    hplot.hideAxis("left"); hplot.hideAxis("bottom"); hplot.setAspectLocked(True)
    hplot.setXRange(-1.25, 1.25); hplot.setYRange(-1.2, 1.32); hplot.setMinimumHeight(250)
    himg = pg.ImageItem(); hplot.addItem(himg)
    th = np.linspace(0, 2 * np.pi, 120)
    hplot.plot(np.cos(th), np.sin(th), pen=pg.mkPen("#9aa3b2", width=2))
    hplot.plot([-0.13, 0, 0.13], [0.99, 1.16, 0.99], pen=pg.mkPen("#9aa3b2", width=2))   # nose
    hxy = head_xy()
    hdots = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(60, 66, 78, 130))
    hdots.setData(pos=hxy); hplot.addItem(hdots)
    for i, c in enumerate(ch):                                       # C3/C4 labelled on the map
        if c in MI_HILITE:
            t = pg.TextItem(c, color="#1f2733", anchor=(0.5, 0.5)); t.setScale(0.7)
            t.setPos(hxy[i, 0], hxy[i, 1] + 0.11); hplot.addItem(t)
    hlay.addWidget(hplot, 1)
    g = np.linspace(-1.08, 1.08, 72); GX, GY = np.meshgrid(g, g); hmask = GX ** 2 + GY ** 2 <= 1.0
    mid.addWidget(hcard, 3)

    bpcard, bplay = _card(QtWidgets)
    bplay.addWidget(_hdr(QtWidgets, "Band power  ·  sensorimotor mean (µV)"))
    bpplot = pg.PlotWidget(); bpplot.setMenuEnabled(False); bpplot.setBackground(CARD)
    bpplot.showGrid(x=False, y=True, alpha=0.15); bpplot.setMinimumHeight(150)
    bpplot.getAxis("bottom").setTicks([[(i, BANDS[i][0]) for i in range(len(BANDS))]])
    bpbar = pg.BarGraphItem(x=list(range(len(BANDS))), height=[0] * len(BANDS), width=0.62,
                            brushes=[pg.mkColor(c) for c in BAND_COLS])
    bpplot.addItem(bpbar); bplay.addWidget(bpplot, 1)
    mid.addWidget(bpcard, 2)
    body.addWidget(mw)

    from montage import SENSORIMOTOR
    sm_idx = [ch.index(c) for c in SENSORIMOTOR if c in ch]

    # ---- right column: spectrum card + quality card ----
    right = QtWidgets.QVBoxLayout(); right.setSpacing(8)
    rw = QtWidgets.QWidget(); rw.setFixedWidth(372); rw.setLayout(right)

    fcard, flay = _card(QtWidgets)
    flay.addWidget(_hdr(QtWidgets, "Spectrum  ·  µV vs Hz  (live, all-ch mean)"))
    fft_plot = pg.PlotWidget(); fft_plot.setMenuEnabled(False); fft_plot.setBackground(CARD)
    fft_plot.setLogMode(False, True)                     # log amplitude (µV)
    fft_plot.setXRange(0, 60, padding=0); fft_plot.setLimits(xMin=0, xMax=fs / 2)
    fft_plot.showGrid(x=True, y=True, alpha=0.12)
    fft_plot.setLabel("bottom", "frequency", units="Hz")
    fft_plot.setMinimumHeight(230)
    for lo, hi, col in [(8, 13, (90, 190, 130, 55)), (13, 30, (90, 140, 210, 45))]:
        reg = pg.LinearRegionItem([lo, hi], movable=False, brush=col)
        reg.setZValue(-10); fft_plot.addItem(reg)       # μ (8–13) & β (13–30) bands
    fft_plot.addItem(pg.InfiniteLine(50, angle=90,
                     pen=pg.mkPen(BAD, style=QtCore.Qt.PenStyle.DashLine)))       # 50 Hz line
    freqs = np.fft.rfftfreq(W, 1 / fs)
    fft_all = fft_plot.plot([], [], pen=pg.mkPen(ACC, width=2))
    fft_post = fft_plot.plot([], [], pen=pg.mkPen(WARN, width=1))                 # posterior (α)
    post_idx = [i for i, c in enumerate(ch) if c in
                {"O1", "O2", "OZ", "PO3", "PO4", "P3", "P4", "PZ"}]
    flay.addWidget(fft_plot, 1)
    right.addWidget(fcard, 3)

    qcard, qlay = _card(QtWidgets)
    qlay.addWidget(_hdr(QtWidgets, "Signal quality  ·  impedance proxy   "
                        "<span style='font-weight:400;font-size:11px'>"
                        f"<span style='color:{GOOD}'>●</span> good "
                        f"<span style='color:{WARN}'>●</span> noisy "
                        f"<span style='color:{BAD}'>●</span> bad</span>"))
    grid = QtWidgets.QGridLayout(); grid.setSpacing(3); qlay.addLayout(grid, 1)
    cells = []
    for i in range(nch):
        r, c0 = i % 16, (i // 16) * 3
        dot = QtWidgets.QLabel("●"); dot.setStyleSheet(f"color:{GOOD};font-size:13px;")
        nm = QtWidgets.QLabel(ch[i]); nm.setStyleSheet(f"color:{TXT};font-size:11px;"); nm.setFixedWidth(38)
        val = QtWidgets.QLabel("–"); val.setStyleSheet(f"color:{SUB};font-size:11px;"); val.setFixedWidth(54)
        grid.addWidget(dot, r, c0); grid.addWidget(nm, r, c0 + 1); grid.addWidget(val, r, c0 + 2)
        cells.append((dot, val))
    right.addWidget(qcard, 4)
    body.addWidget(rw)

    ctx = dict(root=root, plot=plot, curves=curves, cells=cells, disp=disp, raw=raw, filt=filt,
               tvec=tvec, baselines=baselines, fs=fs, nch=nch, stat=stat, note=note, W=W,
               fft_all=fft_all, fft_post=fft_post, freqs=freqs, post_idx=post_idx,
               clip=[100.0], sm_idx=sm_idx, _amp=None,
               head=dict(img=himg, band=head_band, xy=hxy, GX=GX, GY=GY, mask=hmask, g=g),
               bpbar=bpbar,
               ctrls=dict(src=src_cb, host=host_e, port=port_e, rate=rate_cb, low=low_e,
                          high=high_e, notch=notch_cb, conn=btn_conn, rec=btn_rec,
                          task=task_cb, reps=reps_sp, taskbtn=btn_task,
                          car=car_cb, deblink=deblink_cb, calibrate=btn_cal, scale=scale_cb,
                          mode=mode_cb))
    return ctx


def refresh_scope(ctx):
    b = ctx["disp"].snapshot()
    clip = ctx["clip"][0]; half = SPACING_UV * 0.46   # ±clip µV maps to ±half around each baseline
    for i, c in enumerate(ctx["curves"]):
        y = b[i] - b[i].mean()   # center each channel now (don't wait for a slow baseline)
        c.setData(ctx["tvec"], np.clip(y / clip, -1.0, 1.0) * half + ctx["baselines"][i])


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
    ctx["_amp"] = amp                                  # shared with head-map + band-power
    f = ctx["freqs"]
    m = f >= 0.5                                        # skip DC
    allm = np.clip(amp.mean(0)[m], 1e-3, None)
    ctx["fft_all"].setData(f[m], allm)
    if ctx["post_idx"]:
        postm = np.clip(amp[ctx["post_idx"]].mean(0)[m], 1e-3, None)
        ctx["fft_post"].setData(f[m], postm)
    refresh_bandpower(ctx)
    refresh_head(ctx)


def refresh_bandpower(ctx):
    amp = ctx.get("_amp")
    if amp is None:
        return
    f = ctx["freqs"]; sm = ctx["sm_idx"]
    heights = [float(amp[sm][:, (f >= lo) & (f < hi)].mean()) for _, lo, hi in BANDS]
    ctx["bpbar"].setOpts(height=heights)


def refresh_head(ctx):
    amp = ctx.get("_amp")
    if amp is None:
        return
    import pyqtgraph as pg
    from scipy.interpolate import griddata
    h = ctx["head"]
    lo, hi = HEAD_BANDS[h["band"].currentText()]
    f = ctx["freqs"]
    vals = amp[:, (f >= lo) & (f < hi)].mean(1)                 # per-channel band power (µV)
    z = griddata(h["xy"], vals, (h["GX"], h["GY"]), method="cubic")
    zn = griddata(h["xy"], vals, (h["GX"], h["GY"]), method="nearest")
    z[np.isnan(z)] = zn[np.isnan(z)]
    vlo, vhi = np.percentile(vals, 5), np.percentile(vals, 95)
    if vhi <= vlo:
        vhi = vlo + 1e-6
    norm = np.clip((z - vlo) / (vhi - vlo), 0.0, 1.0)
    rgba = pg.colormap.get("viridis").map(norm.ravel(), mode="byte").reshape(norm.shape + (4,))
    rgba[~h["mask"]] = 0                                        # transparent outside the scalp
    h["img"].setImage(rgba, autoLevels=False)
    g0, g1 = h["g"][0], h["g"][-1]
    h["img"].setRect(pg.QtCore.QRectF(g0, g0, g1 - g0, g1 - g0))


def run_live(fs, source_kind, host, port):
    from PyQt6 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(fs, source_kind, host, port)
    ctx["root"].setWindowTitle("Cap32 acquisition")
    rec = {"thread": None, "pab": None}

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

    def apply_scale(_=None):
        txt = ctx["ctrls"]["scale"].currentText()
        digits = "".join(c for c in txt if c.isdigit())
        if digits:
            ctx["clip"][0] = float(digits)
    ctx["ctrls"]["scale"].currentTextChanged.connect(apply_scale)

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
            data, trig, marker, gap = out
            meta = dict(paradigm=dict(kind="free-run (no paradigm)"),
                        acquisition=dict(source=r.kind, sfreq=float(ctx["fs"]),
                                         car=bool(r.car), channels=list(CAP32_CHANNELS)),
                        link=dict(frames_received=int(r.n), frames_lost=int(r.lost),
                                  loss_pct=round(100 * r.lost / max(1, r.n + r.lost), 3)))
            base = save_recording(data, trig, marker, ctx["fs"], CAP32_CHANNELS,
                                  meta=meta, gap=gap)
            ctx["stat"].setText(f"saved {base}.npz  ({data.shape[1]} samples, "
                                f"{data.shape[1]/ctx['fs']:.1f}s)")
    ctx["ctrls"]["rec"].clicked.connect(toggle_rec)

    # ---- MI paradigm ----
    from mi_paradigm import TASK_SETS as _TASK_SETS

    def launch_task():
        r = rec["thread"]
        if not (r and r.is_alive()):
            ctx["stat"].setText("先 Connect 再开始任务"); return
        from mi_paradigm import MiParadigm, make_sequence, Timing, place, MI_TASKS
        c = ctx["ctrls"]
        tasks = _TASK_SETS[c["task"].currentText()]
        reps = c["reps"].value()
        mode = "visual" if c["mode"].currentIndex() else "kinesthetic"
        seq = make_sequence(tasks, reps)
        timing = Timing()
        r.start_rec(); c["rec"].setText("■ Stop rec")

        # ---- trial log: exact sample index of every cue / imagery / rest onset ----
        trials, cur = [], {}

        def on_event(phase, name, code):
            r.set_marker(code)          # 0 except during imagery (the epoching label)
            if phase == "cue":
                cur.clear()
                cur.update(trial=len(trials), task=name,
                           code=MI_TASKS[name].code, cue=r.rec_len())
            elif phase == "imagery":
                cur.setdefault("trial", len(trials)); cur.setdefault("task", name)
                cur["code"] = code or MI_TASKS[name].code
                cur["imagery"] = r.rec_len()
            elif phase in ("rest", "end", "abort") and cur.get("imagery") is not None:
                cur["rest"] = r.rec_len()
                trials.append(dict(cur)); cur.clear()

        def on_done():
            out = r.stop_rec(); c["rec"].setText("● Record")
            r.set_marker(0)
            if out is None:
                ctx["stat"].setText("任务结束但没有数据"); return
            data, trig, marker, gap = out
            meta = dict(
                paradigm=dict(kind="motor-imagery", tasks=tasks, reps=reps,
                              imagery_mode=mode, sequence=seq,
                              timing={k: getattr(timing, k) for k in
                                      ("fixation", "cue", "imagery", "rest")}),
                acquisition=dict(source=r.kind, host=r.host, port=r.port,
                                 sfreq=float(ctx["fs"]), car=bool(r.car),
                                 deblink=r.deblink is not None,
                                 display_filter=dict(low=c["low"].text(), high=c["high"].text(),
                                                     notch50=c["notch"].isChecked()),
                                 channels=list(CAP32_CHANNELS)),
                link=dict(frames_received=int(r.n), frames_lost=int(r.lost),
                          loss_pct=round(100 * r.lost / max(1, r.n + r.lost), 3),
                          samples_filled=int(r.filled)))
            base = save_recording(data, trig, marker, ctx["fs"], CAP32_CHANNELS,
                                  trials=trials, meta=meta, tag="mi", gap=gap)
            ctx["stat"].setText(
                f"✅ 完成 · {len(trials)} trials · 丢包 {meta['link']['loss_pct']}% · "
                f"saved {base}.npz  →  python src/analysis/erd_ers.py {base}.npz")

        pab = MiParadigm(seq, timing, on_event=on_event, send_trigger=r.send,
                         light=True, mode=mode)
        pab.finished.connect(on_done)
        pab.setWindowTitle("MI paradigm — ESC to abort")
        idx, nscr = place(pab)          # 2nd monitor if present, else full-screen here
        pab.start()
        rec["pab"] = pab                # keep a ref so it isn't garbage-collected
        where = f"副屏 {idx}" if nscr > 1 else "全屏(盖住本界面,专心想象)"
        ctx["stat"].setText(f"▶ MI 任务进行中 · {len(seq)} trials · {mode} · "
                            f"提示窗在{where} · ESC 中止")
    ctx["ctrls"]["taskbtn"].clicked.connect(launch_task)

    # ---- live clean toggles ----
    def toggle_car(_=None):
        r = rec["thread"]
        if r:
            r.car = ctx["ctrls"]["car"].isChecked()
    ctx["ctrls"]["car"].stateChanged.connect(toggle_car)

    def toggle_deblink(_=None):
        r = rec["thread"]
        if r:
            r.deblink = rec.get("deblink_op") if ctx["ctrls"]["deblink"].isChecked() else None
    ctx["ctrls"]["deblink"].stateChanged.connect(toggle_deblink)

    def calibrate():
        r = rec["thread"]
        if not (r and r.is_alive()):
            ctx["stat"].setText("先 Connect 再校准"); return
        need = int(15 * ctx["fs"])
        if r.n < need:
            ctx["stat"].setText(f"先采集 ≥15 秒再校准去眨眼 (现在 {r.n/ctx['fs']:.0f}s)"); return
        ctx["stat"].setText("⏳ 校准 ICA 去眨眼中…"); QtWidgets.QApplication.processEvents()
        try:
            from artifacts import LiveDeblink
            valid = min(r.n, r.calib.buf.shape[1])
            buf = r.calib.snapshot()[:, -valid:]
            op = LiveDeblink.calibrate(buf, ctx["fs"], CAP32_CHANNELS, method="eog")
            rec["deblink_op"] = op; r.deblink = op
            db = ctx["ctrls"]["deblink"]; db.setEnabled(True); db.setChecked(True)
            db.setStyleSheet(f"color:{TXT};")
            ctx["stat"].setText(f"✅ 去眨眼已校准 · 移除成分 {op.report['exclude']} "
                                f"({op.report['removed']} 个) · 取消勾选可对比")
        except Exception as e:
            ctx["stat"].setText(f"⚠ 校准失败: {type(e).__name__}: {e}")
    ctx["ctrls"]["calibrate"].clicked.connect(calibrate)

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
    cap = SynthCap(sfreq=fs)
    step = max(1, int(fs // 5))
    for _ in range(int((WINDOW_S + 1) * fs / step)):
        chunk = cap.get_chunk(step)
        car = chunk - np.median(chunk, axis=0, keepdims=True)
        ctx["raw"].append(car)
        ctx["disp"].append(ctx["filt"].process(car).astype(np.float32))
    ctx["stat"].setText("synthetic preview · filtered 1–40 Hz + 50 Hz notch · ▶ MI Task 开始左右手想象范式")
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
