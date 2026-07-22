#!/usr/bin/env python
"""Interactive preprocessing / artifact review — toggle each step, see raw vs cleaned.

Loads one recording and lets you flip every cleaning step on/off and immediately see the
effect: the RAW trace (grey) and the CLEANED trace (blue) are overlaid per channel, so you
can literally watch a blink disappear when ICA is on, or see the untouched data with all
toggles off. A report panel says what each step removed, and "ERD impact" recomputes the
C3/C4 mu-ERD with the current cleaning so you see how it changes the actual MI signal.

Because autoreject and ICA operate on a whole recording (offline), this review UI is where
they live; cap_gui carries only the live-capable toggles (CAR / interp / ICA de-blink).

  python src/analysis/clean_ui.py                                  # newest recording
  python src/analysis/clean_ui.py recordings/cap32_xxx.npz
  python src/analysis/clean_ui.py --synth-blinks                   # demo data with blinks
  python src/analysis/clean_ui.py --screenshot results/clean_ui_preview.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # src/
sys.path.insert(0, str(HERE))                                 # src/analysis/
import load as loadmod                                         # noqa: E402
import artifacts as A                                          # noqa: E402

BG, CARD, LINE = "#eef1f5", "#ffffff", "#dfe3ea"
TXT, SUB = "#1f2733", "#6b7480"
ACC, RAWCOL = "#2b6cb0", "#b8bfc9"
GOOD, WARN, BAD = "#2e9e5b", "#c58a00", "#d1495b"
VIEW_CH = ["FP1", "C3", "C4", "PZ"]     # blink-prone + MI channels
WIN_S = 12.0


def newest_recording():
    rec = sorted((HERE.parents[1] / "recordings").glob("cap32_*.npz"))
    return str(rec[-1]) if rec else None


# ---------------------------------------------------------------- compute layer
class Model:
    def __init__(self, path):
        self.path = path
        self.raw, self.events, self.event_id = loadmod.read_recording(path)
        self.raw.load_data()
        self.fs = self.raw.info["sfreq"]
        self.ch = self.raw.ch_names
        self.raw_uv = self.raw.get_data() * 1e6
        self.cache = {}

    def clean(self, flags: A.CleanFlags):
        key = tuple(sorted(vars(flags).items()))
        if key not in self.cache:
            if not any([flags.notch, flags.l_freq, flags.h_freq, flags.interp,
                        flags.car, flags.ica]):
                self.cache[key] = (self.raw_uv, {"interpolated": [], "ica": None})
            else:
                clean, report = A.preprocess(self.raw, flags)
                self.cache[key] = (clean.get_data() * 1e6, report)
        return self.cache[key]

    def erd_impact(self, flags):
        """Recompute contra/ipsi C3/C4 mu-ERD with the current cleaning (text summary)."""
        import mne
        if not len(self.events):
            return "no events in recording"
        clean, report = self.clean(flags)
        info = self.raw.info
        rawc = mne.io.RawArray(clean * 1e-6, info.copy(), verbose="ERROR")
        ep = mne.Epochs(rawc, self.events, self.event_id, tmin=loadmod.DEFAULT_TMIN,
                        tmax=loadmod.DEFAULT_TMAX, baseline=None, preload=True, verbose="ERROR")
        if flags.autoreject and len(ep) >= 4:
            try:
                ep, _ = A.clean_epochs(ep)
            except Exception:
                pass
        sys.path.insert(0, str(HERE))
        from erd_ers import _tfr, _band_timecourse, MU, IMAGERY
        out = []
        try:
            tfrs = _tfr(ep.copy().pick("eeg"))
            for lab in tfrs:
                contra = "C4" if lab == "left" else "C3"
                ipsi = "C3" if lab == "left" else "C4"
                tt = tfrs[lab].times; w = (tt >= IMAGERY[0]) & (tt <= IMAGERY[1])
                _, yc = _band_timecourse(tfrs[lab], contra, MU)
                _, yi = _band_timecourse(tfrs[lab], ipsi, MU)
                out.append(f"{lab}: {contra} {yc[w].mean():+.0f}%  {ipsi} {yi[w].mean():+.0f}%")
        except Exception as e:
            return f"ERD calc failed: {e}"
        return "  ·  ".join(out) if out else "no conditions"


# ---------------------------------------------------------------- UI
def build(model):
    import pyqtgraph as pg
    from PyQt6 import QtWidgets, QtCore
    pg.setConfigOption("background", CARD); pg.setConfigOption("foreground", "#414852")
    pg.setConfigOptions(antialias=True)

    root = QtWidgets.QWidget()
    root.setStyleSheet(f"background:{BG};color:{TXT};font-family:'Helvetica Neue',Helvetica,Arial;")
    root.resize(1280, 760)
    outer = QtWidgets.QHBoxLayout(root); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(10)

    # ---- left control card ----
    def card():
        f = QtWidgets.QFrame()
        f.setStyleSheet(f"QFrame{{background:{CARD};border:1px solid {LINE};border-radius:8px;}}")
        lay = QtWidgets.QVBoxLayout(f); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(8)
        return f, lay

    lcard, lc = card(); lcard.setFixedWidth(288)
    ttl = QtWidgets.QLabel("Preprocessing"); ttl.setStyleSheet(f"font-size:16px;font-weight:700;color:{TXT};")
    fname = QtWidgets.QLabel(Path(model.path).name); fname.setStyleSheet(f"color:{SUB};font-size:11px;")
    lc.addWidget(ttl); lc.addWidget(fname)

    def chk(text, on=True):
        c = QtWidgets.QCheckBox(text); c.setChecked(on)
        c.setStyleSheet(f"color:{TXT};font-size:13px;padding:3px 0;")
        return c

    hint = QtWidgets.QLabel("勾选=开启处理 · 全部取消=看原始数据")
    hint.setStyleSheet(f"color:{SUB};font-size:11px;")
    lc.addWidget(hint)
    cb_notch = chk("50 Hz notch")
    cb_band = chk("Band-pass 1–40 Hz")
    cb_interp = chk("Bad-channel interpolate")
    cb_car = chk("Common average ref (CAR)")
    cb_ica = chk("ICA — remove eye / artifact")
    method = QtWidgets.QComboBox(); method.addItems(["eog (FP1/FP2 proxy)", "iclabel"])
    method.setStyleSheet(f"background:#f7f9fc;border:1px solid {LINE};border-radius:5px;padding:3px;")
    cb_ar = chk("autoreject (epoch repair)", on=False)
    for w in (cb_notch, cb_band, cb_interp, cb_car, cb_ica):
        lc.addWidget(w)
    mrow = QtWidgets.QHBoxLayout(); ml = QtWidgets.QLabel("   method"); ml.setStyleSheet(f"color:{SUB};")
    mrow.addWidget(ml); mrow.addWidget(method, 1); lc.addLayout(mrow)
    lc.addWidget(cb_ar)
    note = QtWidgets.QLabel("autoreject / ERD 只影响下方“ERD impact”\n(是逐-epoch 的离线步骤)")
    note.setStyleSheet(f"color:{SUB};font-size:10px;"); lc.addWidget(note)

    sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.Shape.HLine); sep.setStyleSheet(f"color:{LINE};")
    lc.addWidget(sep)
    report = QtWidgets.QLabel("—"); report.setWordWrap(True)
    report.setStyleSheet(f"color:{TXT};font-size:11px;"); report.setTextFormat(QtCore.Qt.TextFormat.RichText)
    lc.addWidget(report)
    lc.addStretch(1)
    btn_erd = QtWidgets.QPushButton("Compute ERD impact")
    btn_erd.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    btn_erd.setStyleSheet(f"QPushButton{{background:{ACC};color:#fff;border:none;padding:8px;"
                          "border-radius:6px;font-weight:600;}")
    lc.addWidget(btn_erd)
    erd_lbl = QtWidgets.QLabel(""); erd_lbl.setWordWrap(True); erd_lbl.setStyleSheet(f"color:{TXT};font-size:11px;")
    lc.addWidget(erd_lbl)
    outer.addWidget(lcard)

    # ---- right: signal card ----
    rcard, rc = card()
    hdr = QtWidgets.QLabel(f"Raw vs cleaned   "
                           f"<span style='color:{RAWCOL}'>■</span> raw   "
                           f"<span style='color:{ACC}'>■</span> cleaned")
    hdr.setStyleSheet(f"font-size:13px;font-weight:600;color:{TXT};")
    rc.addWidget(hdr)
    nch = len(VIEW_CH); SP = 220.0
    laxis = pg.AxisItem("left"); laxis.setWidth(46)
    laxis.setTicks([[((nch - 1 - i) * SP, VIEW_CH[i]) for i in range(nch)]])
    plot = pg.PlotWidget(axisItems={"left": laxis}); plot.setMenuEnabled(False); plot.setBackground(CARD)
    plot.showGrid(x=True, y=False, alpha=0.12); plot.setLabel("bottom", "time", units="s")
    labels = []
    raw_curves = [plot.plot([], [], pen=pg.mkPen(RAWCOL, width=1.4)) for _ in VIEW_CH]
    clean_curves = [plot.plot([], [], pen=pg.mkPen(ACC, width=1.0)) for _ in VIEW_CH]
    rc.addWidget(plot, 1)
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    dur = model.raw_uv.shape[1] / model.fs
    slider.setRange(0, max(0, int((dur - WIN_S) * 10)))
    slider.setStyleSheet("QSlider::handle:horizontal{background:%s;border-radius:6px;width:14px;}" % ACC)
    rc.addWidget(slider)
    status = QtWidgets.QLabel("ready"); status.setStyleSheet(f"color:{SUB};font-size:11px;")
    rc.addWidget(status)
    outer.addWidget(rcard, 1)

    ctx = dict(root=root, model=model, plot=plot, raw_curves=raw_curves, clean_curves=clean_curves,
               labels=labels, slider=slider, status=status, report=report, erd_lbl=erd_lbl,
               SP=SP, ctrls=dict(notch=cb_notch, band=cb_band, interp=cb_interp, car=cb_car,
                                 ica=cb_ica, method=method, ar=cb_ar, erd=btn_erd))
    return ctx


def flags_of(ctx):
    c = ctx["ctrls"]
    m = "iclabel" if c["method"].currentText().startswith("iclabel") else "eog"
    f = A.CleanFlags(
        notch=50.0 if c["notch"].isChecked() else 0.0,
        l_freq=1.0 if c["band"].isChecked() else 0.0,
        h_freq=40.0 if c["band"].isChecked() else 0.0,
        interp=c["interp"].isChecked(), car=c["car"].isChecked(),
        ica=c["ica"].isChecked(), ica_method=m)
    f.autoreject = c["ar"].isChecked()
    return f


def recompute(ctx):
    from PyQt6 import QtWidgets
    ctx["status"].setText("⏳ computing…"); QtWidgets.QApplication.processEvents()
    flags = flags_of(ctx)
    clean, rep = ctx["model"].clean(flags)
    ctx["_clean"] = clean
    # report
    parts = []
    interp = rep.get("interpolated") or []
    parts.append(f"<b>Interpolated:</b> {', '.join(interp) if interp else '—'}")
    ica = rep.get("ica")
    if ica and "error" in ica:
        parts.append(f"<b>ICA:</b> <span style='color:{BAD}'>{ica['error']}</span>")
    elif ica:
        ex = ica.get("exclude", [])
        labs = ica.get("labels", [])
        exlabs = ", ".join(f"#{i}:{labs[i].split(' ')[0]}" for i in ex) if ex else "none"
        parts.append(f"<b>ICA ({ica['method']}):</b> removed {ica['removed']}/{ica['n_components']} "
                     f"comps [{exlabs}]")
    else:
        parts.append("<b>ICA:</b> off")
    ctx["report"].setText("<br>".join(parts))
    replot(ctx)
    ctx["status"].setText("updated")


def replot(ctx):
    clean = ctx.get("_clean"); model = ctx["model"]
    if clean is None:
        return
    fs = model.fs; t0 = ctx["slider"].value() / 10.0
    i0, i1 = int(t0 * fs), int((t0 + WIN_S) * fs)
    tt = np.arange(i0, i1) / fs
    for k, c in enumerate(VIEW_CH):
        if c not in model.ch:
            continue
        ci = model.ch.index(c)
        base = (len(VIEW_CH) - 1 - k) * ctx["SP"]
        rawseg = model.raw_uv[ci, i0:i1]; rawseg = rawseg - rawseg.mean()
        clnseg = clean[ci, i0:i1]; clnseg = clnseg - clnseg.mean()
        cl = ctx["SP"] * 0.48
        ctx["raw_curves"][k].setData(tt, np.clip(rawseg, -cl, cl) + base)
        ctx["clean_curves"][k].setData(tt, np.clip(clnseg, -cl, cl) + base)
    ctx["plot"].setXRange(t0, t0 + WIN_S, padding=0)
    ctx["plot"].setYRange(-ctx["SP"] * 0.6, len(VIEW_CH) * ctx["SP"], padding=0)


def run(model):
    from PyQt6 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(model); ctx["root"].setWindowTitle("Cap32 preprocessing review")
    c = ctx["ctrls"]
    for w in (c["notch"], c["band"], c["interp"], c["car"], c["ica"]):
        w.stateChanged.connect(lambda _=None: recompute(ctx))
    c["method"].currentIndexChanged.connect(lambda _=None: recompute(ctx))
    ctx["slider"].valueChanged.connect(lambda _=None: replot(ctx))

    def do_erd():
        ctx["erd_lbl"].setText("⏳ computing ERD…"); QtWidgets.QApplication.processEvents()
        txt = ctx["model"].erd_impact(flags_of(ctx))
        ctx["erd_lbl"].setText("<b>ERD impact</b> (contra should be ⟪more negative⟫):<br>" + txt)
    c["erd"].clicked.connect(do_erd)

    ctx["root"].show()
    QtCore.QTimer.singleShot(50, lambda: recompute(ctx))
    app.exec()


def screenshot(model, out):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ctx = build(model)
    ctx["ctrls"]["method"].setCurrentIndex(0)   # eog proxy (reliable on synth)
    ctx["root"].show(); app.processEvents()
    recompute(ctx); app.processEvents()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ctx["root"].grab().save(str(out)); print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--synth-blinks", action="store_true")
    ap.add_argument("--screenshot", default=None)
    args = ap.parse_args()
    if args.synth_blinks:
        sys.path.insert(0, str(HERE))
        from erd_ers import synth_mi_recording
        path = synth_mi_recording(reps=12, out=HERE.parents[1] / "recordings" / "synth_mi_blinks.npz",
                                  blinks=True)
    else:
        path = args.path or newest_recording()
    if not path:
        raise SystemExit("no recording found — pass a path or use --synth-blinks")
    model = Model(path)
    if args.screenshot:
        screenshot(model, args.screenshot)
    else:
        run(model)


if __name__ == "__main__":
    main()
