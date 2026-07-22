#!/usr/bin/env python
"""Motor-imagery paradigm — timed left / right / feet / rest cues with triggers.

This is the "刺激范式 + 打标" piece: it shows the subject *what to imagine and when*,
and at the exact moment imagery begins it fires an event so that marker lands in the
EEG recording. Afterwards the analysis loader cuts the continuous EEG into labelled
epochs on those markers.

Each trial runs a 4-phase state machine:

    fixation  →  cue (arrow, "prepare")  →  IMAGERY (go, imagine now)  →  rest
      1.5s          1.5s                        4.0s                       2.0s
                                            └── marker fires here ──┘

Two ways to use it:

  1. Embedded in cap_gui (recommended — one process, guaranteed sample alignment):
        pab = MiParadigm(seq, timing, on_event=handler, send_trigger=recv.send)
        pab.finished.connect(on_done); pab.start()
     `on_event(phase, name, code)` is called at every phase boundary. cap_gui uses it to
     stamp the recording's software marker (guaranteed) and, at imagery onset, also push
     the hardware `TXXXX` to the board via `send_trigger`.

  2. Standalone (preview / a separate marker-only session):
        python mi_paradigm.py                       # just the cue window
        python mi_paradigm.py --lsl                  # + push an LSL 'MI_Cues' stream
        python mi_paradigm.py --tasks left right feet --reps 12
        python mi_paradigm.py --screenshot results/mi_paradigm_preview.png
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # src/
from common.mi_events import MI_TASK_CODES       # noqa: E402


# --------------------------------------------------------------------- tasks
@dataclass
class MiTask:
    name: str
    glyph: str          # big central symbol shown during cue + imagery
    label: str          # word under the glyph
    instr: str          # one-line instruction
    color: str

    @property
    def code(self) -> int:
        return MI_TASK_CODES[self.name]


MI_TASKS = {
    "left":  MiTask("left",  "←", "LEFT hand",  "想象左手握拳 · 不要真的动",  "#2b6cb0"),
    "right": MiTask("right", "→", "RIGHT hand", "想象右手握拳 · 不要真的动",  "#2b6cb0"),
    "feet":  MiTask("feet",  "↓", "FEET",       "想象双脚下踩 · 不要真的动",  "#2f855a"),
    "tongue":MiTask("tongue","●", "TONGUE",     "想象舌抵上颚",              "#805ad5"),
    "rest":  MiTask("rest",  "+",      "Rest",       "放松 · 盯着十字 · 什么都不想", "#718096"),
}


@dataclass
class Timing:
    fixation: float = 1.5
    cue: float = 1.5
    imagery: float = 4.0
    rest: float = 2.0

    @property
    def trial(self) -> float:
        return self.fixation + self.cue + self.imagery + self.rest


# --------------------------------------------------------- balanced trial order
def make_sequence(task_names, reps, seed=7):
    """One run: `reps` of each task, shuffled so no task repeats back-to-back."""
    import random
    rng = random.Random(seed)
    pool = list(task_names) * reps
    for _ in range(300):
        rng.shuffle(pool)
        if all(pool[i] != pool[i + 1] for i in range(len(pool) - 1)):
            break
    return pool


# --------------------------------------------------------------- the cue window
def _build_paradigm_class():
    """Import Qt lazily so the module (and make_sequence) works with no display."""
    from PyQt6 import QtWidgets, QtCore, QtGui

    class MiParadigm(QtWidgets.QWidget):
        finished = QtCore.pyqtSignal()
        phase_sig = QtCore.pyqtSignal(str, str, int)   # phase, name, code

        PHASES = ("fixation", "cue", "imagery", "rest")

        def __init__(self, seq, timing=None, on_event=None, send_trigger=None,
                     light=True, parent=None):
            super().__init__(parent)
            self.seq = list(seq)
            self.timing = timing or Timing()
            self.send_trigger = send_trigger
            if on_event is not None:
                self.phase_sig.connect(lambda p, n, c: on_event(p, n, c))
            self._i = -1
            self._phase = None
            self._light = light
            self._build_ui()
            self._timer = QtCore.QTimer(self); self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._next_phase)

        # ---- ui ----
        def _build_ui(self):
            bg, fg, sub = ("#ffffff", "#1f2733", "#6b7480") if self._light else ("#0e1116", "#e8edf4", "#8b95a5")
            self._bg, self._fg, self._sub = bg, fg, sub
            self.setStyleSheet(f"background:{bg};")
            v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
            self.top = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.top.setStyleSheet(f"color:{sub};font-size:16px;font-family:'Helvetica Neue',Arial;")
            self.glyph = QtWidgets.QLabel("+", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.glyph.setStyleSheet(f"color:{fg};font-size:220px;font-weight:300;font-family:'Helvetica Neue',Arial;")
            self.word = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.word.setStyleSheet(f"color:{fg};font-size:40px;font-weight:600;font-family:'Helvetica Neue',Arial;")
            self.instr = QtWidgets.QLabel("按 SPACE 开始 · ESC 退出",
                                          alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.instr.setStyleSheet(f"color:{sub};font-size:22px;font-family:'Helvetica Neue',Arial;")
            v.addStretch(2); v.addWidget(self.top); v.addStretch(1)
            v.addWidget(self.glyph); v.addWidget(self.word); v.addStretch(1)
            self.bar = QtWidgets.QProgressBar(); self.bar.setTextVisible(False); self.bar.setFixedHeight(6)
            self.bar.setStyleSheet(
                f"QProgressBar{{background:{'#eceef1' if self._light else '#1b2029'};border:none;border-radius:3px;}}"
                f"QProgressBar::chunk{{background:#2b6cb0;border-radius:3px;}}")
            v.addWidget(self.instr); v.addStretch(1); v.addWidget(self.bar)
            self.bar.setRange(0, len(self.seq)); self.bar.setValue(0)

        # ---- run control ----
        def start(self):
            """Begin immediately (used when embedded). Standalone waits for SPACE."""
            self.instr.setText("")
            self._i, self._phase = -1, None
            self._next_phase()

        def keyPressEvent(self, e):
            from PyQt6 import QtCore as _c
            if e.key() == _c.Qt.Key.Key_Escape:
                self._abort()
            elif e.key() == _c.Qt.Key.Key_Space and self._phase is None and self._i < 0:
                self.start()

        def _abort(self):
            self._timer.stop()
            self._emit_marker("rest", MI_TASKS["rest"], resting=True)  # clear any live marker
            self.finished.emit()
            self.close()

        def _cur_task(self):
            return MI_TASKS[self.seq[self._i]]

        def _next_phase(self):
            # advance the (trial, phase) cursor
            if self._phase is None or self._phase == "rest":
                self._i += 1
                if self._i >= len(self.seq):
                    self._finish(); return
                self._phase = "fixation"
                self.bar.setValue(self._i)
            else:
                self._phase = self.PHASES[self.PHASES.index(self._phase) + 1]
            self._render()
            self._fire()
            dur = getattr(self.timing, self._phase)
            self._timer.start(int(dur * 1000))

        def _render(self):
            t = self._cur_task()
            self.top.setText(f"trial {self._i + 1} / {len(self.seq)}   ·   {self._phase.upper()}")
            if self._phase == "fixation" or self._phase == "rest":
                self.glyph.setText("+")
                self.glyph.setStyleSheet(f"color:{self._sub};font-size:220px;font-weight:300;font-family:'Helvetica Neue',Arial;")
                self.word.setText("")
                self.instr.setText("准备" if self._phase == "fixation" else "休息")
            elif self._phase == "cue":
                self.glyph.setText(t.glyph)
                self.glyph.setStyleSheet(f"color:{t.color};font-size:220px;font-weight:300;font-family:'Helvetica Neue',Arial;")
                self.word.setText(t.label); self.word.setStyleSheet(
                    f"color:{t.color};font-size:40px;font-weight:600;font-family:'Helvetica Neue',Arial;")
                self.instr.setText(t.instr + "  （准备…）")
            elif self._phase == "imagery":
                self.glyph.setText(t.glyph)
                self.glyph.setStyleSheet(f"color:{t.color};font-size:250px;font-weight:500;font-family:'Helvetica Neue',Arial;")
                self.word.setText(t.label)
                self.instr.setText("▶  现在开始想象")

        def _fire(self):
            """Emit the marker + hardware trigger for the current phase."""
            t = self._cur_task()
            if self._phase == "imagery":
                self._emit_marker("imagery", t, resting=False)
            else:
                self._emit_marker(self._phase, t, resting=True)

        def _emit_marker(self, phase, task, resting):
            code = 0 if resting else task.code
            self.phase_sig.emit(phase, task.name, code)
            if not resting and self.send_trigger is not None:
                try:
                    from common.mi_events import hardware_trigger_bytes
                    self.send_trigger(hardware_trigger_bytes(task.code))
                except Exception:
                    pass

        def _finish(self):
            self._timer.stop()
            self.top.setText("完成 ✓"); self.glyph.setText("✓")
            self.glyph.setStyleSheet(f"color:#2f855a;font-size:220px;font-family:'Helvetica Neue',Arial;")
            self.word.setText(""); self.instr.setText(f"{len(self.seq)} trials 已记录")
            self.bar.setValue(len(self.seq))
            self.phase_sig.emit("end", "", 0)
            self.finished.emit()

    return MiParadigm


def MiParadigm(*a, **k):
    return _build_paradigm_class()(*a, **k)


# ----------------------------------------------------------------- standalone
def run_standalone(seq, timing, use_lsl=False, light=True):
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    outlet = None
    if use_lsl:
        from pylsl import StreamInfo, StreamOutlet, local_clock  # noqa
        outlet = StreamOutlet(StreamInfo("MI_Cues", "Markers", 1, 0, "string", "mi-cues-01"))

    def on_event(phase, name, code):
        if outlet is not None and phase in ("cue", "imagery", "rest", "end"):
            from pylsl import local_clock
            outlet.push_sample([f"{phase}/{name}" if name else phase], local_clock())

    Cls = _build_paradigm_class()
    w = Cls(seq, timing, on_event=on_event, light=light)
    w.setWindowTitle("MI paradigm")
    w.finished.connect(lambda: QtWidgets.QApplication.instance().quit())
    w.showFullScreen()
    app.exec()


def screenshot(seq, timing, out, light=True):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    Cls = _build_paradigm_class()
    w = Cls(seq, timing, light=light); w.resize(1000, 720)
    w._i, w._phase = 0, "imagery"; w._render()     # freeze on an imagery frame for the preview
    w.instr.setText("▶  现在开始想象   ( left-hand imagery )")
    w.show(); app.processEvents()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(out)); print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["left", "right"],
                    choices=list(MI_TASKS), help="tasks to cue (default: left right)")
    ap.add_argument("--reps", type=int, default=15, help="trials per task (default 15)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dark", action="store_true", help="dark cue screen (default: white)")
    ap.add_argument("--lsl", action="store_true", help="also push an LSL 'MI_Cues' marker stream")
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the trial order and exit")
    args = ap.parse_args()

    seq = make_sequence(args.tasks, args.reps, args.seed)
    timing = Timing()
    print(f"{len(args.tasks)} tasks × {args.reps} = {len(seq)} trials, "
          f"~{timing.trial:.0f}s each → ~{len(seq)*timing.trial/60:.1f} min")
    if args.dry_run:
        print(seq); return
    if args.screenshot:
        screenshot(seq, timing, args.screenshot, light=not args.dark)
    else:
        run_standalone(seq, timing, use_lsl=args.lsl, light=not args.dark)


if __name__ == "__main__":
    main()
