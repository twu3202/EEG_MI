#!/usr/bin/env python
"""Motor-imagery paradigm — a full-screen cue window that tells you EXACTLY what to
imagine, right now, in huge type.

Design goal: while you are doing imagery you should NOT be looking at (or thinking about)
the EEG scope. So the cue lives in its own borderless full-screen window — on a SECOND
MONITOR if you have one — showing one unmistakable instruction at a time:

    准备…            ->            现在想象                ->     休息
                                  ← 左  手
                              持续想象握拳的感觉,不要真的动
                                      ⏱ 3

Each trial is a 4-phase state machine; the marker fires at IMAGERY onset, which is what
the analysis epochs on:

    fixation  →  cue (准备, 知道下一个是什么)  →  IMAGERY (打标)  →  rest
      1.5s          1.5s                            4.0s              2.0s

Two ways to use it:

  1. Embedded in cap_gui (recommended — one process, sample-accurate marking):
        pab = MiParadigm(seq, timing, on_event=handler, send_trigger=recv.send)
        pab.finished.connect(on_done); pab.start()
     `on_event(phase, name, code)` fires at every phase boundary; cap_gui uses it to stamp
     the recording's software marker and push the hardware `TXXXX` at imagery onset.

  2. Standalone (practice / preview):
        python mi_paradigm.py                            # practice run, no recording
        python mi_paradigm.py --tasks left right feet --reps 12
        python mi_paradigm.py --mode visual              # visual instead of kinesthetic
        python mi_paradigm.py --screen 1                 # force monitor #1
        python mi_paradigm.py --screenshot out.png --phase imagery
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

FONT = "'PingFang SC','Microsoft YaHei','Helvetica Neue',Helvetica,Arial,sans-serif"


# --------------------------------------------------------------------- tasks
@dataclass
class MiTask:
    name: str
    glyph: str          # arrow / symbol shown next to the word
    cn: str             # the BIG word (what you actually read)
    en: str
    kmi: str            # kinesthetic instruction (feel it)   — cognitive tasks: same text
    vmi: str            # visual instruction (see it)
    color: str
    cat: str = "motor"  # motor | cognitive | rest

    @property
    def code(self) -> int:
        return MI_TASK_CODES[self.name]

    def instr(self, mode="kinesthetic"):
        return self.vmi if mode.startswith("v") else self.kmi


MI_TASKS = {
    "left": MiTask(
        "left", "←", "左手", "LEFT hand",
        "持续想象【左手】握拳发力的感觉 · 不要真的动",
        "在脑中看到自己的【左手】反复握拳 · 不要真的动", "#1d6fbf"),
    "right": MiTask(
        "right", "→", "右手", "RIGHT hand",
        "持续想象【右手】握拳发力的感觉 · 不要真的动",
        "在脑中看到自己的【右手】反复握拳 · 不要真的动", "#c0392b"),
    "feet": MiTask(
        "feet", "↓", "双脚", "FEET",
        "持续想象【双脚】用力下踩的感觉 · 不要真的动",
        "在脑中看到自己的【双脚】反复下踩 · 不要真的动", "#2f855a"),
    "tongue": MiTask(
        "tongue", "●", "舌头", "TONGUE",
        "持续想象【舌尖抵住上颚】的感觉 · 不要真的动",
        "在脑中看到自己的舌尖抵住上颚", "#805ad5"),
    "hands": MiTask(
        "hands", "↑", "双手", "BOTH hands",
        "持续想象【双手】同时握拳发力的感觉 · 不要真的动",
        "在脑中看到自己的【双手】同时反复握拳 · 不要真的动", "#0e7490"),
    "rest": MiTask(
        "rest", "+", "休息", "Rest",
        "放松 · 看着十字 · 什么都不想", "放松 · 看着十字 · 什么都不想", "#6b7480", "rest"),

    # ---- cognitive / non-motor imagery -------------------------------------------
    # These recruit frontal / temporal networks instead of needing fine C3-vs-C4
    # resolution, so on a dry cap they often separate BETTER than two motor classes.
    # Instructions deliberately warn against the confounds that would let a decoder
    # "cheat" on EMG/EOG rather than brain activity.
    "math": MiTask(
        "math", "∑", "连减7", "Mental subtraction",
        "从 300 开始连续减 7:300 → 293 → 286 …  ·  心里默算,不出声、不动嘴",
        "从 300 开始连续减 7:300 → 293 → 286 …  ·  心里默算,不出声、不动嘴",
        "#b45309", "cognitive"),
    "words": MiTask(
        "words", "✎", "想词", "Word association",
        "默想以【花】开头的词,一个接一个  ·  不出声、不动嘴唇和下巴",
        "默想以【花】开头的词,一个接一个  ·  不出声、不动嘴唇和下巴",
        "#0891b2", "cognitive"),
    "song": MiTask(
        "song", "♪", "放歌", "Auditory imagery",
        "在脑中播放一首熟悉的歌  ·  只在心里听,不哼唱、不打拍子",
        "在脑中播放一首熟悉的歌  ·  只在心里听,不哼唱、不打拍子",
        "#be185d", "cognitive"),
    "navigate": MiTask(
        "navigate", "⌂", "走房间", "Spatial navigation",
        "在脑中从家门口走进去,依次走过每个房间  ·  眼睛别乱动",
        "在脑中从家门口走进去,依次走过每个房间  ·  眼睛别乱动",
        "#0d9488", "cognitive"),
    "rotation": MiTask(
        "rotation", "⟳", "旋转", "Mental rotation",
        "在脑中让一个立方体绕轴慢慢旋转  ·  眼睛别乱动",
        "在脑中让一个立方体绕轴慢慢旋转  ·  眼睛别乱动",
        "#7c3aed", "cognitive"),
    "face": MiTask(
        "face", "☺", "想脸", "Familiar face",
        "在脑中清晰地想起一张熟悉的脸,保持住这个画面  ·  眼睛别乱动",
        "在脑中清晰地想起一张熟悉的脸,保持住这个画面  ·  眼睛别乱动",
        "#e11d48", "cognitive"),
}

# ready-made task sets. Mixed motor+cognitive sets are usually the EASIEST to decode on a
# dry cap — different networks beat fine left-vs-right spatial resolution.
TASK_SETS = {
    # Ordered EASIEST → HARDEST for this dry cap, from the first real 30-trial session:
    # left-vs-right gave AUC ~0.60 with permutation p=0.41 (chance), because separating C3
    # from C4 needs fine lateral resolution a dry cap does not deliver. Prefer contrasts
    # that differ in WHICH NETWORK is engaged, not in which hemisphere.
    "① 双手 / 休息":   ["hands", "rest"],
    "② 双手 / 减7":    ["hands", "math"],
    "③ 双手 / 双脚":   ["hands", "feet"],
    "④ 减7 / 放歌":    ["math", "song"],
    "⑤ 手 / 脚 / 减7": ["hands", "feet", "math"],
    "⑥ 认知 4 类":     ["math", "words", "song", "navigate"],
    "⑦ 筛选 8 类":     ["rest", "hands", "feet", "left", "math", "words", "song", "navigate"],
    "⑧ 左手 / 右手":   ["left", "right"],
}

# why each set, shown as a tooltip in the GUI dropdown
TASK_SET_WHY = {
    "① 双手 / 休息":   "先验证到底有没有 MI 信号 —— 最基本的检验,建议第一个跑",
    "② 双手 / 减7":    "运动 vs 认知,调动完全不同的网络,干电极上预期最好分",
    "③ 双手 / 双脚":   "手在外侧 C3/C4、脚在内侧 Cz,空间差异大,不需要左右侧化",
    "④ 减7 / 放歌":    "额区(心算) vs 颞区(听觉想象)",
    "⑤ 手 / 脚 / 减7": "三类:运动外侧 / 运动内侧 / 认知",
    "⑥ 认知 4 类":     "减7 / 想词 / 放歌 / 走房间 —— 全认知任务",
    "⑦ 筛选 8 类":     "跑一轮筛选,用数据帮你选出最可分的组合",
    "⑧ 左手 / 右手":   "⚠ 最难:需要精细的 C3/C4 侧化,本机实测 ≈ 随机 (p=0.41)",
}

# background tint per phase — makes the current state unmistakable, even peripherally
TINT = {"fixation": "#ffffff", "cue": "#fff8e8", "imagery": "#eefaf1", "rest": "#f4f6f8"}


@dataclass
class Timing:
    fixation: float = 1.5
    cue: float = 1.5
    imagery: float = 4.0
    rest: float = 2.0
    break_every: int = 0        # 0 = no breaks; else pause every N trials
    jitter: float = 0.0         # ± random seconds added to fixation (avoid rhythm/anticipation)

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
_CLS = None


def _paradigm_class():
    """Build (once) the Qt widget class — imported lazily so this module works headless."""
    global _CLS
    if _CLS is not None:
        return _CLS
    from PyQt6 import QtWidgets, QtCore

    class MiParadigm(QtWidgets.QWidget):
        finished = QtCore.pyqtSignal()
        phase_sig = QtCore.pyqtSignal(str, str, int)   # phase, task-name, code (code>0 = imagery)

        PHASES = ("fixation", "cue", "imagery", "rest")

        def __init__(self, seq, timing=None, on_event=None, send_trigger=None,
                     light=True, mode="kinesthetic", parent=None):
            super().__init__(parent)
            self.seq = list(seq)
            self.timing = timing or Timing()
            self.send_trigger = send_trigger
            self.mode = mode
            if on_event is not None:
                self.phase_sig.connect(lambda p, n, c: on_event(p, n, c))
            self._i = -1
            self._phase = None
            self._paused = False
            self._done = False      # guard: `finished` must fire exactly once
            self._light = light
            self._rng = __import__("random").Random(11)
            self._build_ui()
            self._timer = QtCore.QTimer(self); self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._next_phase)
            self._clock = QtCore.QElapsedTimer()
            self._phase_ms = 0
            self._tick_t = QtCore.QTimer(self); self._tick_t.timeout.connect(self._tick)

        # ------------------------------------------------------------------ ui
        def _build_ui(self):
            self.setStyleSheet(f"background:#ffffff;")
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(60, 34, 60, 34); v.setSpacing(0)

            self.top = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.top.setStyleSheet(f"color:#98a1ad;font-size:19px;letter-spacing:2px;font-family:{FONT};")
            v.addWidget(self.top)
            v.addStretch(2)

            self.kicker = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.kicker.setStyleSheet(f"color:#8a929e;font-size:36px;letter-spacing:6px;font-family:{FONT};")
            v.addWidget(self.kicker)
            v.addSpacing(6)

            self.word = QtWidgets.QLabel("+", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.word.setStyleSheet(f"color:#1f2733;font-size:200px;font-weight:700;font-family:{FONT};")
            v.addWidget(self.word)
            v.addSpacing(10)

            self.instr = QtWidgets.QLabel("按 SPACE 开始   ·   ESC 退出",
                                          alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.instr.setStyleSheet(f"color:#6b7480;font-size:31px;font-family:{FONT};")
            v.addWidget(self.instr)
            v.addStretch(1)

            self.count = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.count.setStyleSheet(f"color:#b9c0ca;font-size:74px;font-weight:300;font-family:{FONT};")
            v.addWidget(self.count)
            v.addStretch(2)

            self.phase_bar = QtWidgets.QProgressBar()          # time left in THIS phase
            self.phase_bar.setTextVisible(False); self.phase_bar.setFixedHeight(10)
            self.phase_bar.setRange(0, 1000)
            v.addWidget(self.phase_bar)
            v.addSpacing(8)

            self.bar = QtWidgets.QProgressBar()                # overall run progress
            self.bar.setTextVisible(False); self.bar.setFixedHeight(5)
            self.bar.setRange(0, max(1, len(self.seq))); self.bar.setValue(0)
            self.bar.setStyleSheet(
                "QProgressBar{background:#edeff2;border:none;border-radius:2px;}"
                "QProgressBar::chunk{background:#c3cad4;border-radius:2px;}")
            v.addWidget(self.bar)

        def _set_phase_bar(self, color):
            self.phase_bar.setStyleSheet(
                "QProgressBar{background:#e9ecf0;border:none;border-radius:5px;}"
                f"QProgressBar::chunk{{background:{color};border-radius:5px;}}")

        # --------------------------------------------------------- run control
        def start(self):
            """Begin immediately (used when embedded; standalone waits for SPACE)."""
            self._i, self._phase = -1, None
            self._next_phase()

        def keyPressEvent(self, e):
            from PyQt6 import QtCore as _c
            k = e.key()
            if k == _c.Qt.Key.Key_Escape:
                self._abort()
            elif k == _c.Qt.Key.Key_Space:
                if self._paused:                       # resume from a break
                    self._paused = False
                    self._next_phase()
                elif self._phase is None and self._i < 0:
                    self.start()

        def _abort(self):
            self._timer.stop(); self._tick_t.stop()
            if self._done:                  # already finished normally — closing must NOT
                self.close(); return        # re-emit, or the recording gets saved twice
            self._done = True
            self.phase_sig.emit("abort", "", 0)         # clears any live marker
            self.finished.emit()
            self.close()

        def _cur(self):
            return MI_TASKS[self.seq[self._i]]

        def _next_phase(self):
            # advance the (trial, phase) cursor
            if self._phase is None or self._phase == "rest":
                self._i += 1
                if self._i >= len(self.seq):
                    self._finish(); return
                be = self.timing.break_every
                if be and self._i and self._i % be == 0:
                    self._show_break(); return
                self._phase = "fixation"
                self.bar.setValue(self._i)
            else:
                self._phase = self.PHASES[self.PHASES.index(self._phase) + 1]

            dur = float(getattr(self.timing, self._phase))
            if self._phase == "fixation" and self.timing.jitter:
                dur += self._rng.uniform(-self.timing.jitter, self.timing.jitter)
            self._phase_ms = max(200, int(dur * 1000))
            self._render()
            self._fire()
            self._clock.restart(); self._tick_t.start(50); self._tick()
            self._timer.start(self._phase_ms)

        def _tick(self):
            left = max(0, self._phase_ms - self._clock.elapsed())
            self.phase_bar.setValue(int(1000 * left / self._phase_ms))
            # Countdown ONLY during cue. During IMAGERY the digit would change 4x, and each
            # change is a visual transient: the first real recording showed occipital 13-30 Hz
            # +46% (p=0.023) — the ONLY significant effect in the whole dataset — i.e. the
            # screen was driving visual cortex while sensorimotor showed nothing. Keep the
            # imagery screen visually STATIC; the thin phase bar is enough feedback.
            if self._phase == "cue":
                self.count.setText(f"{left/1000:.0f}")
            else:
                self.count.setText("")

        # ------------------------------------------------------------ rendering
        @staticmethod
        def _fit(text, base):
            """Shrink the big word when it's long, so 3–4 char tasks still fit on one line."""
            n = len(text.replace(" ", ""))
            return base if n <= 3 else int(base * (0.80 if n == 4 else 0.66 if n == 5 else 0.55))

        def _big(self, t, base):
            txt = t.cn if t.name == "rest" else f"{t.glyph} {t.cn}"
            return txt, self._fit(txt, base)

        def _render(self):
            t = self._cur()
            ph = self._phase
            self.setStyleSheet(f"background:{TINT[ph]};")
            self.top.setText(f"{self._i + 1} / {len(self.seq)}      {ph.upper()}")

            if ph == "fixation":
                self.kicker.setText("")
                self.word.setText("+")
                self.word.setStyleSheet(f"color:#c3cad4;font-size:200px;font-weight:300;font-family:{FONT};")
                self.instr.setText("放松,准备下一个")
                self._set_phase_bar("#c3cad4")

            elif ph == "cue":
                txt, size = self._big(t, 170)
                self.kicker.setText("准 备")
                self.word.setText(txt)
                self.word.setStyleSheet(f"color:{t.color};font-size:{size}px;font-weight:700;"
                                        f"font-family:{FONT};")
                self.instr.setText("马上开始 —— 先别动,等提示")
                self._set_phase_bar("#e0a800")

            elif ph == "imagery":
                txt, size = self._big(t, 210)
                self.kicker.setText("现 在 想 象" if t.cat != "rest" else "现 在")
                self.word.setText(txt)
                self.word.setStyleSheet(f"color:{t.color};font-size:{size}px;font-weight:800;"
                                        f"font-family:{FONT};")
                self.instr.setText(t.instr(self.mode))
                self._set_phase_bar("#2e9e5b")

            elif ph == "rest":
                self.kicker.setText("")
                self.word.setText("休息")
                self.word.setStyleSheet(f"color:#aab2bd;font-size:120px;font-weight:600;"
                                        f"font-family:{FONT};")
                self.instr.setText("放松,停止想象")
                self._set_phase_bar("#c3cad4")

        def _show_break(self):
            self._paused = True
            self._timer.stop(); self._tick_t.stop()
            done, total = self._i, len(self.seq)
            self.setStyleSheet("background:#ffffff;")
            self.top.setText(f"{done} / {total}      BREAK")
            self.kicker.setText("休 息 一 下")
            self.word.setText("☕")
            self.word.setStyleSheet(f"color:#1f2733;font-size:150px;font-family:{FONT};")
            self.instr.setText(f"已完成 {done} / {total} —— 准备好后按 SPACE 继续")
            self.count.setText(""); self.phase_bar.setValue(0)
            self.phase_sig.emit("break", "", 0)

        # ------------------------------------------------------------- markers
        def _fire(self):
            t = self._cur()
            if self._phase == "imagery" and t.name != "rest":
                self.phase_sig.emit("imagery", t.name, t.code)
                if self.send_trigger is not None:
                    try:
                        from common.mi_events import hardware_trigger_bytes
                        self.send_trigger(hardware_trigger_bytes(t.code))
                    except Exception:
                        pass
            elif self._phase == "imagery":                      # rest trial: label it too
                self.phase_sig.emit("imagery", t.name, t.code)
            else:
                self.phase_sig.emit(self._phase, t.name, 0)     # code 0 -> marker track clear

        def _finish(self):
            self._timer.stop(); self._tick_t.stop()
            self.setStyleSheet("background:#ffffff;")
            self.top.setText(f"{len(self.seq)} / {len(self.seq)}      DONE")
            self.kicker.setText("")
            self.word.setText("✓")
            self.word.setStyleSheet(f"color:#2f855a;font-size:200px;font-family:{FONT};")
            self.instr.setText(f"完成 —— 共 {len(self.seq)} 个 trial,数据已保存")
            self.count.setText(""); self.phase_bar.setValue(0)
            self.bar.setValue(len(self.seq))
            self.phase_sig.emit("end", "", 0)
            if not self._done:
                self._done = True
                self.finished.emit()

    _CLS = MiParadigm
    return _CLS


def MiParadigm(*a, **k):
    return _paradigm_class()(*a, **k)


# ----------------------------------------------------------------- placement
def place(widget, screen_idx=None, windowed=False):
    """Put the cue window on its own screen: prefer a SECOND monitor so the EEG UI can stay
    visible on the main one. Falls back to the primary screen (full-screen covers the UI,
    which is also what we want — no distraction)."""
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    screens = app.screens()
    if screen_idx is None:
        screen_idx = 1 if len(screens) > 1 else 0
    screen_idx = max(0, min(screen_idx, len(screens) - 1))
    geo = screens[screen_idx].geometry()
    if windowed:
        widget.setGeometry(geo.x() + 80, geo.y() + 80, 1100, 760)
        widget.show()
    else:
        widget.setGeometry(geo)
        widget.showFullScreen()
    widget.raise_(); widget.activateWindow()
    return screen_idx, len(screens)


# ----------------------------------------------------------------- standalone
def run_standalone(seq, timing, mode, screen, windowed):
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = _paradigm_class()(seq, timing, mode=mode)
    w.setWindowTitle("MI paradigm")
    w.finished.connect(lambda: QtWidgets.QApplication.instance().quit())
    idx, n = place(w, screen, windowed)
    print(f"cue window on screen {idx} of {n}  ·  SPACE 开始 · ESC 退出")
    app.exec()


def screenshot(seq, timing, out, mode, phase="imagery"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = _paradigm_class()(seq, timing, mode=mode)
    w.resize(1280, 860)
    w._i, w._phase, w._phase_ms = 0, phase, 4000
    w._render()
    w.phase_bar.setValue(720)
    w.count.setText("3" if phase in ("imagery", "cue") else "")
    w.show(); app.processEvents()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(out)); print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["left", "right"], choices=list(MI_TASKS),
                    help="任务名;认知类: math words song navigate rotation face")
    ap.add_argument("--set", dest="task_set", default=None, choices=list(TASK_SETS),
                    help="用预设任务组合(覆盖 --tasks)")
    ap.add_argument("--reps", type=int, default=15, help="trials per task (default 15)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", default="kinesthetic", choices=["kinesthetic", "visual"],
                    help="KMI = feel the movement (default) · VMI = see the movement")
    ap.add_argument("--imagery", type=float, default=4.0, help="imagery seconds per trial")
    ap.add_argument("--break-every", type=int, default=0, help="pause every N trials")
    ap.add_argument("--jitter", type=float, default=0.0, help="± s jitter on fixation")
    ap.add_argument("--screen", type=int, default=None, help="monitor index (default: 2nd if present)")
    ap.add_argument("--windowed", action="store_true", help="windowed instead of full-screen")
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--phase", default="imagery", choices=["fixation", "cue", "imagery", "rest"])
    ap.add_argument("--dry-run", action="store_true", help="print the trial order and exit")
    args = ap.parse_args()

    tasks = TASK_SETS[args.task_set] if args.task_set else args.tasks
    seq = make_sequence(tasks, args.reps, args.seed)
    timing = Timing(imagery=args.imagery, break_every=args.break_every, jitter=args.jitter)
    print(f"{len(tasks)} tasks {tasks} × {args.reps} = {len(seq)} trials, "
          f"~{timing.trial:.0f}s each → ~{len(seq)*timing.trial/60:.1f} min  ·  mode={args.mode}")
    if args.dry_run:
        print(seq); return
    if args.screenshot:
        screenshot(seq, timing, args.screenshot, args.mode, args.phase)
    else:
        run_standalone(seq, timing, args.mode, args.screen, args.windowed)


if __name__ == "__main__":
    main()
