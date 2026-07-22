#!/usr/bin/env python
"""Task-screening paradigm — cue 8-12 candidate mental tasks, push LSL markers.

Records nothing itself: it draws timed cues and publishes an LSL **Markers** stream
('MI_Cues'). Run it alongside `udp_lsl_bridge.py` (which publishes 'Cap32' EEG) and
point **LabRecorder** at both → one time-synced XDF. Later, epoch on the `task/<name>`
markers and feed `src/exploration/task_separability.py`.

Modes:
  (default)     PsychoPy visual cues + LSL markers   (needs psychopy + a display)
  --simulate    no psychopy: run the exact timing + push markers to LSL (test LabRecorder)
  --dry-run     just print the randomized trial sequence (no LSL, no timing)

  python screening_paradigm.py --dry-run
  python screening_paradigm.py --simulate --reps 2
  python screening_paradigm.py --reps 8 --runs 3        # the real session

Install note: DON'T pip-install psychopy into the `eegmi` env (heavy deps can break
torch/mne). Use the PsychoPy standalone app or a separate env; --simulate/--dry-run need
only pylsl.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

# ---- candidate repertoire (motor + non-motor; generators under well-seated central/frontal) ----
@dataclass
class Task:
    name: str
    cue: str            # short on-screen label
    instr: str          # what to actually do
    cat: str            # motor | nonmotor | rest
    color: str = "#e8edf4"


TASKS = [
    Task("rest",        "Rest",           "Relax, eyes on the cross, think of nothing", "rest",     "#8b95a5"),
    Task("left_hand",   "LEFT hand",      "Feel yourself squeezing your LEFT hand (don't move)",  "motor", "#5aa9e6"),
    Task("right_hand",  "RIGHT hand",     "Feel yourself squeezing your RIGHT hand (don't move)", "motor", "#5aa9e6"),
    Task("feet",        "FEET",           "Feel yourself flexing both feet (don't move)", "motor",  "#5aa9e6"),
    Task("tongue",      "TONGUE",         "Feel pressing your tongue to the roof of your mouth", "motor", "#5aa9e6"),
    Task("typing",      "Type / piano",   "Imagine typing or playing piano — a movement sequence", "motor", "#7fd1b8"),
    Task("math",        "Mental math",    "Count down from 300 in steps of 7", "nonmotor", "#e6a15a"),
    Task("words",       "Word gen",       "Silently list words starting with 'S'", "nonmotor", "#e6a15a"),
    Task("song",        "Song",           "Replay a familiar song in your head", "nonmotor", "#e6a15a"),
    Task("navigate",    "Walk home",      "Imagine walking through your home room by room", "nonmotor", "#e6a15a"),
]
TASK_BY_NAME = {t.name: t for t in TASKS}


# --------------------------------------------------------------- trial sequence (testable)
def make_sequence(task_names, reps, runs, seed=7):
    """Balanced, interleaved order — no task repeats back-to-back within a run."""
    import random
    rng = random.Random(seed)
    runs_out = []
    for _ in range(runs):
        pool = task_names * reps
        for _try in range(200):
            rng.shuffle(pool)
            if all(pool[i] != pool[i + 1] for i in range(len(pool) - 1)):
                break
        runs_out.append(list(pool))
    return runs_out


# ------------------------------------------------------------------------- LSL markers
def make_marker_outlet():
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo("MI_Cues", "Markers", 1, 0, "string", "mi-cues-01")
    return StreamOutlet(info)


def push(outlet, msg):
    if outlet is not None:
        from pylsl import local_clock
        outlet.push_sample([msg], local_clock())


# ------------------------------------------------------------------------- timing spec
@dataclass
class Timing:
    fixation: float = 2.0
    cue: float = 1.5
    task: float = 4.0
    rest: float = 2.0     # inter-trial


# --------------------------------------------------------------------- runners
def run_simulate(seq, timing, outlet):
    """Full timing + markers, console output, no psychopy."""
    for r, run in enumerate(seq):
        push(outlet, f"run/start/{r}")
        print(f"\n=== run {r+1}/{len(seq)} ({len(run)} trials) ===")
        for i, name in enumerate(run):
            t = TASK_BY_NAME[name]
            push(outlet, "fixation"); time.sleep(timing.fixation)
            push(outlet, f"cue/{name}"); print(f"  [{i+1:02d}] cue: {t.cue:<14} — {t.instr}")
            time.sleep(timing.cue)
            push(outlet, f"task/{name}")           # <-- the epoching marker
            time.sleep(timing.task)
            push(outlet, "rest"); time.sleep(timing.rest)
        push(outlet, f"run/end/{r}")
    print("\ndone.")


def run_psychopy(seq, timing, outlet):
    from psychopy import visual, core, event
    win = visual.Window(fullscr=True, color="#0e1116", units="norm")
    fix = visual.TextStim(win, text="+", height=0.2, color="#c8ced8")
    cue = visual.TextStim(win, text="", height=0.14, color="#e8edf4", wrapWidth=1.6)
    instr = visual.TextStim(win, text="", height=0.06, color="#8b95a5", pos=(0, -0.25), wrapWidth=1.6)
    go = visual.TextStim(win, text="", height=0.16, color="#e8edf4")

    def wait_draw(stims, dur):
        stims = stims if isinstance(stims, list) else [stims]
        t0 = core.getTime()
        while core.getTime() - t0 < dur:
            for s in stims:
                s.draw()
            win.flip()
            if "escape" in event.getKeys():
                win.close(); core.quit()

    # intro
    cue.text = "Screening session\n\npress SPACE to start"
    cue.draw(); win.flip(); event.waitKeys(keyList=["space"])

    for r, run in enumerate(seq):
        push(outlet, f"run/start/{r}")
        for name in run:
            t = TASK_BY_NAME[name]
            push(outlet, "fixation"); wait_draw(fix, timing.fixation)
            cue.text, cue.color, instr.text = t.cue, t.color, t.instr
            push(outlet, f"cue/{name}"); wait_draw([cue, instr], timing.cue)
            go.text, go.color = t.cue, t.color
            push(outlet, f"task/{name}"); wait_draw(go, timing.task)   # <-- epoching marker
            push(outlet, "rest"); wait_draw(fix, timing.rest)
        push(outlet, f"run/end/{r}")
        cue.text, cue.color = f"break — run {r+1}/{len(seq)} done\n\npress SPACE", "#c8ced8"
        cue.draw(); win.flip(); event.waitKeys(keyList=["space"])
    win.close(); core.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", default=[t.name for t in TASKS],
                    help="subset of task names to screen")
    ap.add_argument("--reps", type=int, default=6, help="trials per task per run")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unknown = [t for t in args.tasks if t not in TASK_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}\navailable: {list(TASK_BY_NAME)}")

    seq = make_sequence(args.tasks, args.reps, args.runs, args.seed)
    total = sum(len(r) for r in seq)
    per = {n: sum(run.count(n) for run in seq) for n in args.tasks}
    print(f"{len(args.tasks)} tasks × {args.reps} reps × {args.runs} runs = {total} trials "
          f"(~{Timing().__dict__['fixation']+Timing().cue+Timing().task+Timing().rest:.0f}s each "
          f"→ ~{total*9/60:.0f} min)\nper-task: {per}")

    if args.dry_run:
        for r, run in enumerate(seq):
            print(f"run {r+1}: {run}")
        return

    outlet = make_marker_outlet()
    print("LSL 'MI_Cues' marker stream open — start udp_lsl_bridge.py + LabRecorder now.")
    timing = Timing()
    (run_simulate if args.simulate else run_psychopy)(seq, timing, outlet)


if __name__ == "__main__":
    main()
