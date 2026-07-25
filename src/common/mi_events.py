"""Canonical MI event codes — the single source of truth shared by the paradigm
(`src/experiment/mi_paradigm.py`), the acquisition GUI (`src/acquisition/cap_gui.py`),
and the analysis loader (`src/analysis/load.py`).

A recording carries an integer label per sample. During a paradigm run the GUI stamps
these small codes (software marker, guaranteed) at *imagery onset*; between trials the
label is 0 (no event). The loader turns the rising edges of this label track into MNE
events and epochs.
"""
from __future__ import annotations

# name -> integer code written into the recording's `marker` track at imagery onset.
# 1–9   motor imagery (sensorimotor / central)
# 10–19 cognitive & non-motor imagery — these often separate BETTER than two motor
#       classes on a dry cap, because they recruit different networks (frontal/temporal)
#       instead of relying on fine C3-vs-C4 spatial resolution. See the screening plan.
MI_TASK_CODES = {
    "rest": 1,
    "left": 2,
    "right": 3,
    "feet": 4,
    "tongue": 5,
    "hands": 6,          # both hands together
    # --- cognitive / non-motor ---
    "math": 10,          # serial subtraction (300 − 7 − 7 …)
    "words": 11,         # word association / verbal fluency
    "song": 12,          # auditory imagery (replay a familiar song)
    "navigate": 13,      # spatial navigation (walk through your home)
    "rotation": 14,      # mental rotation of a 3-D object
    "face": 15,          # imagery of a familiar face
}
CODE_TO_LABEL = {v: k for k, v in MI_TASK_CODES.items()}

# coarse grouping — handy for analysis (motor vs cognitive contrasts) and for the UI
TASK_CATEGORY = {
    "rest": "rest",
    "left": "motor", "right": "motor", "feet": "motor", "tongue": "motor", "hands": "motor",
    "math": "cognitive", "words": "cognitive", "song": "cognitive",
    "navigate": "cognitive", "rotation": "cognitive", "face": "cognitive",
}


def label_of(code: int) -> str:
    return CODE_TO_LABEL.get(int(code), f"code{int(code)}")


def hardware_trigger_bytes(code: int) -> bytes:
    """The 4-char `TXXXX` trigger command sent to the board at imagery onset, so the
    board *also* stamps its own trigger bytes into the stream (sample-accurate if the
    firmware echoes it). The recording's software `marker` track is the primary label;
    this is a redundant hardware path."""
    return b"T" + f"{int(code):04d}".encode("ascii")   # e.g. code 2 -> b"T0002"
