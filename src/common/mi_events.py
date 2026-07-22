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
MI_TASK_CODES = {
    "rest": 1,
    "left": 2,
    "right": 3,
    "feet": 4,
    "tongue": 5,
}
CODE_TO_LABEL = {v: k for k, v in MI_TASK_CODES.items()}


def label_of(code: int) -> str:
    return CODE_TO_LABEL.get(int(code), f"code{int(code)}")


def hardware_trigger_bytes(code: int) -> bytes:
    """The 4-char `TXXXX` trigger command sent to the board at imagery onset, so the
    board *also* stamps its own trigger bytes into the stream (sample-accurate if the
    firmware echoes it). The recording's software `marker` track is the primary label;
    this is a redundant hardware path."""
    return b"T" + f"{int(code):04d}".encode("ascii")   # e.g. code 2 -> b"T0002"
