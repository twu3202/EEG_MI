#!/usr/bin/env python
"""Record a calibrator run for crosstalk analysis, with all 32 channels and full metadata.

Separate from the analysis so a session can be re-analysed without re-recording — the
hardware is the perishable resource, the maths is not.

  python src/acquisition/crosstalk_capture.py --secs 180 --tag fp1
  python src/acquisition/crosstalk_capture.py --secs 180 --tag fp1 --sfreq 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.montage import CAP32_CHANNELS as CH          # noqa: E402

OUT = HERE.parents[1] / "recordings" / "calibrator"


def capture(host, port, sfreq, secs, note):
    from udp_lsl_bridge import UdpSource, parse_packet, board_init, drain
    src = UdpSource(host, port)
    board_init(src, sfreq)
    drain(src)                      # the mode-switch transient is not link quality
    src.sock.settimeout(3.0)
    buf, lost, last = [], 0, None
    t0 = end = None
    for pkt in src.frames():
        p = parse_packet(pkt)
        if p is not None:
            if t0 is None:
                t0 = time.time(); end = t0 + secs
            buf.append(p[0])
            if last is not None:
                lost += (p[1] - last - 1) % 256
            last = p[1]
        if end and time.time() >= end:
            break
    if not buf:
        raise SystemExit("no data — is the Mac on 192.168.4.2 and the cap powered?")
    X = np.asarray(buf, float).T
    dur = time.time() - t0
    return X, dict(frames_received=len(buf), frames_lost=int(lost),
                   loss_pct=round(100 * lost / max(1, len(buf) + lost), 3),
                   nominal_sfreq=sfreq, measured_sfreq=round(len(buf) / dur, 2),
                   duration_s=round(dur, 1), note=note)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=float, default=250.0, choices=[250.0, 500.0, 1000.0])
    ap.add_argument("--secs", type=float, default=180.0)
    ap.add_argument("--freq", type=float, default=7.0, help="generator setting, for metadata")
    ap.add_argument("--tag", required=True, help="e.g. the driven electrode: fp1")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    print(f"recording {a.secs:.0f}s @ {a.sfreq:.0f} Hz …")
    X, link = capture(a.host, a.port, a.sfreq, a.secs, a.note)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"cal_{a.freq:g}hz_{a.tag}_{a.sfreq:.0f}sps_{a.secs:.0f}s.npz"
    np.savez_compressed(OUT / name, data=X.astype(np.float32), fs=float(a.sfreq),
                        ch_names=np.array(CH), meta_json=json.dumps({
                            "kind": "calibrator crosstalk", "date": "2026-08-03",
                            "setup": "cap off head; generator GND -> board GND and REF; "
                                     f"{a.freq:g} Hz sine into ONE channel input "
                                     f"(operator says: {a.tag.upper()}); all other inputs OPEN",
                            "signal": {"freq_hz": a.freq, "waveform": "sine",
                                       "amplitude": "not stated by operator"},
                            "units": "microvolts",
                            "notes": "RAW, pre-CAR, unfiltered, all 32 channels retained",
                            "link": link}, ensure_ascii=False))
    print(f"saved {OUT / name}   {X.shape}")
    print(f"  link: {link['frames_lost']} lost ({link['loss_pct']} %), "
          f"measured {link['measured_sfreq']} Hz vs nominal {a.sfreq:.0f}")


if __name__ == "__main__":
    main()
