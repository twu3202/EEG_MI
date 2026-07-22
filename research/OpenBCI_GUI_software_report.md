# Open-source EEG acquisition GUI / software to borrow from

*For a simple in-house tool for the custom 32-ch ADS1299 WiFi/UDP cap (packet 105 B, `0xA0…0xC0`; `B`/`S`/`TX`). Goal: live display, quality/impedance proxy (no impedance HW), MI trigger/marker injection, standard recording.*

## Headline: a near-identical project already exists
**Cerelog ESP-EEG** — ESP32 + ADS1299 over WiFi/UDP (your exact hardware class). It reaches the OpenBCI GUI + BrainFlow via a **Python UDP→LSL bridge**, shipping both a BrainFlow fork and an OpenBCI-GUI fork. Study it first: https://github.com/Cerelog-ESP-EEG/ESP-EEG
The pattern it validates — **bridge your UDP stream to LSL, then reuse the whole LSL ecosystem** — is the minimum-effort path and the through-line below.

## OpenBCI GUI
- MIT, **Processing 4 (Java)**; v5+ uses **BrainFlow** for all I/O. Repo: https://github.com/OpenBCI/OpenBCI_GUI
- **Cannot ingest a custom/LSL/UDP stream out of the box.** Its Networking widget only *outputs* (Serial/UDP/OSC/**LSL**). Input = BrainFlow board selection (Cyton/Ganglion/Synthetic/Playback/Streaming) — no "read arbitrary stream" mode, and no 32-ch board exists.
- Adapting it = either a custom BrainFlow board (C++) or a GUI fork that reads LSL (the Cerelog route). Polished widgets, heavy toolchain.

## BrainFlow
- MIT, bindings for Python/C++/Java/C#/R/MATLAB/Julia. Repo: https://github.com/brainflow-dev/brainflow
- **No generic "push samples in" board.** Synthetic=fake, Playback=replay its CSV, Streaming=consume another BrainFlow master. To join the ecosystem you must **register a custom board in C++** (guide: https://brainflow.org/2022-11-01-adding-new-boards/). Payoff: all SDKs + OpenBCI GUI board + Timeflux node from one impl. Cost: C++ + maintain a fork.

## Lighter-weight options
| Tool | License | Realtime plot | Custom ingest | Repo |
|---|---|---|---|---|
| **MNE-LSL** (was BSL) | BSD-3 | `StreamViewer` (Qt) | any LSL; `Player` replays FIF→LSL | https://github.com/mne-tools/mne-lsl |
| muse-lsl | BSD | yes | device→LSL template to copy | https://github.com/alexandrebarachant/muse-lsl |
| PyQtGraph | MIT | fast scrolling | you write ingest | https://github.com/pyqtgraph/pyqtgraph |
| pylsl examples | MIT | `ReceiveAndPlot.py` | any LSL | https://github.com/labstreaminglayer/pylsl |
| Timeflux (+timeflux_ui) | MIT | web UI | LSL node, YAML graph, epoching/ML | https://github.com/timeflux/timeflux |
| OpenViBE | AGPL | yes + **Graz MI/CSP scenarios** | Acquisition Server (LSL in) | https://openvibe.inria.fr (v3.7.0, Sep 2025; no official macOS) |
| NeuroPype | proprietary | yes | LSL native | not forkable |
| Bonsai | MIT | visual dataflow | LSL pkgs | https://github.com/bonsai-rx/bonsai (.NET/Windows) |

**MNE-LSL** is the standout modern pick (BSD, active Jul 2025, real-time viewer + tight MNE path). **OpenViBE** has ready MI/CSP scenarios but is AGPL/heavy/no-mac.

## The one piece you must build: UDP→LSL bridge (~50–80 lines)
Expose the cap as an LSL `EEG` stream + a separate `Markers` stream; then MNE-LSL viewer, LabRecorder, Timeflux, OpenViBE, a GUI fork — all work for free. No canonical pip package; copy the pattern from muse-lsl / Cerelog. Sketch:
```python
import socket
from pylsl import StreamInfo, StreamOutlet, local_clock
N_CH, FS = 32, 500                                  # set to your real rate
info = StreamInfo("CustomADS1299", "EEG", N_CH, FS, "float32", "ads1299-cap-01")
chns = info.desc().append_child("channels")
for lbl in CAP32_CHANNELS:                          # from src/common/montage.py
    c = chns.append_child("channel")
    c.append_child_value("label", lbl); c.append_child_value("unit", "microvolts")
eeg = StreamOutlet(info)
mrk = StreamOutlet(StreamInfo("MI_Markers", "Markers", 1, 0, "string", "mi-mrk-01"))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.bind(("0.0.0.0", 8086))
while True:
    pkt, _ = sock.recvfrom(2048)
    if len(pkt) != 105 or pkt[0] != 0xA0 or pkt[-1] != 0xC0:  # frame check
        continue
    eeg.push_sample(parse_105byte_packet(pkt), local_clock())  # -> 32 floats (µV)
    # mrk.push_sample(["left"]) when the MI cue fires
```
Keep `B`/`S`/`TX` as plain UDP `sendto`. Use a separate Markers outlet for MI cues (that's how PsychoPy/LabRecorder expect them).

## Recording format
**LabRecorder → XDF** (captures 32-ch EEG + MI markers, LSL-synced) is the BCI standard; read with **pyxdf** → convert to **MNE Raw**. Archive/interchange in **BDF** (24-bit, matches ADS1299) or **FIF** if you live in MNE.

## Recommendation (minimum effort → working tool)
1. **Write the UDP→LSL bridge** (EEG + Markers). Small, unlocks everything.
2. Instant display: run **MNE-LSL `StreamViewer`** or pylsl **`ReceiveAndPlot.py`**.
3. **~200-line PyQtGraph app** for our specifics: scrolling display + **quality-proxy panel** (per-channel RMS/variance, 50 Hz line-noise ratio, railed/flatline %) + a **marker button** for MI cues. Stimulus from **PsychoPy** with its own LSL Markers.
4. **Record with LabRecorder → XDF**; analyze via **pyxdf → MNE**.
5. Want OpenBCI-GUI polish later? Feed the *same* LSL stream into a GUI fork (study Cerelog) or invest in a BrainFlow custom board.

Ranked effort: **(c) PyQtGraph+pylsl app [recommended]** < (d) Timeflux < (b) BrainFlow custom board < (a) fork OpenBCI GUI.

## Sources
- Cerelog ESP-EEG — https://github.com/Cerelog-ESP-EEG/ESP-EEG
- OpenBCI GUI — https://github.com/OpenBCI/OpenBCI_GUI ; docs — https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/
- BrainFlow — https://github.com/brainflow-dev/brainflow ; add boards — https://brainflow.org/2022-11-01-adding-new-boards/ ; LSL bridge — https://github.com/marles77/openbci-brainflow-lsl
- MNE-LSL — https://github.com/mne-tools/mne-lsl ; muse-lsl — https://github.com/alexandrebarachant/muse-lsl ; pylsl — https://github.com/labstreaminglayer/pylsl
- Timeflux — https://github.com/timeflux/timeflux ; OpenViBE MI — https://openvibe.inria.fr/motor-imagery-bci/ ; Bonsai — https://github.com/bonsai-rx/bonsai
- XDF/MNE I/O — https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html ; PsychoPy+LSL — https://github.com/kaczmarj/psychopy-lsl

*Flags: verify BrainFlow's `lsl://` streamer syntax per installed version; OpenViBE=AGPL / Bonsai=MIT from secondary sources (check LICENSE); no stock 32-ch OpenBCI board, so any BrainFlow/GUI route needs a custom profile regardless.*
