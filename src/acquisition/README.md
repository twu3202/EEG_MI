# Acquisition — cap → LSL bridge + simple 32-ch viewer

Backup in-house acquisition path (see `research/Cerelog_adaptation_notes.md` and
`research/OpenBCI_GUI_software_report.md`). Everything is LSL-based, so LabRecorder,
MNE-LSL, Timeflux, or a forked OpenBCI GUI can also consume the same stream.

```
[cap] --UDP/TCP--> udp_lsl_bridge.py --LSL 'Cap32'(EEG,32ch) + 'Cap32_Markers'--> viewer.py / LabRecorder
```

## Files
| file | what |
|---|---|
| **`cap_gui.py`** | **Main acquisition GUI** (direct UDP/TCP): control bar, 32-ch scope (CAR'd + filtered), **live spectrum**, quality panel, **Record**. |
| `impedance_gui.py` | Impedance **head-map** UI: sends `%`, shows per-electrode kΩ; falls back to passive contact proxy if the board doesn't inject; restores `*` on exit. |
| `impedance.py` | CLI impedance probe (sends `%`, prints per-ch kΩ + amp@31.2Hz, restores `*`). |
| `probe.py` | Socket-level probe (local IPs, init sequence, reframe, per-ch µV + common-mode test). |
| `rt_filter.py` | `RealTimeEEGFilter` — vendor-style Butterworth band/low-pass + 50 Hz iir-notch + baseline removal, stateful `lfilter` for live chunks. |
| `synth.py` | Synthetic 32-ch EEG so the GUI/pipeline run with no hardware. |
| `udp_lsl_bridge.py` | Alternative LSL path: cap → LSL 'Cap32' (for LabRecorder/MNE-LSL/Timeflux interop). |
| `viewer.py` | The LSL-based viewer (pairs with the bridge). |

Two acquisition paths: **`cap_gui.py`** (self-contained, vendor-style — the main one now) or the
**LSL bridge + viewer** (when you want LabRecorder→XDF or other LSL tools). Both share
`common/montage.py` + the `udp_lsl_bridge` parser.

## Use
```bash
conda activate eegmi

# preview the GUI with synthetic data (no hardware) — renders results/cap_gui_preview.png
python src/acquisition/cap_gui.py --screenshot results/cap_gui_preview.png
python src/acquisition/cap_gui.py --source synth                       # live, synthetic
python src/acquisition/cap_gui.py --source udp --host 192.168.4.1 --port 8086   # real cap (UDP)
python src/acquisition/cap_gui.py --source tcp --host 192.168.4.1 --port <tcp_port>  # loss-free

# alternative LSL path (LabRecorder→XDF / Timeflux):
python src/acquisition/udp_lsl_bridge.py --transport udp --host 192.168.4.1 --port 8086
python src/acquisition/viewer.py --source lsl
python src/acquisition/udp_lsl_bridge.py --dry-run                     # parse self-test
```

## Filtering (mirrors the vendor `RealTimeEEGFilter`)
`rt_filter.py`: Butterworth order-4 **band-pass** (lowcut>0) or **low-pass** (lowcut=0) +
optional **50 Hz `iirnotch`** + **baseline/drift removal** (running-mean subtraction),
all via stateful `lfilter` (`zi`) for live streaming. GUI defaults 0–40 Hz + notch on; set
`low=3` for a **3–40 Hz band-pass**. The vendor app uses the exact same design (butter +
iirnotch + running-mean).

## Status (verified on the M5)
- Bridge parse self-test ✅ (`±1000 counts → ±22.35 µV`, sign + trigger correct).
- Viewer renders ✅ (`results/cap_gui_preview.png`).
- LSL two-process data flow ✅ (producer → consumer, 32 ch @ 250 Hz).

## Protocol — CONFIRMED from the vendor app (`fromprovider/main_ui`)
Reverse-read from their PyInstaller build (`utils.parse_24bit_signed`,
`Controller.change_channels_nums`, `utils.UDPReceiver/TCPReceiver`):
- **`FRAME_LEN = n_ch*3 + 9`** → 32 ch = **105 B**. `[0]=0xA0`, `[1]=seq#(0-255)`,
  `[2 : 2+3·n_ch]` = n_ch × **3-byte big-endian signed** samples, then 6 reserved/trigger
  bytes, `[FRAME_LEN-1]=0xC0`.
- **µV = signed24 × 0.02235** (`if v&0x800000: v-=16777216`) — matches our scaling exactly.
- The bridge's `PacketLayout` + `parse_24bit_signed` now mirror this (dry-run verified).

### Board commands (CONFIRMED from vendor `utils.py`/`Controller.py` — LOWERCASE!)
| byte | meaning |
|---|---|
| `b'b'` | start streaming (manual's "B" is WRONG) |
| `b's'` | stop |
| `b'1'`/`b'2'`/`b'3'` | sample rate 250 / 500 / 1000 Hz |
| **`b'*'`** | **EEG-wave mode (normal acquisition)** |
| `b'%'` | impedance mode (injects AC current) |

**Init order that avoids the impedance-mode comb artifact:** `'*'` (EEG mode) → rate → `'b'`
(start). Our `board_init()` does exactly this; the UDP socket also mirrors the vendor
(bind local port **2244**, connect to the board). If you connect and only send `b` (or
uppercase `B`) without `'*'`, the board stays injecting → every channel shows a regular
waveform with an evenly-spaced harmonic comb in the FFT. DC offset is removed in *software*
(running-mean + optional high-pass in `rt_filter.py`), not on the board.

Two things their code confirms:
1. **TCP is already supported** — their `TCPReceiver` uses `TCP_NODELAY`. So the
   loss-free path is available: run our bridge with `--transport tcp` once you know the
   TCP port (ask the vendor; their UDP port is 8086). No firmware change may even be needed.
2. **Per-frame sequence counter** (byte 1) → our bridge now reports **dropped-frame %**
   live and emits `lost/N` markers (same idea as their `check_sequence_continuity`).

Still worth asking the vendor: the exact **sampling rate** (default 250, options 500/1000),
the **TCP port**, whether the 6 reserved bytes carry a real **trigger** (their UDP build
passes trigger=0), and whether the **impedance mode** (`change2Imp`/`ImpedanceMode` +
`WidgetImpedancePlot` exist in their app) actually returns valid readings on this cap.
