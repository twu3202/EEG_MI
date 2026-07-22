# cap32_acquisition — portable toolkit for the 32-ch ADS1299 WiFi cap

Self-contained (flat folder, no parent-project imports). Copy the whole folder anywhere.
Direct UDP acquisition + real-time filtering + recording + electrode-contact check.

## Files
| file | what |
|---|---|
| `cap_gui.py` | **Main GUI**: 32-ch scope (CAR + filter) + live spectrum + quality panel + **Record**. |
| `impedance_gui.py` | Electrode-**contact head map** (sends `%`, shows per-electrode contact %, restores `*`). |
| `impedance.py` | CLI contact probe; `--monitor` logs amp@31.2Hz + amp@50Hz over time. |
| `probe.py` | Socket-level probe (local IPs, init sequence, reframe, per-ch µV, common-mode test). |
| `udp_lsl_bridge.py` | Core: packet parser, UDP/TCP source, board commands, `board_init()`. (Also an optional LSL bridge — needs `pylsl`.) |
| `rt_filter.py` | Streaming Butterworth band/low-pass + 50 Hz notch + baseline removal. |
| `synth.py` | Synthetic 32-ch source (no hardware) for testing/preview. |
| `montage.py` | Channel order, µV scaling, and 2-D scalp positions. |

## Install
```bash
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
```

## Use
```bash
python cap_gui.py --source synth                                   # no hardware, see the UI
python cap_gui.py --source udp --host 192.168.4.1 --port 8086      # real cap
python impedance_gui.py --host 192.168.4.1 --port 8086             # contact head map
python impedance.py --monitor --seconds 40                         # log contact over time
python probe.py                                                    # debug "no data"
python udp_lsl_bridge.py --dry-run                                 # parser self-test
```
Recordings save to `recordings/cap32_<ts>.npz` (+ `_raw.fif` if `mne` installed) with the
per-sample trigger; markers = the 4-byte hardware trigger (`TXXXX`).

## Board protocol (confirmed)
- Join WiFi **ESPBCI** / pw `12345678`; your machine gets **192.168.4.2**. Board = 192.168.4.1:8086.
- Commands go TO 8086; **data comes back to your host on local port 2244** (bind 2244, `sendto` 8086).
- Init: `b'b'` start → `b'1'/'2'/'3'` rate (250/500/1000) → `b'*'` EEG mode. `b'%'` = impedance, `b's'` = stop. (lowercase!)
- Frame = 105 B: `A0` · seq · 32×3 big-endian signed · 2×`00` · 4 trigger bytes · `C0`. µV = signed24 × 0.02235.
- Board streams MTU-sized datagrams (~1472 B) = a byte STREAM; reframe by `A0…C0` (done in `_reframe`).

---

## Migrating to Windows — what changes

**Short answer: no code changes.** It's pure Python + standard sockets + Qt; everything runs
on Windows as-is (fonts already include Segoe UI, paths use `os.path`/`pathlib`). The board's
*own* software is Windows, so board↔Windows is proven. Checklist:

1. **Python + deps**: install Python 3.11 (python.org or Miniconda) → `pip install -r requirements.txt`.
   PyQt6 / pyqtgraph / numpy / scipy all have Windows wheels. (`pip install pywin32` not needed.)
2. **Windows Firewall — the one real gotcha.** On first run Windows will prompt "allow Python to
   communicate on networks" → **Allow** (tick *Private*). Otherwise inbound UDP on port 2244 is
   blocked and you'll see **no data**. To add a rule manually (Admin cmd):
   ```
   netsh advfirewall firewall add rule name="cap32" dir=in action=allow protocol=UDP localport=2244
   ```
3. **WiFi**: connect to `ESPBCI` in Windows Wi-Fi settings. Confirm your IP with `ipconfig` — the
   Wi-Fi adapter should show **192.168.4.2** (the board sends data to .2; if you got .3/.4, another
   client took .2 — reconnect as the only client). Test with `ping 192.168.4.1`.
4. **Close the vendor `main_ui.exe` first.** Only one app can receive the board stream (both bind
   local port 2244) — run either theirs or this, not both.
5. **Run the same commands** (`python cap_gui.py --source udp …`) from an Anaconda Prompt / venv.
6. Cosmetic only: the `--screenshot` mode sets `QT_QPA_PLATFORM=offscreen` (headless) — on a normal
   Windows desktop you don't need it for the live GUI.

Nothing in the protocol, parsing, filtering, or GUI is platform-specific.

> Note: this folder is a **portable copy** of the project's `src/acquisition/` + `src/common/montage.py`.
> Edit in one place to avoid divergence.
