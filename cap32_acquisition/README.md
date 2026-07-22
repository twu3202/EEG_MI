# cap32_acquisition — portable toolkit for the 32-ch ADS1299 WiFi cap

Self-contained (flat folder, no parent-project imports). Copy the whole folder anywhere.
Direct UDP acquisition + real-time filtering + recording + MI paradigm + electrode-contact
check + a full **hardware acceptance-test suite**.

> This is a **portable copy** synced from the project's `src/`. Edit in `src/` and re-run the
> sync, so the two don't diverge.

## Files
| file | what |
|---|---|
| `cap_gui.py` | **Main GUI**: 32-ch scope (CAR + filter) + live spectrum + head-map + band-power + quality + **Record** + **▶ MI Task**. |
| `mi_paradigm.py` · `mi_events.py` | Left/right(/feet/rest) MI cue paradigm + canonical event codes. |
| `impedance_gui.py` | Electrode-**contact head map** (sends `%`, shows per-electrode 31.2 Hz amplitude). |
| `impedance.py` | CLI contact probe; `--monitor` logs amp@31.2 Hz over time. |
| `udp_lsl_bridge.py` | Core: packet parser, UDP/TCP source, board commands, `board_init()`. |
| `rt_filter.py` · `synth.py` · `probe.py` | Streaming filter · synthetic source · socket debug. |
| `montage.py` | Channel order, µV scaling, 2-D scalp positions. |
| `load.py` · `artifacts.py` | Recording loader/epoching + ICA de-blink operator (used by the GUI's Calibrate). |

### Hardware acceptance tests (see `docs/hardware_acceptance.pdf` in the repo)
| file | test | needs |
|---|---|---|
| `test_injection.py` | Prove the **31.2 Hz impedance injection** exists (A/B/A) | cap |
| `impedance_formula_check.py` | Sanity-check the vendor impedance formulas vs physics | — |
| `hardware_check.py` | **Noise floor / DC-offset & rail / crosstalk / mains** (`--test`) | cap (+ signal source for crosstalk) |
| `alpha_check.py` | **Eyes-open/closed alpha** — full-chain acceptance, no equipment | cap |

Every test has a `--demo` mode (synthetic, no hardware) that validates the plots.

## Install
```bash
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
```

## Use
```bash
# acquisition
python cap_gui.py --source synth                                   # no hardware, see the UI
python cap_gui.py --source udp --host 192.168.4.1 --port 8086      # real cap
python impedance_gui.py --host 192.168.4.1 --port 8086             # contact head map

# hardware acceptance (run on the cap; each prints its setup then collects)
python test_injection.py --seconds 6                               # 31.2 Hz injection A/B/A
python hardware_check.py --test noise                              # noise floor (short inputs)
python hardware_check.py --test dc                                 # DC offset / railing
python hardware_check.py --test crosstalk --source-ch C3 --probe-hz 10
python hardware_check.py --test mains                              # 50 Hz pickup
python alpha_check.py --blocks 3 --secs 12                         # eyes open/closed alpha

# validate any script's plots without hardware:
python hardware_check.py --test noise --demo
python alpha_check.py --demo
```
Recordings save to `recordings/cap32_<ts>.npz` (+ `_raw.fif` if `mne` present). Test plots
save to `results/`.

## Board protocol (confirmed)
- Join WiFi **ESPBCI** / pw `12345678`; your machine gets **192.168.4.2**. Board = 192.168.4.1:8086.
- Commands go TO 8086; **data comes back to your host on local port 2244** (bind 2244, `sendto` 8086).
- Init: `b'b'` start → `b'1'/'2'/'3'` rate (250/500/1000) → `b'*'` EEG mode. `b'%'` = impedance, `b's'` = stop. (lowercase!)
- Frame = 105 B: `A0` · seq · 32×3 big-endian signed · 2×`00` · 4 trigger bytes · `C0`. µV = signed24 × 0.02235.
- **Impedance injection is a ~24 nA (not µA) current at 31.2 Hz** — proven by `test_injection.py`
  (see the repo's hardware report). Absolute kΩ from the vendor formula is NOT reliable; use the
  relative 31.2 Hz amplitude for contact.

---

## Migrating to Windows — what changes
**Short answer: no code changes.** Pure Python + standard sockets + Qt. Checklist:
1. **Python + deps**: Python 3.11 → `pip install -r requirements.txt` (all have Windows wheels).
2. **Windows Firewall — the one real gotcha.** On first run, allow Python on **Private** networks,
   else inbound UDP:2244 is blocked → **no data**. Manual rule (Admin):
   ```
   netsh advfirewall firewall add rule name="cap32" dir=in action=allow protocol=UDP localport=2244
   ```
3. **WiFi**: connect to `ESPBCI`; confirm `ipconfig` shows **192.168.4.2**; `ping 192.168.4.1`.
4. **Close the vendor `main_ui.exe` first** — only one app can bind local port 2244.
5. `--screenshot` sets `QT_QPA_PLATFORM=offscreen` (headless); not needed for the live GUI.
