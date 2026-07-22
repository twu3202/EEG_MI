# OpenBCI for a 32-Channel Motor-Imagery BCI — Research Report (2025–2026)

*Comparison against our custom 32-ch ADS1299 dry-electrode WiFi cap. Prices USD, mid-2026 OpenBCI shop; uncertain figures flagged.*

## 1. Hardware paths to 32 channels

| Board | Ch | ADC | Stream rate | Res | Wireless | Price | Stock (mid-2026) |
|---|---|---|---|---|---|---|---|
| Ganglion | 4 | MCP3912 (not ADS1299) | 200 Hz | 24-bit | Simblee BLE | $624.99 | Sold out |
| Cyton | 8 | ADS1299 | **250 Hz** | 24-bit | RFduino BLE | $1,249 | Sold out |
| Cyton + Daisy | 16 | 2× ADS1299 | **125 Hz effective over BLE** | 24-bit | RFduino BLE | $2,499 | In stock |

**Sampling-rate caveat:** with Daisy, streamed rate over the stock BLE dongle drops to **125 Hz** (radio bandwidth split across 16 ch). Internal sampling is still 250 Hz → **SD-card recording is 250 Hz**; the (deprecated) WiFi Shield also restores 250 Hz+.

**Reaching 32 ch — officially NOT supported (a hack):**
- Cannot chain two Cyton boards into one 32-ch stream (ADS1299 clock routing wired only for mainboard + Daisy; OpenBCI staff confirm).
- Realistic route: **two independent Cyton+Daisy boards (16+16)**, each with its own USB dongle (>1 m apart), **merged in software** (LSL / custom BrainFlow).
- **Clocks can't be hardware-synced** — two non-crystal clocks; LSL timestamp alignment leaves residual jitter/drift. "Really close" but not perfectly synced. Tolerable for MI band-power/CSP; tight cross-hemisphere phase not guaranteed.
- **WiFi Shield deprecated / out of production** (packet loss, cyclical noise; one board per shield — does not combine boards). A "Cyton V2 WiFi" is rumored, no timeline.

**Newer 2024–2026 products:** active dev is **Galea** (VR biosensing headset, ~$42,980; only ~8–10 EEG ch @250 Hz + EMG/EOG/EDA/PPG/eye-tracking) and Galea+Pupil-Labs Neon (Oct 2025). **No OpenBCI board raises the EEG ceiling above 16 as of mid-2026.**

## 2. Headset — Ultracortex Mark IV
- Dry passive Ag/AgCl spiky+flat electrodes; configurable wet/gel or active/passive.
- **8 or 16 channels from 35 possible 10-20 nodes**. **Cannot host 32** — 32-ch needs two headsets (unwearable together) or a custom cap → a real gap vs our cap.
- Default 8-ch: Fp1,Fp2,**C3,C4**,P7,P8,O1,O2. 16-ch adds F7,F8,F3,F4,T7,T8,P3,P4.
- Dry: <1 min setup, but higher impedance / more motion & line artifact / lower SNR — a real concern for small MI mu/beta ERD. Most high-accuracy MI uses gel.

## 3. Software stack (OpenBCI's real strength)
- **BrainFlow** — unified SDK; bindings for Python/C++/Java/C#/Julia/MATLAB/R/Node/Rust; returns **NumPy** → MNE `RawArray`; built-in filtering. Recommended path into Python.
- **OpenBCI GUI** — viz/recording + Networking widget (LSL/UDP/OSC/serial).
- **LSL** — de-facto multi-stream time-sync; how you'd merge two boards + sync markers.
- **Timeflux** — real-time streaming pipelines for online BCI.
- Python MI stack (BrainFlow→NumPy→MNE): **MOABB, pyRiemann, braindecode** — same ecosystem regardless of amplifier.

## 4. Approximate cost (USD)
- **16-ch turnkey:** Complete Ultracortex (16-ch dry, incl. Cyton+Daisy) **$2,999**.
- **32-ch hack:** 2× Cyton+Daisy ≈ **$4,998** boards only + a 32-electrode cap (Ultracortex maxes at 16 → 2 helmets $5,998 unwearable together, or custom cap). Realistic all-in **~$5,000–6,500+**, still unsynced dual-board, 125 Hz/board BLE.

## 5. Motor imagery with OpenBCI
- **Official MI tutorial is DEPRECATED** ("no further support"). Its recipe: C3,Cz,C4,P3,Pz,P4,O1,O2,FPz; **CSP + logistic regression**, 2-class, 20–50 trials/class.
- C3=ch3, C4=ch4 on Ultracortex default — canonical motor triplet supported.
- Community MI repos (real): `ZackGoldblum/BCI-Motor-Imagery` (real-time, ShallowConvNet), `vasanza/BCI_Motor_Imagery_Task_OpenBCI`, `mvijay97/Motor-Imagery-BCI` (4-class CSP).
- Realistic accuracy: 2-class left/right **~60–80% within-session**, ~54% cross-session without adaptation; dry → lower end. Not the >90% of curated gel-cap datasets.

## 6. Bottom line — OpenBCI 32-ch vs our custom ADS1299 WiFi dry cap

| Dimension | OpenBCI "32-ch" (2× Cyton+Daisy) | Our custom 32-ch ADS1299 cap |
|---|---|---|
| Channel ceiling | 16/board; 32 only via software merge (hack) | Native 32 in one device |
| Sampling rate | 125 Hz/board over BLE (250 via SD/deprecated WiFi) | ADS1299 up to 16 kSPS; WiFi 250–1000 Hz easily |
| 32-ch clock sync | **Not synchronized** (two clocks, LSL jitter) | **Single clock domain** |
| Amplifier | ADS1299 24-bit | ADS1299 24-bit (parity) |
| 32-ch headset | **None exists** (Ultracortex ≤16) | Already owned |
| Software maturity | **Strong** (BrainFlow/GUI/LSL + MNE stack) | Must build driver / expose LSL/BrainFlow |
| Cost | ~$5,000–6,500+ | Already owned |
| Community | **Large, active** | On our own for HW/firmware |

**Verdict:** OpenBCI's advantages are **software maturity + community**, not hardware. Its 32-ch story is weak (no native 32-ch board or headset, unsynced 2-board hack, 125 Hz BLE ceiling, deprecated MI tutorial/WiFi Shield). **Our custom cap is technically superior for 32-ch MI** (single synchronized clock, native 32 ch, higher/flexible sampling, already owned). Its only real deficit is software/driver maturity — **fully recoverable by exposing the cap's stream via BrainFlow or LSL**, after which the whole OpenBCI/NeuroTechX Python MI stack (MNE, MOABB, pyRiemann, braindecode, timeflux) works identically.

**Actionable takeaway for us:** don't buy OpenBCI for channels; **borrow its software pattern** — wrap the cap's UDP stream as an **LSL** (or BrainFlow) source so every tool "just works", and reuse the community MI repos as references.

**Uncertainties:** Galea EEG ch count (~8–10, config-dependent); Cyton/Ganglion "sold out" mid-2026; "Cyton V2 WiFi" rumored, no specs/date; MI accuracy is subject/algorithm-dependent (60–80% field norm).

## Sources
- Cyton specs — https://docs.openbci.com/Cyton/CytonSpecs/
- Cyton+Daisy (price, 125 Hz) — https://shop.openbci.com/products/cyton-daisy-biosensing-boards-16-channel
- 32-ch not supported — https://openbci.com/forum/index.php?p=/discussion/1850/synchronizing-two-cyton-daisy-boards-in-order-to-get-32-channels
- Two boards via LSL — https://openbci.com/forum/index.php?p=/discussion/364/run-2-openbci-boards-simultaneously-to-get-16-or-32-channels
- WiFi Shield deprecated — https://docs.openbci.com/Deprecated/WiFiShield/WiFiGS/
- Ultracortex Mark IV — https://docs.openbci.com/AddOns/Headwear/MarkIV/
- Complete Ultracortex ($2,999) — https://shop.openbci.com/products/the-complete-headset-eeg
- Galea — https://shop.openbci.com/products/galea
- BrainFlow — https://brainflow.org/ ; OpenBCI↔MNE — https://github.com/openbci-archive/OpenBCI_MNE
- LSL — https://docs.openbci.com/Software/CompatibleThirdPartySoftware/LSL/ ; BrainFlow-LSL bridge — https://github.com/marles77/openbci-brainflow-lsl
- Timeflux — https://timeflux.io
- OpenBCI MI tutorial (deprecated) — https://docs.openbci.com/Deprecated/MotorImagery/
- MI repos — https://github.com/ZackGoldblum/BCI-Motor-Imagery ; https://github.com/vasanza/BCI_Motor_Imagery_Task_OpenBCI ; https://github.com/mvijay97/Motor-Imagery-BCI
- awesome-bci — https://github.com/NeuroTechX/awesome-bci
