# Cerelog ESP-EEG — what we downloaded & how to adapt it to our 32-ch cap

Cloned to `third_party/Cerelog/` (gitignored). Cerelog is an ESP32 + ADS1299 WiFi
biosensing board — the same hardware *class* as our cap, so its software is a direct
template. (It's **8-channel**; ours is 32-ch — the difference is just channel count and
packet layout in the parser.)

## What's in the download
| Repo | Size | What it is / why it matters |
|---|---|---|
| `WiFi_Support/` | 184K | **The key one.** `Python_wifi_LSL.py` (WiFi→LSL bridge) + `esp_hostfw.ino` (ESP32 AP-mode firmware). |
| `Lab-Stream-Layer-LSL-Compatability/` | 172K | `cerelog_lsl.py` — the USB/serial→LSL bridge variant. |
| `How-to-use-OpenBCI-GUI-fork/` | 128K | Instructions to feed the LSL stream into their OpenBCI-GUI fork. |
| `OpenBCI_Fork_New/` | 607M | Full **forked OpenBCI GUI** (Processing) that reads the LSL stream. The "borrow the GUI" asset. |
| `ESP-EEG/` | 57M | Main board repo: `firmware/`, `hardware/`, case, instructions. |
| `Troubleshooting_connection/`, `Robotic_Hand_EMG_.../` | small | Connection help; an EMG demo app. |

## The transport finding (answers "can we avoid UDP?")
**Yes — Cerelog streams EEG data over TCP, not UDP.** From `esp_hostfw.ino`:
- `WiFi.softAP("CERELOG_EEG")` → device hosts its own WiFi AP (same idea as our cap's `ESPBCI` hotspot).
- **TCP server on port 1112** for the data stream, with `client.setNoDelay(true)` (Nagle off → low latency). TCP = reliable, ordered, no dropped samples.
- **UDP port 4445 only for discovery** (broadcast `CERELOG_FIND_ME` → reply `CERELOG_HERE` to auto-find the device IP). No sample data over UDP.

The bridge (`Python_wifi_LSL.py`) mirrors this: a `TcpSerial` client (`SOCK_STREAM` + `TCP_NODELAY`), buffer-based framing (find start marker → check length/end-marker/checksum → parse), then `pylsl` `StreamOutlet.push_sample`.

**Implication for us:** our vendor's current firmware is **UDP-server** (per the manual). Cerelog proves the *same ESP32 class* can do a **TCP server** cleanly — so the strongest fix for UDP loss is to ask the vendor to add a TCP streaming mode and point them at `esp_hostfw.ino` as a working reference. Then our bridge is a TCP client → LSL.

## How Cerelog maps to our cap
| Aspect | Cerelog | Our cap |
|---|---|---|
| MCU / ADC | ESP32 + ADS1299 | ESP-based + ADS1299 |
| AP | `CERELOG_EEG` / `cerelog123` | `ESPBCI` / `12345678` |
| Data transport | **TCP** :1112 | **UDP** :8086 (currently) |
| Discovery | UDP :4445 broadcast | none (fixed IP 192.168.4.1) |
| Channels | 8 | **32** |
| Packet | 37 B, `0xABCD…0xDCBA`, checksum | 105 B, `0xA0…0xC0` |
| µV scale | `(2·VREF/GAIN)/2^24 · 1e6 · corr` | `count × 0.02235` (gain 24) |

## Two paths to a working bridge
**Path A — preferred (loss-free): ask vendor for a TCP mode.** Reference them to
`third_party/Cerelog/WiFi_Support/(Works ) WiFI Firmware  (Device Host)/esp_hostfw.ino`.
Then adapt `Python_wifi_LSL.py`: TCP client to 192.168.4.1:<tcp_port>, reuse the
buffer/framing loop, swap the parser for our 105-B / `0xA0…0xC0` / 32-ch format and the
`×0.02235 µV` scaling. Keep the `B`/`S`/`TX` commands as TCP sends.

**Path B — works with current UDP firmware.** Same bridge, but read from a UDP socket
bound to :8086 instead of the TCP client. De-risk UDP (see the GUI report): large
`SO_RCVBUF`, dedicated AP, and a per-packet sequence counter (ask vendor) so the quality
panel can flag drops. LSL downstream is still TCP/reliable.

## Reuse checklist for our `udp_lsl_bridge.py` (to build next)
- [ ] Socket: TCP client (Path A) or UDP bind :8086 (Path B).
- [ ] Framing: find `0xA0`, take 105 B, verify trailing `0xC0` (+ any checksum), else resync by 1 byte — copy Cerelog's buffer loop.
- [ ] Parse 32 × 3-byte **signed** big-endian → `count × 0.02235` µV (see `src/common/montage.py::ADC_MICROVOLTS_PER_COUNT`). Handle the leading trigger column when present.
- [ ] LSL: `StreamInfo('Cap32','EEG',32,sfreq,'float32',...)` with per-channel labels from `CAP32_CHANNELS`; separate `'Markers'` outlet for MI cues (from `TX` / stimulus).
- [ ] Optional: DC-blocker (Cerelog uses `y = x − x_prev + R·y_prev`, R=0.995) — but prefer doing filtering in MNE downstream.
