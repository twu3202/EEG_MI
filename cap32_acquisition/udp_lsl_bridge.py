# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""WiFi -> LSL bridge for the 32-ch ADS1299 cap (backup acquisition path).

Reads the cap's stream, reframes its `0xA0 … 0xC0` packets, converts counts to µV,
and republishes as two LSL streams so the whole LSL ecosystem (MNE-LSL viewer,
LabRecorder -> XDF, Timeflux, a forked OpenBCI GUI, our own viewer) works:
  - "Cap32"          : EEG, 32 ch, float32 µV
  - "Cap32_Markers"  : Markers, 1 string ch (MI cues from TX / the experiment)

Transport: the vendor firmware is UDP-server now; TCP is preferred once they add it
(Cerelog proves the same ESP32 class does TCP cleanly — see
research/Cerelog_adaptation_notes.md). Both are supported here via --transport.

  python udp_lsl_bridge.py --transport udp --host 192.168.4.1 --port 8086 --sfreq 250
  python udp_lsl_bridge.py --transport tcp --host 192.168.4.1 --port 1112 --sfreq 250
  python udp_lsl_bridge.py --dry-run          # parse self-test, no hardware/LSL needed

NOTE: the exact 105-byte field layout below is a best guess from the vendor manual
(0xA0 start, 0xC0 end, 32×3-byte signed big-endian samples, one trigger column before
0xC0, ×0.02235 µV). VERIFY the header/offset constants against the vendor's final
"105-byte" spec and adjust PACKET_LAYOUT — the framing/LSL code stays the same.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from dataclasses import dataclass

import numpy as np

# --- montage / scaling from our project ---
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from montage import CAP32_CHANNELS, ADC_MICROVOLTS_PER_COUNT  # noqa: E402


@dataclass
class PacketLayout:
    """Frame layout — CONFIRMED from the vendor app (main_ui / utils.parse_24bit_signed,
    Controller.change_channels_nums):  FRAME_LEN = n_ch*3 + 9.
    Byte map, CONFIRMED by the vendor manual `使用手册必读.docx` (32 ch):
      byte[0]              = 0xA0 header
      byte[1]              = sequence counter (00-FF, wraps) — for loss detection
      byte[2 : 98]         = 32 × 3-byte BIG-ENDIAN signed samples (b0=MSB)  (96 B)
      byte[98 : 100]       = 2 reserved bytes (0x00 0x00)
      byte[100 : 104]      = 4 trigger bytes (0 when no trigger; TXXXX sets them)
      byte[104]            = 0xC0 footer
    """
    n_ch: int = 32
    bytes_per_ch: int = 3
    start: int = 0xA0
    end: int = 0xC0
    seq_index: int = 1
    ch_offset: int = 2
    reserved_bytes: int = 2
    trigger_bytes: int = 4
    overhead: int = 9               # A0(1)+seq(1)+reserved(2)+trigger(4)+C0(1)

    @property
    def size(self) -> int:
        return self.n_ch * self.bytes_per_ch + self.overhead  # 32 -> 105

    @property
    def trigger_offset(self) -> int:
        return self.ch_offset + self.n_ch * self.bytes_per_ch + self.reserved_bytes  # 100


LAYOUT = PacketLayout()


def counts_to_uv(raw: int) -> float:
    """3-byte signed ADS1299 count -> µV (gain 24). Vendor's parse_24bit_signed."""
    if raw & 0x800000:                       # 0x800000 = 8388608, sign bit
        raw -= 16777216                      # 2**24, two's complement
    return raw * ADC_MICROVOLTS_PER_COUNT    # ×0.02235 µV/count


def parse_packet(pkt: bytes, layout: PacketLayout = LAYOUT):
    """Validate one frame -> (samples_uv[n_ch], seq, trigger). None if invalid."""
    if len(pkt) != layout.size or pkt[0] != layout.start or pkt[-1] != layout.end:
        return None
    out = np.empty(layout.n_ch, dtype=np.float32)
    o = layout.ch_offset
    for ch in range(layout.n_ch):
        i = o + ch * layout.bytes_per_ch
        raw = int.from_bytes(pkt[i : i + layout.bytes_per_ch], "big", signed=False)
        out[ch] = counts_to_uv(raw)
    seq = pkt[layout.seq_index]
    t = layout.trigger_offset
    trigger = int.from_bytes(pkt[t : t + layout.trigger_bytes], "big")  # 4-byte trigger
    return out, seq, trigger


# ------------------------------------------------------------- board commands
# CONFIRMED from vendor utils.py / Controller.py (lowercase!). The manual's "B/S"
# is wrong. Sending EEG_MODE is REQUIRED — otherwise the board stays in impedance
# mode (AC current injection) and every channel shows a periodic comb-spectrum artifact.
START = b"b"
STOP = b"s"
EEG_MODE = b"*"          # normal EEG acquisition  (Controller.EEGWaveMode)
IMPEDANCE_MODE = b"%"    # impedance measurement   (Controller.ImpedanceMode)
RATE_CMD = {250: b"1", 500: b"2", 1000: b"3"}


def board_init(src, sfreq: float) -> None:
    """Vendor init sequence, in the exact order the app does it
    (utils._connect + Controller.connect + main_ui.apply_mode_for_current_view):
        [socket already connected] -> wait -> START -> rate -> EEG_MODE
    The trailing EEG_MODE ('*') is what pulls the board out of impedance/injection
    mode; without it every channel shows a periodic comb-spectrum artifact."""
    time.sleep(0.5)                              # vendor waits 0.5 s after connect
    src.send(START)                              # b'b'  start streaming
    time.sleep(0.2)
    src.send(RATE_CMD.get(int(sfreq), b"1"))     # b'1'/'2'/'3'  sample rate
    time.sleep(0.1)
    src.send(EEG_MODE)                           # b'*'  EEG-wave mode (leave impedance)
    time.sleep(0.1)


def drain(src, seconds: float = 0.4) -> int:
    """Discard whatever is already sitting in the socket buffer, then let the caller start
    counting from a clean slate.

    `board_init` sends b -> rate -> '*' with sleeps in between and the board RESTARTS its
    stream on each mode switch, so the frames buffered across that transition carry a
    discontinuous sequence counter. Reading them makes the very first second look like a
    ~200-frame burst of "loss" that says nothing about link quality (measured on the real
    cap: 200 lost in second 1, then 0 for the next 119 s). Call this once after board_init
    and reset your `last_seq` to None."""
    import time as _t
    n = 0
    old = src.sock.gettimeout()
    src.sock.settimeout(0.05)
    end = _t.time() + seconds
    try:
        while _t.time() < end:
            try:
                src.sock.recv(65536)
                n += 1
            except OSError:
                break
    finally:
        src.sock.settimeout(old)
        try:
            src.buf.clear()          # drop any half-assembled frame too
        except AttributeError:
            pass
    return n


# --------------------------------------------------------------------------- IO
def _reframe(read, buf: bytearray, layout: PacketLayout):
    """Pull complete `0xA0 … 0xC0` frames out of a growing byte stream.
    `read()` returns the next chunk (b'' -> stop). Works for BOTH transports: the
    board streams MTU-sized chunks (≈1472 B) that don't align to frame boundaries,
    so we buffer and reframe by header + length + footer (like the vendor does)."""
    start = bytes([layout.start])
    while True:
        chunk = read()
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            s = buf.find(start)
            if s == -1:                          # no header yet -> wait for more
                buf.clear()
                break
            if len(buf) - s < layout.size:       # header seen but frame incomplete
                if s:
                    del buf[:s]                  # drop junk before the header
                break
            if buf[s + layout.size - 1] == layout.end:
                yield bytes(buf[s : s + layout.size])
                del buf[: s + layout.size]
            else:
                del buf[: s + 1]                 # false header, resync one byte


class UdpSource:
    """Commands go TO the board (192.168.4.1:8086); DATA comes back to this host on
    local port 2244. We bind 2244 to receive and `sendto` the board for commands.
    We do NOT `connect()` the socket, so recv accepts the board's data whatever
    source port it uses. (Your Mac must actually hold IP 192.168.4.2 on the cap's AP,
    else the board's packets addressed to .2 never reach it.)"""

    def __init__(self, host: str, port: int, local_port: int = 2244):
        self.host, self.cmd_port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        self.sock.bind(("0.0.0.0", local_port))  # board streams data here (2244)
        self.buf = bytearray()

    def send(self, data: bytes) -> None:
        self.sock.sendto(data, (self.host, self.cmd_port))   # command -> board:8086

    def frames(self, layout: PacketLayout = LAYOUT):
        # datagrams are MTU-sized stream chunks, NOT one-frame-each -> reframe
        yield from _reframe(lambda: self.sock.recvfrom(65536)[0], self.buf, layout)


class TcpSource:
    """Stream socket + buffer reframing (Cerelog-style; preferred, loss-free)."""

    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # low latency
        self.buf = bytearray()

    def send(self, data: bytes) -> None:
        self.sock.send(data)

    def frames(self, layout: PacketLayout = LAYOUT):
        yield from _reframe(lambda: self.sock.recv(4096), self.buf, layout)


# --------------------------------------------------------------------------- main
def make_outlets(sfreq: float):
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo("Cap32", "EEG", len(CAP32_CHANNELS), sfreq, "float32", "cap32-ads1299")
    chns = info.desc().append_child("channels")
    for lbl in CAP32_CHANNELS:
        c = chns.append_child("channel")
        c.append_child_value("label", lbl)
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EEG")
    eeg = StreamOutlet(info, chunk_size=1, max_buffered=360)
    mrk = StreamOutlet(StreamInfo("Cap32_Markers", "Markers", 1, 0, "string", "cap32-mrk"))
    return eeg, mrk


def run(args) -> None:
    from pylsl import local_clock

    eeg, mrk = make_outlets(args.sfreq)
    src = UdpSource(args.host, args.port) if args.transport == "udp" else TcpSource(args.host, args.port)
    if not args.no_start:
        board_init(src, args.sfreq)   # EEG_MODE ('*') -> rate -> START ('b')

    print(f">>> bridging {args.transport}://{args.host}:{args.port} -> LSL 'Cap32' ({args.sfreq} Hz)")
    last_seq = None
    lost = n = 0
    try:
        for pkt in src.frames():
            parsed = parse_packet(pkt)
            if parsed is None:
                continue
            sample, seq, trigger = parsed
            eeg.push_sample(sample, local_clock())
            if trigger:                                    # hardware trigger (TXXXX)
                mrk.push_sample([f"trig/{trigger}"])
            if last_seq is not None:                       # drop detection via seq counter
                gap = (seq - last_seq - 1) % 256
                if gap:
                    lost += gap
                    mrk.push_sample([f"lost/{gap}"])
            last_seq = seq
            n += 1
            if n % (int(args.sfreq) * 5) == 0:
                print(f"  {n} samples | dropped {lost} ({100*lost/(n+lost):.2f}%)")
    except KeyboardInterrupt:
        print(f"\nstopping… total dropped {lost}/{n+lost}")
        try:
            src.send(STOP)
        except OSError:
            pass


def dry_run() -> None:
    """Parse a synthetic frame end-to-end — no hardware/LSL needed."""
    pkt = bytearray(LAYOUT.size)
    pkt[0] = LAYOUT.start
    pkt[-1] = LAYOUT.end
    pkt[LAYOUT.seq_index] = 42
    o = LAYOUT.ch_offset
    pkt[o : o + 3] = (1000).to_bytes(3, "big")               # ch0 = +1000 counts
    pkt[o + 3 : o + 6] = ((-1000) & 0xFFFFFF).to_bytes(3, "big")  # ch1 = -1000 (two's comp)
    to = LAYOUT.trigger_offset
    pkt[to : to + 4] = (7).to_bytes(4, "big")               # trigger = 7
    sample, seq, trigger = parse_packet(bytes(pkt))
    print(f"dry-run OK: size={LAYOUT.size}B, {len(sample)} ch, seq={seq}, trigger={trigger}, "
          f"ch0={sample[0]:.2f}µV ch1={sample[1]:.2f}µV (expect ±{1000*ADC_MICROVOLTS_PER_COUNT:.2f})")
    assert seq == 42 and trigger == 7
    assert abs(sample[0] - 1000 * ADC_MICROVOLTS_PER_COUNT) < 1e-3
    assert abs(sample[1] + 1000 * ADC_MICROVOLTS_PER_COUNT) < 1e-3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", default="udp", choices=["udp", "tcp"])
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--no-start", action="store_true", help="don't send 'B' on connect")
    ap.add_argument("--dry-run", action="store_true", help="parse self-test, no hw/LSL")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run(args)


if __name__ == "__main__":
    main()
