# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Socket-level probe — debug 'no data' before touching the GUI.

Binds the local receive port (2244), sends the init sequence (b -> rate -> *) to the
board's command endpoint (192.168.4.1:8086), and prints exactly what comes back and
from where. Shows your local IPs so you can confirm the Mac holds 192.168.4.2.

  python src/acquisition/probe.py
  python src/acquisition/probe.py --start B     # try UPPERCASE start byte
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from montage import ADC_MICROVOLTS_PER_COUNT  # noqa: E402

RATE = {250: b"1", 500: b"2", 1000: b"3"}


def local_ips():
    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    # also probe the route toward the board
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.4.1", 8086)); ips.add(s.getsockname()[0]); s.close()
    except OSError:
        pass
    return sorted(ips)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--cmd-port", type=int, default=8086)
    ap.add_argument("--local-port", type=int, default=2244)
    ap.add_argument("--sfreq", type=int, default=250)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--start", default="b", help="start byte (default lowercase 'b'; try 'B')")
    args = ap.parse_args()

    ips = local_ips()
    print(f"local IPs: {ips}")
    if "192.168.4.2" not in ips:
        print("  ⚠ none of your IPs is 192.168.4.2 — the board sends data to .2, so it may")
        print("    not reach you. Join WiFi 'ESPBCI' and check `ifconfig | grep 192.168.4`.")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass
    s.bind(("0.0.0.0", args.local_port))
    print(f"bound local :{args.local_port}, commands -> {args.host}:{args.cmd_port}")

    for cmd in (args.start.encode(), RATE.get(args.sfreq, b"1"), b"*"):
        try:
            s.sendto(cmd, (args.host, args.cmd_port))
            print(f"  sent {cmd!r}")
        except OSError as e:
            print(f"  ⚠ send {cmd!r} failed: {e}  (route to {args.host}? on the WiFi?)")
        time.sleep(0.2)

    s.settimeout(2.0)
    stream = bytearray()
    ndg, sizes, srcs = 0, set(), set()
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            pkt, addr = s.recvfrom(65536)
            ndg += 1; sizes.add(len(pkt)); srcs.add(addr); stream.extend(pkt)
    except socket.timeout:
        pass

    print(f"\nreceived {ndg} datagrams ({len(stream)} bytes) in {args.seconds:.0f}s | "
          f"sizes={sorted(sizes)} | sources={srcs}")
    if ndg == 0:
        print("NO DATA. Check: (1) Mac joined WiFi 'ESPBCI' (pw 12345678)? "
              "(2) a local IP == 192.168.4.2? (3) `ping 192.168.4.1` works? "
              "(4) board accepts the start byte — retry with `--start B`.")
        return

    # The board streams MTU-sized chunks; reframe the byte stream into 105-B frames.
    FRAME = 105
    frames, i = [], 0
    while i <= len(stream) - FRAME:
        if stream[i] == 0xA0 and stream[i + FRAME - 1] == 0xC0:
            frames.append(bytes(stream[i:i + FRAME])); i += FRAME
        else:
            i += 1
    print(f"reframed {len(frames)} valid 105-B frames (0xA0…0xC0) from the stream")
    if not frames:
        print("⚠ 收到数据但解不出 105B 帧 —— 通道数可能不是 32,或帧格式不同。")
        return

    import numpy as np
    from montage import CAP32_CHANNELS as CH

    def decode(fr):
        v = np.empty(32)
        for c in range(32):
            o = 2 + c * 3
            raw = int.from_bytes(fr[o:o + 3], "big")
            if raw & 0x800000:
                raw -= 16777216
            v[c] = raw * ADC_MICROVOLTS_PER_COUNT
        return v

    M = np.array([decode(fr) for fr in frames])          # (n_frames, 32) µV
    print(f"  seqs (should ++): {[fr[1] for fr in frames[:16]]}")
    print(f"  trigger(frame0): {int.from_bytes(frames[0][100:104], 'big')}")
    raw_mean, raw_std = M.mean(0), M.std(0)
    car = M - np.median(M, axis=1, keepdims=True)        # CAR per timepoint (across channels)
    car_std = car.std(0)

    print("\n  ch    raw_mean(µV)  raw_std(µV) | CAR_std(µV)")
    for c in range(32):
        print(f"  {CH[c]:<4} {raw_mean[c]:11.1f} {raw_std[c]:11.1f} | {car_std[c]:9.1f}")
    print(f"\n  common-mode test: median raw_std={np.median(raw_std):.0f}µV "
          f"-> after CAR={np.median(car_std):.1f}µV")
    if np.median(car_std) < max(1.0, np.median(raw_std) / 3):
        print("  ✅ 大部分是共模(参考漂移);CAR 后大幅下降 —— GUI 已加 CAR,重开应正常。")
    else:
        print("  ⚠ CAR 后仍大 —— 可能不只是共模;看是否某些电极浮空/接触差,或通道顺序/对齐问题。")


if __name__ == "__main__":
    main()
