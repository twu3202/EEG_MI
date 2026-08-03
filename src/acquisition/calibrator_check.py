#!/usr/bin/env python
"""Traceable metrology for the 32-ch cap, using an external µV-level biopotential calibrator.

Why this matters: every hardware test in this project so far has been a SELF-CONSISTENCY
check — alpha_check compares eyes-open to eyes-closed, hardware_check compares channels to
each other, test_injection compares impedance-mode on to off. None of them can detect an
error that affects everything equally. A calibrator with a known amplitude and a known
frequency is the first EXTERNAL reference in the chain, so it can finally answer:

  * is  µV = counts × 0.02235  actually right?   (every amplitude in this project rides on it)
  * do all 32 channels have the SAME gain?
  * how much does a driven channel leak into its neighbours?   (never tested — DIY layout)
  * what is the true input-referred noise floor?
  * is the sample rate really 250.000 Hz?   ← directly relevant to the SSVEP phase problem:
    a 0.4 % clock error rotates a 12 Hz tone by a full cycle in ~2 s, which would by itself
    destroy the phase-locking that eTRCA/TDCA need.

Board (from the silkscreen):
    outputs  J7=GND  J6=50µV  J5=200µV  J4=2000µV
    SW1..6   6-bit frequency, SW1=LSB … SW6=MSB, 1–64 Hz in 1 Hz steps
    SW7      pulse out enable        SW8  frequency sweep enable
             (SW7,SW8) = (0,0) SPOT · (0,1) SWEEP · (1,0) PULSE
    USB-C    charging only — RUN IT ON BATTERY while testing (see below)

Wiring, and the part people get wrong: the ADS1299 measures each electrode against REF, and
needs BIAS/DRL connected or every channel rails. So:

    calibrator GND (J7)  ->  cap REF   AND  cap BIAS/DRL
    calibrator 200µV(J6) ->  the electrode(s) under test

Run the calibrator on BATTERY, not plugged into USB. A charger ties the board to mains earth
and creates a ground loop through the amplifier, which shows up as 50 Hz that swamps a 50 µV
signal. Keep leads short and twisted for the same reason.

    python src/acquisition/calibrator_check.py --gain --freq 10 --nominal 200
    python src/acquisition/calibrator_check.py --crosstalk --freq 10 --driven O1
    python src/acquisition/calibrator_check.py --noise --secs 60
    python src/acquisition/calibrator_check.py --demo          # no hardware, validates plots
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.montage import CAP32_CHANNELS as CH                # noqa: E402

NCH = len(CH)
RESULTS = HERE.parents[1] / "results"
ADC_UV_PER_COUNT = 0.02235          # the constant under test
FULLSCALE = (2 ** 23 - 1) * ADC_UV_PER_COUNT      # 187500 µV — an open input sits here


# ------------------------------------------------------------------ estimation
def fit_tone(x, fs, f0):
    """Least-squares sine fit at f0. Returns (amplitude_peak_uV, phase_rad, residual).

    A projection onto sin/cos rather than an FFT bin, because the record rarely contains a
    whole number of cycles and an FFT bin would then leak and under-read the amplitude."""
    n = len(x)
    t = np.arange(n) / fs
    A = np.column_stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t),
                         np.ones(n), t])          # DC + linear drift nuisance terms
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        if not np.all(np.isfinite(coef)):            # a flat/railed channel is singular here
            return 0.0, 0.0, x
        amp = float(np.hypot(coef[0], coef[1]))
        ph = float(np.arctan2(coef[1], coef[0]))
        return amp, ph, x - A @ coef


def refine_frequency(x, fs, f0, span=0.05, n=4001):
    """True tone frequency by scanning the LS fit around f0 — the clock-accuracy probe.

    NOTE this measures the COMBINED error of the calibrator's crystal and the cap's sample
    clock; it cannot separate them. But a crystal is good to ~20 ppm, so anything above
    ~0.05 % is the acquisition side, not the generator."""
    grid = np.linspace(f0 * (1 - span), f0 * (1 + span), n)
    best = max(grid, key=lambda f: fit_tone(x, fs, f)[0])
    return float(best)


def thd(x, fs, f0, n_harm=5):
    """Total harmonic distortion — catches clipping, a wrong gain setting, or a bad ADC."""
    fund = fit_tone(x, fs, f0)[0]
    h = [fit_tone(x, fs, k * f0)[0] for k in range(2, n_harm + 1) if k * f0 < fs / 2]
    return float(np.sqrt(np.sum(np.square(h))) / fund) if fund > 0 else np.nan


def band_rms(x, fs, band=(1.0, 40.0), notch=None):
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    m = (f >= band[0]) & (f <= band[1])
    for f0 in (notch or []):
        m &= np.abs(f - f0) > 0.6
    return float(np.sqrt(np.trapezoid(P[m], f[m])))


# ------------------------------------------------------------------ acquisition
def record(args, secs):
    if args.demo:
        return _synth(args, secs)
    from udp_lsl_bridge import UdpSource, parse_packet, board_init      # noqa
    src = UdpSource(args.host, args.port)
    board_init(src, args.fs)
    time.sleep(0.4)
    buf, end = [], time.time() + secs
    for pkt in src.frames():
        p = parse_packet(pkt)
        if p is not None:
            buf.append(p[0])
        if time.time() >= end:
            break
    if not buf:
        raise SystemExit("没有收到数据 —— 检查 Mac 是否在 192.168.4.2,帽子是否开机")
    return np.asarray(buf, float).T          # (n_ch, n_samples) µV


def _synth(args, secs):
    """Synthetic cap for validating the analysis without hardware.

    Deliberately imperfect: a 0.3 % clock error, 2 % per-channel gain spread, -46 dB
    crosstalk and a realistic noise floor, so the script's verdicts can be checked."""
    rng = np.random.default_rng(0)
    fs_true = args.fs * 1.003
    n = int(secs * args.fs)
    t = np.arange(n) / fs_true
    driven = CH.index(args.driven) if args.driven in CH else 0
    gains = 1.0 + 0.02 * rng.standard_normal(NCH)
    X = rng.standard_normal((NCH, n)) * 0.9
    X += 3.0 * np.sin(2 * np.pi * 50 * t)
    tone = args.nominal * np.sin(2 * np.pi * args.freq * t)
    if args.crosstalk:
        X[driven] += tone * gains[driven]
        for c in range(NCH):
            if c != driven:
                X[c] += tone * gains[c] * 10 ** (-46 / 20)
    elif not args.noise:
        X += tone[None] * gains[:, None]
    return X


# ------------------------------------------------------------------ tests
def test_gain(X, args):
    print(f"\n=== 增益 / 标度校验  (标称 {args.nominal:.0f} µV @ {args.freq:g} Hz) ===")

    # Find the TRUE tone frequency before measuring any amplitude. If the sample clock is
    # off, the tone is not at the nominal frequency, and fitting at the nominal one lets the
    # projection partially cancel over a long record — a 0.3 % error over 20 s reads a
    # 200 µV tone as ~100 µV. Locking the frequency first removes that error entirely.
    probe = X[int(np.argmax([fit_tone(X[c], args.fs, args.freq)[0] for c in range(NCH)]))]
    f_hat = refine_frequency(probe, args.fs, args.freq)
    if abs(f_hat / args.freq - 1) > 1e-4:
        print(f"  (实测音频在 {f_hat:.5f} Hz 而非 {args.freq:g} Hz,幅度改在实测频率上拟合)")

    amps, phs = [], []
    for c in range(NCH):
        a, p, _ = fit_tone(X[c], args.fs, f_hat)
        amps.append(a); phs.append(p)
    amps = np.array(amps)
    live = amps > 0.15 * np.median(amps[amps > 0]) if np.any(amps > 0) else amps > 0

    print(f"\n  {'通道':<6}{'峰值µV':>9}{'峰峰µV':>9}{'RMS µV':>9}{'/中位数':>9}{'THD%':>7}")
    med = float(np.median(amps[live])) if live.any() else float("nan")
    for c in range(NCH):
        d = thd(X[c], args.fs, args.freq) * 100 if live[c] else np.nan
        flag = "" if live[c] and abs(amps[c] / med - 1) < 0.05 else "  ←"
        print(f"  {CH[c]:<6}{amps[c]:9.2f}{2*amps[c]:9.2f}{amps[c]/np.sqrt(2):9.2f}"
              f"{amps[c]/med:9.3f}{d:7.1f}{flag}")

    spread = (amps[live].std() / med * 100) if live.any() else float("nan")
    print(f"\n  通道间增益离散度 = {spread:.2f}%   (好的板子应 < 2%)")

    # The absolute-scale question. The board's "200µV" could mean peak, pp, or rms; the
    # ratio between channels is convention-free but the absolute number is not.
    print(f"\n  实测中位数:  峰值 {med:.2f} µV | 峰峰 {2*med:.2f} µV | RMS {med/np.sqrt(2):.2f} µV")
    for name, val in (("峰值", med), ("峰峰", 2 * med), ("RMS", med / np.sqrt(2))):
        print(f"    若 '{args.nominal:.0f}µV' 指{name}: 标度误差 = "
              f"{(val/args.nominal - 1)*100:+.2f}%  → 真实 µV/count = "
              f"{ADC_UV_PER_COUNT * args.nominal / val:.5f}")

    err = f_hat / args.freq - 1
    fs_true = args.fs / (1 + err)
    print(f"\n=== 采样时钟 ===")
    print(f"  标称 {args.freq:g} Hz 实测为 {f_hat:.5f} Hz   偏差 {err*1e6:+.0f} ppm ({err*100:+.4f}%)")
    print(f"  → 真实采样率 ≈ {fs_true:.4f} Hz  (标称 {args.fs:g})")

    cyc = abs(err) * 2.0 * args.freq
    print(f"  → 2 秒窗内 {args.freq:g} Hz 的相位漂移 = {cyc:.3f} 周期 ({cyc*360:.0f}°)")
    # The damaging one for template decoders is not within-window drift but the timing error
    # ACCUMULATED across a session: every trial gets cut at a different phase, which is
    # exactly the failure signature already seen (PLV 0.10-0.40, eTRCA/TDCA collapsing).
    for mins, f_ssvep in ((7, 12.0),):
        drift_s = abs(err) * mins * 60
        print(f"  → {mins} 分钟 session 累计时间误差 = {drift_s*1000:.0f} ms "
              f"= {drift_s*f_ssvep:.1f} 个 {f_ssvep:g} Hz 周期")
        if drift_s * f_ssvep > 0.5:
            print("  ** 每个 trial 被切在不同相位上 —— 足以解释 PLV 0.10–0.40 "
                  "和 eTRCA/TDCA 的崩溃 **")
        elif cyc > 0.1:
            print("  ** 窗内漂移已足以损害锁相 **")
        else:
            print("  时钟够准,相位问题不在这里")
    return amps, f_hat, fs_true


def test_crosstalk(X, args):
    """Crosstalk, but only for inputs that are actually TERMINATED.

    Two traps this refuses to walk into, both of which produced a wrong answer on the real
    cap before they were caught:

      * an OPEN input rails, `fit_tone` is then singular and returns exactly 0.0 µV. Reported
        as dB that is -200, which reads as "no crosstalk" when it really means "this channel
        could not have shown crosstalk at all". 25 of 32 channels once landed there, and the
        worst-case figure was computed from the 6 floating ones instead.
      * the generator is not exactly on the nominal frequency. Ours was 7.0226 Hz; fitting
        7.000 over 179 s averaged a steady 24.5 µV source down to 0.31 µV.

    For a real dB number, tie every unused input to the calibrator GND. With open inputs the
    honest output is an upper bound, which is what `crosstalk_analysis.py` computes."""
    rail = (np.abs(X) > 0.97 * FULLSCALE).mean(1)
    dc = np.abs(X.mean(1))
    kind = ["railed" if rail[c] > 0.5 else "floating" if dc[c] > 1000.0 else "terminated"
            for c in range(NCH)]
    live = [c for c in range(NCH) if kind[c] != "railed"]
    if not live:
        raise SystemExit("每个通道都饱和了 —— 检查 GND/BIAS 是否接上")
    head = min(X.shape[1], int(5 * args.fs))            # nominal freq is still valid here
    term = [c for c in live if kind[c] == "terminated"]
    d = max(term or live, key=lambda c: fit_tone(X[c][:head], args.fs, args.freq)[0])
    f0 = refine_frequency(X[d], args.fs, args.freq, span=0.01)
    amps = np.array([fit_tone(X[c], args.fs, f0)[0] if kind[c] != "railed" else np.nan
                     for c in range(NCH)])
    print(f"\n=== 串扰  (被驱动通道 = {CH[d]}, {amps[d]:.2f} µV 峰值 @ {f0:.4f} Hz) ===")
    n_bad = sum(1 for k in kind if k != "terminated")
    if n_bad:
        print(f"\n  ** {n_bad} 个通道不是终接状态:"
              f"{sum(1 for k in kind if k=='railed')} 个开路饱和(零信息)、"
              f"{sum(1 for k in kind if k=='floating')} 个悬空(等于天线)。")
        print("     开路通道无法显示串扰,悬空通道的读数由接线状态决定,不是放大器指标。")
        print("     要得到真正的 dB,把所有不用的输入都接到校准器 GND;"
              "现在只能给上界 —— 用 crosstalk_analysis.py **")
    xt = 20 * np.log10(np.maximum(amps, 1e-9) / amps[d])
    order = [c for c in np.argsort(-np.nan_to_num(amps, nan=-1)) if kind[c] != "railed"]
    print(f"\n  {'通道':<6}{'状态':>11}{'峰值µV':>9}{'串扰dB':>9}")
    for c in order:
        mark = "  ← 被驱动" if c == d else ("  ← 偏高" if xt[c] > -40 else "")
        print(f"  {CH[c]:<6}{kind[c]:>11}{amps[c]:9.3f}{xt[c]:9.1f}{mark}")
    for c in range(NCH):
        if kind[c] == "railed":
            print(f"  {CH[c]:<6}{'railed':>11}{'—':>9}{'—':>9}  ← 开路,无法测量")
    vic = [c for c in order if c != d]
    if vic:
        w = max(vic, key=lambda c: xt[c])
        verdict = "上界" if any(kind[c] != "terminated" for c in vic) else "最差串扰"
        print(f"\n  {verdict} = {xt[w]:.1f} dB ({CH[w]})")
    print("  参考:< -60 dB 很好 | -60~-40 dB 可用 | > -40 dB 说明布线/共地有问题")
    return amps, xt, d


def test_noise(X, args):
    print(f"\n=== 本底噪声 (输入短接到 GND) ===")
    print(f"\n  {'通道':<6}{'1-40Hz RMS':>12}{'50Hz µV':>10}{'峰峰µV':>10}")
    rms = []
    for c in range(NCH):
        r = band_rms(X[c], args.fs, (1.0, 40.0), notch=[50.0])
        h = fit_tone(X[c], args.fs, 50.0)[0]
        rms.append(r)
        flag = "  ←" if r > 3.0 else ""
        print(f"  {CH[c]:<6}{r:12.3f}{h:10.2f}{X[c].max()-X[c].min():10.1f}{flag}")
    rms = np.array(rms)
    print(f"\n  中位本底 = {np.median(rms):.3f} µV RMS (1–40 Hz)")
    print("  参考:ADS1299 @gain24, 250 SPS 典型输入折合噪声约 0.4–1.0 µV RMS")
    if np.median(rms) > 2.0:
        print("  ** 偏高 —— 先确认是在电池供电、导线短且绞合的条件下测的 **")
    return rms


# ------------------------------------------------------------------ plotting
def plot(X, args, amps=None, xt=None, driven=None, rms=None, fs_true=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fam in ("PingFang SC", "Heiti SC", "Arial Unicode MS"):
            if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                break
    except ImportError:
        print("  (无 matplotlib,跳过出图)"); return

    from scipy.signal import welch
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    n = min(X.shape[1], int(2 * args.fs))
    t = np.arange(n) / args.fs
    show = [int(np.argmax(amps))] if amps is not None else list(range(min(4, NCH)))
    for c in show[:4]:
        ax[0].plot(t, X[c, :n], lw=0.8, label=CH[c])
    ax[0].set_xlabel("时间 (s)"); ax[0].set_ylabel("µV")
    ax[0].set_title("波形(前 2 秒)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    f, P = welch(X, fs=args.fs, nperseg=min(X.shape[1], int(4 * args.fs)), axis=-1)
    m = (f >= 0.5) & (f <= 80)
    ax[1].semilogy(f[m], P[:, m].T, lw=0.6, color="#7f8c8d", alpha=0.5)
    ax[1].semilogy(f[m], np.median(P[:, m], 0), lw=1.8, color="#c0392b", label="中位数")
    if not args.noise:
        ax[1].axvline(args.freq, color="#2980b9", ls="--", lw=1, label=f"{args.freq:g} Hz")
    ax[1].axvline(50, color="#e67e22", ls=":", lw=1, label="50 Hz")
    ax[1].set_xlabel("频率 (Hz)"); ax[1].set_ylabel(r"PSD (µV²/Hz)")
    ax[1].set_title("功率谱(全通道)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    if xt is not None:
        o = np.argsort(-amps)
        ax[2].bar(range(NCH - 1), xt[o[1:]], color="#2c3e50")
        ax[2].axhline(-40, color="#c0392b", ls="--", label="-40 dB")
        ax[2].axhline(-60, color="#2E7D46", ls="--", label="-60 dB")
        ax[2].set_xticks(range(NCH - 1))
        ax[2].set_xticklabels([CH[c] for c in o[1:]], rotation=90, fontsize=6)
        ax[2].set_ylabel("串扰 (dB)"); ax[2].set_title(f"相对 {CH[driven]} 的串扰")
        ax[2].legend(fontsize=8)
    elif rms is not None:
        ax[2].bar(range(NCH), rms, color="#2c3e50")
        ax[2].set_xticks(range(NCH)); ax[2].set_xticklabels(CH, rotation=90, fontsize=6)
        ax[2].set_ylabel("µV RMS"); ax[2].set_title("本底噪声 1–40 Hz")
    else:
        ax[2].bar(range(NCH), amps, color="#2c3e50")
        ax[2].axhline(np.median(amps), color="#c0392b", ls="--", label="中位数")
        ax[2].set_xticks(range(NCH)); ax[2].set_xticklabels(CH, rotation=90, fontsize=6)
        ax[2].set_ylabel("峰值 µV"); ax[2].set_title(f"各通道 {args.freq:g} Hz 幅度")
        ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")

    mode = "noise" if args.noise else ("crosstalk" if args.crosstalk else "gain")
    fig.suptitle(f"校准器验收 — {mode}" + (f"  真实 fs≈{fs_true:.3f} Hz" if fs_true else ""),
                 y=1.02)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"calibrator_{mode}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  图: {out}")


def dip_table(freq):
    """Print the DIP switch pattern for a wanted frequency (SW1=LSB … SW6=MSB)."""
    v = int(round(freq)) - 1
    if not 0 <= v <= 63:
        print("  频率需在 1–64 Hz"); return
    bits = [(v >> i) & 1 for i in range(6)]
    print(f"\n  {freq:g} Hz 的拨码(值={v}): " +
          "  ".join(f"SW{i+1}={'ON' if b else 'off'}" for i, b in enumerate(bits)))
    print("  SW7=off SW8=off  (SPOT 定频模式)")
    print("  注意:'1~64Hz' 对应值 0~63,故 频率 = 值 + 1。先用全 off 验证是否为 1 Hz。")


def vhdci_pin(ch_index0):
    """VHDCI-68 pin for a 0-based channel index, per 32导电极接口对应关系.xlsx.

        pin 1      SRB1 / REF
        pin 2-9    CH1-CH8
        pin 10     BIAS (GND electrode)
        pin 11-34  CH9-CH32          (pins 35-68 unused)
    """
    n = ch_index0 + 1                       # CH number, 1-based
    return n + 1 if n <= 8 else n + 2


def pinout():
    print("\n=== VHDCI-68 针脚对照(放大器侧为母座,帽子侧为公口)===")
    print("  pin 1 = REF (SRB1)     pin 10 = BIAS/GND 电极     pin 35-68 空接\n")
    print(f"  {'电极':<6}{'通道':<7}{'pin':>4}    {'电极':<6}{'通道':<7}{'pin':>4}")
    half = (NCH + 1) // 2
    for i in range(half):
        j = i + half
        left = f"  {CH[i]:<6}CH{i+1:<5}{vhdci_pin(i):>4}"
        right = f"    {CH[j]:<6}CH{j+1:<5}{vhdci_pin(j):>4}" if j < NCH else ""
        print(left + right)
    print("\n  ** 通道号↔电极名的对应是本项目推断的,卖家接口表里'电极点位'一列是空的。")
    print("     用 --gain 只驱动单个电极,看哪个通道亮,即可第一次实测验证这张表。 **")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gain", action="store_true", help="增益/标度/时钟校验")
    p.add_argument("--crosstalk", action="store_true", help="串扰:只驱动一个通道")
    p.add_argument("--noise", action="store_true", help="本底噪声:输入短接 GND")
    p.add_argument("--freq", type=float, default=10.0, help="校准器设定频率 Hz")
    p.add_argument("--nominal", type=float, default=200.0, help="校准器设定幅度 µV")
    p.add_argument("--driven", default="O1", help="串扰测试中被驱动的通道(仅 demo 用)")
    p.add_argument("--secs", type=float, default=30.0)
    p.add_argument("--fs", type=float, default=250.0)
    p.add_argument("--host", default="192.168.4.1")
    p.add_argument("--port", type=int, default=8086)
    p.add_argument("--demo", action="store_true", help="无硬件,用合成数据验证分析链")
    p.add_argument("--dip", type=float, help="只打印某频率的拨码设置然后退出")
    p.add_argument("--pinout", action="store_true", help="打印 VHDCI-68 针脚对照然后退出")
    a = p.parse_args()

    if a.pinout:
        pinout(); return
    if a.dip:
        dip_table(a.dip); return
    if not (a.gain or a.crosstalk or a.noise):
        a.gain = True
    dip_table(a.freq)

    print(f"\n采集 {a.secs:g} 秒 …" + ("  [demo 合成数据]" if a.demo else ""))
    X = record(a, a.secs)
    print(f"  得到 {X.shape[0]} 通道 × {X.shape[1]} 采样 ({X.shape[1]/a.fs:.1f} s)")

    amps = xt = driven = rms = fs_true = None
    if a.noise:
        rms = test_noise(X, a)
    elif a.crosstalk:
        amps, xt, driven = test_crosstalk(X, a)
    else:
        amps, _, fs_true = test_gain(X, a)
    plot(X, a, amps=amps, xt=xt, driven=driven, rms=rms, fs_true=fs_true)


if __name__ == "__main__":
    main()
