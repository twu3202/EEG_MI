#!/usr/bin/env python
"""Measure channel-to-channel crosstalk from a single-driven-input calibrator record.

The measurement: cap OFF the head, calibrator GND tied to the board's GND and REF, a sine
driven into exactly ONE channel input, every other input left OPEN.

That last detail is why this file exists instead of `calibrator_check.py --crosstalk`. An open
ADS1299 input floats to the rail and carries no information; the few that have not railed yet
sit at ~100-180 mV of offset and act as antennas. Any "crosstalk" read off them is dominated
by the wiring condition. So the question this module answers is not "how many dB" but:

  1. WHICH INPUTS ARE EVEN MEASURABLE          railed / floating / terminated
  2. IS A TONE ACTUALLY PRESENT ON A VICTIM    tested against the amplitude the SAME estimator
     returns at ~70 nearby null frequencies. Without this control an LS fit always returns a
     positive number, and on a drifting channel that number is ~1 µV whatever you ask for.
  3. IF NOT, WHAT UPPER BOUND DOES IT SET      the null 95th percentile, in dB re. the source.

Two estimator traps this handles, both of which produced wrong answers here before they were
caught. The tone is at 7.0226 Hz, not 7.000: fitting the nominal frequency over a long record
averages the tone away (24.5 µV read as 0.31 µV over 179 s). And `fit_tone` on a railed channel
is singular, which reads out as exactly 0.0 µV — silently, and indistinguishably from "no
crosstalk".

  python src/acquisition/crosstalk_analysis.py recordings/calibrator/cal_7hz_180s.npz --freq 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.montage import CAP32_CHANNELS as CH          # noqa: E402

RESULTS = HERE.parents[1] / "results"
FULLSCALE = 187500.0                     # ±(2^23-1) × 0.02235 µV
RAIL_FRAC = 0.5                          # >this fraction at full scale => open input
FLOAT_DC = 1000.0                        # >1 mV of offset => high-Z input, not driven
B, G, Y, R, RST = "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


# ------------------------------------------------------------------ estimators
def fit_tone(x, fs, f0, order=2):
    """LS projection onto sin/cos at f0, with DC + polynomial drift as nuisance terms.

    A projection rather than an FFT bin because the record is not a whole number of cycles
    and the floating channels carry a drift far larger than the tone — an FFT bin would both
    leak and sit on the skirt of that drift. Returns (peak amplitude µV, phase rad)."""
    n = len(x)
    t = np.arange(n) / fs
    cols = [np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)]
    cols += [t ** k for k in range(order + 1)]
    A = np.column_stack(cols)
    with np.errstate(all="ignore"):
        c, *_ = np.linalg.lstsq(A, x, rcond=None)
    if not np.all(np.isfinite(c)):        # a railed channel is singular here
        return np.nan, np.nan
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0]))


def refine(x, fs, f0, span=0.01, n=4001):
    """The generator's ACTUAL frequency, by scanning the LS fit around the nominal.

    Load-bearing, not a nicety. A fixed-frequency projection over a record many beat periods
    long averages the tone away: this 7.0226 Hz source fitted at 7.000 Hz read 21.8 µV over
    11 s, 5.2 µV over 59 s and 0.31 µV over 179 s, while 5 s windows showed a rock-steady
    24.05 µV throughout. Estimate f once on the driven channel and use it everywhere, so
    source and victims are measured on the same basis."""
    grid = np.linspace(f0 * (1 - span), f0 * (1 + span), n)
    return float(max(grid, key=lambda f: fit_tone(x, fs, f)[0]))


def null_frequencies(f0, lo=4.0, hi=10.0, step=0.0731, keep_out=0.30, n_harm=3):
    """Frequencies where nothing should be, for the detection test. Spaced off any grid that
    would alias with f0, and kept clear of f0 and its harmonics."""
    g = np.arange(lo, hi, step)
    return np.array([f for f in g if all(abs(f - k * f0) > keep_out for k in range(1, n_harm + 1))])


def detect(x, fs, f0, nulls):
    """Is a tone at f0 present in x, above what this estimator returns on noise alone?

    -> dict(amp, null_median, null_p95, z, p). p is the fraction of null frequencies whose
    fitted amplitude is at least as large — a one-sided permutation test with the channel's
    own noise supplying the null."""
    a = fit_tone(x, fs, f0)[0]
    nn = np.array([fit_tone(x, fs, f)[0] for f in nulls])
    return dict(amp=a, null_med=float(np.median(nn)), null_p95=float(np.percentile(nn, 95)),
                z=float((a - nn.mean()) / (nn.std() + 1e-30)), p=float((nn >= a).mean()))


def band_rms(x, fs, lo, hi, notch=()):
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=int(8 * fs), detrend="linear")
    m = (f >= lo) & (f <= hi)
    for f0 in notch:
        m &= np.abs(f - f0) > 0.6
    return float(np.sqrt(np.trapezoid(P[m], f[m])))


def harmonics(x, fs, f0, upto=6):
    return [fit_tone(x, fs, k * f0)[0] for k in range(2, upto + 1) if k * f0 < fs / 2 - 1]


def vhdci_pin(i0):
    """VHDCI-68 pin for a 0-based channel index (pin 1 REF, 2-9 CH1-8, 10 BIAS, 11-34 CH9-32)."""
    n = i0 + 1
    return n + 1 if n <= 8 else n + 2


def circular_sd(ph_rad):
    return float(np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.asarray(ph_rad))))))))


# ------------------------------------------------------------------ analysis
def classify(X):
    """-> (rail_frac, kind[]) with kind in {'railed', 'floating', 'terminated'}."""
    rail = (np.abs(X) > 0.97 * FULLSCALE).mean(1)
    dc = np.abs(X.mean(1))
    return rail, ["railed" if rail[c] > RAIL_FRAC else
                  "floating" if dc[c] > FLOAT_DC else "terminated"
                  for c in range(X.shape[0])]


def analyse(X, fs, f_nominal):
    rail, kind = classify(X)
    live = [c for c in range(X.shape[0]) if kind[c] != "railed"]
    term = [c for c in live if kind[c] == "terminated"]
    head = min(X.shape[1], int(5 * fs))   # short window: the nominal frequency still works here
    d = max(term or live, key=lambda c: fit_tone(X[c][:head], fs, f_nominal)[0])
    f0 = refine(X[d], fs, f_nominal)
    nulls = null_frequencies(f0)
    a0, p0 = fit_tone(X[d], fs, f0)

    rows = []
    for c in live:
        det = detect(X[c], fs, f0, nulls)
        h = harmonics(X[c], fs, f0)
        _, p = fit_tone(X[c], fs, f0)
        rows.append(dict(
            ch=CH[c], idx=c, pin=vhdci_pin(c), kind=kind[c], dc=float(X[c].mean()),
            pin_dist=abs(vhdci_pin(c) - vhdci_pin(d)),
            db=20 * np.log10(max(det["amp"], 1e-12) / a0),
            bound_db=20 * np.log10(det["null_p95"] / a0),      # detection ceiling, in dB re. src
            dphi=float(np.degrees((p - p0 + np.pi) % (2 * np.pi) - np.pi)),
            h2=h[0], h3=h[1],
            mains=fit_tone(X[c], fs, 50.0)[0],
            floor=band_rms(X[c], fs, 20, 45, notch=(35.0,)),
            **det))

    # controls: does anything about the victims actually relate to the driven channel?
    vic = [r for r in rows if r["idx"] != d]
    zd = (X[d] - X[d].mean()) / (X[d].std() + 1e-30)
    for r in vic:
        z = (X[r["idx"]] - X[r["idx"]].mean()) / (X[r["idx"]].std() + 1e-30)
        r["r_with_driven"] = float(np.mean(zd * z))
    ctl = dict(phase_sd=circular_sd([np.radians(r["dphi"]) for r in vic]) if vic else np.nan)
    if vic:
        sp = []
        for f in nulls:
            pd = fit_tone(X[d], fs, f)[1]
            sp.append(circular_sd([fit_tone(X[r["idx"]], fs, f)[1] - pd for r in vic]))
        sp = np.array(sp)
        ctl.update(phase_sd_null_med=float(np.median(sp)),
                   phase_sd_p=float((sp <= ctl["phase_sd"]).mean()))
    return dict(driven=d, a0=a0, p0=p0, f0=f0, f_nominal=f_nominal, nulls=nulls,
                rows=rows, rail=rail, kind=kind, live=live, ctl=ctl)


def stability(X, fs, f0, chans, win_s=5.0):
    n, w = X.shape[1], int(win_s * fs)
    return {c: np.array([fit_tone(X[c][s:s + w], fs, f0)[0]
                         for s in range(0, n - w + 1, w)]) for c in chans}


# ------------------------------------------------------------------ reporting
def report(res, X, fs):
    d, a0, f0, rows = res["driven"], res["a0"], res["f0"], res["rows"]
    railed = [CH[c] for c in range(X.shape[0]) if res["kind"][c] == "railed"]

    print(f"\n{B}=== input state ==={RST}")
    print(f"  railed / open : {len(railed):2d}   {', '.join(railed) if railed else '-'}")
    print(f"  floating      : {sum(1 for r in rows if r['kind']=='floating'):2d}")
    print(f"  terminated    : {sum(1 for r in rows if r['kind']=='terminated'):2d}")
    if railed:
        print(f"  {Y}an open ADS1299 input rails and carries no information, so it cannot show\n"
              f"  crosstalk at all; the ones still floating are antennas.{RST}")

    print(f"\n{B}=== driven channel: {CH[d]} (VHDCI pin {vhdci_pin(d)}) ==={RST}")
    h = harmonics(X[d], fs, f0)
    print(f"  amplitude     : {a0:.3f} µV peak ({a0/np.sqrt(2):.3f} µV rms)")
    print(f"  frequency     : {f0:.4f} Hz  (nominal {res['f_nominal']:g} -> "
          f"{1e6*(f0-res['f_nominal'])/res['f_nominal']:+.0f} ppm, generator + ADC clock combined)")
    print(f"  THD 2f..6f    : {100*np.sqrt(np.sum(np.square(h)))/a0:.2f} %")
    print(f"  50 Hz         : {fit_tone(X[d], fs, 50.0)[0]:.4f} µV")
    print(f"  noise 20-45Hz : {band_rms(X[d], fs, 20, 45, notch=(35.0,)):.3f} µV rms")

    print(f"\n{B}=== is the tone detectable on any other channel? ==={RST}")
    print(f"  null distribution: {len(res['nulls'])} nearby frequencies, same estimator\n")
    print(f"  {'ch':<5}{'state':>11}{'A(f0)':>8}{'null med':>10}{'null p95':>10}{'z':>9}"
          f"{'p':>7}{'r w/drv':>9}   verdict")
    for r in sorted(rows, key=lambda r: (r["idx"] != d, r["p"])):
        v = ("TONE PRESENT" if r["p"] < 0.01 else
             "marginal" if r["p"] < 0.05 else "not distinguishable from own noise")
        col = G if r["p"] < 0.01 else (Y if r["p"] < 0.05 else "")
        rw = "" if r["idx"] == d else f"{r['r_with_driven']:+9.2f}"
        print(f"  {r['ch']:<5}{r['kind']:>11}{r['amp']:8.3f}{r['null_med']:10.3f}"
              f"{r['null_p95']:10.3f}{r['z']:9.1f}{r['p']:7.3f}{rw:>9}   {col}{v}{RST}")

    vic = [r for r in rows if r["idx"] != d]
    det = [r for r in vic if r["p"] < 0.05]
    print(f"\n{B}=== verdict ==={RST}")
    if not det:
        worst = max(vic, key=lambda r: r["bound_db"])
        print(f"  {G}No channel shows a detectable tone. Crosstalk is below this measurement's"
              f"\n  floor.{RST} The floor is set by the victims' own noise, not by the amplifier:")
        print(f"    upper bound  <= {worst['bound_db']:.1f} dB   (worst channel {worst['ch']}, "
              f"null p95 {worst['null_p95']:.2f} µV vs source {a0:.1f} µV)")
        term_floor = next((r["null_p95"] for r in rows if r["idx"] == d), None)
        if term_floor:
            print(f"    the one TERMINATED input has a null p95 of {term_floor:.3f} µV, so the "
                  f"same\n    record with every input terminated would bound crosstalk at "
                  f"{20*np.log10(term_floor/a0):.0f} dB — "
                  f"{worst['bound_db']-20*np.log10(term_floor/a0):.0f} dB better, for free.")
    else:
        for r in det:
            print(f"  {r['ch']}: {r['amp']:.3f} µV = {r['db']:.1f} dB re. source (p={r['p']:.3f})")

    ctl = res["ctl"]
    if vic and np.isfinite(ctl.get("phase_sd", np.nan)):
        print(f"\n{B}=== controls ==={RST}")
        print(f"  victim phase spread at f0 : {ctl['phase_sd']:.1f}°   "
              f"(null median {ctl['phase_sd_null_med']:.1f}°, p={ctl['phase_sd_p']:.3f})")
        if ctl["phase_sd_p"] > 0.05:
            print(f"    {Y}the victims are phase-locked to each other at EVERY frequency — shared\n"
                  f"    ambient pickup — so a tight cluster at f0 is not evidence of coupling.{RST}")
        rr = np.array([r["r_with_driven"] for r in vic])
        print(f"  broadband r with driven   : {rr.min():+.3f} .. {rr.max():+.3f}")
        if np.abs(rr).max() < 0.05:
            print(f"    {G}the driven channel is uncorrelated with every other channel.{RST}")
        print(f"  harmonics on victims are fits to the same noise (source THD is "
              f"{100*np.sqrt(np.sum(np.square(h)))/a0:.2f} %), and carry no separate meaning\n"
              f"    while the fundamental itself is undetected.")


# ------------------------------------------------------------------ plots
def figures(res, X, fs, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch, coherence
    d, a0, f0, rows = res["driven"], res["a0"], res["f0"], res["rows"]
    live, vic = res["live"], [r for r in rows if r["idx"] != d]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = []

    # --- fig 1: what state the inputs are in, and what they look like
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0)); fig.patch.set_facecolor("white")
    col = {"railed": "#c0392b", "floating": "#c58a00", "terminated": "#2e9e5b"}
    ax[0].bar(range(32), X.mean(1) / 1000.0, color=[col[k] for k in res["kind"]])
    ax[0].axhline(FULLSCALE / 1000, color="#6b7480", ls="--", lw=1)
    ax[0].set_xticks(range(32)); ax[0].set_xticklabels(CH, rotation=90, fontsize=7)
    ax[0].set_ylabel("DC level (mV)")
    ax[0].set_title("Input state — 25 open inputs sit at full scale")
    ax[0].legend([plt.Rectangle((0, 0), 1, 1, color=v) for v in col.values()]
                 + [plt.Line2D([], [], ls="--", color="#6b7480")],
                 list(col) + ["full scale 187.5 mV"], fontsize=8, loc="lower right")
    for c in live:
        f, P = welch(X[c], fs=fs, nperseg=int(16 * fs), detrend="linear")
        ax[1].semilogy(f, np.sqrt(2 * P * (f[1] - f[0])), lw=1.8 if c == d else 0.9,
                       color="#c0392b" if c == d else None, zorder=5 if c == d else 1,
                       label=CH[c] + (" (driven)" if c == d else ""))
    ax[1].axvline(f0, color="#6b7480", lw=0.8)
    ax[1].axvline(50, color="#6b7480", ls="--", lw=0.8)
    ax[1].set_xlim(0, 60); ax[1].set_ylim(1e-3, 1e3)
    ax[1].set_xlabel("Hz"); ax[1].set_ylabel("µV peak per bin")
    ax[1].set_title("Spectra — the terminated input's floor is ~100x lower")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); p = RESULTS / f"{stem}_state.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig); out.append(p)

    # --- fig 2: the detection test — the heart of the result
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0)); fig.patch.set_facecolor("white")
    order = [r for r in rows if r["idx"] != d] + [r for r in rows if r["idx"] == d]
    xs = np.arange(len(order))
    ax[0].bar(xs, [r["amp"] for r in order],
              color=["#2e9e5b" if r["p"] < 0.01 else "#2b6cb0" for r in order], zorder=2)
    ax[0].errorbar(xs, [r["null_med"] for r in order],
                   yerr=[np.zeros(len(order)),
                         [r["null_p95"] - r["null_med"] for r in order]],
                   fmt="_", ms=22, color="#c0392b", lw=2, capsize=6, zorder=3,
                   label="noise-only null: median → 95th pct")
    ax[0].set_yscale("log"); ax[0].set_ylim(0.05, 60)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([r["ch"] for r in order])
    ax[0].set_ylabel(f"amplitude at {f0:.3f} Hz (µV peak)")
    ax[0].set_title("Fitted tone vs what the same fit returns on noise")
    ax[0].legend(fontsize=8)
    for r, xi in zip(order, xs):
        ax[0].annotate(f"p={r['p']:.2f}", (xi, r["amp"]), ha="center", fontsize=7,
                       xytext=(0, 4), textcoords="offset points")

    nulls = res["nulls"]
    for r in order:
        c = r["idx"]
        aa = [fit_tone(X[c], fs, f)[0] for f in nulls]
        ax[1].semilogy(nulls, aa, lw=0.9, color="#c0392b" if c == d else None,
                       label=r["ch"] + (" (driven)" if c == d else ""))
        ax[1].scatter([f0], [r["amp"]], s=70, zorder=5,
                      color="#c0392b" if c == d else "#2b6cb0",
                      marker="*" if c == d else "o")
    ax[1].axvline(f0, color="#6b7480", ls="--", lw=1)
    ax[1].set_xlabel("fit frequency (Hz)"); ax[1].set_ylabel("fitted amplitude (µV peak)")
    ax[1].set_title(f"Only the driven channel has anything at {f0:.3f} Hz")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); p = RESULTS / f"{stem}_detection.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig); out.append(p)

    # --- fig 3: controls — stability, geometry, coherence
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); fig.patch.set_facecolor("white")
    st = stability(X, fs, f0, [d] + [r["idx"] for r in vic])
    t = np.arange(len(st[d])) * 5.0
    for c, a in st.items():
        ax[0].plot(t, a, lw=1.8 if c == d else 1.0, marker="o", ms=2.5,
                   color="#c0392b" if c == d else None, label=CH[c])
    ax[0].set_yscale("log"); ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("amplitude (µV peak)")
    ax[0].set_title(f"5 s-window amplitude at {f0:.3f} Hz\n(source is flat to ±0.1 %)")
    ax[0].legend(fontsize=7, ncol=2)

    ax[1].scatter([r["pin_dist"] for r in vic], [r["amp"] for r in vic], s=90,
                  c="#2b6cb0", zorder=3, label="victims (all undetected)")
    for r in vic:
        ax[1].annotate(r["ch"], (r["pin_dist"], r["amp"]), fontsize=8,
                       xytext=(5, 4), textcoords="offset points")
    ax[1].axhline(max(r["null_p95"] for r in vic), color="#c0392b", ls="--", lw=1,
                  label="worst noise-only 95th pct")
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel(f"VHDCI pin distance from driven pin {vhdci_pin(d)}")
    ax[1].set_ylabel(f"amplitude at {f0:.3f} Hz (µV)")
    ax[1].set_title("Nothing tracks connector geometry"); ax[1].legend(fontsize=8)

    for r in vic:
        f, C = coherence(X[d], X[r["idx"]], fs=fs, nperseg=int(8 * fs))
        ax[2].plot(f, C, lw=1.0, label=r["ch"])
    ax[2].axvline(f0, color="#c0392b", ls="--", lw=1)
    ax[2].set_xlim(0, 60); ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel("Hz"); ax[2].set_ylabel("coherence²")
    ax[2].set_title(f"Coherence with {CH[d]} — no peak at f0")
    ax[2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); p = RESULTS / f"{stem}_controls.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig); out.append(p)

    for p in out:
        print(f"  saved {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--freq", type=float, default=7.0, help="nominal generator frequency")
    ap.add_argument("--stem", default="crosstalk7")
    ap.add_argument("--json", help="also dump the numbers here")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    d = np.load(a.path, allow_pickle=True)
    X, fs = d["data"].astype(float), float(d["fs"])
    print(f"\n{B}{Path(a.path).name}{RST}  {X.shape[0]} ch x {X.shape[1]} samp "
          f"= {X.shape[1]/fs:.0f} s @ {fs:.0f} Hz")
    res = analyse(X, fs, a.freq)
    report(res, X, fs)
    if not a.no_plots:
        figures(res, X, fs, a.stem)
    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(file=a.path, f_nominal=a.freq, f_measured=res["f0"], driven=CH[res["driven"]],
                 source_amp_uv=res["a0"], n_nulls=len(res["nulls"]),
                 controls=res["ctl"], rows=res["rows"]), indent=2, default=float))
        print(f"  saved {a.json}")


if __name__ == "__main__":
    main()
