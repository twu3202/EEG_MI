# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Sanity-check the vendor's two impedance formulas against physics (Ohm's law).

Injection-based impedance is just Ohm's law: a known AC current I is injected and the
voltage V at the injection frequency is measured, so  Z = V / I  — a straight line
THROUGH THE ORIGIN with slope 1/I. We compare that to the vendor's two formulas:

    #1   Z = 32.1073 · V1     − 3983     (V1 = 31.2 Hz amplitude, µV)
    #2   Z = 0.00115  · Vpeak − 4483     (Vpeak in raw ADC counts)

and show they (a) have unphysical negative intercepts and (b) imply injection currents
~600× apart, so they contradict each other. Saves results/impedance_formula_analysis.png.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

SCALE = (4.5e6 / (2 ** 23 - 1)) / 24        # µV per ADC count ≈ 0.02235
RESULTS = Path(__file__).resolve().parents[2] / "results"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.linspace(0, 3000, 600)           # 31.2 Hz pickup amplitude (µV)
    f1 = (32.1073 * v - 3982.98) / 1000                      # vendor #1 (V1 in µV) -> kΩ
    f2 = (0.00115 * (v / SCALE) - 4483.0) / 1000            # vendor #2 (Vpeak in counts) -> kΩ
    phys_24n = (1e-6 / 24e-9) * v / 1000                    # Z=V/I, I=24 nA
    phys_24u = (1e-6 / 24e-6) * v / 1000                    # Z=V/I, I=24 µA

    fig, ax = plt.subplots(figsize=(10.5, 6.4)); fig.patch.set_facecolor("white")
    ax.axhspan(-15, 0, color="#f3d6d6", alpha=0.6, zorder=0)                 # impossible Z<0
    ax.text(60, -8, "Z < 0  (physically impossible)", color="#b23b3b", fontsize=10)
    ax.plot(v, phys_24n, color="#2e9e5b", lw=2.4, label="physics  Z=V/I, I=24 nA  (through 0)")
    ax.plot(v, phys_24u, color="#2e9e5b", lw=2.0, ls=":", label="physics  Z=V/I, I=24 µA  (through 0)")
    ax.plot(v, f1, color="#2b6cb0", lw=2.4, label="vendor #1: 32.1·V1−3983  → implies ~24 nA")
    ax.plot(v, f2, color="#c0392b", lw=2.4, label="vendor #2: 0.00115·Vpeak−4483  → implies ~24 µA")
    ax.axhline(0, color="#333", lw=0.8)
    ax.plot(3982.98 / 32.1073, 0, "o", color="#2b6cb0")
    ax.annotate("#1 hits Z=0 at 124 µV\n(good electrodes → negative)", (124, 0),
                (400, 14), color="#2b6cb0", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2b6cb0"))
    ax.annotate("#2 is negative across the whole\nnormal range (Z=0 only at 87 mV)",
                (2600, f2[-1]), (1500, -13), color="#c0392b", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_xlim(0, 3000); ax.set_ylim(-15, 100)
    ax.set_xlabel("measured 31.2 Hz amplitude  V1 (µV)")
    ax.set_ylabel("impedance Z (kΩ)")
    ax.set_title("Vendor impedance formulas vs Ohm's law — the two disagree ~600×")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.15)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "impedance_formula_analysis.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", out)

    # printed summary
    print(f"\nformula #1 slope 32.1073 Ω/µV  -> I = {1e-6/32.1073*1e9:5.1f} nA")
    print(f"formula #2 slope {0.00115/SCALE:.4f} Ω/µV -> I = {1e-6/(0.00115/SCALE)*1e6:5.1f} µA")
    print(f"ratio of implied currents: ~{(1e-6/(0.00115/SCALE))/(1e-6/32.1073):.0f}×  -> mutually inconsistent")


if __name__ == "__main__":
    main()
