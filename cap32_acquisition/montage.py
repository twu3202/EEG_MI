"""Our 32-ch dry-cap montage (standard 10-20) + helpers.

Channel order matches `fromprovider/32导电极接口对应关系.xlsx` (REF/GND and the
unwired M1/M2 removed). Reused by the ERD/ERS sanity checks and by the
foundation-model channel-name mapping later on.
"""
from __future__ import annotations

# 32 scalp channels, in acquisition order.
CAP32_CHANNELS = [
    "FP1", "FP2", "AF3", "AF4", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6", "P3", "P4", "P7", "P8",
    "PO3", "PO4", "O1", "O2", "FZ", "CZ", "PZ", "OZ",
]

# Channels that carry most of the MI mu/beta ERD/ERS signal.
SENSORIMOTOR = [
    "FC5", "FC1", "FC2", "FC6",
    "C3", "CZ", "C4",
    "CP5", "CP1", "CP2", "CP6",
]

# ADS1299: sample_value * (VREF / (2**23 - 1)) / gain  ->  Volts
#   VREF = 4.5 V, gain = 24  ->  LSB ≈ 0.5364 µV, /24 ≈ 0.02235 µV per count
ADC_MICROVOLTS_PER_COUNT = (4.5e6 / (2 ** 23 - 1)) / 24  # ≈ 0.02235 µV


def make_info(sfreq: float, ch_names=None):
    """Build an MNE Info with the standard_1020 montage attached."""
    import mne

    ch_names = list(ch_names) if ch_names is not None else CAP32_CHANNELS
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


if __name__ == "__main__":
    print(f"{len(CAP32_CHANNELS)} channels: {CAP32_CHANNELS}")
    print(f"sensorimotor subset ({len(SENSORIMOTOR)}): {SENSORIMOTOR}")
    print(f"ADC scale: {ADC_MICROVOLTS_PER_COUNT:.5f} µV / count")


# 2D scalp-map positions (top-down, normalized to the unit head circle), precomputed
# from MNE's standard_1020 montage so the impedance head map needs no MNE dependency.
CAP32_XY = {
    "FP1": (-0.2853, 0.9431), "FP2": (0.2751, 0.9524), "AF3": (-0.3256, 0.8762),
    "AF4": (0.3303, 0.8846), "F3": (-0.4819, 0.6521), "F4": (0.4826, 0.6633),
    "F7": (-0.6711, 0.5515), "F8": (0.6830, 0.5699), "FC1": (-0.3290, 0.3960),
    "FC2": (0.3215, 0.4000), "FC5": (-0.7368, 0.3264), "FC6": (0.7443, 0.3386),
    "C3": (-0.6247, 0.0403), "C4": (0.6270, 0.0472), "T7": (-0.8024, -0.0011),
    "T8": (0.7967, 0.0083), "CP1": (-0.3427, -0.2966), "CP2": (0.3555, -0.2946),
    "CP5": (-0.7592, -0.2896), "CP6": (0.7801, -0.2854), "P3": (-0.5080, -0.5942),
    "P4": (0.5188, -0.5921), "P7": (-0.6916, -0.5438), "P8": (0.6831, -0.5402),
    "PO3": (-0.3522, -0.8027), "PO4": (0.3404, -0.8027), "O1": (-0.2851, -0.9123),
    "O2": (0.2748, -0.9095), "FZ": (-0.0042, 0.7031), "CZ": (-0.0034, 0.0636),
    "PZ": (-0.0041, -0.6162), "OZ": (-0.0061, -0.9354),
}
