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
