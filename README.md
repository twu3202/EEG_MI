# EEG_MI

Exploring **Motor Imagery (MI)** decoding with a 32-ch dry-electrode EEG cap
(TI ADS1299, WiFi/UDP), aiming for **minimal per-subject retraining**, plus
testing **EEG foundation models** on data we collect.

- Hardware / montage notes and the MI + foundation-model survey:
  [`research/MI_and_EEG_FM_survey.md`](research/MI_and_EEG_FM_survey.md)
- Vendor docs (protocol etc., mostly not needed): `fromprovider/`

## Layout
```
src/
  common/montage.py                  # our 32-ch 10-20 montage + ADC scaling
  baselines/moabb_riemann_baseline.py# CSP/Riemannian MI baseline (CPU)
  baselines/riemann_alignment.py     # + Euclidean Alignment (calibration-free)
  foundation/                        # frozen-backbone linear probing (MPS/server)
  acquisition/                       # cap → LSL bridge + 32-ch viewer (+ synth source)
research/                            # survey + notes (MI/FM, OpenBCI, GUI, Cerelog)
results/                            # metrics + plots + ui_preview.png (gitignored)
checkpoints/  third_party/          # FM weights + repos incl. Cerelog (gitignored)
```

## Environment
```bash
conda activate eegmi                 # Python 3.11
pip install -r requirements-cpu.txt  # baselines (no GPU)
pip install -r requirements-dl.txt   # torch (MPS) + braindecode + FM deps
```

## Quickstart — verify the baseline on public data
```bash
conda activate eegmi
python src/baselines/moabb_riemann_baseline.py --dataset 2b --subjects 4 --eval cross
# larger / standard benchmark:
python src/baselines/moabb_riemann_baseline.py --dataset 2a --subjects 9 --eval cross
```
Datasets download automatically via MOABB. `2b` = BCI IV-2b (small, fast),
`2a` = BCI IV-2a (22-ch, the standard benchmark).

## Roadmap
1. ✅ Baselines on public MI data (this repo).
2. ⬜ Foundation-model linear probes on the same MOABB data (`src/foundation`).
3. ⬜ Wire in our cap: UDP → MNE `Raw` (montage in `common/montage.py`),
   record a left/right-hand MI pilot, check C3/C4 ERD/ERS, then rerun 1–2 on it.
