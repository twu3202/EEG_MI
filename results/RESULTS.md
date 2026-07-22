# Empirical results — public-data verification

Cross-subject (leave-one-subject-out) MI, 2-class **left vs right hand**, ROC-AUC.
All run on the M5 Mac (classic = CPU; CBraMod = MPS). Reproduce with commands below.

## BCI IV-2a (`BNCI2014_001`, 22 ch, 9 subjects)

| Method | ROC-AUC | Deep training? |
|---|---:|---|
| **TS+LR + Euclidean Alignment** | **0.780** | none |
| **TS+LR** (Riemannian tangent space + logistic reg) | 0.777 | none |
| **CSP+LDA** | 0.772 | none |
| **MDM + Euclidean Alignment** | 0.759 | none |
| **MDM** (min. distance to Riemannian mean) | 0.760 | none |
| LaBraM-base frozen backbone + linear probe | 0.595 | none (frozen) |
| CBraMod frozen backbone + linear probe | 0.560 | none (frozen) |
| raw-flatten floor | 0.521 | — |
| chance | 0.500 | — |

Frozen-probe embeddings all keep per-channel structure (pool over time only) and use
broadband 0.5–45 Hz input at µV/100 scaling. EEGPT not run here — its checkpoint is on
figshare (blocked from automated download); the encoder is documented in
`src/foundation/README.md` and runs once the .ckpt is placed in `checkpoints/`.

### Euclidean Alignment (calibration-free, per-subject) — `src/baselines/riemann_alignment.py`
| Pipeline | none | +EA | Δ |
|---|---:|---:|---:|
| TS+LR | 0.771 | 0.780 | +0.008 |
| MDM   | 0.744 | 0.759 | +0.014 |

EA is a small but free win on clean 2a (tangent space is already robust); it typically
helps more under the bigger domain shift we expect from dry electrodes / cross-session,
so it stays in the pipeline.

## BCI IV-2b (`BNCI2014_004`, 3 ch, 4 subjects)

| Method | ROC-AUC |
|---|---:|
| MDM | 0.753 |
| CSP+LDA | 0.749 |
| TS+LR | 0.741 |
| CBraMod frozen probe | 0.61 |
| raw-flatten floor | 0.497 |

## Takeaways
1. **Classic Riemannian (TS+LR) is the strongest calibration-free MI decoder** at ~0.77
   cross-subject — right in the published MOABB range, with zero deep training. This is
   our recommended baseline for the cap.
2. **A frozen foundation-model backbone + linear probe substantially underperforms**
   (CBraMod 0.56 vs 0.77). This is expected: MI's signal is a subtle *spatial* mu/beta
   ERD/ERS contrast (C3 vs C4); a generically self-supervised backbone doesn't expose it
   linearly. Foundation models close the gap mainly when the **backbone is fine-tuned**
   (as in the CBraMod paper), not frozen.
3. **Gotcha we hit and fixed:** averaging encoder features over channels drives MI to
   chance (0.50) — it erases the left/right lateralization. Always keep per-channel
   structure for MI probes.
4. Cross-subject variance is huge (per-subject AUC 0.48–0.99). A few minutes of
   target-subject calibration + alignment (Euclidean/Riemannian) is where the real gains are.

## Reproduce
```bash
conda activate eegmi
# classic baselines
python src/baselines/moabb_riemann_baseline.py --dataset 2a --subjects 9 --eval cross
# foundation-model probe (needs bash src/foundation/setup_foundation.sh first)
python src/foundation/probe_moabb.py --dataset 2a --subjects 9 --encoder cbramod --fmin 0.5 --fmax 45
python src/foundation/probe_moabb.py --dataset 2a --subjects 9 --encoder raw   # floor
```

## Next levers (to actually beat 0.77 with less/zero calibration)
- Add **Euclidean/Riemannian alignment** (pyRiemann `transfer`) to the classic pipeline — the
  single biggest calibration-free win.
- **Fine-tune** CBraMod/EEGPT/LaBraM (not frozen) on pooled source subjects; probe was only the
  cheap first look.
- Compare **EEGPT** (256 Hz native) and **LaBraM** (weights already in-repo) encoders.
