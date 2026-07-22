# EEG Foundation-Model probing

Goal: reuse a **frozen** pretrained EEG backbone and fit only a light **linear/MLP
probe** on labelled MI data — i.e. no full retraining. This is how we'll test the
32-ch cap data later, and it's how we get an apples-to-apples comparison against
the pyRiemann baselines.

## Where things run
- **M5 Mac (MPS):** fine for *inference / linear-probing* of the small models
  (EEGPT ≈ 10M params, CBraMod). Use `device="mps"`.
- **GPU server:** needed only for *fine-tuning the backbone* or *pretraining*.

## Setup
```bash
bash src/foundation/setup_foundation.sh   # clone repos + fetch weights
```
Repos land in `third_party/`, weights in `checkpoints/`.

## Candidates (all channel-flexible, weights public)
| Model   | Repo                     | Input it expects            | License |
|---------|--------------------------|-----------------------------|---------|
| EEGPT   | `BINE022/EEGPT`          | 58ch/256Hz/4s, patch 64     | see repo |
| CBraMod | `wjq-learning/CBraMod`   | (ch, seg, 200 pts/patch)=200Hz | MIT   |
| LaBraM  | `935963004/LaBraM`       | 200Hz, patch, channel names | see repo |

## Probe recipe (identical across models)
1. Pull MI epochs from MOABB (same `LeftRightImagery` paradigm as the baseline)
   so the comparison is fair.
2. Resample to the model's native rate (EEGPT 256 Hz, CBraMod/LaBraM 200 Hz);
   map our channel names to the model's channel embedding.
3. Forward pass through the **frozen** backbone → per-trial embedding.
4. Fit `LogisticRegression` (or a 1-layer MLP) on embeddings; evaluate
   cross-subject exactly like the baseline.

`probe_moabb.py` implements this loop; encoders live in `encoders.py`.

## Status (cross-subject 2a, ROC-AUC — see ../../results/RESULTS.md)
| encoder | wired? | 2a AUC | notes |
|---|---|---|---|
| `raw`     | ✅ | 0.521 | floor |
| `cbramod` | ✅ (MPS) | 0.560 | weights in `checkpoints/CBraMod/` |
| `labram`  | ✅ (MPS) | 0.595 | weights ship in `third_party/LaBraM/checkpoints/` |
| `eegpt`   | ⬜ | — | needs the figshare checkpoint (below) |

Run: `python src/foundation/probe_moabb.py --dataset 2a --subjects 9 --encoder labram --fmin 0.5 --fmax 45`
(set `PYTORCH_ENABLE_MPS_FALLBACK=1` for the transformer ops MPS doesn't cover).

All frozen probes underperform the classic Riemannian baseline (~0.78) — expected: MI
is a subtle *spatial* ERD contrast that a frozen generic backbone doesn't expose linearly.
FMs need **fine-tuning** (not frozen probing) to compete.

### Gotchas already handled (in encoders.py)
- **Pool over time only, keep channels** — averaging over channels drives MI to chance.
- **Scale = µV/100** (`scale=0.01`); MOABB returns µV-scale here. Feed **broadband** 0.5–45 Hz.
- **LaBraM + timm 1.x**: `_timm1x_shim()` aliases the old `timm.models.layers/.registry`.
- **PyTorch 2.6 `torch.load`**: LaBraM ckpt needs `weights_only=False` (trusted repo file).

### EEGPT (not yet wired — needs manual download)
Checkpoint is on figshare (blocked from automated fetch):
https://figshare.com/s/e37df4f8a907a866df4b → `eegpt_mcae_58chs_4s_large4E.ckpt`
(58 ch, **256 Hz**, 4 s, patch 64). Download it into `checkpoints/`, then mirror
`third_party/EEGPT/downstream/linear_probe_EEGPT_BCIC2A.py` (`EEGPTClassifier` in
`downstream/Modules/models/EEGPT_mcae_finetune.py`) as an `EEGPTEncoder`. Note its native
rate is 256 Hz — run the probe with `--sfreq 256`.
