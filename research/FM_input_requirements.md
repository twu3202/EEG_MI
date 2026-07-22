# What input "stage" of EEG each foundation model expects

Read from the actual preprocessing code in the downloaded repos (not from memory).
Answers: these models don't take *raw* EEG, and they don't take hand-engineered MI
band-power either — they take **cleaned continuous EEG, resampled + broadband-filtered
+ amplitude-normalized, then cut into ~1 s patches.**

## Per-model input contract

| Model (pipeline) | Resample | Band-pass | Notch | Reference | Amplitude scaling | Window → patches | Channels |
|---|---|---|---|---|---|---|---|
| **CBraMod** — BCI IV-2a finetune (`preprocessing_bciciv2a.py`) | 250→**200 Hz** | **0.3–40 Hz** Butterworth o5 | — | **CAR** (subtract per-timepoint channel mean) | **÷100** → µV/100 (O(0.1–1)) | 2–6 s (4 s) → `(22, 4, 200)` = 4×1 s | native (criss-cross attn) |
| **CBraMod** — pretrain (TUEG) | **200 Hz** | 0.3–75 Hz | 60 Hz | — | **÷100** | 1 s / 200-pt patches | flexible |
| **LaBraM** — pretrain + downstream (`dataset_maker`, `data_preprocess.py`) | **200 Hz** | **0.1–75 Hz** | **50 Hz** | dataset-native | **÷100** (`normalization(x)=x/100`) | 1 s / 200-pt patches | channel-name embeddings + cls token |
| **EEGPT** — BCI IV-2a probe (`LoadData.py`, `linear_probe_EEGPT_BCIC2A.py`) | **256 Hz**, 4 s | ~**0–38 Hz** IIR (lowpass) | — | native | **×1e6 → µV** (NO ÷100, O(10)) | patch = **64 samples** | 22→19 via 1×1 conv + 19 named channel-ids |

## Time domain or frequency domain? Exact tensor format & size
**You feed all of them TIME-DOMAIN EEG (before FFT).** Any spectral transform happens
*inside* the model — you never hand-FFT the signal.

| Model | Input tensor (float32) | 1 patch | Rate | Scaling | Extra arg | What it does with FFT |
|---|---|---|---|---|---|---|
| **CBraMod** | `(B, C, S, 200)` — (batch, ch, segments, 200 samp) | 200 samp = 1 s | 200 Hz | µV **÷100** | — | time-domain in; **internally** `rfft`→\|·\|(101 bins)→linear, *added* to a time-conv embed (uses both) |
| **LaBraM** | `(B, C, A, 200)` + `input_chans` (len C+1) | 200 samp = 1 s | 200 Hz | µV **÷100** | channel-id list | time-domain in, **TemporalConv** patch embed (no input FFT; FFT was only the VQ-tokenizer's pretrain *target*) |
| **EEGPT** | `(B, C, T)` (T = n_patches × 64) | **64 samp** ≈ 0.25 s | ~250–256 Hz | µV (**×1e6**, no ÷100) | channel-id list; 22→19 via 1×1 conv | time-domain only, `Conv2d(1,·,(1,64))` — **no FFT anywhere** |
| *BIOT* (in repo, not our pick) | `(B, C, T)` | STFT frame | 200 Hz | — | — | **after-FFT**: `torch.stft(n_fft=200, hop=100)` first, then patches the spectrogram |

Example, a 4 s epoch: CBraMod/LaBraM → 4 s×200 Hz = 800 samp → `(B, C, 4, 200)`;
EEGPT → 4 s×256 Hz ≈ 1024 samp → `(B, C, 1024)` (16 patches of 64). All `float32`.

Takeaway: give them **raw time-domain µV patches** (÷100 for CBraMod/LaBraM, plain µV for
EEGPT). CBraMod computes its own FFT internally; only BIOT expects you to think in spectra,
and even it runs the STFT inside the model.

## The common thread (what "stage" they want)
1. **Resampled** to a fixed rate — 200 Hz (CBraMod/LaBraM) or 256 Hz (EEGPT).
2. **Broadband**, not MI-band — roughly 0.1–75 Hz (pretraining) or ~0.3–40 Hz (CBraMod 2a);
   **not** the classic 8–30 Hz μ/β band. Foundation models want the full spectral content.
3. **Notch'd** for line noise where the pipeline sets it (LaBraM 50, CBraMod-pretrain 60).
4. **Amplitude-normalized to O(1)** — CBraMod & LaBraM use **µV÷100**; EEGPT uses **µV** directly.
5. **Lightly referenced** — CBraMod's 2a pipeline applies **CAR**; LaBraM/EEGPT keep the
   recording's native reference.
6. **Epoched into short windows then split into ~1 s patches** (or 64-sample patches for EEGPT).

So: *cleaned, resampled, filtered, normalized, patched continuous EEG* — between "raw" and
"feature-engineered."

## How this compares to what our probe currently feeds (and how to improve)
`probe_moabb.py` currently feeds: `LeftRightImagery(fmin=0.5, fmax=45, resample=200)`, µV,
`÷100` for CBraMod/LaBraM, channel-preserving pooling — **no CAR, no notch**.

Mismatches that likely cost the frozen-probe a few points:
- **No CAR** for CBraMod (its 2a pipeline expects it).
- **No 50 Hz notch** for LaBraM (its pipeline applies one).
- **Band edges** slightly off (LaBraM native 0.1–75 vs our 0.5–45).
- **EEGPT not run**, and when we do it needs **256 Hz + µV (no ÷100)**, not our 200 Hz/÷100.

**Levers to match native preprocessing (should raise the probe numbers):**
- CBraMod: resample 200, band-pass 0.3–40, **apply CAR**, ÷100.
- LaBraM: resample 200, band-pass 0.1–75 + **notch 50**, ÷100.
- EEGPT: resample **256**, band-pass ~0.3–38, **µV (no ÷100)**, feed the 19 named channels.

## Takeaway for our 32-ch cap
- These models want **broadband** data (~0.1–75 Hz) + **CAR** + **µV/100** (or µV for EEGPT) —
  all of which our cap can produce (CAR needs no M1/M2, consistent with §hardware notes).
- Our classic Riemannian baseline instead wants the **narrow 8–30 Hz MI band** — so the two
  approaches want *different* preprocessing; keep separate pipelines per model.
- When we fine-tune (the real FM path), replicate each model's native preprocessing above.
