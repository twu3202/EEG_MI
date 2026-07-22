#!/usr/bin/env bash
# Clone EEG foundation-model repos and fetch pretrained weights.
# Works on the M5 Mac (inference via MPS) or the GPU server.
# Run from repo root:  bash src/foundation/setup_foundation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FM_DIR="$ROOT/third_party"
CKPT_DIR="$ROOT/checkpoints"
mkdir -p "$FM_DIR" "$CKPT_DIR"

echo "== EEGPT (NeurIPS 2024, 10M params, strong MI linear-probe) =="
[ -d "$FM_DIR/EEGPT" ] || git clone --depth 1 https://github.com/BINE022/EEGPT "$FM_DIR/EEGPT"
echo "  -> checkpoint (eegpt_mcae_58chs_4s_large4E.ckpt) is linked from the EEGPT repo README"
echo "     (Google Drive / release). Download it into: $CKPT_DIR/"

echo "== CBraMod (ICLR 2025, MIT, criss-cross, channel-flexible) =="
[ -d "$FM_DIR/CBraMod" ] || git clone --depth 1 https://github.com/wjq-learning/CBraMod "$FM_DIR/CBraMod"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download weighting666/CBraMod pretrained_weights.pth \
    --local-dir "$CKPT_DIR/CBraMod" || echo "  (hf download failed — grab manually from HF)"
else
  echo "  -> pip install huggingface_hub, then:"
  echo "     huggingface-cli download weighting666/CBraMod pretrained_weights.pth --local-dir $CKPT_DIR/CBraMod"
fi

echo "== LaBraM (ICLR 2024 spotlight, channel-name embeddings) =="
[ -d "$FM_DIR/LaBraM" ] || git clone --depth 1 https://github.com/935963004/LaBraM "$FM_DIR/LaBraM"
echo "  -> checkpoints/labram-base.pth ships in the repo."

echo
echo "Done. Repos in $FM_DIR, weights in $CKPT_DIR."
echo "Next: extract frozen embeddings on MOABB MI data and fit a linear probe"
echo "      (see src/foundation/README.md)."
