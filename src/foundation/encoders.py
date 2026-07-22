"""Frozen foundation-model encoders that turn MI epochs into per-trial embeddings.

Each encoder exposes `.encode(X)` where X is (n_trials, n_ch, n_times) already
resampled to the model's native rate, and returns (n_trials, emb_dim). They run
on the M5 GPU via MPS. Kept separate from probe_moabb.py so the probe loop stays
model-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CBRAMOD_DIR = ROOT / "third_party" / "CBraMod"
CBRAMOD_CKPT = ROOT / "checkpoints" / "CBraMod" / "pretrained_weights.pth"
# LaBraM's model code lives inside the EEGPT repo's downstream/Modules; weights ship in the LaBraM repo.
LABRAM_CODE_DIR = ROOT / "third_party" / "EEGPT" / "downstream"
LABRAM_CKPT = ROOT / "third_party" / "LaBraM" / "checkpoints" / "labram-base.pth"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CBraModEncoder:
    """CBraMod (ICLR'25) frozen backbone. Native input: 200 Hz, 1 s patches.

    CBraMod's own BCI IV-2a loader feeds `data/100` on a µV-scale signal; MOABB
    returns Volts, so we multiply by 1e4 (V->µV, then /100) to match.
    """

    name = "cbramod-frozen"
    sfreq = 200
    patch = 200  # points per patch (1 s @ 200 Hz)

    def __init__(self, scale: float = 0.01, device: torch.device | None = None):
        sys.path.insert(0, str(CBRAMOD_DIR))
        from models.cbramod import CBraMod  # noqa: E402
        import torch.nn as nn

        self.device = device or pick_device()
        self.model = CBraMod().to(self.device)
        self.model.load_state_dict(torch.load(CBRAMOD_CKPT, map_location=self.device))
        self.model.proj_out = nn.Identity()  # expose raw 200-d patch features
        self.model.eval()
        self.scale = scale
        self._logged = False

    def _to_patches(self, X: np.ndarray) -> np.ndarray:
        n, ch, t = X.shape
        seg = t // self.patch
        if seg == 0:
            raise ValueError(f"epoch too short: {t} samples < one {self.patch}-pt patch")
        X = X[:, :, : seg * self.patch]
        return X.reshape(n, ch, seg, self.patch)

    @torch.no_grad()
    def encode(self, X, batch_size: int = 64) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32) * self.scale
        if not self._logged:
            print(f"  [cbramod] scaled input std={X.std():.3f} (target ~0.1-1)")
            self._logged = True
        X = self._to_patches(X)
        embs = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(self.device)
            feats = self.model(xb)            # (b, ch, seg, 200)
            # Pool over TIME/segments only, KEEP channels -> (b, ch, 200) -> flatten.
            # Averaging over channels would erase the C3/C4 lateralization MI needs.
            emb = feats.mean(dim=2).reshape(feats.shape[0], -1)  # (b, ch*200)
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


def _timm1x_shim() -> None:
    """LaBraM imports the pre-1.0 timm paths (timm.models.layers / .registry).
    Alias them onto the installed timm 1.x so the model file imports cleanly."""
    import sys
    import types
    import timm.layers as _tl
    from timm.models import register_model as _rm

    sys.modules.setdefault("timm.models.layers", _tl)
    _reg = types.ModuleType("timm.models.registry")
    _reg.register_model = _rm
    sys.modules.setdefault("timm.models.registry", _reg)


class LaBraMEncoder:
    """LaBraM base (ICLR'24) frozen backbone. Native input: 200 Hz, 1 s patches.
    Same µV/100 scaling convention as CBraMod. Keeps per-channel token structure."""

    name = "labram-frozen"
    sfreq = 200
    patch = 200

    def __init__(self, scale: float = 0.01, device: torch.device | None = None):
        _timm1x_shim()
        sys.path.insert(0, str(LABRAM_CODE_DIR))
        import Modules.LaBraM.modeling_finetune  # noqa: F401  (registers the model)
        from timm.models import create_model

        self.device = device or pick_device()
        ckpt = torch.load(LABRAM_CKPT, map_location="cpu", weights_only=False)  # trusted repo ckpt
        state = {
            k[len("student."):]: v
            for k, v in ckpt["model"].items()
            if k.startswith("student.")
        }
        model = create_model(
            "labram_base_patch200_200", qkv_bias=False, rel_pos_bias=True,
            num_classes=4, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
            drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
            use_rel_pos_bias=True, use_abs_pos_emb=True, init_values=0.1,
        )
        model.load_state_dict(state, strict=False)
        self.model = model.eval().to(self.device)
        self.scale = scale
        self._logged = False

    @torch.no_grad()
    def encode(self, X, batch_size: int = 64) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32) * self.scale
        n, C, T = X.shape
        a = T // self.patch
        if a == 0:
            raise ValueError(f"epoch too short: {T} < one {self.patch}-pt patch")
        X = X[:, :, : a * self.patch].reshape(n, C, a, self.patch)
        if not self._logged:
            print(f"  [labram] scaled std={X.std():.3f}, patched shape={X.shape}")
            self._logged = True
        input_chans = list(range(C + 1))  # index 0 = cls, 1..C = channels
        embs = []
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(self.device)
            tok = self.model.forward_features(
                xb, input_chans=input_chans, return_all_tokens=True
            )  # (b, 1 + C*a, D)
            tok = tok[:, 1:, :].reshape(xb.shape[0], C, a, -1).mean(dim=2)  # keep channels
            embs.append(tok.reshape(xb.shape[0], -1).float().cpu().numpy())
        return np.concatenate(embs, axis=0)
