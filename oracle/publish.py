"""Publish the converted GFPGAN v1.4 weights to mlx-community.

  GFPGANv1.4 -> mlx-community/GFPGANv1.4-fp32

Run:  .venv/bin/python publish.py [--dry-run]
"""
import os
import sys

from huggingface_hub import HfApi

API = HfApi()
HERE = os.path.dirname(os.path.abspath(__file__))

REPO = "mlx-community/GFPGANv1.4-fp32"

CARD = """---
license: apache-2.0
library_name: mlx
tags:
- mlx
- face-restoration
- image-restoration
- stylegan2
- gfpgan
base_model: TencentARC/GFPGAN
---

# GFPGANv1.4-fp32 (MLX)

[GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) blind face restoration converted to MLX
NHWC safetensors for Apple Silicon. 87,143,276 parameters (`params_ema`), fp32.

- **Architecture:** `GFPGANv1Clean` — degradation-removal U-Net + StyleGAN2 (clean) decoder
  with channel-split SFT. 512×512 aligned face crops, RGB in [-1, 1].
- **Layout:** MLX NHWC. Conv `(O,kH,kW,I)`; the 5-D modulated-conv kernels `(1,O,kH,kW,I)`;
  every other 4-D tensor (biases, stored noise buffers, constant input) takes the same
  `(0,2,3,1)` transpose. Keys mirror the upstream state dict exactly.
- **Noise:** the checkpoint's stored noise buffers ship in the file (`randomize_noise=False`
  is the deterministic parity mode).
- **dtype:** fp32. fp16 is DISQUALIFIED — the e2e gate collapses (cosine ≈ −0.3, PSNR 8 dB);
  bf16 gates clean at 48 dB if you need it, but restoration ships fp32 here.

## License

Apache-2.0. The upstream repo carries third-party carve-outs (NVIDIA StyleGAN2,
DFDNet) that do NOT touch this artifact: the clean architecture contains no NVIDIA-derived
code, and the v1.4 decoder prior was trained from scratch by BasicSR
(`StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth`, Apache-2.0), not from NVIDIA's FFHQ
weights. Trained on FFHQ (dataset compilation CC-BY-NC-SA — the standard unsettled
dataset-to-weights question that applies to every face model).

## Consume

Swift (Apple Silicon): `mlx-gfpgan-swift` — `GFPGANMLXCore.GFPGANv1Clean` +
`MLXGFPGAN.GFPGANRestorePackage` (MLXEngine `imageRestore`, Vision-based detect/align/paste).

Converted with `oracle/convert.py` in that repo; parity vs the PyTorch reference:
key contract 285/285 tensors, per-stage taps 94/94 ≤ 5e-4 relative (fp32, CPU stream).
"""


def main():
    dry = "--dry-run" in sys.argv
    src = os.path.join(HERE, "converted", "GFPGANv1.4", "model.safetensors")
    assert os.path.exists(src), src
    print(f"{'[dry-run] ' if dry else ''}{src} -> {REPO}")
    if dry:
        print(CARD)
        return
    API.create_repo(REPO, repo_type="model", exist_ok=True)
    API.upload_file(path_or_fileobj=src, path_in_repo="model.safetensors",
                    repo_id=REPO, repo_type="model")
    API.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print("✅ published")


main()
