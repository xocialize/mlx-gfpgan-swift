# mlx-gfpgan-swift

MLX-Swift port of **[GFPGAN v1.4](https://github.com/TencentARC/GFPGAN)** — blind face
restoration (U-Net degradation removal + StyleGAN2 "clean" decoder with channel-split SFT) —
wrapped as an MLXEngine `imageRestore` package with an Apple-Vision detect → align →
restore → paste-back pipeline.

```
Sources/GFPGANMLXCore   the network: GFPGANv1Clean + StyleGAN2GeneratorCSFT (NHWC, isomorphic
                        to gfpganv1_clean_arch.py / stylegan2_clean_arch.py)
Sources/MLXGFPGAN       the ModelPackage: Vision face detect/align (FFHQ 5-point template),
                        per-face restore, feathered paste-back, strength dial (contract 1.30.0)
Sources/Gate            gfpgan-gate — parity gates vs the PyTorch oracle (CPU stream) +
                        --bench / --fp16 / --bf16 dtype gates (GPU stream)
Sources/AlignGate       gfpgan-align-gate — Vision-vs-facexlib agreement gate (crop IoU)
Sources/Validate        gfpgan-validate — authoritative split footprint through the real engine
oracle/                 torch oracle: convert.py · gen_goldens.py (114 goldens) · verify.py ·
                        gen_align_fixtures.py (facexlib ground truth) · publish.py
```

## Weights

`mlx-community/GFPGANv1.4-fp32` (348.6 MB fp32, `params_ema`, 285 tensors, MLX NHWC layout,
stored noise buffers included). Local conversion: `oracle/convert.py`.

**fp32 deliberately.** fp16 collapses end-to-end (cosine ≈ −0.3, PSNR 8 dB — measured, the
dtype gate exists for exactly this); bf16 gates clean (48 dB) but the weights are small
enough that restoration follows the sibling-package fp32 precedent.

## Gates (all green, 2026-07-27, M5 Max)

| Gate | Result |
|---|---|
| S0 key contract | 285/285 tensors, 87,143,276 params, strict both ways |
| S1 primitives | 7/7 (bilinear, modulated conv ±demod, StyleConv, ToRGB) rel ≤ 3e-6 |
| S2 blocks | 2/2 (ResBlock down/up) rel ≤ 2e-6 |
| S3 full model | 94/94 per-stage taps on 512² rand + real face, final image rel 6.6e-6 |
| Align (vs facexlib) | 7/7 faces, crop IoU 0.86–0.97, landmark dist ≤ 0.11×interocular |
| Conformance | 12/12 offline (MAT, CAN, manifest, strength, footprint, align math) |
| Validate (real engine) | floor 0.36 GB · peak 1.81 GB · act 1.45 GB · 1.7 s @1297×1920 ×2 faces |

Noise doctrine: every parity number uses the checkpoint's **stored noise buffers**
(`randomize_noise=False`) — deterministic on both sides, no cross-framework RNG. The package
defaults to the same; `randomizeNoise: true` restores upstream's stochastic default.

## Run the gates

```bash
swift run gfpgan-gate --s0 oracle/converted/GFPGANv1.4/model.safetensors
swift run gfpgan-gate --all oracle/goldens oracle/converted/GFPGANv1.4/model.safetensors
swift run gfpgan-gate --bf16 oracle/goldens oracle/converted/GFPGANv1.4/model.safetensors
swift run gfpgan-align-gate oracle/align_fixtures
swift run gfpgan-validate oracle/converted/GFPGANv1.4/model.safetensors photo.png
```

## License

Apache-2.0 (port code and weights). Upstream carries third-party carve-outs (NVIDIA
StyleGAN2, DFDNet) that this port never touches: clean arch only, and the v1.4 decoder prior
is BasicSR's from-scratch StyleGAN2, not NVIDIA's. Residual: FFHQ training data
(CC-BY-NC-SA dataset compilation). See `Sources/GFPGANMLXCore/GFPGAN.swift` header.
