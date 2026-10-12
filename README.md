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

**Engine-materialized weights read in place since 0.2.1 (AB-A-0112).** MLXEngine downloads the declared
`weightSources` into its model store before `load()`; `load()` reads that copy (flat or snapshot layout) instead of
fetching a second one through swift-transformers' `HubApi`. It also reads `<root>/models/<org>/<name>/`, where earlier
versions downloaded, so existing installs aren't fetched again. `HubApi` remains only the backstop outside the engine
(no store, or nothing materialized); an explicit `weightsURL` bypasses all of it.

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

## GPU numerics: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

GFPGAN has 21 such convs per 512² face:

- the U-Net ResBlocks and SFT condition heads, at 128/256 channels and 64²/128²;
- six StyleGAN2 modulated convs, at 512/256/128 channels from 64² to 256². At batch 1 these are
  dense conv2d calls.

The S0–S3 gates pin the CPU device, so this never showed.

Both kinds of site now take a route (`model.convRoute`, type `GFPGANConvRoute`). **Default
`.conv3d`.**

Measurements: aligned 512² face, production fp32, stored noise, GPU against the CPU lane.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Output | 1.0e-3 · max 1.06e-2 · 2 levels | **7.5e-7 · exact-class** |
| Forward time, 512² | 266 ms | 259 ms |

- The raw loss is below 8-bit visibility. The route is still the default because it is free here
  and it makes the GPU lane parity-clean: S3's max-rel ≤ 5e-4 would read 9.99e-3 on the raw GPU
  lane.
- GFPGAN has no TF32-eligible matmuls, so the route alone closes the gap.
- Environment override: `GFPGAN_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- Gate: `GFPGAN_LANE=1 swift test -c release -Xswiftc -enable-testing --filter GPULaneTests`.

## 16 bits end to end (0.3.0)

The surface declares `ToolDescriptor.rawImageFormats = [.rawBGRA8, .rawRGBA16]` (MLXToolKit contract 1.52.0,
AB-T-0230). A `.rawRGBA16` image comes back `.rawRGBA16`; a PNG or TIFF deeper than 8 bits comes back as a 16-bit PNG.
- **Every pixel hop of the face pass runs at 16 bits** (`restoreDeep`). The FFHQ crop is warped into a 16-bit canvas
  (`FaceAlign.warpDeep`), and the restored face is rounded to 16 bits and warped back into one. The feather mask is warped
  at 16 bits (`FaceAlign.warpedMaskDeep`), and the composite runs in float. An 8-bit face inside a 16-bit frame would be
  an 8-bit plateau inside the paste mask, a visible tone step in a sky behind a portrait.
- Faces are found on the image's 8-bit rendering, as the 8-bit path finds them. No face: the input comes back as it was.
- **G-6 gate** (2026-10-12, `mlxengine-image/WIP/g6-16bit-gate`, `faces`): on a portrait in front of a 16-bit sky,
  the 16-bit output's smooth-area flat-neighbour share in the band just inside the paste edge is 0.042, against
  0.90 for the 8-bit output, and the edge shows no step at ×8 contrast. What remains is GFPGAN's own re-render of the
  background inside its crop (about 2 steps darker), feathered, the same at 8 bits.
- 8-bit inputs take the 8-bit path: 62/62 outputs (7 images × png, png at strength 0.5, rawBGRA8, jpeg, with NAFNet and
  the other face model alongside) byte-identical to an output of v0.2.1, in release builds on the GPU.

## Faces in a fixed order (0.3.0)

Vision returns the same faces in no fixed order from run to run, and the paste-back composites them in order. Where two
faces' paste masks overlap, the output depended on the run: across 13 runs of v0.2.1 on GFPGAN's `Blake_Lively`, every
variant came out two ways. `FaceAlign.detectFaces` now sorts faces left to right by the left eye (then top to bottom),
so the output is deterministic and matches one of the two.

## License

Apache-2.0 (port code and weights). Upstream carries third-party carve-outs (NVIDIA
StyleGAN2, DFDNet) that this port never touches: clean arch only, and the v1.4 decoder prior
is BasicSR's from-scratch StyleGAN2, not NVIDIA's. Residual: FFHQ training data
(CC-BY-NC-SA dataset compilation). See `Sources/GFPGANMLXCore/GFPGAN.swift` header.
