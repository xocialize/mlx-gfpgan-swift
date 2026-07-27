"""GFPGAN weight conversion: GFPGANv1.4.pth -> safetensors in MLX NHWC layout.

Three things worth knowing, one of which differs from the sibling Restormer port:

1. **Unwrap `raw['params_ema']`.** GFPGAN releases carry BOTH `params` and `params_ema`;
   `GFPGANer` prefers `params_ema` and so do we. (Restormer had only `params`.)
2. Every 4-D tensor takes the SAME `(0,2,3,1)` transpose — this is not an accident:
   conv `(O,I,kH,kW) -> (O,kH,kW,I)`, StyleConv/ToRGB bias `(1,C,1,1) -> (1,1,1,C)`,
   noise buffers `(1,1,H,W) -> (1,H,W,1)`, constant input `(1,512,4,4) -> (1,4,4,512)`.
3. The **5-D ModulatedConv2d** weights `(1,O,I,k,k) -> (1,O,k,k,I)` via `(0,1,3,4,2)`, so
   after style modulation the Swift side reshapes straight to conv layout `(b*O,k,k,I)`.

`style_mlp` is dead at inference (`input_is_latent=True`) but is converted anyway — the key
set stays isomorphic with upstream and the Swift strict load proves it.

Run:  .venv/bin/python convert.py [GFPGANv1.4 ...]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

STEMS = sys.argv[1:] or ["GFPGANv1.4"]

for stem in STEMS:
    src = f"weights/{stem}.pth"
    if not os.path.exists(src):
        print(f"[skip] {src} not present")
        continue

    raw = torch.load(src, map_location="cpu", weights_only=False)
    key = "params_ema" if "params_ema" in raw else "params"
    sd = raw[key]

    out_dir = os.path.join("converted", stem)
    os.makedirs(out_dir, exist_ok=True)

    converted, stats = {}, {"conv4d": 0, "modconv5d": 0, "passthrough": 0}
    for k, v in sd.items():
        a = v.detach().cpu().numpy().astype(np.float32)
        if a.ndim == 4:
            a = np.transpose(a, (0, 2, 3, 1))
            stats["conv4d"] += 1
        elif a.ndim == 5:
            a = np.transpose(a, (0, 1, 3, 4, 2))
            stats["modconv5d"] += 1
        else:
            stats["passthrough"] += 1
        converted[k] = np.ascontiguousarray(a)

    total = sum(int(np.prod(v.shape)) for v in converted.values())
    print(f"=== {stem} ===")
    print(f"  wrapper: raw['{key}']   tensors: {len(converted)}")
    print(f"  transforms: conv4d {stats['conv4d']} · modconv5d {stats['modconv5d']} "
          f"· passthrough {stats['passthrough']}")
    print(f"  params: {total:,}  ({total * 4 / 1e6:.2f} MB fp32)")

    meta = {"format": "pt", "source": f"TencentARC/GFPGAN {stem}.pth ({key})",
            "license": "Apache-2.0 (see LICENSE; clean arch, no NVIDIA-derived code path; "
                       "decoder prior trained from scratch by BasicSR)",
            "layout": "MLX NHWC; conv (O,kH,kW,I); modconv (1,O,kH,kW,I)", "params": str(total)}
    save_file(converted, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    with open(os.path.join(out_dir, "CONVERSION.json"), "w") as f:
        json.dump({"stem": stem, "wrapper": key, "transforms": stats, "params": total}, f, indent=2)
    sz = os.path.getsize(os.path.join(out_dir, "model.safetensors")) / 1e6
    print(f"  written: {out_dir}/model.safetensors ({sz:.2f} MB)\n")
