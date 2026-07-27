"""Spot-check converted/GFPGANv1.4/model.safetensors against the source .pth.

Round-trips the layout transforms on every tensor and requires bit-equality —
this is a materialization check as much as a math check (a lazily-saved tensor
would read back as zeros).

Run:  .venv/bin/python verify.py
"""
import numpy as np
import torch
from safetensors.numpy import load_file

sd = torch.load("weights/GFPGANv1.4.pth", map_location="cpu", weights_only=False)["params_ema"]
st = load_file("converted/GFPGANv1.4/model.safetensors")

assert set(st) == set(sd), f"key sets differ: {set(st) ^ set(sd)}"

bad = 0
for k, v in sd.items():
    a = v.detach().numpy().astype(np.float32)
    b = st[k]
    if a.ndim == 4:
        b = np.transpose(b, (0, 3, 1, 2))
    elif a.ndim == 5:
        b = np.transpose(b, (0, 1, 4, 2, 3))
    if not np.array_equal(a, b):
        print(f"  MISMATCH {k}")
        bad += 1
    if np.all(b == 0) and np.any(a != 0):
        print(f"  ZEROED {k}")
        bad += 1

print(f"{'✅' if bad == 0 else '❌'} {len(sd)} tensors checked, {bad} bad")
