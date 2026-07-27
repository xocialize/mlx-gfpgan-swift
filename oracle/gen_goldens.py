"""GFPGAN oracle — per-sub-op goldens for the Swift port.

fp32, CPU-torch, numpy-seeded, C-contiguous, PyTorch NCHW. The Swift gate transposes to
NHWC, runs, transposes back, compares.

Noise doctrine: every golden runs `randomize_noise=False`, i.e. the checkpoint's stored
`stylegan_decoder.noises.noise{0..14}` buffers. That is an upstream-supported mode, ships
with the weights, and makes both sides bit-deterministic — no cross-framework RNG to
reconcile. (Upstream's default is fresh noise per call; the Swift package offers both and
defaults to stored. The learned per-layer noise weights are tiny, so the visual delta is
negligible — but parity REQUIRES the stored mode.)

The full-model taps come from a hand-replicated forward (upstream mutates activations with
`F.leaky_relu_`, which poisons hook-captured tensors). The replica is SELF-CHECKING: its
final image is asserted bit-equal against a direct `model(x)` call before anything is saved.

Run:  .venv/bin/python gen_goldens.py
Out:  goldens/*.npy  +  goldens/MANIFEST.txt  +  goldens/*.png (eyeball)
"""
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_grad_enabled(False)

# --- import the upstream archs without pulling basicsr/cv2/facexlib ---------------------
UP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")


class _Registry:
    def register(self, *a, **k):
        return lambda cls: cls


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_stub("basicsr")
_stub("basicsr.utils")
_stub("basicsr.utils.registry", ARCH_REGISTRY=_Registry())
_stub("basicsr.archs")
# no-op is safe: load_state_dict(strict=True) overwrites every parameter afterwards
_stub("basicsr.archs.arch_util", default_init_weights=lambda *a, **k: None)

import importlib.util

_pkg = types.ModuleType("gfp_archs")
_pkg.__path__ = [os.path.join(UP, "gfpgan", "archs")]
sys.modules["gfp_archs"] = _pkg
for name in ("stylegan2_clean_arch", "gfpganv1_clean_arch"):
    spec = importlib.util.spec_from_file_location(
        f"gfp_archs.{name}", os.path.join(UP, "gfpgan", "archs", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"gfp_archs.{name}"] = mod
    spec.loader.exec_module(mod)

A = sys.modules["gfp_archs.gfpganv1_clean_arch"]

# --- model: the exact GFPGANer('clean') construction ------------------------------------
STEM = "GFPGANv1.4"
OUT = "goldens"
os.makedirs(OUT, exist_ok=True)

raw = torch.load(f"weights/{STEM}.pth", map_location="cpu", weights_only=False)
sd = raw["params_ema" if "params_ema" in raw else "params"]
model = A.GFPGANv1Clean(
    out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
    fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1,
    sft_half=True)
model.load_state_dict(sd, strict=True)
model.eval()

manifest = []


def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(os.path.join(OUT, name + ".npy"), a)
    manifest.append(f"{name + '.npy':44s} {str(a.shape):26s} "
                    f"min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}")
    print(f"  saved {name}.npy  {tuple(a.shape)}")


def dump(name, t):
    save(name, t.detach().cpu().numpy())


def seeded(seed, *shape):
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.standard_normal(shape, dtype=np.float32))


dec = model.stylegan_decoder
stored_noise = [getattr(dec.noises, f"noise{i}") for i in range(dec.num_layers)]

print("\n=== 1. Bilinear interpolate — the raw op, both directions ===")
xb = seeded(6001, 1, 8, 16, 16)
dump("bilinear_in", xb)
dump("bilinear_up2_out", F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False))
dump("bilinear_down2_out", F.interpolate(xb, scale_factor=0.5, mode="bilinear", align_corners=False))

print("\n=== 2. ResBlock — down and up variants ===")
xr = seeded(6002, 1, 32, 64, 64)
dump("resblock_down_in", xr)
dump("resblock_down_out", model.conv_body_down[0](xr))    # 32 -> 64 ch, /2

xu = seeded(6003, 1, 256, 8, 8)
dump("resblock_up_in", xu)
dump("resblock_up_out", model.conv_body_up[0](xu))        # 256 -> 256 ch, x2

print("\n=== 3. ModulatedConv2d — demodulated (StyleConv) and plain (ToRGB) ===")
xm = seeded(6004, 1, 512, 4, 4)
sm = seeded(6005, 1, 512)
dump("modconv_in", xm)
dump("modconv_style", sm)
dump("modconv_demod_out", dec.style_conv1.modulated_conv(xm, sm))
dump("modconv_plain_out", dec.to_rgb1.modulated_conv(xm, sm))

print("\n=== 4. StyleConv (stored noise0) and ToRGB (with skip) ===")
dump("styleconv_out", dec.style_conv1(xm, sm, noise=stored_noise[0]))
skip = seeded(6006, 1, 3, 4, 4)
dump("torgb_skip_in", skip)
# to_rgb1 has upsample=False; exercise the upsample+skip path with to_rgbs[0] instead
x8 = seeded(6007, 1, 512, 8, 8)
s8 = seeded(6008, 1, 512)
sk4 = seeded(6009, 1, 3, 4, 4)
dump("torgb_up_in", x8)
dump("torgb_up_style", s8)
dump("torgb_up_skip", sk4)
dump("torgb_up_out", dec.to_rgbs[0](x8, s8, sk4))
dump("torgb1_out", dec.to_rgb1(xm, sm, None))

print("\n=== 5. Full model — replicated forward with per-stage taps (self-checked) ===")


def full_taps(tag, x):
    dump(f"{tag}_in", x)

    # encoder
    feat = F.leaky_relu(model.conv_body_first(x), negative_slope=0.2)
    dump(f"{tag}_feat_first", feat)
    unet_skips = []
    for i in range(model.log_size - 2):
        feat = model.conv_body_down[i](feat)
        unet_skips.insert(0, feat)
        dump(f"{tag}_down{i}", feat)
    feat = F.leaky_relu(model.final_conv(feat), negative_slope=0.2)
    dump(f"{tag}_final_conv", feat)

    # style code
    style_code = model.final_linear(feat.view(feat.size(0), -1))
    dump(f"{tag}_style_code", style_code)
    style_code = style_code.view(style_code.size(0), -1, model.num_style_feat)

    # decode + SFT conditions
    conditions = []
    for i in range(model.log_size - 2):
        feat = feat + unet_skips[i]
        feat = model.conv_body_up[i](feat)
        dump(f"{tag}_up{i}", feat)
        scale = model.condition_scale[i](feat)
        conditions.append(scale.clone())
        shift = model.condition_shift[i](feat)
        conditions.append(shift.clone())
        dump(f"{tag}_scale{i}", scale)
        dump(f"{tag}_shift{i}", shift)

    # stylegan decoder (input_is_latent=True, stored noise), tapped per level
    latent = style_code  # ndim==3 branch: per-layer latents pass through untouched
    out = dec.constant_input(latent.shape[0])
    out = dec.style_conv1(out, latent[:, 0], noise=stored_noise[0])
    dump(f"{tag}_dec_conv1", out)
    skip = dec.to_rgb1(out, latent[:, 1])

    i = 1
    for li, (conv1, conv2, noise1, noise2, to_rgb) in enumerate(
            zip(dec.style_convs[::2], dec.style_convs[1::2], stored_noise[1::2],
                stored_noise[2::2], dec.to_rgbs)):
        out = conv1(out, latent[:, i], noise=noise1)
        if i < len(conditions):
            out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
            out_sft = out_sft * conditions[i - 1] + conditions[i]
            out = torch.cat([out_same, out_sft], dim=1)
        out = conv2(out, latent[:, i + 1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)
        dump(f"{tag}_dec_out{li}", out)
        dump(f"{tag}_dec_skip{li}", skip)
        i += 2

    dump(f"{tag}_image", skip)
    return skip


def reference_forward(x):
    image, _ = model(x, return_rgb=False, randomize_noise=False)
    return image


# 5a. seeded input in the [-1,1] input contract
g = np.random.default_rng(7100)
x_rand = torch.from_numpy((g.random((1, 3, 512, 512), dtype=np.float32) * 2 - 1))
img_replica = full_taps("full_rand", x_rand)
img_direct = reference_forward(x_rand)
assert torch.equal(img_replica, img_direct), "replicated forward diverged from model(x)!"
print("  ✅ replica == model(x) bit-exact (rand)")

# 5b. a real aligned face (upstream test fixture), production preprocessing
face = Image.open(os.path.join(UP, "inputs", "cropped_faces", "Julia_Roberts_crop.png"))
face = face.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
face_np = np.asarray(face, dtype=np.float32) / 255.0          # HWC RGB [0,1]
x_face = torch.from_numpy(np.ascontiguousarray(
    ((face_np - 0.5) / 0.5).transpose(2, 0, 1)))[None]        # NCHW [-1,1]
img_replica = full_taps("full_face", x_face)
img_direct = reference_forward(x_face)
assert torch.equal(img_replica, img_direct), "replicated forward diverged from model(x)!"
print("  ✅ replica == model(x) bit-exact (face)")

# eyeball PNGs
for tag, img in (("full_face_in", x_face), ("full_face_out", img_direct)):
    arr = img[0].numpy().transpose(1, 2, 0)
    arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, tag + ".png"))
print("  saved eyeball PNGs (full_face_in/out)")

with open(os.path.join(OUT, "MANIFEST.txt"), "w") as f:
    f.write("GFPGAN v1.4 PyTorch goldens — fp32, CPU, PyTorch NCHW, C-contiguous.\n")
    f.write(f"checkpoint: weights/{STEM}.pth  (state dict under raw['params_ema'])\n")
    f.write("constructor: GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,\n")
    f.write("             num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)\n")
    f.write("noise: stored buffers (randomize_noise=False) — REQUIRED for parity\n")
    f.write("input contract: RGB, 512x512, [-1,1] (img/255 - 0.5)/0.5; output [-1,1] clamp\n\n")
    f.write("\n".join(manifest) + "\n")

print(f"\n✅ {len(manifest)} goldens written to {OUT}/")
