"""Alignment ground truth — facexlib's detect→align, run ONCE offline (#43b: the
unshippable dependency is the ORACLE, not a runtime dep).

For each fixture image, runs the exact upstream helper GFPGANer uses
(`FaceRestoreHelper(upscale=1, face_size=512, det_model='retinaface_resnet50')`) and dumps,
per face:
  - the 5 source landmarks (image px, top-left origin)
  - the 2x3 affine matrix (image → 512² crop, cv2 convention)
  - the cropped 512² face PNG

The Swift `gfpgan-gate --align` compares the Vision-based FaceAlign against these on
crop-quadrilateral IoU + landmark distance. RetinaFace weights download on first run.

Run:  .venv/bin/python gen_align_fixtures.py
Out:  align_fixtures/<image>/{faces.json, face<i>.png}
"""
import json
import os

import cv2
import numpy as np
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

IMAGES = ["upstream/inputs/whole_imgs/00.jpg",
          "upstream/inputs/whole_imgs/10045.png",
          "upstream/inputs/whole_imgs/Blake_Lively.jpg"]
OUT = "align_fixtures"

helper = FaceRestoreHelper(
    upscale_factor=1, face_size=512, crop_ratio=(1, 1),
    det_model="retinaface_resnet50", save_ext="png", use_parse=False,
    device="cpu", model_rootpath="weights/facexlib")

for path in IMAGES:
    name = os.path.splitext(os.path.basename(path))[0]
    odir = os.path.join(OUT, name)
    os.makedirs(odir, exist_ok=True)

    helper.clean_all()
    helper.read_image(cv2.imread(path))
    n = helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    helper.align_warp_face()

    faces = []
    for i, (lm, mat, crop) in enumerate(
            zip(helper.all_landmarks_5, helper.affine_matrices, helper.cropped_faces)):
        cv2.imwrite(os.path.join(odir, f"face{i}.png"), crop)
        faces.append({"landmarks5": np.asarray(lm, dtype=float).tolist(),
                      "affine": np.asarray(mat, dtype=float).tolist()})

    h, w = helper.input_img.shape[:2]
    with open(os.path.join(odir, "faces.json"), "w") as f:
        json.dump({"image": path, "width": w, "height": h, "faces": faces}, f, indent=2)
    print(f"{name}: {n} face(s) -> {odir}/")

print("✅ alignment fixtures written")
