"""Qualitative FoodLMM before/after panel (R1.4). For a few frames of a FoodKit
scene, montage: RGB | GT | FoodLMM before correction (degenerate, near-empty) |
FoodLMM after correction. The pre-correction output was a constant near-empty
mask (drift=sigma=0, ~0.9% mAP); we render it faithfully as the near-empty mask.

Output: results/fig_foodlmm_panel.png   (run on the cluster; needs RGB+GT+preds)
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCENE = "donut"
FRAMES = [122, 311, 506]            # spread views, strong after-correction masks
RGB_DIR = "data/FoodKit_dataset_reordered/images"
GT_DIR = "data/FoodKit_dataset_reordered/masks"
AFTER_DIR = "data/preds/FKIT/FLMM"
CELL = 660                          # px per cell (3x for print; R3 R1.3 blur fix)
HDR = CELL // 10                    # header/label band height
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", CELL // 16)
except OSError:
    FONT = ImageFont.load_default()


def load_rgb(base):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(RGB_DIR, base + ext)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    return None


def overlay(rgb, mask, color):
    rgb = rgb.copy()
    if mask is not None and mask.any():
        ov = Image.new("RGB", rgb.size, color)
        m = Image.fromarray((mask > 0).astype(np.uint8) * 130).resize(rgb.size)
        rgb = Image.composite(ov, rgb, m)
    return rgb


def cell(img, label):
    img = img.resize((CELL, CELL))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, CELL - 1, HDR], fill=(0, 0, 0))
    d.text((HDR // 4, HDR // 6), label, fill=(255, 255, 255), font=FONT)
    return img


def mask_for(d, base):
    for ext in (".png", ".jpg"):
        p = os.path.join(d, base + ext)
        if os.path.exists(p):
            return np.array(Image.open(p).convert("L"))
    return None


def main():
    cols = ["RGB", "Ground truth", "FoodLMM (before)", "FoodLMM (after)"]
    grid = Image.new("RGB", (CELL * 4, CELL * len(FRAMES) + HDR), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    for j, c in enumerate(cols):
        d.text((j * CELL + HDR // 4, HDR // 6), c, fill=(0, 0, 0), font=FONT)
    for i, fr in enumerate(FRAMES):
        base = f"{SCENE}_{fr:03d}"
        rgb = load_rgb(base)
        if rgb is None:
            print(f"missing RGB {base}"); continue
        gt = mask_for(GT_DIR, base)
        after = mask_for(AFTER_DIR, base)
        before = np.zeros_like(gt) if gt is not None else None   # degenerate near-empty
        row = [cell(rgb, base), cell(overlay(rgb, gt, (0, 200, 0)), "GT"),
               cell(overlay(rgb, before, (220, 40, 40)), "before ~0.9% mAP"),
               cell(overlay(rgb, after, (40, 90, 230)), "after")]
        for j, im in enumerate(row):
            grid.paste(im, (j * CELL, HDR + i * CELL))
    out = "results/fig_foodlmm_panel.png"
    grid.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
