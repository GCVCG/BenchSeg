"""R3 (ESWA R1.4): how much does the automatically generated seed bound FoodMem?

FoodMem writes the SeTR-MLA mask of the first frame with a non-empty prediction (the seed chosen by
track_pipeline.py with n_seed=1) into XMem2's permanent memory, where it stays fixed. Per scene, relate
the seed's IoU to the IoU of the propagated masks on all other frames.

  python src/seed_quality.py --seed_preds_root <dir with <P>/SETR_MLA>   -> results/seed_quality.csv + summary
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

PARTS = ["FKIT", "MTF", "N5K", "VF"]


def seed_frame(pred_dir, scene_rows):
    for _, r in scene_rows.sort_values("frame").iterrows():
        p = os.path.join(pred_dir, r.filename)
        if os.path.exists(p) and (np.asarray(Image.open(p).convert("L")) > 0).any():
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_preds_root", default="data/preds")
    ap.add_argument("--iou_root", default="results/per_frame_iou")
    ap.add_argument("--out", default="results/seed_quality.csv")
    a = ap.parse_args()
    rows = []
    for p in PARTS:
        setr = pd.read_csv(f"{a.iou_root}/{p}/SETR_MLA.csv")
        fm = pd.read_csv(f"{a.iou_root}/{p}/FoodMem.csv")
        for sc, g in setr.groupby("scene"):
            s = seed_frame(os.path.join(a.seed_preds_root, p, "SETR_MLA"), g)
            if s is None:
                continue
            rest = fm[(fm.scene == sc) & (fm.frame != s.frame)]
            rows.append(dict(partition=p, scene=sc, seed_frame=int(s.frame), seed_iou=s.iou,
                             foodmem_iou=rest.iou.mean(), setr_iou=g[g.frame != s.frame].iou.mean()))
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    r = spearmanr(d.seed_iou, d.foodmem_iou)
    hi, lo = d[d.seed_iou >= 0.9], d[d.seed_iou < 0.8]
    print(f"scenes={len(d)}  Spearman rho={r.correlation:.2f} (p={r.pvalue:.1e})")
    print(f"seed IoU>=0.9: {len(hi)} scenes, FoodMem IoU {hi.foodmem_iou.mean():.3f}")
    print(f"seed IoU<0.8 : {len(lo)} scenes, FoodMem IoU {lo.foodmem_iou.mean():.3f}  "
          f"{lo[['partition', 'scene', 'seed_iou', 'foodmem_iou']].round(3).to_dict('records')}")
    print(f"FoodMem > SeTR-MLA (same frames) in {(d.foodmem_iou > d.setr_iou).mean() * 100:.0f}% of scenes")


if __name__ == "__main__":
    main()
