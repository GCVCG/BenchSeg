"""R3 (ESWA R1.4): speed of XMem2's permanent memory vs XMem's original memory, same network and code.

Per scene, frames and the first M non-empty SeTR-MLA seeds are laid out exactly as in track_pipeline.py,
then XMem2 runs once per memory mode, back to back, alternating which mode goes first. Time is XMem2's
own model-only timer (per-frame step + post-processing; excludes model loading and image I/O).
Needs an idle GPU. Appends to --out:
  partition,scene,frames,M,mode,proc_s,preload_s
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_frame_iou import split_scene_frame  # noqa: E402
from track_pipeline import list_images, run_tracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--seed_preds_dir", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--n_seeds", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--venv_py", required=True)
    ap.add_argument("--tracker_dir", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    by_scene = defaultdict(list)
    for f in list_images(a.img_dir):
        base = os.path.splitext(f)[0]
        scene, frame, _ = split_scene_frame(base)
        by_scene[scene].append((frame or 0, base, f))
    new = not os.path.exists(a.out)
    done = set() if new else {(r["partition"], r["scene"], r["M"], r["mode"]) for r in csv.DictReader(open(a.out))}
    out = open(a.out, "a", newline="")
    w = csv.writer(out)
    if new:
        w.writerow(["partition", "scene", "frames", "M", "mode", "proc_s", "preload_s"])
    for k, (scene, frames) in enumerate(sorted(by_scene.items(), key=lambda t: str(t[0]))):
        todo = {(M, md) for M in a.n_seeds for md in (["permanent"] if M == 1 else ["permanent", "original"])
                if (a.partition, str(scene), str(M), md) not in done}
        if not todo:
            continue
        frames.sort()
        fdir = os.path.join(a.tmp, "frames")
        shutil.rmtree(a.tmp, ignore_errors=True)
        os.makedirs(fdir)
        cands = []
        for i, (_, base, f) in enumerate(frames):
            Image.open(os.path.join(a.img_dir, f)).convert("RGB").save(os.path.join(fdir, f"{i:06d}.jpg"))
            m = (np.array(Image.open(os.path.join(a.seed_preds_dir, base + ".png")).convert("L")) > 0)
            if m.any():
                cands.append((i, m.astype(np.uint8) * 255))
        for M in a.n_seeds:
            sdir = os.path.join(a.tmp, f"seeds_M{M}")
            os.makedirs(sdir)
            for i, m in cands[:M]:
                Image.fromarray(m).save(os.path.join(sdir, f"{i:06d}.png"))
            # M=1: both modes keep the single seed permanently -> identical work, time once
            modes = ["permanent"] if M == 1 else (["permanent", "original"] if k % 2 == 0 else ["original", "permanent"])
            for mode in modes:
                if (M, mode) not in todo:
                    continue
                odir = os.path.join(a.tmp, f"out_{M}_{mode}")
                os.makedirs(odir, exist_ok=True)
                cmd = [a.venv_py, "process_video.py", "--video", fdir, "--masks", sdir, "--output", odir, "--print_fps"]
                if mode == "original":
                    cmd.append("--original_memory_mechanism")
                rc, so, se = run_tracker(cmd, a.tracker_dir, 300 + 1.5 * len(frames), odir)
                proc = re.search(r"TOTAL PROCESSING TIME: ([\d.]+)s", so)
                pre = re.search(r"TOTAL PRELOADING TIME: ([\d.]+)s", so)
                if rc != 0 or not proc:
                    print(f"{scene} M{M} {mode}: FAILED\n{se[-500:]}", flush=True)
                    continue
                w.writerow([a.partition, scene, len(frames), M, mode, proc.group(1), pre.group(1)])
                out.flush()
                print(f"{a.partition}/{scene} M{M} {mode}: {len(frames) / float(proc.group(1)):.1f} fps", flush=True)
    shutil.rmtree(a.tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
