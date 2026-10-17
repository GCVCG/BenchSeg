"""Per-scene seed->track orchestrator for hybrid methods (FoodMem, *+XMem2/SAM2).

For each scene (an ordered frame sequence), it:
  1. lays out the scene's frames as frame_000000.jpg, frame_000001.jpg, ...
  2. seeds the first --n_seed frames with a segmentor's predicted mask
     (binarised foreground), named frame_000000.png, ...
  3. runs the tracker (XMem2 process_video.py or SAM2 sam2_track_from_mask.py)
     to propagate across the whole scene
  4. maps the propagated masks back to <scene>_<frame>.png in --out_dir.

Runs inside the python310 SIF; calls the tracker's own venv python via subprocess
(same container, so the venv resolves).
"""
import argparse, glob, os, re, shutil, signal, subprocess, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_frame_iou import split_scene_frame  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def run_tracker(cmd, cwd, timeout, odir, tries=3):
    """XMem2 intermittently deadlocks (no CPU, no progress). With a timeout, kill the whole process
    group (its image-saver child too) and retry on a clean output dir. Returns (returncode, stdout, stderr).
    Observed cause: an image-saver worker dies and the main process waits on it forever."""
    for attempt in range(1, tries + 1):
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
            return p.returncode, out, err
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            print(f"  tracker hung for {timeout:.0f}s (attempt {attempt}/{tries}) -> retry", flush=True)
            shutil.rmtree(odir, ignore_errors=True)
            os.makedirs(odir)
    return -1, "", "tracker hung on every attempt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--seed_preds_dir", required=True, help="segmentor masks named <scene>_<frame>.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tracker", required=True, choices=["xmem2", "sam2", "sam3"])
    ap.add_argument("--n_seed", type=int, default=1)
    ap.add_argument("--seed_select", choices=["first", "random"], default="first",
                    help="which n_seed non-empty frames to seed from: the first M, or M random (same budget)")
    ap.add_argument("--seed_rng", type=int, default=0, help="base RNG seed for --seed_select random")
    ap.add_argument("--venv_py", required=True, help="path to tracker venv python")
    ap.add_argument("--tracker_dir", required=True, help="tracker repo dir")
    ap.add_argument("--xmem_ckpt", default="saves/XMem.pth")
    ap.add_argument("--xmem_max_mid", type=int, default=None, help="XMem2 max_mid_term_frames (working memory size); None=default 10")
    ap.add_argument("--xmem_mem_every", type=int, default=None, help="XMem2 mem_every (memory update interval); None=default 5")
    ap.add_argument("--xmem_min_mid", type=int, default=None, help="XMem2 min_mid_term_frames; None=default 5 (must be < max_mid)")
    ap.add_argument("--xmem_original_memory", action="store_true",
                    help="XMem's original memory: only the first seed is permanent, later seeds enter working memory")
    ap.add_argument("--sam2_ckpt", default="")
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--tmp_root", default="data/_track_tmp")
    ap.add_argument("--tracker_timeout_per_frame", type=float, default=0,
                    help="per-scene tracker timeout = 300s + this*frames, then kill+retry (0 = no timeout)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    # unique temp root per (method, partition) so concurrent jobs don't collide
    a.tmp_root = os.path.join(a.tmp_root, a.out_dir.strip("/").replace("/", "_"))
    shutil.rmtree(a.tmp_root, ignore_errors=True)
    # group images by scene, ordered by frame
    by_scene = defaultdict(list)
    for f in list_images(a.img_dir):
        base = os.path.splitext(f)[0]
        scene, frame, _ = split_scene_frame(base)
        by_scene[scene].append((frame if frame is not None else 0, base, f))
    for s in by_scene:
        by_scene[s].sort(key=lambda t: t[0])

    for scene, frames in by_scene.items():
        tmp = os.path.join(a.tmp_root, str(scene))
        fdir, sdir, odir = (os.path.join(tmp, x) for x in ("frames", "seeds", "out"))
        for d in (fdir, sdir, odir):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        # lay out all frames with integer names (works for both XMem2 and SAM2)
        # zero-pad names so XMem2/SAM's sorted(os.listdir()) keeps numeric frame order
        # (unpadded "10.jpg" sorts before "2.jpg" and breaks multi-seed mask matching)
        idx_to_base = {}
        for i, (frame, base, f) in enumerate(frames):
            idx_to_base[i] = base
            Image.open(os.path.join(a.img_dir, f)).convert("RGB").save(os.path.join(fdir, f"{i:06d}.jpg"))
        # candidate seed frames = those with a NON-empty predicted mask (in frame order)
        candidates = []  # (i, mask_uint8)
        for i, (frame, base, f) in enumerate(frames):
            sp = os.path.join(a.seed_preds_dir, base + ".png")
            if os.path.exists(sp):
                m = (np.array(Image.open(sp).convert("L")) > 0).astype(np.uint8) * 255
                if m.any():
                    candidates.append((i, m))
        # select n_seed of them: the first M, or M random within the same budget
        if a.seed_select == "random" and len(candidates) > a.n_seed:
            # per-scene deterministic RNG so the draw is reproducible across reruns
            rng = np.random.default_rng(a.seed_rng * 1_000_003 + (hash(str(scene)) & 0xFFFFFFFF))
            pick = sorted(rng.choice(len(candidates), size=a.n_seed, replace=False).tolist())
            chosen = [candidates[k] for k in pick]
        else:
            chosen = candidates[:a.n_seed]
        first_seed_idx = chosen[0][0] if chosen else None
        for i, m in chosen:
            Image.fromarray(m).save(os.path.join(sdir, f"{i:06d}.png"))

        def fill_empty():
            # write all-zero masks for frames with no output (so the scene is complete)
            for i, (frame, base, f) in enumerate(frames):
                op = os.path.join(a.out_dir, base + ".png")
                if not os.path.exists(op):
                    sz = Image.open(os.path.join(a.img_dir, f)).size
                    Image.fromarray(np.zeros((sz[1], sz[0]), np.uint8)).save(op)

        if not list_images(sdir):
            print(f"scene {scene}: no non-empty seed -> all-empty masks", flush=True)
            fill_empty()
            shutil.rmtree(tmp, ignore_errors=True)
            continue

        if a.tracker == "xmem2":
            cmd = [a.venv_py, "process_video.py", "--video", os.path.abspath(fdir),
                   "--masks", os.path.abspath(sdir), "--output", os.path.abspath(odir)]
            if a.xmem_max_mid is not None:
                cmd += ["--max_mid_term_frames", str(a.xmem_max_mid)]
            if a.xmem_mem_every is not None:
                cmd += ["--mem_every", str(a.xmem_mem_every)]
            if a.xmem_min_mid is not None:
                cmd += ["--min_mid_term_frames", str(a.xmem_min_mid)]
            if a.xmem_original_memory:
                cmd += ["--original_memory_mechanism"]
        elif a.tracker == "sam2":
            cmd = [a.venv_py, os.path.abspath(os.path.join(os.path.dirname(__file__), "sam2_track_from_mask.py")),
                   "--video_path", os.path.abspath(fdir), "--mask_dir", os.path.abspath(sdir),
                   "--out_dir", os.path.abspath(odir), "--checkpoint", a.sam2_ckpt, "--config", a.sam2_cfg,
                   "--n_masks", "1", "--frame_idx", str(first_seed_idx or 0)]
        else:  # sam3: checkpoint auto-resolves from the HF cache (load_from_HF)
            cmd = [a.venv_py, os.path.abspath(os.path.join(os.path.dirname(__file__), "sam3_track_from_mask.py")),
                   "--video_path", os.path.abspath(fdir), "--mask_dir", os.path.abspath(sdir),
                   "--out_dir", os.path.abspath(odir),
                   "--n_masks", "1", "--frame_idx", str(first_seed_idx or 0)]
        timeout = 300 + a.tracker_timeout_per_frame * len(frames) if a.tracker_timeout_per_frame > 0 else None
        rc, _, err = run_tracker(cmd, a.tracker_dir, timeout, odir)
        if rc != 0:
            print(f"scene {scene}: tracker FAILED -> all-empty\n{err[-800:]}", flush=True)
            fill_empty()
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        # collect outputs (xmem2 may write to out/masks/)
        src = odir
        if os.path.isdir(os.path.join(odir, "masks")):
            src = os.path.join(odir, "masks")
        for of in list_images(src):
            m = re.search(r"\d+", of)
            if not m:
                continue
            i = int(m.group())
            base = idx_to_base.get(i)
            if base is None:
                continue
            mask = (np.array(Image.open(os.path.join(src, of)).convert("L")) > 0).astype(np.uint8) * 255
            Image.fromarray(mask).save(os.path.join(a.out_dir, base + ".png"))
        fill_empty()  # any frames the tracker skipped -> empty (keep scene complete)
        print(f"scene {scene}: {len(frames)} frames tracked", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"done -> {a.out_dir} ({len(list_images(a.out_dir))} masks)", flush=True)


if __name__ == "__main__":
    main()
