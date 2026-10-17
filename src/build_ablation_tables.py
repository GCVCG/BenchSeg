"""Build the mask-count (M, first vs random) and memory-size ablation LaTeX tables
from the per-partition ablation_spatial CSVs (all 4 partitions, complete V&F).
Outputs: results/tables/tab_masks_ablation.tex, results/tables/tab_memory_ablation.tex
"""
import csv
import glob
import os
import numpy as np

PARTS = ["N5K", "VF", "MTF", "FKIT"]


def load(pattern, skip_mem=False):
    d = {}
    for f in glob.glob(pattern):
        if skip_mem and "mem" in f:
            continue
        for r in csv.DictReader(open(f)):
            d[(r["method"], r["partition"])] = r
    return d


def meanp(d, m, key="mAP"):
    vs = [float(d[(m, p)][key]) for p in PARTS if (m, p) in d]
    return np.mean(vs) if len(vs) == 4 else None


def style(v, vals):
    """bold best, uline 2nd, italic 3rd among vals."""
    s = sorted(set(round(x, 2) for x in vals if x is not None), reverse=True)
    r = round(v, 2)
    if s and r == s[0]:
        return f"\\textbf{{{v:.2f}}}"
    if len(s) > 1 and r == s[1]:
        return f"\\uline{{{v:.2f}}}"
    if len(s) > 2 and r == s[2]:
        return f"\\textit{{{v:.2f}}}"
    return f"{v:.2f}"


def masks_table():
    d = load("results/ablation_spatial_[NMVF]*.csv", skip_mem=True)
    DISP = {"Y+X2": "YOLO+XMem2", "S+X2": "SegMan+XMem2", "SF+X2": "SegMan-FT+XMem2"}
    L = [r"\begin{table}[htb]", r"\centering", r"\footnotesize",
         r"\caption{\change{Mask-count ablation: number of seed masks $M$ and the seed-selection "
         r"strategy (first $M$ frames vs.\ $M$ random frames within the same budget), for the three "
         r"XMem2 hybrids. Mean mAP (\%) over the four partitions; random reports mean$\pm$std over "
         r"three draws. $M{=}1$ first is the base hybrid. Best per combo in \textbf{bold}.}}",
         r"\label{tab:masks_ablation}", r"\begin{tabular}{llcc}", r"\toprule",
         r"\textbf{Combination} & \textbf{$M$} & \textbf{first} & \textbf{random} \\"]
    for C in ["Y+X2", "S+X2", "SF+X2"]:
        L.append(r"\midrule")
        firsts = [meanp(d, C if M == 1 else f"{C}_M{M}") for M in (1, 3, 6, 9)]
        for i, M in enumerate([1, 3, 6, 9]):
            fm = C if M == 1 else f"{C}_M{M}"
            fv = meanp(d, fm)
            rs = [meanp(d, f"{C}_M{M}_r{dd}") for dd in (0, 1, 2)]
            rs = [x for x in rs if x is not None]
            rtxt = f"${np.mean(rs):.2f}\\pm{np.std(rs, ddof=1):.2f}$" if len(rs) == 3 else "--"
            combo = DISP[C] if i == 0 else ""
            fcell = style(fv, firsts) if fv is not None else "--"
            L.append(f"{combo} & {M} & {fcell} & {rtxt} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("results/tables/tab_masks_ablation.tex", "w").write("\n".join(L) + "\n")
    print("wrote tab_masks_ablation.tex")


def masks_additional_table():
    """Body rows of tab:masks_ablation_additional: Precision/F1/IoU/Accuracy per partition (FKit, MTF,
    N5k, V&F), first-M seeds, mean row + std row per M; best/2nd/3rd per column within each combo."""
    d = load("results/ablation_spatial_[NMVF]*.csv", skip_mem=True)
    order, keys = ["FKIT", "MTF", "N5K", "VF"], ["precision", "f1", "iou", "accuracy"]
    shade = r"\cellcolor{gray!12}"
    L = []
    for c, (C, disp) in enumerate([("Y+X2", "YOLO+XMem2"), ("SF+X2", "SegMan-FT+XMem2")]):
        if c:
            L.append(r"\midrule")
        meth = [C if M == 1 else f"{C}_M{M}" for M in (1, 3, 6, 9)]
        col = {(k, p): [float(d[(m, p)][k]) for m in meth] for k in keys for p in order}
        for i, (M, m) in enumerate(zip((1, 3, 6, 9), meth)):
            sh = shade if M in (3, 9) else ""
            means = [sh + style(float(d[(m, p)][k]), col[(k, p)]) for k in keys for p in order]
            stds = [sh + f"{float(d[(m, p)][k + '_std']):.2f}" for k in keys for p in order]
            lead = rf"\multirow{{8}}{{*}}{{{disp}}}" if i == 0 else ""
            L.append(f"{lead} & {sh}\\multirow[t]{{2}}{{*}}{{{M}}} & " + " & ".join(means) + r" \\")
            L.append(f" & {sh} & " + " & ".join(stds) + r" \\")
    open("results/tables/tab_masks_ablation_additional_body.tex", "w").write("\n".join(L) + "\n")
    print("wrote tab_masks_ablation_additional_body.tex")


def temporal_grouped_table():
    """Body rows of tab:temporal_grouped_method: C_t, FR, dIoU, sigmaIoU per partition (FKit, MTF, N5k,
    V&F) + Global (mean over partitions; std across partitions), first-M seeds, from the ablation per-frame
    IoU with the same metric profile as the main temporal table (corrected_minimal)."""
    import build_temporal_table as bt
    import temporal_metrics as tm
    order, keys = ["FKIT", "MTF", "N5K", "VF"], ["continuity", "flicker", "drift", "sigma"]
    higher = {"continuity"}
    shade = r"\cellcolor{gray!12}"

    def mark(v, vals, up):
        s = sorted(set(round(x, 1) for x in vals), reverse=up)
        r, t = round(v, 1), f"{v:.1f}"
        return next((f"{w}{{{t}}}" for w, x in zip((r"\textbf", r"\underline", r"\textit"), s) if r == x), t)

    L = []
    for c, (C, disp) in enumerate([("Y+X2", "YOLO+XMem2"), ("SF+X2", "SegMan-FT+XMem2")]):
        if c:
            L.append(r"\midrule")
        meth = [C if M == 1 else f"{C}_M{M}" for M in (1, 3, 6, 9)]
        paths = [f"results/ablation_per_frame_iou/{p}/{m}.csv" for p in order for m in meth]
        rows = {(r["method"], r["partition"]): r for r in bt.build_table(paths, tm.TemporalConfig(), "corrected_minimal")}
        val = {m: {(k, p): 100 * rows[(m, p)][k] for k in keys for p in order} for m in meth}
        std = {m: {(k, p): 100 * rows[(m, p)][k + "_std"] for k in keys for p in order} for m in meth}
        glob_ = {m: {k: np.mean([val[m][(k, p)] for p in order]) for k in keys} for m in meth}
        gstd = {m: {k: np.std([val[m][(k, p)] for p in order], ddof=1) for k in keys} for m in meth}
        for i, (M, m) in enumerate(zip((1, 3, 6, 9), meth)):
            sh = shade if M in (3, 9) else ""
            means = [sh + mark(val[m][(k, p)], [val[x][(k, p)] for x in meth], k in higher) for k in keys for p in order]
            means += [sh + mark(glob_[m][k], [glob_[x][k] for x in meth], k in higher) for k in keys]
            stds = [sh + f"{std[m][(k, p)]:.1f}" for k in keys for p in order] + [sh + f"{gstd[m][k]:.1f}" for k in keys]
            lead = rf"\multirow{{8}}{{*}}{{{disp}}}" if i == 0 else ""
            L.append(f"{lead} & {sh}\\multirow[t]{{2}}{{*}}{{{M}}} & " + " & ".join(means) + r" \\")
            L.append(f" & {sh} & " + " & ".join(stds) + r" \\")
    open("results/tables/tab_temporal_grouped_method_body.tex", "w").write("\n".join(L) + "\n")
    print("wrote tab_temporal_grouped_method_body.tex")


def memory_table():
    d = load("results/ablation_spatial_mem_*.csv")
    rows = [("4", "FoodMem_mem4"), ("10 (default)", "FoodMem"),
            ("20", "FoodMem_mem20"), ("40", "FoodMem_mem40")]
    L = [r"\begin{table}[htb]", r"\centering", r"\footnotesize",
         r"\caption{\change{Memory-size ablation: XMem2 working-memory size "
         r"(\texttt{max\_mid\_term\_frames}) in FoodMem. Mean over the four partitions. "
         r"Performance is essentially flat across sizes, indicating FoodMem is robust to the "
         r"working-memory budget.}}",
         r"\label{tab:memory_ablation}", r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Working memory} & \textbf{mAP} & \textbf{Recall} & \textbf{IoU} \\", r"\midrule"]
    maps = [meanp(d, m, "mAP") for _, m in rows]
    for lbl, m in rows:
        vals = [meanp(d, m, k) for k in ("mAP", "recall", "iou")]
        if any(v is None for v in vals):
            continue
        cells = [style(vals[0], maps)] + [f"{v:.2f}" for v in vals[1:]]
        L.append(f"{lbl} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("results/tables/tab_memory_ablation.tex", "w").write("\n".join(L) + "\n")
    print("wrote tab_memory_ablation.tex")


if __name__ == "__main__":
    os.makedirs("results/tables", exist_ok=True)
    masks_table()
    masks_additional_table()
    temporal_grouped_table()
    memory_table()
