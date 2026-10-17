"""Rows of tab:permanent_memory (R3, ESWA R1.4) from results/ablation_permanent_memory.csv (spatial,
src/ablation_metrics.py) and results/xmem_timing.csv (src/xmem_timing.py). Prints the LaTeX body."""
import pandas as pd

PARTS = ["N5K", "MTF", "VF", "FKIT"]
d = pd.read_csv("results/ablation_permanent_memory.csv").drop_duplicates(["method", "partition"], keep="last")
t = pd.read_csv("results/xmem_timing.csv")
t["tot"] = t.proc_s + t.preload_s
fps = t.groupby(["M", "mode"]).apply(lambda g: g.frames.sum() / g.tot.sum(), include_groups=False)


def row(method, M, mode):
    g = d[d.method == method].set_index("partition")
    assert set(PARTS) <= set(g.index), (method, list(g.index))
    cells = [f"{g.loc[p, 'mAP']:.2f}" for p in PARTS]
    cells += [f"{g.loc[PARTS, 'mAP'].mean():.2f}", f"{g.loc[PARTS, 'iou'].mean():.2f}"]
    cells.append(f"{fps[(M, mode)]:.1f}" if (M, mode) in fps.index else "--")
    return cells


lines = ["1 & both & identical & " + " & ".join(row("FoodMem_M1", 1, "permanent")) + r" \\", r"\midrule"]
for M in (3, 6, 9):
    # shaded second row with a negative \multirow, as in the other tables (shading would hide a forward multirow)
    lines.append(r" & XMem2 & permanent & " + " & ".join(row(f"FoodMem_M{M}", M, "permanent")) + r" \\")
    lines.append(rf"\rowcolor{{gray!12}} \multirow{{-2}}{{*}}{{{M}}} & XMem & original & "
                 + " & ".join(row(f"FoodMem_M{M}_xmem", M, "original")) + r" \\")
    if M != 9:
        lines.append(r"\midrule")
print("\n".join(lines))
