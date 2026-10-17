#!/bin/bash
# R3 (ESWA R1.4) ablation: XMem++ permanent memory vs XMem's original memory in FoodMem
# (SeTR-MLA seeds + XMem2), M seed masks ("first" selection). M=1 is identical by construction
# (one seed -> one permanent frame in both modes) and doubles as the reproduction check.
# Runs locally (RTX 3090); W holds the XMem2 copy patched with --original_memory_mechanism.
#   W=/media/amughrabi/NVME_4TB/benchseg_r3 LANES=3 bash xmem_memory_ablation.sh
set -u
cd "$(dirname "$0")"
W=${W:-/media/amughrabi/NVME_4TB/benchseg_r3}; LANES=${LANES:-3}
declare -A IMGS=( [N5K]=n5k_reordered [MTF]=mtf_foodMem_reordered [VF]=vf_reordered [FKIT]=FoodKit_dataset_reordered )
mkdir -p "$W/logs"

job(){ # partition M mode(xmempp|xmem)
  local P=$1 M=$2 MODE=$3 name="FoodMem_M$2"; local extra=""
  [ "$MODE" = xmem ] && { name="${name}_xmem"; extra="--xmem_original_memory"; }
  local out="$W/preds/$P/$name" log="$W/logs/${P}_${name}.log"
  grep -q '^done ->' "$log" 2>/dev/null && ! grep -q 'tracker FAILED' "$log" && { echo "skip $P/$name"; return; }
  rm -rf "$out"
  python3 src/track_pipeline.py --img_dir "$W/data/${IMGS[$P]}/images" --seed_preds_dir "$W/preds/$P/SETR_MLA" \
    --out_dir "$out" --tracker xmem2 --n_seed "$M" --seed_select first \
    --venv_py "$W/venv/bin/python" --tracker_dir "$W/XMem2" --tmp_root "$W/_track_tmp" \
    --tracker_timeout_per_frame 1.5 $extra > "$log" 2>&1
  echo "$P/$name: $(grep -c 'tracker FAILED' "$log") failed scenes, $(ls "$out" | wc -l) masks"
}
export -f job; export W
# small partitions first (all configs), then FoodKit; within each, the M=3/M=9 pairs before M=6
{ for P in N5K VF MTF FKIT; do
    echo "$P 1 xmempp"
    for M in 3 9 6; do echo "$P $M xmempp"; echo "$P $M xmem"; done
  done; } | xargs -P "$LANES" -L 1 bash -c 'declare -A IMGS=( [N5K]=n5k_reordered [MTF]=mtf_foodMem_reordered [VF]=vf_reordered [FKIT]=FoodKit_dataset_reordered ); job "$0" "$1" "$2"'
echo XMEM_ABLATION_DONE
