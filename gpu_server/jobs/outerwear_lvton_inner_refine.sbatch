#!/usr/bin/env bash
#SBATCH --job-name=lvton-inner-refine
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/inner_refine_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_lvton_20260925
PROJECT=$BASE/eval/outerwear_top_20260924/repo
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
MODEL=$BASE/shared_models/flux2-klein-4b-e7b7dc27

export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=$TASK/matplotlib XDG_CACHE_HOME=$TASK/cache
mkdir -p "$TASK/output" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
EXTRA_ARGS=()
if [ "${INNER_REFINE_BOX:-0}" = "1" ]; then
  EXTRA_ARGS+=(--fill-component-box)
fi
if [ "${INNER_REFINE_NO_REPEAT:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no-repeat-graphics)
fi

"$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/refine_layered_inner.py" \
  --original "$TASK/input/MEN-Jackets_Vests-id_00005047-01_4_full.jpg" \
  --candidate "$TASK/input/lvton_denim05047_composite.png" \
  --source-labels "$TASK/input/MEN-Jackets_Vests-id_00005047-01_4_full_segm.png" \
  --label-source official-oracle \
  --inner-label 1 \
  --exclude-label 0 --exclude-label 5 --exclude-label 13 \
  --exclude-label 14 --exclude-label 15 \
  --protected-mask "$TASK/input/dior_control_denim05047_two_flux_protected.png" \
  --product "$TASK/input/MS7050470_clean.jpg" \
  --model "$MODEL" \
  --margin-ratio "${INNER_REFINE_MARGIN:-0.012}" \
  --seed 42 --steps 4 \
  --output "${INNER_REFINE_OUTPUT:-$TASK/output/denim05047_lvton_inner_refined.png}" \
  "${EXTRA_ARGS[@]}"
