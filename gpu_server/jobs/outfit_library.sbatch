#!/usr/bin/env bash
#SBATCH --job-name=outfit-library
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/data1/dsl01
#SBATCH --output=/mnt/data1/dsl01/logs/outfit_library_%j.log
set -euo pipefail
BASE=/mnt/data1/dsl01
PROJECT=${FITTA_LIBRARY_PROJECT:?Set FITTA_LIBRARY_PROJECT to an isolated checkout}
OUTPUT=${FITTA_LIBRARY_OUTPUT:?Set FITTA_LIBRARY_OUTPUT to a private output directory}
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache MPLCONFIGDIR=$OUTPUT/matplotlib XDG_CACHE_HOME=$OUTPUT/cache
export FASHION_ATTRIBUTE_HEADS_PATH=$BASE/26-2_Modeling_Fashion/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export FASHION_DATASETS_DIR=$BASE/26-2_Modeling_Fashion/datasets
export FASHION_ENABLE_VTON=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
# One writer per run, including when resuming from another Slurm job.
exec 9>"$OUTPUT/.writer.lock"
flock -n 9 || { echo "Another outfit-library job is writing to $OUTPUT" >&2; exit 1; }
cd "$PROJECT"
SCRIPT=ai_fashion_recommender/scripts/precompute_outfits.py
if [ "${FITTA_LIBRARY_STAGE:-all}" = inspect ]; then
  "$PYTHON" ai_fashion_recommender/scripts/inspect_offline_outfits.py \
    --root "${FITTA_LIBRARY_SOURCE:-$OUTPUT}" --output "$OUTPUT/evidence" \
    --people "${FITTA_LIBRARY_PEOPLE:-$BASE/eval/deepfashion_mm/images}"
  exit 0
fi
if [ "${FITTA_LIBRARY_STAGE:-all}" = compare ]; then
  "$PYTHON" "$SCRIPT" index --root "$OUTPUT" --all-contexts
  "$PYTHON" "$SCRIPT" plan --root "$OUTPUT"
  for backend in catvton flux; do
    "$PYTHON" "$SCRIPT" render --root "$OUTPUT" \
      --people "${FITTA_LIBRARY_PEOPLE:-$BASE/eval/deepfashion_mm/images}" \
      --shoe-model "$BASE/eval/shoes_vton_20260919/flux-klein-4b" \
      --evidence "$OUTPUT/evidence" --lower-backend "$backend" --max-renders 2 \
      --reference-name MEN-Jackets_Vests-id_00005047-01_4_full.jpg \
      --combination-ids a103b1ab50a1b35566bc bbde0934cc28bb1c397f
    "$PYTHON" "$SCRIPT" render --root "$OUTPUT" \
      --people "${FITTA_LIBRARY_PEOPLE:-$BASE/eval/deepfashion_mm/images}" \
      --shoe-model "$BASE/eval/shoes_vton_20260919/flux-klein-4b" \
      --evidence "$OUTPUT/evidence" --lower-backend "$backend" --max-renders 12 \
      --combination-ids a103b1ab50a1b35566bc bbde0934cc28bb1c397f
  done
  exit 0
fi
if [ "${FITTA_LIBRARY_STAGE:-all}" != render ]; then
  "$PYTHON" "$SCRIPT" prepare --root "$OUTPUT" --device "${FITTA_LIBRARY_DEVICE:-auto}"
  "$PYTHON" "$SCRIPT" index --root "$OUTPUT" --all-contexts
  "$PYTHON" "$SCRIPT" plan --root "$OUTPUT"
fi
if [ "${FITTA_LIBRARY_STAGE:-all}" != prepare ]; then
  if [ "${FITTA_LIBRARY_STAGE:-all}" = all ]; then
    "$PYTHON" ai_fashion_recommender/scripts/inspect_offline_outfits.py \
      --root "$OUTPUT" --output "$OUTPUT/evidence" \
      --people "${FITTA_LIBRARY_PEOPLE:-$BASE/eval/deepfashion_mm/images}"
  fi
  RENDER_ARGS=()
  if [ -n "${FITTA_LIBRARY_REFERENCE:-}" ]; then
    RENDER_ARGS+=(--reference-name "$FITTA_LIBRARY_REFERENCE")
  fi
  if [ -n "${FITTA_LIBRARY_COMBINATION_IDS:-}" ]; then
    read -r -a COMBINATION_IDS <<< "$FITTA_LIBRARY_COMBINATION_IDS"
    RENDER_ARGS+=(--combination-ids "${COMBINATION_IDS[@]}")
  fi
  "$PYTHON" "$SCRIPT" render --root "$OUTPUT" \
    --people "${FITTA_LIBRARY_PEOPLE:-$BASE/eval/deepfashion_mm/images}" \
    --shoe-model "$BASE/eval/shoes_vton_20260919/flux-klein-4b" \
    --evidence "${FITTA_LIBRARY_EVIDENCE:-$OUTPUT/evidence}" \
    --lower-backend "${FITTA_LIBRARY_LOWER_BACKEND:-catvton}" \
    --shoe-reference-mode "${FITTA_LIBRARY_SHOE_REFERENCE_MODE:-baseline}" \
    --max-renders "${FITTA_LIBRARY_MAX_RENDERS:-0}" "${RENDER_ARGS[@]}"
fi
