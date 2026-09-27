#!/usr/bin/env bash
#SBATCH --job-name=outer-lvton
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_lvton_20260925
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/job_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_lvton_20260925
FASHN=$BASE/eval/outerwear_vton15_20260925
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$TASK/deps:$TASK/external:$FASHN/deps:$FASHN/repo/src:$BASE/web-site-packages:$BASE/site-packages
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export MPLCONFIGDIR=$TASK/matplotlib XDG_CACHE_HOME=$TASK/cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TASK/output"

if [[ -n "${LVTON_MANIFEST:-}" ]]; then
  "$PYTHON" "$TASK/external/eval_layering_vton_batch.py" \
    --manifest "$LVTON_MANIFEST" \
    --model-dir "$TASK/qwen_base" \
    --external-source-dir "$TASK/external" \
    --pose-weights-dir "$TASK/checkpoints" \
    --steps "${LVTON_STEPS:-30}" \
    --seed "${LVTON_SEED:-42}" \
    --text-device cpu
else
  PERSON=${LVTON_PERSON:?Set LVTON_PERSON}
  GARMENT=${LVTON_GARMENT:?Set LVTON_GARMENT}
  OUT=${LVTON_OUT:?Set LVTON_OUT}
  DESCRIPTION=${LVTON_DESCRIPTION:?Set LVTON_DESCRIPTION}
  "$PYTHON" "$TASK/external/eval_layering_vton_outerwear.py" \
    --person "$PERSON" \
    --garment "$GARMENT" \
    --model-dir "$TASK/qwen_base" \
    --external-source-dir "$TASK/external" \
    --pose-weights-dir "$TASK/checkpoints" \
    --output "$OUT" \
    --description "$DESCRIPTION" \
    --steps "${LVTON_STEPS:-30}" \
    --seed "${LVTON_SEED:-42}" \
    --text-device cpu
fi
