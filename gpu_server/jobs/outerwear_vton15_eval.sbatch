#!/usr/bin/env bash
#SBATCH --job-name=outer-vton15
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_vton15_20260925
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_vton15_20260925/job_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_vton15_20260925
PERSON=${VTON15_PERSON:?Set VTON15_PERSON}
GARMENT=${VTON15_GARMENT:?Set VTON15_GARMENT}
OUT=${VTON15_OUT:?Set VTON15_OUT}
GARMENT_TYPE=${VTON15_GARMENT_TYPE:?Set VTON15_GARMENT_TYPE to model or flat-lay}
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11

export PYTHONPATH=$TASK/deps:$BASE/web-site-packages:$BASE/site-packages:$TASK/repo/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=8
export MPLCONFIGDIR=$TASK/matplotlib XDG_CACHE_HOME=$TASK/cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

EXTRA_ARGS=()
if [ "${VTON15_MASKED:-0}" = "1" ]; then
  EXTRA_ARGS+=(--masked)
fi
if [ -n "${VTON15_COAT_PROPOSAL:-}" ]; then
  EXTRA_ARGS+=(--coat-proposal-mask "$VTON15_COAT_PROPOSAL")
fi
if [ -n "${VTON15_TARGET_BODY_MASK:-}" ]; then
  EXTRA_ARGS+=(--target-body-mask "$VTON15_TARGET_BODY_MASK")
fi
if [ "${VTON15_STUDIO_BACKGROUND_OUTSIDE_TARGET:-0}" = "1" ]; then
  EXTRA_ARGS+=(--studio-background-outside-target)
fi
if [ -n "${VTON15_DEBUG_PREPROCESS_DIR:-}" ]; then
  EXTRA_ARGS+=(--debug-preprocess-dir "$VTON15_DEBUG_PREPROCESS_DIR")
fi

"$PYTHON" "$TASK/eval_fashn_vton15_outerwear.py" \
  --person "$PERSON" \
  --garment "$GARMENT" \
  --garment-photo-type "$GARMENT_TYPE" \
  --weights-dir "$TASK/weights" \
  --out "$OUT" \
  --steps "${VTON15_STEPS:-30}" \
  --seed "${VTON15_SEED:-42}" \
  --guidance "${VTON15_GUIDANCE:-1.5}" \
  "${EXTRA_ARGS[@]}"
