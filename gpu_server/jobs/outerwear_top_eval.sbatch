#!/usr/bin/env bash
#SBATCH --job-name=outerwear-top
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_top_20260924/job_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_top_20260924
PROJECT=${OUTERWEAR_TOP_PROJECT:-$TASK/repo}
PERSON_IMAGE=${OUTERWEAR_TOP_PERSON:?Set OUTERWEAR_TOP_PERSON to an outerwear photo}
TOP_IMAGE=${OUTERWEAR_TOP_REFERENCE:?Set OUTERWEAR_TOP_REFERENCE to a top product image}
OUTPUT_IMAGE=${OUTERWEAR_TOP_OUTPUT:-$TASK/output/result_flux_direct.png}
MODEL_PATH=${OUTERWEAR_TOP_MODEL:-$BASE/shared_models/flux2-klein-4b-e7b7dc27}
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11

export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASHION_DATASETS_DIR=$BASE/releases/fitta_current/datasets
MODEL_DIR=$BASE/releases/fitta_current/ai_fashion_recommender/models
export FASHION_ATTRIBUTE_HEADS_PATH=$MODEL_DIR/fashion_attribute_heads_augmented.pt
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=$TASK/matplotlib XDG_CACHE_HOME=$TASK/cache

mkdir -p "$(dirname "$OUTPUT_IMAGE")" "$TASK/debug" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$PROJECT"

EXTRA_ARGS=()
if [ "${OUTERWEAR_TOP_TWO_PASS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--second-pass-catvton)
fi
if [ "${OUTERWEAR_TOP_TWO_STAGE_FLUX:-0}" = "1" ]; then
  EXTRA_ARGS+=(--second-pass-flux)
fi
if [ -n "${OUTERWEAR_TOP_SECOND_STAGE_MASK:-}" ]; then
  EXTRA_ARGS+=(--second-stage-mask "$OUTERWEAR_TOP_SECOND_STAGE_MASK")
fi
if [ -n "${OUTERWEAR_TOP_SHAPE_REFERENCE:-}" ]; then
  EXTRA_ARGS+=(--coatless-shape-reference "$OUTERWEAR_TOP_SHAPE_REFERENCE")
fi
if [ -n "${OUTERWEAR_TOP_COAT_PROPOSAL_MASK:-}" ]; then
  EXTRA_ARGS+=(--coat-proposal-mask "$OUTERWEAR_TOP_COAT_PROPOSAL_MASK")
fi
if [ -n "${OUTERWEAR_TOP_ACCESSORY_MASK:-}" ]; then
  EXTRA_ARGS+=(--accessory-mask "$OUTERWEAR_TOP_ACCESSORY_MASK")
fi
if [ "${OUTERWEAR_TOP_BILATERAL_SLEEVES:-0}" = "1" ]; then
  EXTRA_ARGS+=(--bilateral-sleeves)
fi
if [ "${OUTERWEAR_TOP_ALLOW_UNCONFIRMED:-0}" = "1" ]; then
  EXTRA_ARGS+=(--allow-unconfirmed-outerwear)
fi
if [ "${OUTERWEAR_TOP_FIT_GUIDANCE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--fit-guidance)
fi
if [ "${OUTERWEAR_TOP_RESTORE_STUDIO:-0}" = "1" ]; then
  EXTRA_ARGS+=(--restore-studio-background)
fi
if [ "${OUTERWEAR_TOP_RELEASE_POCKET_HANDS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--release-pocket-hands)
fi
if [ -n "${OUTERWEAR_TOP_OFFICIAL_SEGMENTATION:-}" ]; then
  EXTRA_ARGS+=(--official-segmentation "$OUTERWEAR_TOP_OFFICIAL_SEGMENTATION")
fi
if [ "${OUTERWEAR_TOP_ORACLE_OUTER_ONLY:-0}" = "1" ]; then
  EXTRA_ARGS+=(--oracle-outer-only)
fi
if [ "${OUTERWEAR_TOP_ORACLE_INNER_REFERENCE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--oracle-inner-reference)
fi
if [ "${OUTERWEAR_TOP_PRESERVE_PANTS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--preserve-observed-pants)
fi

"$PYTHON" ai_fashion_recommender/scripts/eval_outerwear_top.py \
  --person "$PERSON_IMAGE" \
  --top-image "$TOP_IMAGE" \
  --model "$MODEL_PATH" \
  --steps "${OUTERWEAR_TOP_STEPS:-4}" \
  --guidance "${OUTERWEAR_TOP_GUIDANCE:-1.0}" \
  --top-fit "${OUTERWEAR_TOP_FIT:-레귤러핏}" \
  --prompt-variant "${OUTERWEAR_TOP_PROMPT_VARIANT:-baseline}" \
  --out "$OUTPUT_IMAGE" \
  --debug-dir "$TASK/debug" \
  "${EXTRA_ARGS[@]}"
