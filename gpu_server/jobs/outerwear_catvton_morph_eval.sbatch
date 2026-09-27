#!/usr/bin/env bash
#SBATCH --job-name=coat-morph-ab
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_top_20260924/catvton_morph_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_top_20260924
PROJECT=${CATVTON_MORPH_PROJECT:-$TASK/repo}
PERSON=${CATVTON_MORPH_PERSON:?Set CATVTON_MORPH_PERSON}
GARMENT=${CATVTON_MORPH_GARMENT:?Set CATVTON_MORPH_GARMENT}
OUTPUT=${CATVTON_MORPH_OUTPUT:?Set CATVTON_MORPH_OUTPUT}
VARIANT=${CATVTON_MORPH_VARIANT:?Set CATVTON_MORPH_VARIANT}
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11

export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASHION_DATASETS_DIR=$BASE/releases/fitta_current/datasets
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=$TASK/matplotlib XDG_CACHE_HOME=$TASK/cache
mkdir -p "$(dirname "$OUTPUT")" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$PROJECT"

EXTRA_ARGS=()
if [ "$VARIANT" = "custom-target" ] || [ "$VARIANT" = "custom-target-standard" ]; then
  EXTRA_ARGS+=(
    --pose-target-mask "${CATVTON_MORPH_POSE_TARGET:?Set CATVTON_MORPH_POSE_TARGET}"
    --old-inner-proposal-mask "${CATVTON_MORPH_INNER_PROPOSAL:?Set CATVTON_MORPH_INNER_PROPOSAL}"
    --proposal-source "${CATVTON_MORPH_PROPOSAL_SOURCE:?Set CATVTON_MORPH_PROPOSAL_SOURCE}"
    --protected-mask "${CATVTON_MORPH_PROTECTED:?Set CATVTON_MORPH_PROTECTED}"
  )
fi

"$PYTHON" ai_fashion_recommender/scripts/eval_catvton_upper_morphology.py \
  --person "$PERSON" --garment "$GARMENT" --out "$OUTPUT" \
  --catvton-repo "$BASE/26-2_Modeling_Fashion/third_party/CatVTON" \
  --variant "$VARIANT" --steps "${CATVTON_MORPH_STEPS:-50}" \
  --seed "${CATVTON_MORPH_SEED:-42}" \
  --garment-cache-dir "$TASK/catvton_morph_garment_cache" \
  "${EXTRA_ARGS[@]}"
