#!/usr/bin/env bash
#SBATCH --job-name=outerwear-zones-catvton
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/zones_catvton_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_lvton_20260925
PROJECT=$BASE/eval/outerwear_top_20260924/repo
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASHION_DATASETS_DIR=$BASE/releases/fitta_current/datasets
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=$TASK/matplotlib XDG_CACHE_HOME=$TASK/cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

for specification in \
  "denim05047:MS7050470_clean.jpg" \
  "longcoat01582:MS6697352_clean.jpg" \
  "puffer05412:MS6595807_clean.jpg" \
  "fur07123:MS6697352_clean.jpg"
do
  case_id=${specification%%:*}
  garment=${specification#*:}
  "$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/eval_catvton_upper_morphology.py" \
    --person "$TASK/input/${case_id}_occlusion_aware.png" \
    --garment "$TASK/input/$garment" \
    --out "$TASK/output/${case_id}_zones_catvton_standard.png" \
    --catvton-repo "$BASE/26-2_Modeling_Fashion/third_party/CatVTON" \
    --variant standard --steps 50 --seed 42 \
    --garment-cache-dir "$TASK/catvton_morph_garment_cache"
done
