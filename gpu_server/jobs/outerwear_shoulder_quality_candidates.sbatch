#!/usr/bin/env bash
#SBATCH --job-name=outerwear-shoulder-candidates
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/shoulder_candidates_%j.log
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
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TASK/output/shoulder_candidates"

for specification in \
  'fur07123:fur07123_pocket_release.png:MS6697352_clean.jpg' \
  'longcoat01582:longcoat01582_occlusion_aware.png:MS6697352_clean.jpg' \
  'puffer05412:puffer05412_occlusion_aware.png:MS6595807_clean.jpg' \
  'denim05047:denim05047_occlusion_aware.png:MS7050470_clean.jpg'
do
  IFS=: read -r case_id person garment <<< "$specification"
  for seed in 7 123; do
    "$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/eval_catvton_upper_morphology.py" \
      --person "$TASK/input/$person" \
      --garment "$TASK/input/$garment" \
      --out "$TASK/output/shoulder_candidates/${case_id}_seed${seed}.png" \
      --catvton-repo "$BASE/26-2_Modeling_Fashion/third_party/CatVTON" \
      --variant standard --steps 50 --seed "$seed" \
      --garment-cache-dir "$TASK/catvton_morph_garment_cache"
  done
done
