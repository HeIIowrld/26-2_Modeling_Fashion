#!/usr/bin/env bash
#SBATCH --job-name=outerwear-pocket-hand-ab
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/pocket_hand_%j.log
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

"$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/eval_catvton_upper_morphology.py" \
  --person "$TASK/input/fur07123_pocket_release.png" \
  --garment "$TASK/input/MS6697352_clean.jpg" \
  --out "$TASK/output/fur07123_pocket_release_catvton_standard.png" \
  --catvton-repo "$BASE/26-2_Modeling_Fashion/third_party/CatVTON" \
  --variant standard --steps 50 --seed 42 \
  --garment-cache-dir "$TASK/catvton_morph_garment_cache"
