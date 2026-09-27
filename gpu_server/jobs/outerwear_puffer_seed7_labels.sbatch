#!/usr/bin/env bash
#SBATCH --job-name=outerwear-puffer-labels
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/puffer_seed7_labels_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_lvton_20260925
PROJECT=$BASE/eval/outerwear_top_20260924/repo
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASHION_DATASETS_DIR=$BASE/releases/fitta_current/datasets
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
"$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/export_fashn_labels.py" \
  --image "$TASK/output/shoulder_candidates/puffer05412_seed7.png" \
  --output "$TASK/output/shoulder_candidates/puffer05412_seed7_fashn.png" \
  --device cpu
