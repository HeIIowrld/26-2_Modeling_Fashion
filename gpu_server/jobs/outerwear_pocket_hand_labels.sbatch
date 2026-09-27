#!/usr/bin/env bash
#SBATCH --job-name=outerwear-pocket-hand-labels
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/pocket_hand_labels_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_lvton_20260925
PROJECT=$BASE/eval/outerwear_top_20260924/repo
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

mkdir -p "$TASK/output/generated_labels"
"$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/export_fashn_labels.py" \
  --image "$TASK/input/fur07123_occlusion_aware.png" \
  --output "$TASK/output/generated_labels/fur07123_occlusion_aware_fashn.png" \
  --device cpu
"$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/export_fashn_labels.py" \
  --image "$TASK/input/fur07123_pocket_release.png" \
  --output "$TASK/output/generated_labels/fur07123_pocket_release_fashn.png" \
  --device cpu
