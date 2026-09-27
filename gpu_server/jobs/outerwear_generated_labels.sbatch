#!/usr/bin/env bash
#SBATCH --job-name=outerwear-generated-labels
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_lvton_20260925/generated_labels_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_lvton_20260925
PROJECT=$BASE/eval/outerwear_top_20260924/repo
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

mkdir -p "$TASK/output/generated_labels"
for case_id in puffer05412 fur07123 longcoat01582 denim05047
do
  "$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/export_fashn_labels.py" \
    --image "$TASK/output/${case_id}_remove.png" \
    --output "$TASK/output/generated_labels/${case_id}_remove_fashn.png" \
    --device cpu
done
