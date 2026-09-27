#!/usr/bin/env bash
#SBATCH --job-name=layer-corrector-eval
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_top_20260924/repo
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_layer_20260925/eval_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_layer_20260925
PROJECT=$BASE/eval/outerwear_top_20260924/repo
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$TASK/deps:$BASE/eval/outerwear_vton15_20260925/deps:$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

"$PYTHON" "$PROJECT/ai_fashion_recommender/scripts/eval_layer_corrector.py" \
  --dataset-root "$BASE/eval/deepfashion_mm" \
  --checkpoint "$TASK/weights/schp-lip-20-int8-static.onnx" \
  --cache-dir "$TASK/cache" \
  --review-dir "$TASK/review_allcases" \
  --unlabeled-review-dir "$TASK/review_allcases" \
  --output "$TASK/output/layer_corrector_allcases_cv.csv"
