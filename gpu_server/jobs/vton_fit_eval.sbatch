#!/usr/bin/env bash
#SBATCH --job-name=vton-fit-eval
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/vton_fit_20260914
#SBATCH --output=/mnt/data1/dsl01/logs/vton_fit_eval_%j.out
#SBATCH --error=/mnt/data1/dsl01/logs/vton_fit_eval_%j.out
# VTON 핏 보존·품질 평가. 운영 서비스(fitta-web)와 폴더·캐시를 공유하지 않는다.
#   sbatch --export=ALL,EVAL_ARGS="--smoke" /mnt/data1/dsl01/eval/vton_fit_20260914/eval.sbatch
set -euo pipefail
B=/mnt/data1/dsl01
export EVAL_ROOT=$B/eval/vton_fit_20260914
export EVAL_CATALOG=$B/releases/fitta_current/ai_fashion_recommender/data/products_musinsa_enriched.csv
export EVAL_GARMENT_RAW=$B/releases/fitta_current/datasets/garments/raw
export HOME=$B
export PYTHONPATH=$B/site-packages:$EVAL_ROOT/repo/ai_fashion_recommender/src
export HF_HOME=$B/hf_cache
export MPLCONFIGDIR=$B/.cache/matplotlib
export FASHION_DATASETS_DIR=$B/releases/fitta_current/datasets
export FASHION_ATTRIBUTE_HEADS_PATH=$B/releases/fitta_current/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
echo "=== $(hostname) job=${SLURM_JOB_ID:-none} $(date) args=${EVAL_ARGS:-} ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
$PY $EVAL_ROOT/eval_vton_fit.py ${EVAL_ARGS:-}
echo "=== 끝 $(date) ==="
