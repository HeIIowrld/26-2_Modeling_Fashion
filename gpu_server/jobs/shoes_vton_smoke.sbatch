#!/usr/bin/env bash
#SBATCH --job-name=shoes-vton-smoke
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/shoes_vton_20260919/repo
#SBATCH --output=/mnt/data1/dsl01/eval/shoes_vton_20260919/smoke_%j.log
set -euo pipefail
B=/mnt/data1/dsl01
TASK=$B/eval/shoes_vton_20260919
export PYTHONPATH=$B/web-site-packages:$B/site-packages:$TASK/repo/ai_fashion_recommender/src
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export FASHION_DATASETS_DIR=$B/releases/fitta_current/datasets
export MPLCONFIGDIR=$TASK/matplotlib
cd "$TASK/repo"
$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 ai_fashion_recommender/scripts/smoke_shoe_tryon.py \
  --people "$B/eval/vton_fit_20260914/people" \
  --catalog "$B/releases/fitta_current/ai_fashion_recommender/data/products_musinsa_enriched.csv" \
  --model "$TASK/flux-klein-4b" --output "$TASK/out" --shoe-manifest "$TASK/shoe_inputs/manifest.json"
