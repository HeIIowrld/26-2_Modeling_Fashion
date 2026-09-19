#!/usr/bin/env bash
# 사람 6명 × 신발 4개 확대 검증. FASHION_SHOE_PROMPT_FILE 을 주면 프롬프트 A/B 실행이 된다.
#SBATCH --job-name=shoes-vton-broad
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/shoes_vton_20260919/repo
#SBATCH --output=/mnt/data1/dsl01/eval/shoes_vton_20260919/broad_%j.log
set -uo pipefail
B=/mnt/data1/dsl01
TASK=$B/eval/shoes_vton_20260919
export PYTHONPATH=$B/web-site-packages:$B/site-packages:$TASK/repo/ai_fashion_recommender/src
export HF_HOME=$B/hf_cache
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export FASHION_DATASETS_DIR=$B/releases/fitta_current/datasets
export MPLCONFIGDIR=$TASK/matplotlib
OUT=${SHOES_BROAD_OUT:-$TASK/out_broad}
PROMPT=${SHOES_BROAD_PROMPT:-}
cd "$TASK/repo"
echo "=== start $(date) out=$OUT prompt=${PROMPT:-기본}"
$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 ai_fashion_recommender/scripts/smoke_shoe_tryon.py \
  --people "$B/eval/vton_fix_20260915/people_dfmm" --people-limit 6 --shoe-limit 4 --skip-outfit \
  --catalog "$B/releases/fitta_current/ai_fashion_recommender/data/products_musinsa_enriched.csv" \
  --model "$TASK/flux-klein-4b" --output "$OUT" --shoe-manifest "$TASK/shoe_inputs/manifest_broad.json" \
  ${PROMPT:+--prompt-file "$PROMPT"}
echo "=== exit=$? $(date)"
