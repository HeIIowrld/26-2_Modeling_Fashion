#!/usr/bin/env bash
#SBATCH --job-name=vton-gate-audit
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/vton_review_20260915
#SBATCH --output=/mnt/data1/dsl01/logs/vton_gate_audit_%j.out
#SBATCH --error=/mnt/data1/dsl01/logs/vton_gate_audit_%j.out
# 원본 평가/운영은 읽기만 한다. 합성을 다시 실행하지 않는 품질 게이트 감사.
set -euo pipefail
B=/mnt/data1/dsl01
export EVAL_ROOT=$B/eval/vton_review_20260915
export EVAL_CATALOG=$B/releases/fitta_current/ai_fashion_recommender/data/products_musinsa_enriched.csv
export EVAL_GARMENT_RAW=$B/releases/fitta_current/datasets/garments/raw
export PYTHONPATH=$B/site-packages:$EVAL_ROOT/repo/ai_fashion_recommender/src
export HF_HOME=$B/hf_cache
export XDG_CACHE_HOME=$B/.cache
export MPLCONFIGDIR=$EVAL_ROOT/out/matplotlib
export FASHION_DATASETS_DIR=$B/releases/fitta_current/datasets
export FASHION_ATTRIBUTE_HEADS_PATH=$B/releases/fitta_current/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
if [[ "${VTON_AUDIT_MODE:-gates}" == "upper-torso" ]]; then
    $PY $EVAL_ROOT/eval_vton_fit.py \
        --selection-refs $B/eval/vton_fit_20260914/out/refs.json \
        --mask-mode upper-torso --offset "${VTON_OFFSET:-0}" --limit "${VTON_LIMIT:-0}"
else
    $PY $EVAL_ROOT/audit_vton_gates.py --baseline-root $B/eval/vton_fit_20260914
fi
