#!/usr/bin/env bash
#SBATCH --job-name=fitta-web
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --oversubscribe
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/data1/dsl01
#SBATCH --output=/mnt/data1/dsl01/logs/fitta_web_%j.out
#SBATCH --error=/mnt/data1/dsl01/logs/fitta_web_%j.out
set -euo pipefail

BASE=/mnt/data1/dsl01
PROJECT=${FITTA_PROJECT:-$BASE/26-2_Modeling_Fashion}
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
WEB_HOST=${FASHION_WEB_HOST:-0.0.0.0}
WEB_PORT=${FASHION_WEB_PORT:-8000}

cd "$PROJECT"

export HOME=$BASE
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache
export MPLCONFIGDIR=$BASE/.cache/matplotlib
export FASHION_DATASETS_DIR=$PROJECT/datasets
export FASHION_PRODUCTS_CSV=$PROJECT/ai_fashion_recommender/data/products_musinsa_enriched.csv
export FASHION_ATTRIBUTE_HEADS_PATH=$PROJECT/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export FASHION_ENABLE_VTON=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$MPLCONFIGDIR" "$BASE/logs"

echo "=== FITTA service / node=$(hostname) / job=${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "프로젝트: $PROJECT"
echo "GPU worker API: http://$WEB_HOST:$WEB_PORT (private; 192.168.0.110 gateway only)"

exec "$PYTHON" -m uvicorn \
  --app-dir "$PROJECT/web" \
  --host "$WEB_HOST" \
  --port "$WEB_PORT" \
  --workers 1 \
  app:app
