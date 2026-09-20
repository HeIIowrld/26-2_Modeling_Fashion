#!/usr/bin/env bash
#SBATCH --job-name=fitta-web
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --oversubscribe
# 시간이 끝나면 작업을 놓고 우선순위 최하위로 다시 줄을 선다. 8시간이면 하루 세 번
# 이 추첨을 겪는데, 같은 노드의 다른 사용자는 12~24시간을 쓴다. 같은 24시간으로 맞춰
# 재배치 횟수를 하루 한 번으로 줄인다. 더 늘리지 않는 건 공용 노드이기 때문이다.
#SBATCH --time=24:00:00
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
# 검증된 DPM++ 25스텝 프리셋으로 순위별 배치 결과의 대기 시간을 줄인다.
export FASHION_VTON_PRESET=${FASHION_VTON_PRESET:-fast}
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
