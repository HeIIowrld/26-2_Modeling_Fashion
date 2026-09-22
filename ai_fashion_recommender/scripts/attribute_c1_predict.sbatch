#!/usr/bin/env bash
#SBATCH --job-name=attribute-c1
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/data1/dsl01
#SBATCH --output=/mnt/data1/dsl01/logs/attribute_c1_%j.log
set -euo pipefail
BASE=/mnt/data1/dsl01
PROJECT=${C1_PROJECT:?Set C1_PROJECT to the isolated benchmark directory}
CHECKPOINT=${C1_CHECKPOINT:?Set C1_CHECKPOINT to the verified server checkpoint}
EXPECTED=${C1_CHECKPOINT_SHA256:?Set C1_CHECKPOINT_SHA256 to the verified hash}
BACKBONE=${C1_BACKBONE_DIR:?Set C1_BACKBONE_DIR to the server OpenCLIP snapshot directory}
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=$BASE/web-site-packages:$BASE/site-packages:$PROJECT/ai_fashion_recommender/src
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$PROJECT"
"$PYTHON" ai_fashion_recommender/scripts/benchmark_product_attributes.py predict \
  --labels ai_fashion_recommender/reports/attribute_c1/pool.csv \
  --images ai_fashion_recommender/data/cache/attribute_c1/images \
  --checkpoint "$CHECKPOINT" --expected-sha256 "$EXPECTED" --device "${C1_DEVICE:-cuda}" \
  --backbone-dir "$BACKBONE" \
  --output ai_fashion_recommender/reports/attribute_c1/predictions.json
