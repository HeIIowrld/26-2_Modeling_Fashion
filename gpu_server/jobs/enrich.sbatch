#!/usr/bin/env bash
#SBATCH --job-name=enrich
#SBATCH --partition=partition1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/data1/dsl01
#SBATCH --output=/mnt/data1/dsl01/logs/enrich_%j.out
#SBATCH --error=/mnt/data1/dsl01/logs/enrich_%j.out
set -euo pipefail
B=/mnt/data1/dsl01; PROJ=$B/26-2_Modeling_Fashion
export HOME=$B PYTHONPATH=$B/site-packages:$PROJ/ai_fashion_recommender/src
# HF_HUB_OFFLINE 은 일부러 켜지 않는다. 켜면 transformers 가 "저장소에 없는 파일"
# (timm/ViT-B-16-SigLIP 의 config.json 등)을 캐시된 .no_exist 표시로 처리하지 못하고
# OSError 를 낸다. 빼면 연결이 6초 만에 실패한 뒤 캐시로 대체되어 정상 동작한다.
export HF_HOME=$B/hf_cache MPLCONFIGDIR=$B/.cache/matplotlib
export FASHION_DATASETS_DIR=$PROJ/datasets
export FASHION_ATTRIBUTE_HEADS_PATH=$PROJ/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"
echo "CPU 실행"
cd $PROJ/ai_fashion_recommender
$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 scripts/enrich_catalog.py
