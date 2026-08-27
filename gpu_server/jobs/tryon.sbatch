#!/usr/bin/env bash
#SBATCH --job-name=catvton
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl01
#SBATCH --output=/mnt/data1/dsl01/logs/catvton_%j.out
#SBATCH --error=/mnt/data1/dsl01/logs/catvton_%j.out
#
# 제출:  sbatch /mnt/data1/dsl01/jobs/tryon.sbatch   (master에서는 /data1/dsl01/jobs/...)
# 확인:  squeue -u $USER      취소: scancel <JobID>
#
# ※ 이 클러스터의 특성 (확인 완료)
#   - 같은 저장소가 master=/data1/dsl01, 계산노드=/mnt/data1/dsl01 로 경로가 다르다.
#     Slurm 지시자는 계산 노드에서 해석되므로 반드시 /mnt/data1 을 쓴다.
#   - HOME 환경변수가 /data1/dsl01 로 상속되는데 계산 노드엔 그 경로가 없다. 직접 덮는다.
#   - 계산 노드에는 인터넷이 없다. 모델은 master에서 미리 받아둔 캐시를 오프라인으로 쓴다.
#   - 계산 노드 기본 파이썬은 3.8 이라 프로젝트 요구(3.9+)에 못 미친다.
#     공유 저장소에 넣어둔 3.11 을 절대경로로 부른다.
set -euo pipefail

B=/mnt/data1/dsl01
PROJ=$B/26-2_Modeling_Fashion
PY=$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11

export HOME=$B
export PYTHONPATH=$B/site-packages:$PROJ/ai_fashion_recommender/src
# HF_HUB_OFFLINE 은 일부러 켜지 않는다.
# 켜면 transformers 가 "저장소에 없는 파일"(timm/ViT-B-16-SigLIP 의 config.json 등)을
# 캐시된 .no_exist 표시로 처리하지 못하고 OSError 를 낸다. 플래그를 빼면 연결이
# 6초 만에 실패한 뒤 캐시로 대체되어 정상 동작한다. 계산 노드는 인터넷이 없으므로
# 실제 다운로드는 어차피 일어나지 않는다.
export HF_HOME=$B/hf_cache
export              # 인터넷 없음. 캐시에 없으면 조용히 굶지 말고 바로 실패하게
export MPLCONFIGDIR=$B/.cache/matplotlib
export FASHION_DATASETS_DIR=$PROJ/datasets
# 배포 모델: models/CHECKSUMS.json 이 augmented(22,341 crop, sha256 3c560c9d…)를
# "최종 채택 배포 모델"로, fashion_attribute_heads.pt(baseline)를 "rollback용"으로 적고 있다.
# 그런데 config.py 기본값은 baseline 쪽이라 채택 모델을 명시적으로 가리킨다.
export FASHION_ATTRIBUTE_HEADS_PATH=$PROJ/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR" "$B/logs"

echo "=== 노드 $(hostname) / JobID ${SLURM_JOB_ID:-none} / $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo

cd "$PROJ/ai_fashion_recommender"
$PY scripts/test_tryon.py --vton --count "${COUNT:-2}"

echo
echo "=== 결과 파일 ==="
ls -lt "$PROJ/ai_fashion_recommender/outputs" | head -20
