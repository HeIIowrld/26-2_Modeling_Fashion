#!/usr/bin/env bash
#SBATCH --job-name=vton-fix-eval
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=07:00:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/vton_fix_20260915
#SBATCH --output=/mnt/data1/dsl01/logs/vton_fix_eval_%j.out
#SBATCH --error=/mnt/data1/dsl01/logs/vton_fix_eval_%j.out
# 2026-09-15 후속 해결 작업의 격리 GPU 평가. 운영(fitta-web)과 폴더·캐시를 공유하지 않고
# 기존 평가 폴더(vton_fit_20260914)는 읽기만 한다.
#   sbatch --export=ALL,MODE=replay                vton_fix_eval.sbatch   # 기존 480쌍 품질 검사 재생
#   sbatch --export=ALL,MODE=dfmm                  vton_fix_eval.sbatch   # DeepFashion 착장 판정 정답 비교
#   sbatch --export=ALL,MODE=synth,EVAL_ARGS="--mask-mode upper-agnostic" vton_fix_eval.sbatch
set -euo pipefail
B=/mnt/data1/dsl01
export EVAL_ROOT=$B/eval/vton_fix_20260915
export EVAL_CATALOG=$B/releases/fitta_current/ai_fashion_recommender/data/products_musinsa_enriched.csv
export EVAL_GARMENT_RAW=$B/releases/fitta_current/datasets/garments/raw
export HOME=$B
export PYTHONPATH=$B/site-packages:$EVAL_ROOT/repo/ai_fashion_recommender/src
export HF_HOME=$B/hf_cache
export XDG_CACHE_HOME=$B/.cache
export MPLCONFIGDIR=$EVAL_ROOT/out/matplotlib
export FASHION_DATASETS_DIR=$B/releases/fitta_current/datasets
export FASHION_ATTRIBUTE_HEADS_PATH=$B/releases/fitta_current/ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$B/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
SCRIPTS=$EVAL_ROOT/repo/ai_fashion_recommender/scripts
echo "=== $(hostname) job=${SLURM_JOB_ID:-none} mode=${MODE:-synth} $(date) args=${EVAL_ARGS:-} ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
case "${MODE:-synth}" in
    replay)
        $PY $SCRIPTS/replay_tryon_quality.py $B/eval/vton_fit_20260914 \
            --out $EVAL_ROOT/out/quality_replay_baseline.jsonl ${EVAL_ARGS:-} ;;
    dfmm)
        $PY $SCRIPTS/eval_outfit_gates_dfmm.py --root $B/eval/deepfashion_mm \
            --out $EVAL_ROOT/out/dfmm_outfit.jsonl ;;
    synth)
        $PY $EVAL_ROOT/eval_vton_fit.py \
            --selection-refs $B/eval/vton_fit_20260914/out/refs.json ${EVAL_ARGS:-} ;;
    multi)
        # 사용자당 GPU 제한 때문에 GPU 한 장에서 여러 합성 실험을 병렬로 돌린다(합성 1개 VRAM ≈ 4GB).
        # MULTI_ARGS는 ';'로 구분한다. 첫 실행이 레퍼런스 캐시를 채운 뒤 나머지를 띄워 캐시 쓰기 경합을 피한다.
        IFS=';' read -ra RUNS <<< "${MULTI_ARGS:?}"
        if [[ "${WITH_ANALYSIS:-0}" == 1 ]]; then
            MODE=replay bash "$0" > $EVAL_ROOT/out/multi_${SLURM_JOB_ID:-local}_replay.log 2>&1 &
            MODE=dfmm bash "$0" > $EVAL_ROOT/out/multi_${SLURM_JOB_ID:-local}_dfmm.log 2>&1 &
        fi
        for i in "${!RUNS[@]}"; do
            log=$EVAL_ROOT/out/multi_${SLURM_JOB_ID:-local}_$i.log
            $PY $EVAL_ROOT/eval_vton_fit.py \
                --selection-refs $B/eval/vton_fit_20260914/out/refs.json ${RUNS[$i]} > "$log" 2>&1 &
            [[ $i == 0 ]] && until grep -q "\[레퍼런스\]\|Traceback" "$log" 2>/dev/null; do sleep 10; done
        done
        wait ;;
esac
echo "=== 끝 $(date) ==="
