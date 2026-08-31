#!/usr/bin/env bash
set -euo pipefail

# hpcmaster의 사용자 systemd에서 실행한다. 실제 GPU worker API는 반드시 Slurm이
# 할당한 계산 노드에서만 실행하고, 이 프로세스는 제출·감시·취소만 담당한다.
PROJECT_MASTER=${FITTA_PROJECT_MASTER:-/data1/dsl01/releases/fitta_current}
PROJECT_COMPUTE=${FITTA_PROJECT_COMPUTE:-/mnt/data1/dsl01/releases/fitta_current}
STATE_DIR=${FITTA_SERVICE_STATE_DIR:-/data1/dsl01/.local/state/fitta-web}
JOB_FILE=$STATE_DIR/job_id
JOB_SCRIPT=$PROJECT_MASTER/gpu_server/jobs/web_service.sbatch
SERVICE_USER=$(id -un)
JOB_ID=

mkdir -p "$STATE_DIR"

find_existing_job() {
    /usr/bin/squeue \
        --noheader \
        --user "$SERVICE_USER" \
        --name fitta-web \
        --states PENDING,RUNNING,CONFIGURING \
        --format '%A' \
        | head -n 1 \
        | tr -d '[:space:]'
}

stop_job() {
    trap - TERM INT
    if [[ -n "${JOB_ID:-}" ]]; then
        echo "FITTA service: Slurm job $JOB_ID 취소"
        /usr/bin/scancel "$JOB_ID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! /usr/bin/squeue --noheader --jobs "$JOB_ID" | grep -q .; then
                break
            fi
            sleep 1
        done
    fi
    rm -f "$JOB_FILE"
    exit 0
}
trap stop_job TERM INT

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "FITTA service: Slurm 스크립트가 없습니다: $JOB_SCRIPT" >&2
    exit 1
fi

JOB_ID=$(find_existing_job)
if [[ -n "$JOB_ID" ]]; then
    echo "FITTA service: 실행 중인 Slurm job $JOB_ID 인계"
else
    JOB_ID=$(
        /usr/bin/sbatch \
            --parsable \
            --export="ALL,FITTA_PROJECT=$PROJECT_COMPUTE,FASHION_WEB_HOST=0.0.0.0,FASHION_WEB_PORT=8000" \
            "$JOB_SCRIPT"
    )
    JOB_ID=${JOB_ID%%;*}
    echo "FITTA service: Slurm job $JOB_ID 제출"
fi
printf '%s\n' "$JOB_ID" >"$JOB_FILE"

while /usr/bin/squeue --noheader --jobs "$JOB_ID" | grep -q .; do
    sleep 5
done

FINAL_STATE=$(
    /usr/bin/sacct --jobs "$JOB_ID" --allocations --noheader --parsable2 --format State \
        | head -n 1 \
        | tr -d '[:space:]'
)
rm -f "$JOB_FILE"
echo "FITTA service: Slurm job $JOB_ID 종료 (${FINAL_STATE:-UNKNOWN})" >&2

# systemd가 새 GPU 작업을 다시 제출하도록 비정상 종료로 보고한다.
exit 1
