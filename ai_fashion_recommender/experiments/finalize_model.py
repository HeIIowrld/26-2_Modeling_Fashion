"""최종 모델 보존 — 체크섬 기록, metrics.json 생성, 배포 모델 무변경 확인.

파일을 지우거나 배포 체크포인트를 덮어쓰지 않는다. 기록만 남긴다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from eval_lib import BASELINE_CHECKPOINT, PROJECT_DIR, REPORT_DIR
from fashion_attribute_model import load_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS

AUGMENTED = BASELINE_CHECKPOINT.parent / "fashion_attribute_heads_augmented.pt"
# 배포 체크포인트는 2026-08-14 학습분이며 이번 작업에서 한 번도 쓰지 않았다.
BASELINE_EXPECTED_SIZE = 13665395


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(2)

    print("=" * 78)
    print("1. 배포 체크포인트 무변경 확인")
    print("=" * 78)
    baseline_size = BASELINE_CHECKPOINT.stat().st_size
    baseline_mtime = datetime.fromtimestamp(BASELINE_CHECKPOINT.stat().st_mtime)
    unchanged = baseline_size == BASELINE_EXPECTED_SIZE
    print(f"  {BASELINE_CHECKPOINT.name}")
    print(f"    크기 {baseline_size:,} bytes  수정시각 {baseline_mtime:%Y-%m-%d %H:%M}")
    print(f"    → {'변경 없음 OK' if unchanged else '크기가 다릅니다. 확인 필요'}")
    baseline_hash = sha256(BASELINE_CHECKPOINT)
    print(f"    sha256 {baseline_hash}")

    print()
    print("=" * 78)
    print("2. 최종 후보 모델 검증")
    print("=" * 78)
    heads, payload = load_attribute_heads(AUGMENTED, "cpu")
    parameters = sum(p.numel() for p in heads.parameters())
    augmented_hash = sha256(AUGMENTED)
    print(f"  {AUGMENTED.name}")
    print(f"    크기 {AUGMENTED.stat().st_size:,} bytes")
    print(f"    sha256 {augmented_hash}")
    print(f"    백본 {payload['backbone_model_id']}  헤드 파라미터 {parameters:,}")
    print(f"    라벨 스키마 검증: 17 태스크 / {sum(len(t.labels) for t in ATTRIBUTE_TASKS.values())} 라벨 → 로드 성공")

    support = payload["label_support"]
    blocked = [
        f"{task}|{label}"
        for task, labels in support.items()
        for label, count in labels.items()
        if count < int(payload.get("minimum_label_examples", 5))
    ]
    print(f"    표본 부족으로 여전히 차단되는 라벨 {len(blocked)}개: {', '.join(blocked) if blocked else '없음'}")

    comparison = json.loads((REPORT_DIR / "14_final_comparison.json").read_text(encoding="utf-8"))
    comparison_new = json.loads(
        (REPORT_DIR / "15_final_comparison_newval.json").read_text(encoding="utf-8")
    )
    integrity = json.loads((REPORT_DIR / "13_split_integrity.json").read_text(encoding="utf-8"))

    metrics = {
        "checkpoint": AUGMENTED.name,
        "sha256": augmented_hash,
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "backbone_model_id": payload["backbone_model_id"],
        "head_parameters": parameters,
        "hidden_dim": payload["hidden_dim"],
        "minimum_label_examples": payload.get("minimum_label_examples"),
        "thresholds": payload.get("thresholds"),
        "training": {
            "recipe": "R2_mixup (dropout 0.4, weight_decay 1e-3, lr 5e-4 cosine, mixup alpha 0.4, batch 256, 100 epoch)",
            "seeds": 3,
            "train_crops_total": 22341,
            "composition": {
                "original": 4789,
                "fashionpedia_train_round1": 6797,
                "fashionpedia_train_round2": 10755,
            },
            "fit_crops": integrity["fit_dev"]["fit_crops"],
            "dev_crops": integrity["fit_dev"]["dev_crops"],
            "dev_usage": "early stopping, 레시피/에폭 선택, 임계값 선택 (성능 보고에는 사용하지 않음)",
        },
        "evaluation": {
            "existing_val_1195": {
                "usage": "validation set — 최종 성능 보고용. 2차 보강 모델의 학습·선택에는 사용하지 않음.",
                "caveat": "배포 모델은 원본 학습 코드에서 이 셋으로 early stopping과 임계값을 선택했으므로 배포 모델에 유리한 비교다.",
                "baseline": comparison["overall"]["baseline"],
                "augmented": comparison["overall"]["augmented"],
            },
            "new_val_1716": {
                "usage": "validation set — Fashionpedia train split의 20%. 두 모델 모두 학습에 사용하지 않음.",
                "caveat": "2차 보강 설계 시 1차 결과를 참고했으므로 완전 독립 test set이 아니다.",
                "baseline": comparison_new["overall"]["baseline"],
                "augmented": comparison_new["overall"]["augmented"],
            },
        },
        "split_integrity": {
            "train_eval_image_leak": integrity["train_eval_leak_images"],
            "verdict": integrity["verdict"],
            "fit_dev_note": (
                "fit/dev 는 crop 단위로 나뉘어 같은 이미지가 양쪽에 걸쳐 있다"
                f" (dev crop의 {integrity['fit_dev']['contaminated_ratio'] * 100:.0f}%)."
                " dev 는 선택에만 쓰였고 보고 성능에는 쓰이지 않았다."
            ),
        },
        "per_task_existing_val": {
            name: {
                "baseline_score": value["baseline"]["score"],
                "augmented_score": value["augmented"]["score"],
                "baseline_macro_f1": value["baseline"].get("macro_f1"),
                "augmented_macro_f1": value["augmented"].get("macro_f1"),
                "samples": value["augmented"]["samples"],
            }
            for name, value in comparison["per_task"].items()
        },
        "blocked_labels": blocked,
    }
    metrics_path = AUGMENTED.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  metrics 저장: {metrics_path.name}")

    checksums = {
        "generated": metrics["created"],
        "note": "배포 체크포인트는 이번 작업에서 수정하지 않았다.",
        "files": {
            BASELINE_CHECKPOINT.name: {
                "sha256": baseline_hash, "bytes": baseline_size,
                "role": "기존 배포 모델 (변경 없음)",
            },
            AUGMENTED.name: {
                "sha256": augmented_hash, "bytes": AUGMENTED.stat().st_size,
                "role": "최종 후보 모델 (2차 보강)",
            },
        },
    }
    for extra in ("fashion_attribute_heads_finetuned.pt", "fashion_attribute_heads_detailnone.pt"):
        path = BASELINE_CHECKPOINT.parent / extra
        if path.is_file():
            checksums["files"][extra] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
                "role": "중간 실험 산출물 (배포 대상 아님)",
            }
    checksum_path = BASELINE_CHECKPOINT.parent / "CHECKSUMS.json"
    checksum_path.write_text(json.dumps(checksums, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  체크섬 저장: {checksum_path.name}")

    print()
    print("=" * 78)
    print("3. models/ 최종 상태")
    print("=" * 78)
    for path in sorted(BASELINE_CHECKPOINT.parent.glob("*.pt")):
        print(f"  {path.name:<45} {path.stat().st_size:>12,} bytes  "
              f"{datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
