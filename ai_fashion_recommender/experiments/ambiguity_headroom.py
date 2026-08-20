"""오차의 성격 분석 — 남은 오답이 "틀린 예측"인가 "구분 불가능한 이웃 라벨"인가.

레시피 개선이 통하지 않았으므로 병목이 학습 방법이 아니라는 뜻이다.
혼동 행렬 상위 쌍을 하나로 합쳤을 때 정확도가 얼마나 오르는지 재면
남은 오차 중 몇 %가 "인접 라벨 구분 실패"인지 정량화된다.

출력: reports/07_ambiguity_headroom.json
"""

from __future__ import annotations

import json

import torch

from eval_lib import (
    BASELINE_CHECKPOINT,
    REPORT_DIR,
    VAL_CACHE,
    head_logits,
    load_cache,
)
from fashion_attribute_model import load_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS

# 진단 단계에서 관찰된 상위 혼동 쌍. 외형이 실제로 겹치는 조합만 골랐다.
MERGE_GROUPS: dict[str, list[list[str]]] = {
    "pant_leg_shape": [["와이드", "팔라초"], ["스트레이트", "테이퍼드"], ["플레어", "부츠컷"]],
    "lower_fit": [["스트레이트핏", "테이퍼드핏"], ["와이드핏", "플레어핏"]],
    "upper_fit": [["여유핏", "오버핏"], ["슬림핏", "레귤러핏"]],
    "pant_length": [["크롭·앵클", "풀렝스"]],
    "upper_length": [["크롭 기장", "기본 기장"]],
    "lower_subtype": [["조거·스웨트팬츠", "트랙팬츠"], ["슬랙스", "치노 팬츠"]],
    "neckline": [["라운드넥", "보트넥"]],
    "collar": [["셔츠 칼라", "스탠드 칼라"]],
    "sleeve_length": [["7부 소매", "긴팔"]],
    "silhouette": [["A라인", "X라인"]],
    "lower_length": [["미디·7부 기장", "무릎 기장"]],
    "category": [["티셔츠", "탑"], ["셔츠", "블라우스"], ["팬츠", "청바지"]],
}


def main() -> None:
    device = "cpu"
    val_cache = load_cache(VAL_CACHE)
    heads, payload = load_attribute_heads(BASELINE_CHECKPOINT, device)
    support_map = payload.get("label_support", {})
    minimum = int(payload.get("minimum_label_examples", 5))
    logits = head_logits(heads, val_cache, device)

    report = {}
    header = f"{'task':<16}{'orig':>8}{'merged':>8}{'Δ':>8}{'errors':>8}{'fixed':>7}{'share':>8}  합친 쌍"
    print("=" * 110)
    print("인접 라벨을 합쳤을 때의 정확도 (배포 체크포인트, val)")
    print("=" * 110)
    print(header)
    print("-" * 110)
    for task_name, groups in MERGE_GROUPS.items():
        task = ATTRIBUTE_TASKS[task_name]
        if task.multi_label:
            continue
        mask = val_cache["valid"][task_name]
        if not bool(mask.any()):
            continue
        expected = val_cache["targets"][task_name][mask].long()
        selected = logits[task_name][mask].float()
        allowed = [
            i for i, label in enumerate(task.labels)
            if int(support_map.get(task_name, {}).get(label, 0)) >= minimum
        ]
        masked = selected.clone()
        blocked = sorted(set(range(len(task.labels))) - set(allowed))
        if blocked:
            masked[:, blocked] = float("-inf")
        predicted = masked.argmax(dim=-1)

        # 라벨 인덱스를 그룹 대표 인덱스로 매핑
        mapping = {i: i for i in range(len(task.labels))}
        index_of = {label: i for i, label in enumerate(task.labels)}
        applied = []
        for group in groups:
            indices = [index_of[label] for label in group if label in index_of]
            if len(indices) < 2:
                continue
            root = min(indices)
            for i in indices:
                mapping[i] = root
            applied.append("+".join(group))
        lookup = torch.tensor([mapping[i] for i in range(len(task.labels))])

        original = float((predicted == expected).float().mean())
        merged = float((lookup[predicted] == lookup[expected]).float().mean())
        errors = int((predicted != expected).sum())
        fixed = errors - int((lookup[predicted] != lookup[expected]).sum())
        share = fixed / errors if errors else 0.0
        print(
            f"{task_name:<16}{original:>8.3f}{merged:>8.3f}{merged - original:>+8.3f}"
            f"{errors:>8}{fixed:>7}{share:>8.1%}  {', '.join(applied)}"
        )
        report[task_name] = {
            "original_accuracy": round(original, 4),
            "merged_accuracy": round(merged, 4),
            "delta": round(merged - original, 4),
            "errors": errors,
            "errors_fixed_by_merge": fixed,
            "share_of_errors_from_adjacent_labels": round(share, 4),
            "merged_groups": applied,
        }

    total_errors = sum(v["errors"] for v in report.values())
    total_fixed = sum(v["errors_fixed_by_merge"] for v in report.values())
    print()
    print(
        f"합계: 오답 {total_errors}건 중 {total_fixed}건({total_fixed / total_errors:.1%})이 "
        "인접 라벨 혼동이다."
    )
    report["_total"] = {
        "errors": total_errors,
        "fixed": total_fixed,
        "share": round(total_fixed / total_errors, 4) if total_errors else None,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "07_ambiguity_headroom.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {REPORT_DIR / '07_ambiguity_headroom.json'}")


if __name__ == "__main__":
    main()
