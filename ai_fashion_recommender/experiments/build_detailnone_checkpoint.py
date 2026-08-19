"""권장안 체크포인트 — 다른 15개 헤드는 BASE 그대로 두고 detail 헤드만 D50으로 교체한다.

헤드가 서로 파라미터를 공유하지 않으므로 태스크 단위 교체가 가능하다.
D50 = Fashionpedia 빈 detail 행의 50%만 `디테일 없음`으로 채워 학습한 것.
전량(D) 대비 단추 손실을 절반으로 줄이면서 `디테일 없음` 능력의 97%를 얻는다.
"""

from __future__ import annotations

import json

import torch

from eval_lib import (
    BASELINE_CHECKPOINT,
    REPORT_DIR,
    TRAIN_CACHE,
    VAL_CACHE,
    evaluate_logits,
    head_logits,
    load_cache,
    overall_summary,
    split_train_dev,
    subset,
)
from evaluate_relabel import assemble, label_support, tune_multi_threshold
from fashion_attribute_model import save_attribute_checkpoint
from fashion_attribute_schema import ATTRIBUTE_TASKS
from finetune import RUN_DIR
from run_relabel_experiment import CACHE_DIR, RELABEL_RUN_DIR

OUTPUT = BASELINE_CHECKPOINT.parent / "fashion_attribute_heads_detailnone.pt"
PANTS_TASKS = ("lower_subtype", "pant_leg_shape", "pant_length")


def main() -> None:
    torch.set_num_threads(4)
    device = "cpu"
    original_train = load_cache(TRAIN_CACHE)
    original_val = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(original_train)
    fit = subset(original_train, fit_indices)
    dev = subset(original_train, dev_indices)

    base_runs = torch.load(RUN_DIR / "R2_mixup.pt", map_location="cpu", weights_only=False)["runs"]
    heads, picked = assemble(base_runs, fit)

    d50 = torch.load(RELABEL_RUN_DIR / "D50.pt", map_location="cpu", weights_only=False)
    d50_train = load_cache(CACHE_DIR / "fashion_attributes_train_D50.pt")
    d50_fit = subset(d50_train, fit_indices)
    d50_dev = subset(d50_train, dev_indices)
    d50_heads, d50_picked = assemble(d50["runs"], d50_fit)
    heads.heads["detail"].load_state_dict(d50_heads.heads["detail"].state_dict())
    heads.eval()
    picked["detail"] = {**d50_picked["detail"], "source": "D50"}

    # label_support: detail 만 D50 통계를 써야 `디테일 없음`이 게이팅에서 풀린다.
    support = label_support(fit)
    support["detail"] = label_support(d50_fit)["detail"]
    minimum = 5

    dev_logits = head_logits(heads, dev, device)
    d50_dev_logits = head_logits(heads, d50_dev, device)
    thresholds: dict[str, float] = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if task.multi_label:
            # detail 은 D50 라벨 정의로 임계값을 골라야 한다.
            if task_name == "detail":
                thresholds[task_name] = tune_multi_threshold(
                    d50_dev_logits, d50_dev, task_name, support, minimum
                )
            else:
                thresholds[task_name] = tune_multi_threshold(
                    dev_logits, dev, task_name, support, minimum
                )
        elif task_name in PANTS_TASKS:
            expected = dev["targets"][task_name][dev["valid"][task_name]]
            selected = dev_logits[task_name][dev["valid"][task_name]].float()
            allowed = [
                i for i, label in enumerate(task.labels)
                if int(support.get(task_name, {}).get(label, 0)) >= minimum
            ]
            masked = selected.clone()
            blocked = sorted(set(range(len(task.labels))) - set(allowed))
            if blocked:
                masked[:, blocked] = float("-inf")
            confidence, predicted = masked.softmax(dim=-1).max(dim=-1)
            chosen = task.minimum_confidence
            for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
                if threshold < task.minimum_confidence:
                    continue
                accepted = confidence >= threshold
                if float(accepted.float().mean()) < 0.20:
                    continue
                if float((predicted[accepted] == expected[accepted]).float().mean()) >= 0.80:
                    chosen = threshold
                    break
            thresholds[task_name] = chosen
        else:
            thresholds[task_name] = task.minimum_confidence

    d50_val = load_cache(CACHE_DIR / "fashion_attributes_val_D50.pt")
    strict = evaluate_logits(
        head_logits(heads, original_val, device), original_val, thresholds,
        label_support=support, minimum_label_examples=minimum,
    )
    relabeled = evaluate_logits(
        head_logits(heads, d50_val, device), d50_val, thresholds,
        label_support=support, minimum_label_examples=minimum,
    )

    print("원본 val 라벨 기준 전체:", overall_summary(strict))
    print(f"  detail  micro {strict['detail']['micro_f1']:.4f}  단추 F1 "
          f"{strict['detail']['per_label']['단추']['f1']:.3f}")
    print(f"  재라벨 val: detail {relabeled['detail']['samples']}행 "
          f"micro {relabeled['detail']['micro_f1']:.4f}  "
          f"디테일 없음 F1 {relabeled['detail']['per_label']['디테일 없음']['f1']:.3f}")

    save_attribute_checkpoint(
        OUTPUT,
        heads,
        backbone_model_id=fit["backbone_model_id"],
        thresholds=thresholds,
        training_summary={
            "note": (
                "detail 헤드만 D50(Fashionpedia 빈 detail 행의 50%를 `디테일 없음`으로 채운 학습)이고 "
                "나머지 16개 헤드는 R2_mixup BASE. train_fit(4071) 학습, train_dev(718) 선택, val 평가 전용."
            ),
            "picked": picked,
            "metrics_original_labels": strict,
            "metrics_relabeled": relabeled,
        },
        label_support=support,
        minimum_label_examples=minimum,
    )
    (REPORT_DIR / "10_detailnone_checkpoint.json").write_text(
        json.dumps(
            {"thresholds": thresholds, "picked": picked,
             "strict": strict, "relabeled": relabeled},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {OUTPUT}")


if __name__ == "__main__":
    main()
