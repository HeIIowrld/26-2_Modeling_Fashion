"""1~3단계 실험 공용 유틸.

설계 원칙
- 학습 대상은 캐시된 FashionSigLIP 임베딩 위의 17개 헤드뿐이므로 재학습이 저렴하다.
- 기존 학습은 val 셋으로 early stopping과 임계값을 동시에 고른 뒤 같은 val로 성능을 보고했다.
  이 실험은 train을 train_fit/train_dev로 쪼개 dev에서만 고르고 val은 끝까지 손대지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fashion_attribute_schema import ATTRIBUTE_TASKS  # noqa: E402

TRAIN_CACHE = PROJECT_DIR / "data" / "cache" / "fashion_attributes_train.pt"
VAL_CACHE = PROJECT_DIR / "data" / "cache" / "fashion_attributes_val.pt"
BASELINE_CHECKPOINT = PROJECT_DIR / "models" / "fashion_attribute_heads.pt"
REPORT_DIR = PROJECT_DIR / "reports"

DEV_FRACTION = 0.15
SPLIT_SEED = 20260818


def load_cache(path: str | Path) -> dict[str, Any]:
    from fashion_attribute_training import load_embedding_cache

    return load_embedding_cache(path)


def subset(cache: dict[str, Any], indices) -> dict[str, Any]:
    return {
        "version": 1,
        "backbone_model_id": cache["backbone_model_id"],
        "features": cache["features"][indices],
        "targets": {name: value[indices] for name, value in cache["targets"].items()},
        "valid": {name: value[indices] for name, value in cache["valid"].items()},
        "image_paths": [cache["image_paths"][int(i)] for i in indices],
    }


def split_train_dev(cache: dict[str, Any], *, dev_fraction: float = DEV_FRACTION, seed: int = SPLIT_SEED):
    """출처별로 같은 비율을 떼어내 dev가 한쪽 데이터에 쏠리지 않게 한다."""
    import torch

    sources = [Path(path).as_posix().split("/") for path in cache["image_paths"]]
    # .../data/<source>/images/<file> 구조에서 source 폴더명을 찾는다.
    groups: dict[str, list[int]] = {}
    for index, parts in enumerate(sources):
        key = parts[-3] if len(parts) >= 3 else "unknown"
        groups.setdefault(key, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    dev_indices: list[int] = []
    fit_indices: list[int] = []
    for key in sorted(groups):
        members = torch.tensor(groups[key])
        order = members[torch.randperm(len(members), generator=generator)]
        cut = int(round(len(order) * dev_fraction))
        dev_indices.extend(int(i) for i in order[:cut])
        fit_indices.extend(int(i) for i in order[cut:])
    fit_indices.sort()
    dev_indices.sort()
    return torch.tensor(fit_indices), torch.tensor(dev_indices)


def head_logits(heads, cache: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    heads.eval()
    with torch.inference_mode():
        return {name: value.cpu() for name, value in heads(cache["features"].to(device)).items()}


def evaluate_logits(
    logits: dict[str, Any],
    cache: dict[str, Any],
    thresholds: dict[str, float],
    *,
    label_support: dict[str, dict[str, int]] | None = None,
    minimum_label_examples: int = 5,
) -> dict[str, dict[str, Any]]:
    """추론 시 게이팅(라벨 마스킹 + 임계값)을 반영한 지표.

    단일분류: accuracy / macro_recall / macro_f1 / accepted_coverage / accepted_accuracy
    다중분류: micro_f1 / macro_f1
    """
    import torch

    results: dict[str, dict[str, Any]] = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = cache["valid"][task_name]
        count = int(mask.sum())
        if count == 0:
            results[task_name] = {"samples": 0, "score": None}
            continue
        expected = cache["targets"][task_name][mask]
        selected = logits[task_name][mask].float()
        support_map = (label_support or {}).get(task_name)
        supported = [
            index
            for index, label in enumerate(task.labels)
            if support_map is None or int(support_map.get(label, 0)) >= minimum_label_examples
        ]
        threshold = float(thresholds.get(task_name, task.minimum_confidence))

        if task.multi_label:
            probabilities = selected.sigmoid()
            blocked = sorted(set(range(len(task.labels))) - set(supported))
            if blocked:
                probabilities[:, blocked] = 0.0
            predicted = probabilities >= threshold
            expected_bool = expected.bool()
            tp = int((predicted & expected_bool).sum())
            fp = int((predicted & ~expected_bool).sum())
            fn = int((~predicted & expected_bool).sum())
            denominator = 2 * tp + fp + fn
            micro_f1 = 2 * tp / denominator if denominator else 0.0
            per_label = {}
            f1_values = []
            for index, label in enumerate(task.labels):
                label_expected = expected_bool[:, index]
                label_predicted = predicted[:, index]
                ltp = int((label_predicted & label_expected).sum())
                lfp = int((label_predicted & ~label_expected).sum())
                lfn = int((~label_predicted & label_expected).sum())
                ldenominator = 2 * ltp + lfp + lfn
                label_f1 = 2 * ltp / ldenominator if ldenominator else None
                per_label[label] = {
                    "support": int(label_expected.sum()),
                    "f1": round(label_f1, 4) if label_f1 is not None else None,
                    "blocked": index not in supported,
                }
                if int(label_expected.sum()) > 0:
                    f1_values.append(label_f1 or 0.0)
            results[task_name] = {
                "samples": count,
                "multi_label": True,
                "micro_f1": round(micro_f1, 4),
                "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
                "score": round(micro_f1, 4),
                "threshold": threshold,
                "per_label": per_label,
            }
            continue

        masked = selected.clone()
        blocked = sorted(set(range(len(task.labels))) - set(supported))
        if blocked and len(supported) >= 1:
            masked[:, blocked] = float("-inf")
        probabilities = masked.softmax(dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        accuracy = float((predicted == expected).float().mean())
        accepted = confidence >= threshold
        accepted_accuracy = (
            float((predicted[accepted] == expected[accepted]).float().mean())
            if bool(accepted.any()) else None
        )
        per_label = {}
        recalls = []
        f1_values = []
        for index, label in enumerate(task.labels):
            label_expected = expected == index
            label_predicted = predicted == index
            ltp = int((label_predicted & label_expected).sum())
            predicted_count = int(label_predicted.sum())
            label_support_count = int(label_expected.sum())
            recall = ltp / label_support_count if label_support_count else None
            precision = ltp / predicted_count if predicted_count else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall else (0.0 if label_support_count else None)
            )
            per_label[label] = {
                "support": label_support_count,
                "recall": round(recall, 4) if recall is not None else None,
                "precision": round(precision, 4) if precision is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "blocked": index not in supported,
            }
            if label_support_count:
                recalls.append(recall or 0.0)
                f1_values.append(f1 or 0.0)
        confusion = _top_confusions(predicted, expected, task.labels)
        results[task_name] = {
            "samples": count,
            "multi_label": False,
            "accuracy": round(accuracy, 4),
            "macro_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
            "score": round(accuracy, 4),
            "threshold": threshold,
            "accepted_coverage": round(float(accepted.float().mean()), 4),
            "accepted_accuracy": round(accepted_accuracy, 4) if accepted_accuracy is not None else None,
            "per_label": per_label,
            "top_confusions": confusion,
        }
    return results


def _top_confusions(predicted, expected, labels: tuple[str, ...], limit: int = 6):
    counts: dict[tuple[str, str], int] = {}
    for want, got in zip(expected.tolist(), predicted.tolist()):
        if want == got:
            continue
        key = (labels[int(want)], labels[int(got)])
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: -item[1])[:limit]
    return [{"truth": key[0], "predicted": key[1], "count": value} for key, value in ordered]


def overall_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """태스크 평균과 표본 가중 평균을 함께 본다."""
    scores = [(name, value) for name, value in metrics.items() if value.get("score") is not None]
    macro = [value.get("macro_f1") for _, value in scores if value.get("macro_f1") is not None]
    total = sum(value["samples"] for _, value in scores)
    weighted = sum(value["score"] * value["samples"] for _, value in scores) / total if total else 0.0
    return {
        "tasks": len(scores),
        "mean_score": round(sum(value["score"] for _, value in scores) / len(scores), 4),
        "sample_weighted_score": round(weighted, 4),
        "mean_macro_f1": round(sum(macro) / len(macro), 4) if macro else None,
    }
