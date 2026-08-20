from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from config import FASHION_SIGLIP_MODEL_ID
from fashion_attribute_dataset import AttributeRecord, encode_record_targets, load_attribute_csv
from fashion_attribute_model import (
    GEOMETRY_DIM,
    PREPROCESS_MODES,
    PREPROCESS_SQUASH,
    TASK_PREPROCESSING,
    resolve_task_preprocessing,
    apply_preprocess_mode,
    build_attribute_heads,
    geometry_vector,
    save_attribute_checkpoint,
)
from fashion_attribute_schema import ATTRIBUTE_TASKS


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.15
    patience: int = 5
    seed: int = 42
    minimum_label_examples: int = 5
    # crop의 종횡비 등 기하 특징을 임베딩과 함께 헤드에 넣을지 여부.
    #
    # 기본값은 False다. 5,984 crop으로 A/B한 결과 기하 의존 속성의 평균 macro-F1이
    # 0.642 → 0.637로 오히려 소폭 낮아졌다. 이유는 두 가지다.
    #   1) 바지 속성은 crop이 옷 단위가 아니다(tight 비율 0%). 종횡비가 상품 사진 규격이다.
    #   2) 상의는 crop이 옷 단위인데도 종횡비가 핏을 구분하지 못한다. bbox 종횡비는 핏보다
    #      기장에 지배되어, 오버핏이 슬림핏보다 오히려 좁고 길쭉하게 측정된다.
    # 핏은 "옷 폭 ÷ 몸 기준폭"이라 학습 표본에 어깨·골반 좌표가 있어야 한다.
    # 자세한 수치는 README의 '기하 특징 A/B 결과' 참조.
    use_geometry: bool = False


class FrozenFashionSigLIPEncoder:
    """FashionSigLIP은 고정하고 정규화된 이미지 임베딩만 만든다."""

    def __init__(
        self,
        model_id: str = FASHION_SIGLIP_MODEL_ID,
        device: str = "auto",
        preprocessing: str = PREPROCESS_SQUASH,
    ) -> None:
        import open_clip
        import torch

        if preprocessing not in PREPROCESS_MODES:
            raise ValueError(f"알 수 없는 전처리 방식입니다: {preprocessing}")
        self.preprocessing = preprocessing
        self.model_id = model_id
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{model_id}", device=self.device
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_pil_batch(self, images: list[Image.Image]):
        import torch

        batch = torch.stack([
            self.preprocess(apply_preprocess_mode(image.convert("RGB"), self.preprocessing))
            for image in images
        ]).to(self.device)
        with torch.inference_mode():
            return self.model.encode_image(batch, normalize=True).float().cpu()


def _crop_box(record: AttributeRecord, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """crop 영역을 한 곳에서만 계산한다.

    임베딩에 쓰는 crop과 기하 특징에 쓰는 crop이 달라지면 학습과 추론이 어긋난다.
    """
    image_width, image_height = image_size
    if record.bbox is None:
        return 0, 0, image_width, image_height
    x, y, width, height = record.bbox
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(image_width, math.ceil(x + width))
    bottom = min(image_height, math.ceil(y + height))
    if right <= left or bottom <= top:
        raise ValueError(f"이미지 범위를 벗어난 bbox입니다: {record.image_path}, {record.bbox}")
    return left, top, right, bottom


def _crop_record(record: AttributeRecord) -> Image.Image:
    image = Image.open(record.image_path).convert("RGB")
    box = _crop_box(record, image.size)
    if box == (0, 0, image.width, image.height):
        return image
    return image.crop(box)


def _record_geometry(record: AttributeRecord) -> list[float]:
    """추론과 같은 정의로 crop의 기하 특징을 만든다. 픽셀은 읽지 않는다."""
    with Image.open(record.image_path) as image:
        left, top, right, bottom = _crop_box(record, image.size)
    return geometry_vector(right - left, bottom - top, tight_crop=record.bbox is not None)


def build_embedding_cache(
    records: list[AttributeRecord],
    encoder: FrozenFashionSigLIPEncoder,
    output_path: str | Path,
    *,
    batch_size: int = 64,
    reuse_cache_paths: list[str | Path] | None = None,
) -> Path:
    import torch

    if not records:
        raise ValueError("임베딩을 만들 학습 레코드가 없습니다.")
    record_keys = [str(record.image_path).replace("\\", "/").lower() for record in records]
    reused_features: dict[int, Any] = {}
    for cache_path in reuse_cache_paths or []:
        cache = load_embedding_cache(cache_path)
        if cache["backbone_model_id"] != encoder.model_id:
            raise ValueError("재사용 캐시의 FashionSigLIP 백본이 현재 백본과 다릅니다.")
        if cache["preprocessing"] != encoder.preprocessing:
            raise ValueError(
                "재사용 캐시의 crop 처리 방식이 다릅니다: "
                f"캐시={cache['preprocessing']}, 현재={encoder.preprocessing}"
            )
        cached = {
            path.replace("\\", "/").lower(): feature
            for path, feature in zip(cache["image_paths"], cache["features"])
        }
        for index, key in enumerate(record_keys):
            if index not in reused_features and key in cached:
                reused_features[index] = cached[key]

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(output.suffix + ".partial.pt")
    encoded_indices: list[int] = []
    encoded_batches = []
    if partial_path.is_file():
        partial = _torch_load_cache(partial_path)
        if partial.get("record_keys") != record_keys or partial.get("backbone_model_id") != encoder.model_id:
            raise ValueError(f"입력 CSV와 맞지 않는 중간 캐시입니다. 확인 후 삭제하세요: {partial_path}")
        encoded_indices = [int(index) for index in partial.get("indices", [])]
        if encoded_indices:
            encoded_batches.append(partial["features"])
        print(f"[FashionSigLIP 임베딩] 중간 결과 {len(encoded_indices)}장 재개", flush=True)

    completed_indices = set(reused_features) | set(encoded_indices)
    missing_indices = [index for index in range(len(records)) if index not in completed_indices]
    for start in range(0, len(missing_indices), batch_size):
        batch_indices = missing_indices[start:start + batch_size]
        images = [_crop_record(records[index]) for index in batch_indices]
        encoded_batches.append(encoder.encode_pil_batch(images))
        encoded_indices.extend(batch_indices)
        encoded = torch.cat(encoded_batches, dim=0)
        torch.save(
            {
                "version": 1,
                "backbone_model_id": encoder.model_id,
                "record_keys": record_keys,
                "indices": encoded_indices,
                "features": encoded,
            },
            partial_path,
        )
        total_completed = len(completed_indices) + min(start + batch_size, len(missing_indices))
        if start == 0 or total_completed == len(records) or total_completed % (batch_size * 5) == 0:
            print(
                f"[FashionSigLIP 임베딩] {total_completed}/{len(records)} "
                f"(기존 특징 재사용 {len(reused_features)})",
                flush=True,
            )

    encoded_map = {
        index: feature
        for index, feature in zip(encoded_indices, torch.cat(encoded_batches, dim=0))
    } if encoded_indices else {}
    features = torch.stack([
        reused_features[index] if index in reused_features else encoded_map[index]
        for index in range(len(records))
    ])

    target_rows = [encode_record_targets(record) for record in records]
    targets = {}
    valid = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        values = [row[0][task_name] for row in target_rows]
        targets[task_name] = torch.tensor(values, dtype=torch.float32 if task.multi_label else torch.long)
        valid[task_name] = torch.tensor([row[1][task_name] for row in target_rows], dtype=torch.bool)

    geometry = torch.tensor([_record_geometry(record) for record in records], dtype=torch.float32)
    torch.save(
        {
            "version": 2,
            "backbone_model_id": encoder.model_id,
            "preprocessing": encoder.preprocessing,
            "features": features,
            "geometry": geometry,
            "targets": targets,
            "valid": valid,
            "image_paths": [str(record.image_path) for record in records],
        },
        output,
    )
    if partial_path.is_file():
        partial_path.unlink()
    return output


def _torch_load_cache(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_embedding_cache(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("version") not in (1, 2):
        raise ValueError("지원하지 않는 임베딩 캐시 버전입니다.")
    length = len(payload["features"])
    payload.setdefault("preprocessing", PREPROCESS_SQUASH)
    if "geometry" not in payload:
        # 버전 1 캐시에는 기하 특징이 없다. 중립값으로 채워 두되,
        # 이 캐시로는 기하 특징을 쓰는 헤드를 학습할 수 없다.
        payload["geometry"] = None
    # 새 속성 헤드가 추가되어도 기존 FashionSigLIP 특징은 다시 계산하지 않는다.
    # 과거 캐시에 없는 task는 미주석(valid=False)으로 안전하게 보강한다.
    for task_name, task in ATTRIBUTE_TASKS.items():
        if task_name not in payload["targets"]:
            shape = (length, len(task.labels)) if task.multi_label else (length,)
            dtype = torch.float32 if task.multi_label else torch.long
            payload["targets"][task_name] = torch.zeros(shape, dtype=dtype)
        if task_name not in payload["valid"]:
            payload["valid"][task_name] = torch.zeros(length, dtype=torch.bool)
    if any(len(value) != length for value in payload["targets"].values()):
        raise ValueError("임베딩과 target 개수가 다릅니다.")
    return payload


def merge_embedding_caches(cache_paths: list[str | Path], output_path: str | Path) -> Path:
    """같은 백본과 라벨 스키마로 만든 캐시를 순서대로 합친다."""
    import torch

    if len(cache_paths) < 2:
        raise ValueError("합칠 임베딩 캐시를 두 개 이상 지정해야 합니다.")
    caches = [load_embedding_cache(path) for path in cache_paths]
    model_ids = {cache["backbone_model_id"] for cache in caches}
    if len(model_ids) != 1:
        raise ValueError("백본이 다른 임베딩 캐시는 합칠 수 없습니다.")
    modes = {cache["preprocessing"] for cache in caches}
    if len(modes) != 1:
        raise ValueError("crop 처리 방식이 다른 임베딩 캐시는 합칠 수 없습니다.")
    feature_dims = {int(cache["features"].shape[1]) for cache in caches}
    if len(feature_dims) != 1:
        raise ValueError("특징 차원이 다른 임베딩 캐시는 합칠 수 없습니다.")
    expected_tasks = set(ATTRIBUTE_TASKS)
    for cache in caches:
        if set(cache["targets"]) != expected_tasks or set(cache["valid"]) != expected_tasks:
            raise ValueError("라벨 task 구성이 다른 임베딩 캐시는 합칠 수 없습니다.")

    geometries = [cache["geometry"] for cache in caches]
    merged_geometry = None if any(item is None for item in geometries) else torch.cat(geometries, dim=0)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "backbone_model_id": caches[0]["backbone_model_id"],
            "preprocessing": caches[0]["preprocessing"],
            "features": torch.cat([cache["features"] for cache in caches], dim=0),
            "geometry": merged_geometry,
            "targets": {
                task_name: torch.cat([cache["targets"][task_name] for cache in caches], dim=0)
                for task_name in ATTRIBUTE_TASKS
            },
            "valid": {
                task_name: torch.cat([cache["valid"][task_name] for cache in caches], dim=0)
                for task_name in ATTRIBUTE_TASKS
            },
            "image_paths": [path for cache in caches for path in cache["image_paths"]],
        },
        output,
    )
    return output


def filter_embedding_cache(
    cache_path: str | Path,
    output_path: str | Path,
    *,
    exclude_path_fragments: list[str],
) -> Path:
    """특정 데이터 보강분만 제외해 기존 FashionSigLIP 특징을 재사용한다."""
    import torch

    if not exclude_path_fragments:
        raise ValueError("제외할 경로 조각을 한 개 이상 지정해야 합니다.")
    cache = load_embedding_cache(cache_path)
    normalized_fragments = [fragment.replace("\\", "/").lower() for fragment in exclude_path_fragments]
    keep = torch.tensor(
        [
            not any(fragment in path.replace("\\", "/").lower() for fragment in normalized_fragments)
            for path in cache["image_paths"]
        ],
        dtype=torch.bool,
    )
    if not bool(keep.any()):
        raise ValueError("필터링 후 남는 임베딩이 없습니다.")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "backbone_model_id": cache["backbone_model_id"],
            "preprocessing": cache["preprocessing"],
            "features": cache["features"][keep],
            "geometry": None if cache["geometry"] is None else cache["geometry"][keep],
            "targets": {name: values[keep] for name, values in cache["targets"].items()},
            "valid": {name: values[keep] for name, values in cache["valid"].items()},
            "image_paths": [path for path, use in zip(cache["image_paths"], keep.tolist()) if use],
        },
        output,
    )
    return output


def _select_features(cache: dict[str, Any], routed: dict[str, Any] | None, indices=None, device: str = "cpu"):
    """라우팅이 있으면 {전처리: 특징} 사전을, 없으면 텐서 하나를 돌려준다."""
    if routed is None:
        features = cache["features"]
        return (features if indices is None else features[indices]).to(device)
    return {
        mode: (item["features"] if indices is None else item["features"][indices]).to(device)
        for mode, item in routed.items()
    }


def load_routed_caches(cache_paths: dict[str, str | Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """전처리 방식별 캐시를 읽고, 같은 표본을 같은 순서로 담고 있는지 확인한다.

    속성마다 다른 crop 처리를 쓰려면 한 표본의 여러 임베딩이 나란히 놓여야 한다.
    """
    caches = {mode: load_embedding_cache(path) for mode, path in cache_paths.items()}
    reference = next(iter(caches.values()))
    for mode, cache in caches.items():
        if cache["preprocessing"] != mode:
            raise ValueError(f"{mode} 자리에 {cache['preprocessing']} 캐시가 들어왔습니다.")
        if cache["image_paths"] != reference["image_paths"]:
            raise ValueError("전처리별 캐시의 표본 순서가 다릅니다. 같은 CSV로 다시 만드세요.")
    return caches, reference


def _cache_geometry(cache: dict[str, Any], device: str, *, use_geometry: bool = True):
    """헤드가 기하 특징을 쓸 때만 캐시에서 꺼내 올린다."""
    geometry = cache.get("geometry")
    if not use_geometry or geometry is None:
        return None
    return geometry.to(device)


def _multitask_loss(heads, features, targets, valid, *, pos_weights=None, class_weights=None, geometry=None):
    import torch
    import torch.nn.functional as functional

    logits = heads(features, geometry)
    losses = []
    task_losses = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = valid[task_name]
        if not bool(mask.any()):
            continue
        if task.multi_label:
            weight = None if pos_weights is None else pos_weights.get(task_name)
            loss = functional.binary_cross_entropy_with_logits(
                logits[task_name][mask], targets[task_name][mask].float(), pos_weight=weight
            )
        else:
            weight = None if class_weights is None else class_weights.get(task_name)
            loss = functional.cross_entropy(
                logits[task_name][mask], targets[task_name][mask].long(), weight=weight
            )
        losses.append(loss)
        task_losses[task_name] = float(loss.detach().cpu())
    if not losses:
        raise ValueError("현재 batch에 학습 가능한 속성 라벨이 없습니다.")
    return torch.stack(losses).mean(), task_losses


def _positive_weights(cache: dict[str, Any], device: str) -> dict[str, Any]:
    weights = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if not task.multi_label:
            continue
        mask = cache["valid"][task_name]
        target = cache["targets"][task_name][mask].float()
        if not len(target):
            continue
        positives = target.sum(dim=0)
        negatives = len(target) - positives
        # 극소수 라벨이 loss를 지배하지 않도록 최대 20배로 제한한다.
        weights[task_name] = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0).to(device)
    return weights


def _single_class_weights(cache: dict[str, Any], device: str) -> dict[str, Any]:
    """Fashionpedia의 긴 꼬리 분포가 다수 클래스 쪽으로만 학습을 끌지 않게 한다."""
    import torch

    weights = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if task.multi_label:
            continue
        mask = cache["valid"][task_name]
        target = cache["targets"][task_name][mask].long()
        if not len(target):
            continue
        counts = torch.bincount(target, minlength=len(task.labels)).float()
        present = counts > 0
        task_weights = torch.zeros_like(counts)
        # 역빈도보다 완만한 제곱근 역빈도로 희소 클래스의 과도한 증폭을 막는다.
        task_weights[present] = torch.sqrt(counts[present].sum() / counts[present])
        task_weights[present] /= task_weights[present].mean()
        weights[task_name] = task_weights.clamp_max(10.0).to(device)
    return weights


def _label_support(cache: dict[str, Any]) -> dict[str, dict[str, int]]:
    import torch

    support = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = cache["valid"][task_name]
        target = cache["targets"][task_name][mask]
        if task.multi_label:
            counts = target.float().sum(dim=0) if len(target) else torch.zeros(len(task.labels))
        else:
            counts = (
                torch.bincount(target.long(), minlength=len(task.labels))
                if len(target)
                else torch.zeros(len(task.labels), dtype=torch.long)
            )
        support[task_name] = {
            label: int(counts[index]) for index, label in enumerate(task.labels)
        }
    return support


def _metrics(heads, cache: dict[str, Any], device: str, thresholds: dict[str, float], routed: dict[str, Any] | None = None) -> dict[str, dict[str, float | int | None]]:
    import torch

    heads.eval()
    with torch.inference_mode():
        logits = heads(_select_features(cache, routed, None, device), _cache_geometry(cache, device))
    results = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = cache["valid"][task_name]
        count = int(mask.sum())
        if count == 0:
            results[task_name] = {"samples": 0, "score": None}
            continue
        expected = cache["targets"][task_name][mask]
        selected = logits[task_name][mask].cpu()
        if task.multi_label:
            predicted = selected.sigmoid() >= thresholds[task_name]
            expected_bool = expected.bool()
            true_positive = int((predicted & expected_bool).sum())
            false_positive = int((predicted & ~expected_bool).sum())
            false_negative = int((~predicted & expected_bool).sum())
            denominator = 2 * true_positive + false_positive + false_negative
            score = 2 * true_positive / denominator if denominator else 0.0
            per_label = {}
            for index, label in enumerate(task.labels):
                label_expected = expected_bool[:, index]
                label_predicted = predicted[:, index]
                tp = int((label_predicted & label_expected).sum())
                fp = int((label_predicted & ~label_expected).sum())
                fn = int((~label_predicted & label_expected).sum())
                label_denominator = 2 * tp + fp + fn
                per_label[label] = {
                    "support": int(label_expected.sum()),
                    "f1": round(2 * tp / label_denominator, 4) if label_denominator else None,
                }
            results[task_name] = {
                "samples": count,
                "micro_f1": round(score, 4),
                "score": round(score, 4),
                "threshold": thresholds[task_name],
                "per_label": per_label,
            }
        else:
            probabilities = selected.softmax(dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            accuracy = float((predicted == expected).float().mean())
            threshold = float(thresholds.get(task_name, task.minimum_confidence))
            accepted = confidence >= threshold
            accepted_accuracy = (
                float((predicted[accepted] == expected[accepted]).float().mean())
                if bool(accepted.any()) else None
            )
            per_label = {}
            for index, label in enumerate(task.labels):
                label_expected = expected == index
                label_predicted = predicted == index
                tp = int((label_predicted & label_expected).sum())
                predicted_count = int(label_predicted.sum())
                support = int(label_expected.sum())
                per_label[label] = {
                    "support": support,
                    "recall": round(tp / support, 4) if support else None,
                    "precision": round(tp / predicted_count, 4) if predicted_count else None,
                }
            results[task_name] = {
                "samples": count,
                "accuracy": round(accuracy, 4),
                "score": round(accuracy, 4),
                "threshold": threshold,
                "accepted_coverage": round(float(accepted.float().mean()), 4),
                "accepted_accuracy": round(accepted_accuracy, 4) if accepted_accuracy is not None else None,
                "per_label": per_label,
            }
    return results


def _tune_multilabel_thresholds(heads, cache: dict[str, Any], device: str, routed: dict[str, Any] | None = None) -> dict[str, float]:
    import torch

    heads.eval()
    with torch.inference_mode():
        logits = heads(_select_features(cache, routed, None, device), _cache_geometry(cache, device))
    thresholds = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if not task.multi_label:
            continue
        mask = cache["valid"][task_name]
        if not bool(mask.any()):
            thresholds[task_name] = task.minimum_confidence
            continue
        probabilities = logits[task_name][mask].sigmoid().cpu()
        expected = cache["targets"][task_name][mask].bool()
        best_threshold, best_f1 = task.minimum_confidence, -1.0
        for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            predicted = probabilities >= threshold
            tp = int((predicted & expected).sum())
            fp = int((predicted & ~expected).sum())
            fn = int((~predicted & expected).sum())
            denominator = 2 * tp + fp + fn
            f1 = 2 * tp / denominator if denominator else 0.0
            if f1 > best_f1:
                best_threshold, best_f1 = threshold, f1
        thresholds[task_name] = best_threshold
    return thresholds


def _tune_pants_single_thresholds(
    heads,
    cache: dict[str, Any],
    device: str,
    *,
    target_accuracy: float = 0.80,
    routed: dict[str, Any] | None = None,
) -> dict[str, float]:
    """새 하의 세부 축은 오답을 줄이도록 검증셋 정확도 기준으로 보정한다."""
    import torch

    selected_tasks = {"lower_subtype", "pant_leg_shape", "pant_length"}
    heads.eval()
    with torch.inference_mode():
        logits = heads(_select_features(cache, routed, None, device), _cache_geometry(cache, device))
    thresholds = {}
    for task_name in selected_tasks:
        task = ATTRIBUTE_TASKS[task_name]
        mask = cache["valid"][task_name]
        if not bool(mask.any()):
            thresholds[task_name] = task.minimum_confidence
            continue
        expected = cache["targets"][task_name][mask].long()
        confidence, predicted = logits[task_name][mask].softmax(dim=-1).cpu().max(dim=-1)
        chosen = task.minimum_confidence
        for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            if threshold < task.minimum_confidence:
                continue
            accepted = confidence >= threshold
            if float(accepted.float().mean()) < 0.20:
                continue
            accuracy = float((predicted[accepted] == expected[accepted]).float().mean())
            if accuracy >= target_accuracy:
                chosen = threshold
                break
        thresholds[task_name] = chosen
    return thresholds


def train_attribute_heads(
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    output_checkpoint: str | Path,
    *,
    config: TrainingConfig | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    import torch

    settings = config or TrainingConfig()
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if isinstance(train_cache_path, dict) != isinstance(val_cache_path, dict):
        raise ValueError("train/val 캐시 지정 방식이 서로 다릅니다.")
    if isinstance(train_cache_path, dict):
        train_routed, train_cache = load_routed_caches(train_cache_path)
        val_routed, val_cache = load_routed_caches(val_cache_path)
        if set(train_routed) != set(val_routed):
            raise ValueError("train/val의 전처리 종류가 다릅니다.")
        task_routing = {
            task: mode for task, mode in TASK_PREPROCESSING.items() if task in ATTRIBUTE_TASKS
        }
        unavailable = set(task_routing.values()) - set(train_routed)
        if unavailable:
            raise ValueError(f"라우팅에 필요한 전처리 캐시가 없습니다: {sorted(unavailable)}")
        checkpoint_preprocessing = task_routing
    else:
        train_routed = val_routed = None
        train_cache = load_embedding_cache(train_cache_path)
        val_cache = load_embedding_cache(val_cache_path)
        task_routing = None
        checkpoint_preprocessing = train_cache["preprocessing"]
    if train_cache["backbone_model_id"] != val_cache["backbone_model_id"]:
        raise ValueError("train/val 임베딩 캐시의 백본이 다릅니다.")
    if train_cache["preprocessing"] != val_cache["preprocessing"]:
        raise ValueError("train/val 임베딩 캐시의 crop 처리 방식이 다릅니다.")
    input_dim = int(train_cache["features"].shape[1])
    if int(val_cache["features"].shape[1]) != input_dim:
        raise ValueError("train/val 임베딩 차원이 다릅니다.")

    use_geometry = settings.use_geometry
    if use_geometry and (train_cache["geometry"] is None or val_cache["geometry"] is None):
        raise ValueError(
            "기하 특징이 없는 예전 캐시입니다. 캐시를 다시 만들거나 use_geometry=False로 학습하세요."
        )
    geometry_dim = GEOMETRY_DIM if use_geometry else 0
    train_geometry = _cache_geometry(train_cache, selected_device, use_geometry=use_geometry)
    val_geometry = _cache_geometry(val_cache, selected_device, use_geometry=use_geometry)

    heads = build_attribute_heads(
        input_dim, settings.hidden_dim, settings.dropout, geometry_dim=geometry_dim,
        task_preprocessing=task_routing,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    pos_weights = _positive_weights(train_cache, selected_device)
    class_weights = _single_class_weights(train_cache, selected_device)
    train_size = len(train_cache["features"])
    best_task_states: dict[str, dict[str, Any]] = {}
    best_task_losses = {task_name: float("inf") for task_name in ATTRIBUTE_TASKS}
    stale_task_epochs = {task_name: 0 for task_name in ATTRIBUTE_TASKS}
    history = []

    for epoch in range(1, settings.epochs + 1):
        heads.train()
        permutation = torch.randperm(train_size)
        epoch_losses = []
        for start in range(0, train_size, settings.batch_size):
            indices = permutation[start:start + settings.batch_size]
            features = _select_features(train_cache, train_routed, indices, selected_device)
            targets = {name: value[indices].to(selected_device) for name, value in train_cache["targets"].items()}
            valid = {name: value[indices].to(selected_device) for name, value in train_cache["valid"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _multitask_loss(
                heads,
                features,
                targets,
                valid,
                pos_weights=pos_weights,
                class_weights=class_weights,
                geometry=None if train_geometry is None else train_geometry[indices],
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        heads.eval()
        with torch.inference_mode():
            val_loss, val_task_losses = _multitask_loss(
                heads,
                _select_features(val_cache, val_routed, None, selected_device),
                {name: value.to(selected_device) for name, value in val_cache["targets"].items()},
                {name: value.to(selected_device) for name, value in val_cache["valid"].items()},
                geometry=val_geometry,
            )
        row = {
            "epoch": epoch,
            "train_loss": round(sum(epoch_losses) / len(epoch_losses), 6),
            "val_loss": round(float(val_loss.cpu()), 6),
        }
        history.append(row)
        print(
            f"[속성 헤드] epoch {epoch}/{settings.epochs} "
            f"train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f}",
            flush=True,
        )
        for task_name, task_loss in val_task_losses.items():
            if task_loss < best_task_losses[task_name] - 1e-5:
                best_task_losses[task_name] = task_loss
                best_task_states[task_name] = {
                    name: value.detach().cpu().clone()
                    for name, value in heads.heads[task_name].state_dict().items()
                }
                stale_task_epochs[task_name] = 0
            else:
                stale_task_epochs[task_name] += 1
        tracked_tasks = set(val_task_losses)
        if tracked_tasks and all(stale_task_epochs[name] >= settings.patience for name in tracked_tasks):
            break

    if not best_task_states:
        raise RuntimeError("학습 중 유효한 체크포인트를 만들지 못했습니다.")
    for task_name, state in best_task_states.items():
        heads.heads[task_name].load_state_dict(state)
    heads.to(selected_device).eval()
    thresholds = _tune_multilabel_thresholds(heads, val_cache, selected_device, val_routed)
    thresholds.update(_tune_pants_single_thresholds(heads, val_cache, selected_device, routed=val_routed))
    metrics = _metrics(heads, val_cache, selected_device, thresholds, val_routed)
    label_support = _label_support(train_cache)
    summary = {
        "device": selected_device,
        "backbone_model_id": train_cache["backbone_model_id"],
        "train_samples": train_size,
        "val_samples": len(val_cache["features"]),
        "config": asdict(settings),
        "history": history,
        "metrics": metrics,
        "label_support": label_support,
    }
    save_attribute_checkpoint(
        output_checkpoint,
        heads,
        backbone_model_id=train_cache["backbone_model_id"],
        thresholds=thresholds,
        training_summary=summary,
        label_support=label_support,
        minimum_label_examples=settings.minimum_label_examples,
        preprocessing=checkpoint_preprocessing,
    )
    report_path = Path(output_checkpoint).expanduser().resolve().with_suffix(".metrics.json")
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_embedding_caches(
    annotation_csv: str | Path,
    image_root: str | Path,
    train_cache: str | Path,
    val_cache: str | Path,
    *,
    model_id: str = FASHION_SIGLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 64,
    reuse_cache_paths: list[str | Path] | None = None,
    preprocessing: str = PREPROCESS_SQUASH,
) -> tuple[Path, Path]:
    train_records = load_attribute_csv(annotation_csv, image_root, split="train")
    val_records = load_attribute_csv(annotation_csv, image_root, split="val")
    if not train_records or not val_records:
        raise ValueError("CSV에 train과 val split이 모두 있어야 합니다.")
    encoder = FrozenFashionSigLIPEncoder(model_id=model_id, device=device, preprocessing=preprocessing)
    return (
        build_embedding_cache(
            train_records,
            encoder,
            train_cache,
            batch_size=batch_size,
            reuse_cache_paths=reuse_cache_paths,
        ),
        build_embedding_cache(
            val_records,
            encoder,
            val_cache,
            batch_size=batch_size,
            reuse_cache_paths=reuse_cache_paths,
        ),
    )
