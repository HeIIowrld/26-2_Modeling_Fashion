"""고정 FashionSigLIP 임베딩으로 공용 핏 헤드를 학습한다."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import FASHION_SIGLIP_MODEL_ID
from fashion_attribute_training import FrozenFashionSigLIPEncoder
from fit_vision_dataset import FitVisionRecord, load_fit_vision_csv
from fit_vision_model import (
    FIT_GEOMETRY_DIM,
    FIT_FEATURE_MODES,
    _crop_views,
    apply_fit_feature_mode,
    build_fit_vision_heads,
    fit_geometry_vector,
    save_fit_vision_checkpoint,
)
from fit_vision_schema import FIT_VISION_TASKS


@dataclass
class FitVisionTrainingConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.20
    patience: int = 6
    seed: int = 42
    feature_mode: str = "rgb_mask_geometry"


def _load_mask(record: FitVisionRecord, size: tuple[int, int]) -> np.ndarray | None:
    if record.mask_path is None:
        return None
    mask = Image.open(record.mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask) > 127


def build_fit_embedding_cache(
    records: list[FitVisionRecord],
    output_path: str | Path,
    *,
    model_id: str = FASHION_SIGLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 32,
) -> Path:
    import torch

    if not records:
        raise ValueError("핏 임베딩을 만들 레코드가 없습니다.")
    encoder = FrozenFashionSigLIPEncoder(model_id, device, preprocessing="letterbox")
    context_features, garment_features, geometries = [], [], []
    for start in range(0, len(records), batch_size):
        context_batch, garment_batch = [], []
        for record in records[start:start + batch_size]:
            image = Image.open(record.image_path).convert("RGB")
            context, garment, local_mask = _crop_views(image, _load_mask(record, image.size))
            context_batch.append(context)
            garment_batch.append(garment)
            geometries.append(fit_geometry_vector(local_mask))
        context_features.append(encoder.encode_pil_batch(context_batch))
        garment_features.append(encoder.encode_pil_batch(garment_batch))
        print(f"[핏 임베딩] {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    targets, valid = {}, {}
    for task_name, task in FIT_VISION_TASKS.items():
        index = {label: i for i, label in enumerate(task.labels)}
        targets[task_name] = torch.tensor([
            index.get(record.labels.get(task_name, ""), -1) for record in records
        ], dtype=torch.long)
        valid[task_name] = targets[task_name] >= 0
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "backbone_model_id": model_id,
        "context_features": torch.cat(context_features),
        "garment_features": torch.cat(garment_features),
        "geometry": torch.tensor(geometries, dtype=torch.float32),
        "targets": targets,
        "valid": valid,
        "image_paths": [str(record.image_path) for record in records],
        "source_domains": [record.source_domain for record in records],
        "group_ids": [record.group_id for record in records],
    }, output)
    return output


def load_fit_cache(path: str | Path) -> dict:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("version") != 1 or int(payload["geometry"].shape[1]) != FIT_GEOMETRY_DIM:
        raise ValueError("지원하지 않는 핏 임베딩 캐시입니다.")
    return payload


def _macro_f1(expected, predicted, label_count: int) -> float:
    scores = []
    for label in range(label_count):
        tp = int(((expected == label) & (predicted == label)).sum())
        fp = int(((expected != label) & (predicted == label)).sum())
        fn = int(((expected == label) & (predicted != label)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(sum(scores) / len(scores))


def _classification_metrics(expected, predicted, labels: tuple[str, ...]) -> dict:
    per_class = {}
    for index, label in enumerate(labels):
        tp = int(((expected == index) & (predicted == index)).sum())
        fp = int(((expected != index) & (predicted == index)).sum())
        fn = int(((expected == index) & (predicted != index)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        per_class[label] = 2 * precision * recall / max(1e-12, precision + recall)
    active_indices = [index for index in range(len(labels)) if bool((expected == index).any())]
    active_scores = [per_class[labels[index]] for index in active_indices]
    return {
        "samples": int(expected.numel()),
        "accuracy": float((expected == predicted).float().mean()),
        "macro_f1": _macro_f1(expected, predicted, len(labels)),
        # Small, explicitly scoped experiments may intentionally train only a
        # subset of the production label contract. Keep the full-contract score
        # above, and report the supported-label score separately.
        "active_macro_f1": float(sum(active_scores) / max(1, len(active_scores))),
        "active_labels": [labels[index] for index in active_indices],
        "per_class_f1": per_class,
    }


def _evaluate_cache(heads, cache: dict, tensors) -> dict:
    """Evaluate one cache without changing the checkpoint-selection split."""
    import torch

    metrics = {}
    with torch.inference_mode():
        context, garment, geometry = tensors(cache)
        tasks = [name for name in FIT_VISION_TASKS if bool(cache["valid"][name].any())]
        outputs = heads(context, garment, geometry, tasks)
    for name in tasks:
        task = FIT_VISION_TASKS[name]
        mask = cache["valid"][name]
        expected = cache["targets"][name][mask]
        predicted = outputs[name].cpu()[mask].argmax(dim=-1)
        metrics[name] = _classification_metrics(expected, predicted, task.labels)
        metrics[name]["by_domain"] = {}
        for domain in sorted(set(cache.get("source_domains", []))):
            domain_mask = torch.tensor(
                [value == domain for value in cache["source_domains"]], dtype=torch.bool
            ) & mask
            if not domain_mask.any():
                continue
            domain_expected = cache["targets"][name][domain_mask]
            domain_predicted = outputs[name].cpu()[domain_mask].argmax(dim=-1)
            metrics[name]["by_domain"][domain] = _classification_metrics(
                domain_expected, domain_predicted, task.labels
            )
    return metrics


def train_fit_vision_heads(
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    output_checkpoint: str | Path,
    *,
    test_cache_path: str | Path | None = None,
    config: FitVisionTrainingConfig | None = None,
    device: str = "auto",
) -> dict:
    import torch
    import torch.nn.functional as functional

    settings = config or FitVisionTrainingConfig()
    if settings.feature_mode not in FIT_FEATURE_MODES:
        raise ValueError(f"feature_mode은 다음 중 하나여야 합니다: {FIT_FEATURE_MODES}")
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    train, val = load_fit_cache(train_cache_path), load_fit_cache(val_cache_path)
    test = load_fit_cache(test_cache_path) if test_cache_path else None
    if train["backbone_model_id"] != val["backbone_model_id"]:
        raise ValueError("train/val 핏 캐시의 백본이 다릅니다.")
    if test is not None and train["backbone_model_id"] != test["backbone_model_id"]:
        raise ValueError("train/test 핏 캐시의 백본이 다릅니다.")
    input_dim = int(train["context_features"].shape[1])
    heads = build_fit_vision_heads(input_dim, settings.hidden_dim, settings.dropout).to(selected_device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    best_state, best_loss, stale, history = None, float("inf"), 0, []

    def tensors(cache, indices=None):
        pick = slice(None) if indices is None else indices
        values = (cache["context_features"][pick].to(selected_device),
                  cache["garment_features"][pick].to(selected_device),
                  cache["geometry"][pick].to(selected_device))
        return apply_fit_feature_mode(*values, settings.feature_mode)

    def loss_for(cache, outputs, indices=None):
        pick = slice(None) if indices is None else indices
        losses = []
        for name in outputs:
            valid = cache["valid"][name][pick].to(selected_device)
            if valid.any():
                expected = cache["targets"][name][pick].to(selected_device)
                losses.append(functional.cross_entropy(outputs[name][valid], expected[valid]))
        if not losses:
            raise ValueError("선택한 batch에 유효한 핏 라벨이 없습니다.")
        return sum(losses) / len(losses)

    size = len(train["context_features"])
    for epoch in range(1, settings.epochs + 1):
        heads.train()
        order = torch.randperm(size)
        train_losses = []
        for start in range(0, size, settings.batch_size):
            indices = order[start:start + settings.batch_size]
            context, garment, geometry = tensors(train, indices)
            tasks = [name for name in FIT_VISION_TASKS if bool(train["valid"][name][indices].any())]
            optimizer.zero_grad(set_to_none=True)
            outputs = heads(context, garment, geometry, tasks)
            loss = loss_for(train, outputs, indices)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        heads.eval()
        with torch.inference_mode():
            context, garment, geometry = tensors(val)
            tasks = [name for name in FIT_VISION_TASKS if bool(val["valid"][name].any())]
            outputs = heads(context, garment, geometry, tasks)
            val_loss = float(loss_for(val, outputs).cpu())
        row = {"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses), "val_loss": val_loss}
        history.append(row)
        print(f"[핏 헤드] epoch {epoch}/{settings.epochs} train={row['train_loss']:.5f} val={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in heads.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is None:
        raise RuntimeError("유효한 핏 체크포인트를 만들지 못했습니다.")
    heads.load_state_dict(best_state)
    heads.to(selected_device).eval()

    support = {}
    for name, task in FIT_VISION_TASKS.items():
        support[name] = {label: int((train["targets"][name][train["valid"][name]] == i).sum())
                         for i, label in enumerate(task.labels)}
    metrics = _evaluate_cache(heads, val, tensors)
    summary = {"device": selected_device, "config": asdict(settings), "history": history,
               "metrics": metrics, "label_support": support,
               "deployment_status": "research_only_until_human_label_review"}
    if test is not None:
        summary["test_metrics"] = _evaluate_cache(heads, test, tensors)
    save_fit_vision_checkpoint(output_checkpoint, heads,
        backbone_model_id=train["backbone_model_id"], label_support=support,
        training_summary=summary)
    Path(output_checkpoint).with_suffix(".metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def prepare_fit_caches(
    csv_path, image_root, train_cache, val_cache, test_cache=None, **kwargs
):
    records = load_fit_vision_csv(csv_path, image_root)
    outputs = [
        build_fit_embedding_cache([r for r in records if r.split == "train"], train_cache, **kwargs),
        build_fit_embedding_cache([r for r in records if r.split == "val"], val_cache, **kwargs),
    ]
    if test_cache:
        outputs.append(build_fit_embedding_cache(
            [r for r in records if r.split == "test"], test_cache, **kwargs
        ))
    return tuple(outputs)
