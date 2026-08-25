from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from config import FASHION_SIGLIP_MODEL_ID
from fashion_attribute_training import FrozenFashionSigLIPEncoder
from layering_model import (
    LAYER_COMPONENT_CATEGORIES,
    LAYERING_ROI_NAMES,
    build_layering_heads,
    layering_roi_crops,
    save_layering_checkpoint,
)


@dataclass(frozen=True)
class LayeringRecord:
    image_path: Path
    split: str
    is_layered: int
    inner_category: str | None = None
    outer_category: str | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class LayeringTrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.15
    patience: int = 5
    seed: int = 42
    component_loss_weight: float = 0.50


def _parse_state(value: str, *, source: str) -> int:
    normalized = value.strip().lower().replace("_", " ")
    if normalized in {"1", "true", "layered", "겹쳐입음", "레이어드"}:
        return 1
    if normalized in {"0", "false", "single", "single garment", "단일", "단일 옷", "단일 상의"}:
        return 0
    raise ValueError(f"{source} is_layered는 0/1, 단일 옷/겹쳐입음 중 하나여야 합니다: {value}")


def _parse_bbox(row: dict[str, str], *, source: str):
    values = [(row.get(name) or "").strip() for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]
    if not any(values):
        return None
    if not all(values):
        raise ValueError(f"{source} bbox_x, bbox_y, bbox_w, bbox_h를 모두 입력해야 합니다.")
    bbox = tuple(float(value) for value in values)
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"{source} bbox_w와 bbox_h는 0보다 커야 합니다.")
    return bbox


def load_layering_csv(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    *,
    split: str | None = None,
    require_images: bool = True,
) -> list[LayeringRecord]:
    annotation = Path(csv_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else annotation.parent
    records = []
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "is_layered"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("레이어드 CSV에는 image_path와 is_layered 열이 필요합니다.")
        for line_number, row in enumerate(reader, start=2):
            row_split = (row.get("split") or "train").strip().lower()
            if split and row_split != split.lower():
                continue
            source = f"{annotation.name}:{line_number}"
            image_path = Path((row.get("image_path") or "").strip()).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
            image_path = image_path.resolve()
            if require_images and not image_path.is_file():
                raise FileNotFoundError(f"{source} 이미지가 없습니다: {image_path}")
            state = _parse_state(row.get("is_layered") or "", source=source)
            inner = (row.get("inner_category") or "").strip() or None
            outer = (row.get("outer_category") or "").strip() or None
            for field_name, value in (("inner_category", inner), ("outer_category", outer)):
                if value and value not in LAYER_COMPONENT_CATEGORIES:
                    raise ValueError(f"{source} {field_name}에 정의되지 않은 라벨입니다: {value}")
            if state == 0 and (inner or outer):
                raise ValueError(f"{source} 단일 옷에는 inner_category/outer_category를 입력하지 마세요.")
            records.append(
                LayeringRecord(image_path, row_split, state, inner, outer, _parse_bbox(row, source=source))
            )
    return records


def _upper_crop(record: LayeringRecord) -> Image.Image:
    image = Image.open(record.image_path).convert("RGB")
    if record.bbox is None:
        return image
    x, y, width, height = record.bbox
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(image.width, math.ceil(x + width)), min(image.height, math.ceil(y + height))
    if right <= left or bottom <= top:
        raise ValueError(f"이미지 범위를 벗어난 bbox입니다: {record.image_path}, {record.bbox}")
    return image.crop((left, top, right, bottom))


def build_layering_embedding_cache(
    records: list[LayeringRecord],
    encoder: FrozenFashionSigLIPEncoder,
    output_path: str | Path,
    *,
    batch_size: int = 16,
) -> Path:
    import torch

    if not records:
        raise ValueError("레이어드 임베딩을 만들 레코드가 없습니다.")
    batches = []
    for start in range(0, len(records), batch_size):
        selected = records[start:start + batch_size]
        flattened = []
        for record in selected:
            rois = layering_roi_crops(_upper_crop(record))
            flattened.extend(rois[name] for name in LAYERING_ROI_NAMES)
        encoded = encoder.encode_pil_batch(flattened)
        batches.append(encoded.reshape(len(selected), len(LAYERING_ROI_NAMES), -1))
        print(
            f"[레이어드 ROI 임베딩] {min(start + batch_size, len(records))}/{len(records)}",
            flush=True,
        )
    categories = {label: index for index, label in enumerate(LAYER_COMPONENT_CATEGORIES)}
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "backbone_model_id": encoder.model_id,
            "roi_names": list(LAYERING_ROI_NAMES),
            "features": torch.cat(batches, dim=0),
            "layering_targets": torch.tensor([record.is_layered for record in records], dtype=torch.long),
            "inner_targets": torch.tensor(
                [categories.get(record.inner_category, -1) for record in records], dtype=torch.long
            ),
            "outer_targets": torch.tensor(
                [categories.get(record.outer_category, -1) for record in records], dtype=torch.long
            ),
            "image_paths": [str(record.image_path) for record in records],
        },
        output,
    )
    return output


def _load_cache(path: str | Path):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("version") != 1 or tuple(payload.get("roi_names", ())) != LAYERING_ROI_NAMES:
        raise ValueError("현재 코드와 맞지 않는 레이어드 임베딩 캐시입니다.")
    return payload


def prepare_layering_caches(
    annotation_csv: str | Path,
    image_root: str | Path,
    train_cache: str | Path,
    val_cache: str | Path,
    *,
    model_id: str = FASHION_SIGLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 16,
) -> tuple[Path, Path]:
    train = load_layering_csv(annotation_csv, image_root, split="train")
    val = load_layering_csv(annotation_csv, image_root, split="val")
    if not train or not val:
        raise ValueError("레이어드 CSV에 train과 val split이 모두 필요합니다.")
    encoder = FrozenFashionSigLIPEncoder(model_id=model_id, device=device)
    return (
        build_layering_embedding_cache(train, encoder, train_cache, batch_size=batch_size),
        build_layering_embedding_cache(val, encoder, val_cache, batch_size=batch_size),
    )


def _metrics(heads, cache, device: str) -> dict:
    import torch

    heads.eval()
    with torch.inference_mode():
        output = heads(cache["features"].to(device))
    state_prediction = output["layering"].argmax(dim=-1).cpu()
    state_accuracy = float((state_prediction == cache["layering_targets"]).float().mean())
    result = {"layering_accuracy": round(state_accuracy, 4)}
    for name in ("inner", "outer"):
        expected = cache[f"{name}_targets"]
        valid = expected >= 0
        result[f"{name}_samples"] = int(valid.sum())
        result[f"{name}_accuracy"] = (
            round(float((output[f"{name}_category"].argmax(dim=-1).cpu()[valid] == expected[valid]).float().mean()), 4)
            if bool(valid.any()) else None
        )
    return result


def train_layering_heads(
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    output_checkpoint: str | Path,
    *,
    config: LayeringTrainingConfig | None = None,
    device: str = "auto",
) -> dict:
    import torch
    import torch.nn.functional as functional

    settings = config or LayeringTrainingConfig()
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    train, val = _load_cache(train_cache_path), _load_cache(val_cache_path)
    if train["backbone_model_id"] != val["backbone_model_id"]:
        raise ValueError("train/val 레이어드 캐시의 백본이 다릅니다.")
    input_dim = int(train["features"].shape[-1])
    heads = build_layering_heads(input_dim, settings.hidden_dim, settings.dropout).to(selected_device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    state_counts = torch.bincount(train["layering_targets"], minlength=2).float().clamp_min(1)
    state_weights = (state_counts.sum() / state_counts).to(selected_device)
    state_weights /= state_weights.mean()
    best_loss, stale, best_state = float("inf"), 0, None
    history = []
    size = len(train["features"])

    for epoch in range(1, settings.epochs + 1):
        heads.train()
        permutation = torch.randperm(size)
        losses = []
        for start in range(0, size, settings.batch_size):
            indices = permutation[start:start + settings.batch_size]
            output = heads(train["features"][indices].to(selected_device))
            state_target = train["layering_targets"][indices].to(selected_device)
            loss = functional.cross_entropy(output["layering"], state_target, weight=state_weights)
            for name in ("inner", "outer"):
                target = train[f"{name}_targets"][indices].to(selected_device)
                valid = target >= 0
                if bool(valid.any()):
                    loss = loss + settings.component_loss_weight * functional.cross_entropy(
                        output[f"{name}_category"][valid], target[valid]
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        heads.eval()
        with torch.inference_mode():
            output = heads(val["features"].to(selected_device))
            validation_loss = functional.cross_entropy(
                output["layering"], val["layering_targets"].to(selected_device), weight=state_weights
            )
            for name in ("inner", "outer"):
                target = val[f"{name}_targets"].to(selected_device)
                valid = target >= 0
                if bool(valid.any()):
                    validation_loss = validation_loss + settings.component_loss_weight * functional.cross_entropy(
                        output[f"{name}_category"][valid], target[valid]
                    )
        row = {
            "epoch": epoch,
            "train_loss": round(sum(losses) / max(len(losses), 1), 6),
            "val_loss": round(float(validation_loss.cpu()), 6),
        }
        history.append(row)
        print(
            f"[레이어드 헤드] epoch {epoch}/{settings.epochs} "
            f"train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f}",
            flush=True,
        )
        if row["val_loss"] < best_loss - 1e-5:
            best_loss, stale = row["val_loss"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in heads.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is None:
        raise RuntimeError("레이어드 헤드 체크포인트를 만들지 못했습니다.")
    heads.load_state_dict(best_state)
    heads.to(selected_device).eval()
    metrics = _metrics(heads, val, selected_device)
    summary = {
        "device": selected_device,
        "backbone_model_id": train["backbone_model_id"],
        "train_samples": size,
        "val_samples": len(val["features"]),
        "config": asdict(settings),
        "history": history,
        "metrics": metrics,
    }
    save_layering_checkpoint(
        output_checkpoint,
        heads,
        backbone_model_id=train["backbone_model_id"],
        training_summary=summary,
    )
    report = Path(output_checkpoint).expanduser().resolve().with_suffix(".metrics.json")
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
