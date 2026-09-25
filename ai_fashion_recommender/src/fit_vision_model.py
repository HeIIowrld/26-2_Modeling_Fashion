"""FashionSigLIP 위에 얹는 사용자/상품 공용 핏 분류 헤드."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fashion_attribute_model import PREPROCESS_LETTERBOX, apply_preprocess_mode
from fit_vision_schema import FIT_VISION_TASKS, USABLE_QUALITY, tasks_for_category


FIT_VISION_CHECKPOINT_VERSION = 1
FIT_GEOMETRY_LEVELS = (0.10, 0.25, 0.45, 0.65, 0.85)
FIT_GEOMETRY_DIM = 10
FIT_FEATURE_MODES = (
    "rgb_only",
    "masked_only",
    "rgb_mask",
    "rgb_geometry",
    "rgb_mask_geometry",
)


@dataclass(frozen=True)
class FitVisionPrediction:
    labels: list[str]
    scores: dict[str, float]
    confidence: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_geometry_vector(mask: np.ndarray | None) -> list[float]:
    """의류 마스크의 폭 변화를 크기와 무관한 10차원 특징으로 만든다."""
    if mask is None:
        return [0.0] * FIT_GEOMETRY_DIM
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2 or binary.sum() < 20:
        return [0.0] * FIT_GEOMETRY_DIM
    ys, xs = np.where(binary)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = binary[y1:y2, x1:x2]
    height, width = crop.shape
    widths = []
    band = max(1, int(round(height * 0.025)))
    for level in FIT_GEOMETRY_LEVELS:
        row = min(height - 1, max(0, int(round((height - 1) * level))))
        samples = []
        for y in range(max(0, row - band), min(height, row + band + 1)):
            columns = np.where(crop[y])[0]
            if len(columns):
                samples.append(columns.max() - columns.min() + 1)
        widths.append(float(np.median(samples)) / max(width, 1) if samples else 0.0)
    area_ratio = float(crop.mean())
    aspect = math.tanh(math.log(max(width, 1) / max(height, 1)))
    top, middle, hem = widths[1], widths[2], widths[-1]
    return [
        aspect, area_ratio, *widths,
        middle / max(top, 1e-6),
        hem / max(middle, 1e-6),
        1.0,
    ]


def _crop_views(image: Image.Image, mask: np.ndarray | None) -> tuple[Image.Image, Image.Image, np.ndarray | None]:
    rgb = np.asarray(image.convert("RGB"))
    if mask is None:
        return image.convert("RGB"), image.convert("RGB"), None
    binary = np.asarray(mask).astype(bool)
    if binary.shape != rgb.shape[:2] or binary.sum() < 20:
        return image.convert("RGB"), image.convert("RGB"), None
    ys, xs = np.where(binary)
    margin = max(2, int(round(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.08)))
    y1, y2 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    x1, x2 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    context = rgb[y1:y2, x1:x2].copy()
    local_mask = binary[y1:y2, x1:x2]
    isolated = context.copy()
    isolated[~local_mask] = 255
    return Image.fromarray(context), Image.fromarray(isolated), local_mask


def build_fit_vision_heads(input_dim: int, hidden_dim: int = 256, dropout: float = 0.20):
    import torch
    import torch.nn as nn

    class FitVisionHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.norm = nn.LayerNorm(input_dim * 2)
            self.geometry = nn.Sequential(
                nn.LayerNorm(FIT_GEOMETRY_DIM),
                nn.Linear(FIT_GEOMETRY_DIM, 32),
                nn.GELU(),
            )
            self.heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(input_dim * 2 + 32, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, len(task.labels)),
                )
                for name, task in FIT_VISION_TASKS.items()
            })

        def forward(self, context_features, garment_features, geometry, tasks=None):
            shared = self.norm(torch.cat((context_features, garment_features), dim=-1))
            fused = torch.cat((shared, self.geometry(geometry)), dim=-1)
            selected = tasks or tuple(self.heads)
            return {name: self.heads[name](fused) for name in selected}

    return FitVisionHeads()


def apply_fit_feature_mode(context, garment, geometry, mode: str):
    """동일 구조에서 RGB/Mask/Geometry ablation을 재현 가능하게 만든다."""
    if mode not in FIT_FEATURE_MODES:
        raise ValueError(f"지원하지 않는 핏 feature mode입니다: {mode}")
    if mode == "rgb_only":
        return context, garment.new_zeros(garment.shape), geometry.new_zeros(geometry.shape)
    if mode == "masked_only":
        return context.new_zeros(context.shape), garment, geometry.new_zeros(geometry.shape)
    if mode == "rgb_mask":
        return context, garment, geometry.new_zeros(geometry.shape)
    if mode == "rgb_geometry":
        return context, garment.new_zeros(garment.shape), geometry
    return context, garment, geometry


def save_fit_vision_checkpoint(
    path: str | Path,
    heads,
    *,
    backbone_model_id: str,
    thresholds: dict[str, float] | None = None,
    label_support: dict[str, dict[str, int]] | None = None,
    training_summary: dict[str, Any] | None = None,
) -> Path:
    import torch

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": FIT_VISION_CHECKPOINT_VERSION,
        "backbone_model_id": backbone_model_id,
        "input_dim": heads.input_dim,
        "hidden_dim": heads.hidden_dim,
        "dropout": heads.dropout,
        "tasks": {name: list(task.labels) for name, task in FIT_VISION_TASKS.items()},
        "thresholds": thresholds or {
            name: task.minimum_confidence for name, task in FIT_VISION_TASKS.items()
        },
        "label_support": label_support or {},
        "state_dict": heads.state_dict(),
        "training_summary": training_summary or {},
    }, output)
    return output


def load_fit_vision_heads(path: str | Path, device: str = "cpu"):
    import torch

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("version") != FIT_VISION_CHECKPOINT_VERSION:
        raise ValueError(f"지원하지 않는 핏 비전 체크포인트입니다: {payload.get('version')}")
    expected = {name: list(task.labels) for name, task in FIT_VISION_TASKS.items()}
    if payload.get("tasks") != expected:
        raise ValueError("핏 비전 체크포인트의 라벨 순서가 현재 코드와 다릅니다.")
    heads = build_fit_vision_heads(payload["input_dim"], payload["hidden_dim"], payload["dropout"])
    heads.load_state_dict(payload["state_dict"])
    heads.to(device).eval()
    return heads, payload


class FitVisionPredictor:
    """하나의 체크포인트를 사용자 crop과 상품 착용 사진에 공통 사용한다."""

    def __init__(self, checkpoint_path, *, image_encoder, preprocess, model_id: str, device: str):
        self.device = device
        self.image_encoder = image_encoder
        self.preprocess = preprocess
        self.heads, self.metadata = load_fit_vision_heads(checkpoint_path, device)
        if self.metadata["backbone_model_id"] != model_id:
            raise ValueError("핏 비전 헤드와 FashionSigLIP 백본이 다릅니다.")
        self.thresholds = self.metadata.get("thresholds", {})
        self.label_support = self.metadata.get("label_support", {})
        self.feature_mode = (
            self.metadata.get("training_summary", {}).get("config", {}).get(
                "feature_mode", "rgb_mask_geometry"
            )
        )

    def predict(self, image, *, category: str, mask: np.ndarray | None = None):
        import torch

        tasks = tasks_for_category(category)
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        context, isolated, local_mask = _crop_views(pil, mask)
        batch = torch.stack([
            self.preprocess(apply_preprocess_mode(view, PREPROCESS_LETTERBOX))
            for view in (context, isolated)
        ]).to(self.device)
        with torch.inference_mode():
            encoded = self.image_encoder.encode_image(batch, normalize=True).float()
            geometry = torch.tensor(
                [fit_geometry_vector(local_mask)], dtype=torch.float32, device=self.device
            )
            context_features, garment_features, geometry = apply_fit_feature_mode(
                encoded[:1], encoded[1:], geometry, self.feature_mode
            )
            logits = self.heads(context_features, garment_features, geometry, tasks)
        results = {}
        for name in tasks:
            task = FIT_VISION_TASKS[name]
            probabilities = logits[name][0].softmax(dim=-1).cpu()
            index = int(probabilities.argmax())
            label = task.labels[index]
            confidence = float(probabilities[index])
            support = self.label_support.get(name, {})
            supported = not support or int(support.get(label, 0)) > 0
            accepted = supported and confidence >= float(self.thresholds.get(name, task.minimum_confidence))
            results[name] = FitVisionPrediction(
                [label] if accepted else [],
                {value: float(probabilities[i]) for i, value in enumerate(task.labels)},
                confidence,
                accepted,
            )
        quality = results.get("quality")
        if quality and quality.accepted and quality.labels != [USABLE_QUALITY]:
            for name in tasks:
                if name != "quality":
                    value = results[name]
                    results[name] = FitVisionPrediction([], value.scores, value.confidence, False)
        return results
