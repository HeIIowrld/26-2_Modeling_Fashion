from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


LAYERING_CHECKPOINT_VERSION = 1
LAYERING_LABELS = ("단일 옷", "겹쳐입음")
LAYER_COMPONENT_CATEGORIES = (
    "티셔츠", "폴로 셔츠", "셔츠", "블라우스", "니트", "가디건",
    "후드티", "재킷", "블레이저", "코트", "베스트", "탑",
)
LAYERING_ROI_NAMES = (
    "global", "neck_collar", "left_cuff", "right_cuff", "hem", "placket",
)


@dataclass(frozen=True)
class LayeringHeadPrediction:
    state: str
    confidence: float
    accepted: bool
    inner_category: str = "종류 불확실"
    outer_category: str = "종류 불확실"
    component_confidence: float = 0.0
    scores: dict[str, float] | None = None
    roi_attention: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    pixels = (
        max(0, min(width - 1, int(round(left * width)))),
        max(0, min(height - 1, int(round(top * height)))),
        max(1, min(width, int(round(right * width)))),
        max(1, min(height, int(round(bottom * height)))),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        return image.copy()
    return image.crop(pixels)


def layering_roi_crops(image: str | Path | Image.Image) -> dict[str, Image.Image]:
    """상의 crop에서 레이어드 단서가 나타나는 고정 ROI를 만든다.

    학습과 추론이 같은 좌표계를 사용하도록 포즈 좌표 대신 상의 bbox 상대좌표를
    사용한다. global은 실루엣·중첩 경계, 나머지는 칼라·커프스·밑단·플래킷을 본다.
    """
    pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    return {
        "global": pil,
        "neck_collar": _relative_crop(pil, (0.20, 0.00, 0.80, 0.38)),
        "left_cuff": _relative_crop(pil, (0.00, 0.32, 0.32, 0.86)),
        "right_cuff": _relative_crop(pil, (0.68, 0.32, 1.00, 0.86)),
        "hem": _relative_crop(pil, (0.08, 0.66, 0.92, 1.00)),
        "placket": _relative_crop(pil, (0.34, 0.10, 0.66, 0.90)),
    }


def build_layering_heads(input_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
    import torch
    import torch.nn as nn

    class LayeringHeads(nn.Module):
        """ROI별 위치 임베딩과 attention으로 여섯 레이어드 단서를 결합한다."""

        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.roi_names = LAYERING_ROI_NAMES
            self.norm = nn.LayerNorm(input_dim)
            self.position = nn.Parameter(torch.zeros(len(LAYERING_ROI_NAMES), input_dim))
            nn.init.normal_(self.position, std=0.02)
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
            )
            fused_dim = input_dim * 2

            def head(output_dim: int):
                return nn.Sequential(
                    nn.LayerNorm(fused_dim),
                    nn.Linear(fused_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )

            self.layering = head(len(LAYERING_LABELS))
            self.inner_category = head(len(LAYER_COMPONENT_CATEGORIES))
            self.outer_category = head(len(LAYER_COMPONENT_CATEGORIES))

        def forward(self, roi_features):
            if roi_features.dim() != 3:
                raise ValueError("레이어드 특징은 [batch, roi, feature] 형태여야 합니다.")
            if roi_features.shape[1] != len(self.roi_names):
                raise ValueError(
                    f"레이어드 ROI가 {len(self.roi_names)}개 필요합니다: {self.roi_names}"
                )
            normalized = self.norm(roi_features) + self.position.unsqueeze(0)
            weights = self.attention(normalized).squeeze(-1).softmax(dim=1)
            attended = (normalized * weights.unsqueeze(-1)).sum(dim=1)
            # 전체 상의 특징을 항상 남겨 작은 ROI만 보고 오판하는 것을 막는다.
            fused = torch.cat([normalized[:, 0], attended], dim=-1)
            return {
                "layering": self.layering(fused),
                "inner_category": self.inner_category(fused),
                "outer_category": self.outer_category(fused),
                "roi_attention": weights,
            }

    return LayeringHeads()


def save_layering_checkpoint(
    path: str | Path,
    heads,
    *,
    backbone_model_id: str,
    single_max_probability: float = 0.30,
    layered_min_probability: float = 0.70,
    component_min_probability: float = 0.45,
    training_summary: dict[str, Any] | None = None,
) -> Path:
    import torch

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": LAYERING_CHECKPOINT_VERSION,
            "backbone_model_id": backbone_model_id,
            "input_dim": heads.input_dim,
            "hidden_dim": heads.hidden_dim,
            "dropout": heads.dropout,
            "roi_names": list(LAYERING_ROI_NAMES),
            "layering_labels": list(LAYERING_LABELS),
            "component_categories": list(LAYER_COMPONENT_CATEGORIES),
            "thresholds": {
                "single_max_probability": float(single_max_probability),
                "layered_min_probability": float(layered_min_probability),
                "component_min_probability": float(component_min_probability),
            },
            "state_dict": heads.state_dict(),
            "training_summary": training_summary or {},
        },
        output,
    )
    return output


def _torch_load(path: str | Path, device: str):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_layering_heads(path: str | Path, device: str = "cpu"):
    payload = _torch_load(path, device)
    if payload.get("version") != LAYERING_CHECKPOINT_VERSION:
        raise ValueError(f"지원하지 않는 레이어드 체크포인트입니다: {payload.get('version')}")
    if tuple(payload.get("roi_names", ())) != LAYERING_ROI_NAMES:
        raise ValueError("레이어드 체크포인트의 ROI 구성이 현재 코드와 다릅니다.")
    if tuple(payload.get("layering_labels", ())) != LAYERING_LABELS:
        raise ValueError("레이어드 체크포인트의 상태 라벨이 현재 코드와 다릅니다.")
    if tuple(payload.get("component_categories", ())) != LAYER_COMPONENT_CATEGORIES:
        raise ValueError("레이어드 체크포인트의 안옷·겉옷 라벨이 현재 코드와 다릅니다.")
    heads = build_layering_heads(
        int(payload["input_dim"]), int(payload["hidden_dim"]), float(payload["dropout"])
    )
    heads.load_state_dict(payload["state_dict"])
    heads.to(device).eval()
    return heads, payload


class LayeringPredictor:
    """기존 FashionSigLIP 백본을 공유해 멀티 ROI 레이어드 헤드를 실행한다."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        image_encoder,
        preprocess,
        model_id: str,
        device: str,
    ) -> None:
        self.device = device
        self.image_encoder = image_encoder
        self.preprocess = preprocess
        self.heads, self.metadata = load_layering_heads(checkpoint_path, device)
        if self.metadata["backbone_model_id"] != model_id:
            raise ValueError(
                "레이어드 헤드와 이미지 백본이 다릅니다. "
                f"학습 백본={self.metadata['backbone_model_id']}, 현재 백본={model_id}"
            )
        self.thresholds = self.metadata.get("thresholds", {})

    def predict(self, upper_crop: str | Path | Image.Image) -> LayeringHeadPrediction:
        import torch

        rois = layering_roi_crops(upper_crop)
        batch = torch.stack([self.preprocess(rois[name]) for name in LAYERING_ROI_NAMES]).to(self.device)
        with torch.inference_mode():
            features = self.image_encoder.encode_image(batch, normalize=True).float().unsqueeze(0)
            output = self.heads(features)
            probabilities = output["layering"][0].softmax(dim=-1).cpu()
            inner_probabilities = output["inner_category"][0].softmax(dim=-1).cpu()
            outer_probabilities = output["outer_category"][0].softmax(dim=-1).cpu()
            attention = output["roi_attention"][0].cpu()

        layered_probability = float(probabilities[LAYERING_LABELS.index("겹쳐입음")])
        layered_threshold = float(self.thresholds.get("layered_min_probability", 0.70))
        single_threshold = float(self.thresholds.get("single_max_probability", 0.30))
        if layered_probability >= layered_threshold:
            state, accepted = "레이어드", True
        elif layered_probability <= single_threshold:
            state, accepted = "단일 상의", True
        else:
            state, accepted = "판단 보류", False

        inner_index = int(inner_probabilities.argmax())
        outer_index = int(outer_probabilities.argmax())
        inner_confidence = float(inner_probabilities[inner_index])
        outer_confidence = float(outer_probabilities[outer_index])
        component_threshold = float(self.thresholds.get("component_min_probability", 0.45))
        inner = (
            LAYER_COMPONENT_CATEGORIES[inner_index]
            if state == "레이어드" and inner_confidence >= component_threshold
            else "종류 불확실"
        )
        outer = (
            LAYER_COMPONENT_CATEGORIES[outer_index]
            if state == "레이어드" and outer_confidence >= component_threshold
            else "종류 불확실"
        )
        return LayeringHeadPrediction(
            state=state,
            confidence=max(layered_probability, 1.0 - layered_probability),
            accepted=accepted,
            inner_category=inner,
            outer_category=outer,
            component_confidence=min(inner_confidence, outer_confidence),
            scores={"단일 옷": 1.0 - layered_probability, "겹쳐입음": layered_probability},
            roi_attention={name: float(attention[index]) for index, name in enumerate(LAYERING_ROI_NAMES)},
        )
