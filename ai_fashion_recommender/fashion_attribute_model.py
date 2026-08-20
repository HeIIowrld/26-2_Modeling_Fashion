from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fashion_attribute_schema import ATTRIBUTE_TASKS


CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2)

# FashionSigLIP 전처리는 crop을 224x224 정사각형으로 눌러 종횡비를 없앤다.
# 핏·다리 모양·기장처럼 비율이 핵심인 속성은 그 정보 없이는 맞힐 수 없으므로
# crop 자체에서 잰 기하 특징을 임베딩 옆에 따로 붙인다.
GEOMETRY_DIM = 2

# crop을 224x224로 만드는 방식.
#   squash    : FashionSigLIP 기본 전처리. 정사각형으로 눌러 종횡비를 없앤다.
#   letterbox : 종횡비를 유지한 채 여백을 채워 정사각형으로 만든다.
# 학습에 쓴 방식을 체크포인트에 적어 두고 추론에서 그대로 재현한다.
PREPROCESS_SQUASH = "squash"
PREPROCESS_LETTERBOX = "letterbox"
PREPROCESS_MODES = (PREPROCESS_SQUASH, PREPROCESS_LETTERBOX)


def letterbox_image(image, fill: tuple[int, int, int] = (255, 255, 255)):
    """종횡비를 유지한 채 정사각형 캔버스 가운데에 놓는다.

    기본 전처리는 crop을 눌러 정사각형으로 만들기 때문에 스키니와 와이드가 같은 입력이 된다.
    여백을 채워 정사각형을 만들면 옷의 실제 비율이 인코더에 그대로 들어간다.
    """
    from PIL import Image

    side = max(image.width, image.height)
    if side == image.width == image.height:
        return image
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


# 속성마다 필요한 정보가 달라 crop 처리 방식을 나눈다.
#   letterbox: 옷 전체의 비율·실루엣이 답을 정하는 속성
#   squash   : 넥라인·패턴처럼 국소 디테일이 답을 정하는 속성
#              (레터박스는 여백만큼 옷이 작아져 세부 해상도가 떨어진다)
# 시드 3회 비교에서 기하 의존 속성은 +0.012, 국소 디테일 속성은 -0.012로 갈렸다.
TASK_PREPROCESSING = {
    "upper_length": PREPROCESS_LETTERBOX,
    "lower_length": PREPROCESS_LETTERBOX,
    "pant_length": PREPROCESS_LETTERBOX,
    "upper_fit": PREPROCESS_LETTERBOX,
    "lower_fit": PREPROCESS_LETTERBOX,
    "pant_leg_shape": PREPROCESS_LETTERBOX,
    "silhouette": PREPROCESS_LETTERBOX,
    "sleeve_length": PREPROCESS_LETTERBOX,
    "category": PREPROCESS_SQUASH,
    "lower_subtype": PREPROCESS_SQUASH,
    "neckline": PREPROCESS_SQUASH,
    "collar": PREPROCESS_SQUASH,
    "sleeve_shape": PREPROCESS_SQUASH,
    "detail": PREPROCESS_SQUASH,
    "lower_detail": PREPROCESS_SQUASH,
    "pattern": PREPROCESS_SQUASH,
    "material": PREPROCESS_SQUASH,
}


def resolve_task_preprocessing(preprocessing) -> dict[str, str]:
    """체크포인트의 preprocessing 값을 속성별 방식으로 펼친다.

    문자열이면 모든 속성이 같은 방식, 사전이면 속성별로 다른 방식이다.
    """
    if isinstance(preprocessing, dict):
        missing = set(ATTRIBUTE_TASKS) - set(preprocessing)
        if missing:
            raise ValueError(f"전처리 방식이 지정되지 않은 속성이 있습니다: {sorted(missing)}")
        return dict(preprocessing)
    return {task: preprocessing for task in ATTRIBUTE_TASKS}


def apply_preprocess_mode(image, mode: str):
    if mode == PREPROCESS_LETTERBOX:
        return letterbox_image(image)
    if mode != PREPROCESS_SQUASH:
        raise ValueError(f"알 수 없는 전처리 방식입니다: {mode}")
    return image


def geometry_vector(width: int, height: int, tight_crop: bool) -> list[float]:
    """임베딩에 덧붙일 기하 특징.

    학습과 추론이 반드시 같은 방식으로 계산해야 하므로, 실제로 임베딩에 들어가는
    crop의 크기만 사용한다.

    - 종횡비: 좁고 긴 옷과 넓은 옷을 가르는 신호. log 후 tanh로 범위를 묶는다.
    - tight_crop: 옷에 딱 맞게 자른 crop인지. 상품 사진 전체를 쓴 학습 표본은
      종횡비가 옷이 아니라 사진의 성질이라 헤드가 구분할 수 있어야 한다.
    """
    import math

    ratio = max(float(width), 1.0) / max(float(height), 1.0)
    return [math.tanh(math.log(ratio)), 1.0 if tight_crop else 0.0]


@dataclass(frozen=True)
class AttributePrediction:
    labels: list[str]
    scores: dict[str, float]
    confidence: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _torch_load(path: str | Path, map_location: str):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 2.5 이하 호환
        return torch.load(path, map_location=map_location)


def build_attribute_heads(
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    geometry_dim: int = 0,
    task_preprocessing: dict[str, str] | None = None,
):
    import torch
    import torch.nn as nn

    class AttributeHead(nn.Module):
        """임베딩만 정규화하고 기하 특징은 그 뒤에 붙인다.

        둘을 합쳐 LayerNorm하면 768차원 임베딩에 묻혀 기하 신호가 사라진다.
        """

        def __init__(self, label_count: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(input_dim)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim + geometry_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, label_count),
            )

        def forward(self, features, geometry=None):
            normalized = self.norm(features)
            if geometry_dim:
                if geometry is None:
                    raise ValueError("이 헤드는 기하 특징을 함께 받아야 합니다.")
                normalized = torch.cat([normalized, geometry], dim=-1)
            return self.mlp(normalized)

    class MultiAttributeHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.geometry_dim = geometry_dim
            self.heads = nn.ModuleDict(
                {
                    task_name: AttributeHead(len(task.labels))
                    for task_name, task in ATTRIBUTE_TASKS.items()
                }
            )

        def forward(self, features, geometry=None):
            """features는 텐서 하나이거나 {전처리 방식: 텐서} 사전이다."""
            if not isinstance(features, dict):
                return {
                    task_name: head(features, geometry) for task_name, head in self.heads.items()
                }
            routing = self.task_preprocessing
            return {
                task_name: head(features[routing[task_name]], geometry)
                for task_name, head in self.heads.items()
            }

    module = MultiAttributeHeads()
    module.task_preprocessing = dict(task_preprocessing or {})
    return module


def build_legacy_attribute_heads(input_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
    """버전 1 체크포인트를 그대로 읽기 위한 예전 구조 (기하 특징 없음)."""
    import torch.nn as nn

    class LegacyMultiAttributeHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.geometry_dim = 0
            self.heads = nn.ModuleDict(
                {
                    task_name: nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, len(task.labels)),
                    )
                    for task_name, task in ATTRIBUTE_TASKS.items()
                }
            )

        def forward(self, features, geometry=None):
            return {task_name: head(features) for task_name, head in self.heads.items()}

    return LegacyMultiAttributeHeads()


def save_attribute_checkpoint(
    path: str | Path,
    heads,
    *,
    backbone_model_id: str,
    thresholds: dict[str, float] | None = None,
    training_summary: dict[str, Any] | None = None,
    label_support: dict[str, dict[str, int]] | None = None,
    minimum_label_examples: int = 1,
    preprocessing: str = PREPROCESS_SQUASH,
) -> Path:
    import torch

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "backbone_model_id": backbone_model_id,
        "input_dim": heads.input_dim,
        "hidden_dim": heads.hidden_dim,
        "dropout": heads.dropout,
        "geometry_dim": getattr(heads, "geometry_dim", 0),
        "preprocessing": preprocessing,
        "tasks": {
            name: {
                "labels": list(task.labels),
                "multi_label": task.multi_label,
                "minimum_confidence": task.minimum_confidence,
            }
            for name, task in ATTRIBUTE_TASKS.items()
        },
        "thresholds": thresholds or {
            name: task.minimum_confidence for name, task in ATTRIBUTE_TASKS.items() if task.multi_label
        },
        "label_support": label_support or {},
        "minimum_label_examples": int(minimum_label_examples),
        "state_dict": heads.state_dict(),
        "training_summary": training_summary or {},
    }
    torch.save(payload, output)
    return output


def load_attribute_heads(path: str | Path, device: str = "cpu"):
    payload = _torch_load(path, device)
    version = payload.get("version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(f"지원하지 않는 속성 헤드 체크포인트 버전입니다: {version}")
    expected = {
        name: {"labels": list(task.labels), "multi_label": task.multi_label}
        for name, task in ATTRIBUTE_TASKS.items()
    }
    actual = {
        name: {"labels": value["labels"], "multi_label": value["multi_label"]}
        for name, value in payload.get("tasks", {}).items()
    }
    if actual != expected:
        raise ValueError("체크포인트의 속성 라벨 스키마가 현재 코드와 다릅니다.")
    if version == 1:
        heads = build_legacy_attribute_heads(
            payload["input_dim"], payload["hidden_dim"], payload["dropout"]
        )
    else:
        heads = build_attribute_heads(
            payload["input_dim"],
            payload["hidden_dim"],
            payload["dropout"],
            geometry_dim=int(payload.get("geometry_dim", 0)),
            task_preprocessing=resolve_task_preprocessing(
                payload.get("preprocessing", PREPROCESS_SQUASH)
            ),
        )
    heads.load_state_dict(payload["state_dict"])
    heads.to(device).eval()
    return heads, payload


class FashionAttributePredictor:
    """이미 로드된 FashionSigLIP 이미지 인코더와 학습된 작은 헤드를 결합한다."""

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
        self.heads, self.metadata = load_attribute_heads(checkpoint_path, device)
        expected_model = self.metadata["backbone_model_id"]
        if expected_model != model_id:
            raise ValueError(
                "속성 헤드와 이미지 백본이 다릅니다. "
                f"학습 백본={expected_model}, 현재 백본={model_id}"
            )
        self.thresholds = self.metadata.get("thresholds", {})
        self.label_support = self.metadata.get("label_support", {})
        self.minimum_label_examples = int(self.metadata.get("minimum_label_examples", 1))
        self.geometry_dim = int(self.metadata.get("geometry_dim", 0))
        # 예전 체크포인트에는 이 값이 없다. 그때는 기본 전처리로 학습했다.
        self.preprocessing = self.metadata.get("preprocessing", PREPROCESS_SQUASH)
        self.task_preprocessing = resolve_task_preprocessing(self.preprocessing)
        # 추론에서 만들어야 하는 임베딩 종류. 하나면 인코딩도 한 번이면 된다.
        self.required_modes = sorted(set(self.task_preprocessing.values()))

    @property
    def uses_geometry(self) -> bool:
        return self.geometry_dim > 0

    def _geometry_tensor(self, geometry, batch_size: int):
        """기하 특징을 쓰지 않는 체크포인트면 None, 쓰는데 값이 없으면 오류."""
        import torch

        if not self.uses_geometry:
            return None
        if geometry is None:
            raise ValueError(
                "이 체크포인트는 기하 특징이 필요합니다. geometry_vector()로 만든 값을 넘기세요."
            )
        tensor = torch.as_tensor(geometry, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        return tensor

    def predict(
        self,
        image,
        *,
        tasks: list[str] | None = None,
        geometry=None,
    ) -> dict[str, AttributePrediction]:
        import torch
        from PIL import Image

        selected = tasks or list(ATTRIBUTE_TASKS)
        unknown = set(selected) - set(ATTRIBUTE_TASKS)
        if unknown:
            raise KeyError(f"정의되지 않은 속성 task입니다: {sorted(unknown)}")
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        if geometry is None and self.uses_geometry:
            geometry = geometry_vector(pil.width, pil.height, tight_crop=True)
        features = {}
        for mode in self.required_modes:
            tensor = self.preprocess(apply_preprocess_mode(pil, mode)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                features[mode] = self.image_encoder.encode_image(tensor, normalize=True).float()
        single = features[self.required_modes[0]] if len(self.required_modes) == 1 else features
        return self.predict_features(single, tasks=tasks, geometry=geometry)

    def predict_features(
        self,
        features,
        *,
        tasks: list[str] | None = None,
        geometry=None,
    ) -> dict[str, AttributePrediction]:
        """이미 계산한 FashionSigLIP 특징을 받아 백본 중복 실행을 피한다."""
        import torch

        results: dict[str, AttributePrediction] = {}
        selected = tasks or list(ATTRIBUTE_TASKS)
        unknown = set(selected) - set(ATTRIBUTE_TASKS)
        if unknown:
            raise KeyError(f"정의되지 않은 속성 task입니다: {sorted(unknown)}")
        if isinstance(features, dict):
            missing = set(self.required_modes) - set(features)
            if missing:
                raise ValueError(f"이 체크포인트에 필요한 임베딩이 없습니다: {sorted(missing)}")
            features = {mode: value.to(self.device).float() for mode, value in features.items()}
            batch_size = next(iter(features.values())).shape[0]
        else:
            if len(self.required_modes) > 1:
                raise ValueError(
                    "속성마다 다른 crop 처리를 쓰는 체크포인트입니다. "
                    f"{self.required_modes} 임베딩을 사전으로 넘기세요."
                )
            features = features.to(self.device).float()
            batch_size = features.shape[0]
        geometry_tensor = self._geometry_tensor(geometry, batch_size)
        with torch.inference_mode():
            logits = self.heads(features, geometry_tensor)
        for task_name in selected:
            task = ATTRIBUTE_TASKS[task_name]
            task_logits = logits[task_name][0]
            support = self.label_support.get(task_name)
            supported = [
                index
                for index, label in enumerate(task.labels)
                if support is None or int(support.get(label, 0)) >= self.minimum_label_examples
            ]
            if task.multi_label:
                probabilities = task_logits.sigmoid().cpu()
                threshold = float(self.thresholds.get(task_name, task.minimum_confidence))
                indices = [
                    index for index in supported if float(probabilities[index]) >= threshold
                ]
                # 무지/없음과 실제 양성 속성이 동시에 선택되면 실제 속성을 우선한다.
                empty_labels = {"무지", "디테일 없음"}
                if any(task.labels[index] not in empty_labels for index in indices):
                    indices = [index for index in indices if task.labels[index] not in empty_labels]
                labels = [task.labels[index] for index in indices]
                confidence = max(
                    (float(probabilities[index]) for index in indices),
                    default=max((float(probabilities[index]) for index in supported), default=0.0),
                )
                accepted = bool(indices)
                scores = {
                    label: float(probabilities[index]) if index in supported else 0.0
                    for index, label in enumerate(task.labels)
                }
            else:
                masked_logits = task_logits.clone()
                unsupported = set(range(len(task.labels))) - set(supported)
                if unsupported:
                    masked_logits[list(unsupported)] = float("-inf")
                probabilities = masked_logits.softmax(dim=-1).cpu() if supported else task_logits.softmax(dim=-1).cpu()
                index = int(probabilities.argmax())
                confidence = float(probabilities[index])
                threshold = float(self.thresholds.get(task_name, task.minimum_confidence))
                labels = [task.labels[index]] if len(supported) >= 2 and confidence >= threshold else []
                accepted = bool(labels)
                scores = {
                    label: float(probabilities[i]) if i in supported else 0.0
                    for i, label in enumerate(task.labels)
                }
            results[task_name] = AttributePrediction(labels, scores, confidence, accepted)
        return results


def fuse_measured_and_learned(
    measured_label: str,
    measured_confidence: float,
    prediction: AttributePrediction | None,
) -> tuple[str, float, str]:
    """마스크 측정과 분류 결과를 합친다. 충돌 시 더 강한 근거를 사용한다."""
    unavailable = not measured_label or any(word in measured_label for word in ("불가", "보류", "불확실"))
    measured = measured_label.replace(" 추정", "")
    measured_confidence = max(0.0, min(1.0, measured_confidence))
    if prediction is None or not prediction.accepted:
        return measured_label, measured_confidence, "mask"
    learned = prediction.labels[0]
    if unavailable:
        return learned, prediction.confidence, "trained_head"
    if measured == learned or measured in learned or learned in measured:
        return learned, min(1.0, 0.55 * measured_confidence + 0.55 * prediction.confidence), "fused_agreement"
    if prediction.confidence >= measured_confidence + 0.15:
        return learned, prediction.confidence, "trained_head"
    return measured_label, measured_confidence, "mask"
