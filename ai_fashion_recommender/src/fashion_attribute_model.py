from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fashion_attribute_schema import ATTRIBUTE_TASKS


CHECKPOINT_VERSION = 1


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


def build_attribute_heads(input_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
    import torch.nn as nn

    class MultiAttributeHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
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

        def forward(self, features):
            return {task_name: head(features) for task_name, head in self.heads.items()}

    return MultiAttributeHeads()


def save_attribute_checkpoint(
    path: str | Path,
    heads,
    *,
    backbone_model_id: str,
    thresholds: dict[str, float] | None = None,
    training_summary: dict[str, Any] | None = None,
    label_support: dict[str, dict[str, int]] | None = None,
    minimum_label_examples: int = 1,
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
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"지원하지 않는 속성 헤드 체크포인트 버전입니다: {payload.get('version')}")
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
    heads = build_attribute_heads(payload["input_dim"], payload["hidden_dim"], payload["dropout"])
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

    def predict(
        self,
        image,
        *,
        tasks: list[str] | None = None,
    ) -> dict[str, AttributePrediction]:
        import torch
        from PIL import Image

        selected = tasks or list(ATTRIBUTE_TASKS)
        unknown = set(selected) - set(ATTRIBUTE_TASKS)
        if unknown:
            raise KeyError(f"정의되지 않은 속성 task입니다: {sorted(unknown)}")
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        image_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.image_encoder.encode_image(image_tensor, normalize=True).float()
        return self.predict_features(features, tasks=tasks)

    def predict_features(
        self,
        features,
        *,
        tasks: list[str] | None = None,
    ) -> dict[str, AttributePrediction]:
        """이미 계산한 FashionSigLIP 특징을 받아 백본 중복 실행을 피한다."""
        import torch

        results: dict[str, AttributePrediction] = {}
        selected = tasks or list(ATTRIBUTE_TASKS)
        unknown = set(selected) - set(ATTRIBUTE_TASKS)
        if unknown:
            raise KeyError(f"정의되지 않은 속성 task입니다: {sorted(unknown)}")
        with torch.inference_mode():
            logits = self.heads(features.to(self.device).float())
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
