"""고정 FashionSigLIP 위의 독립 신발 헤드. 의류 체크포인트는 변경하지 않는다."""
from __future__ import annotations

from pathlib import Path

SHOE_LABELS = ("스니커즈", "러닝화", "로퍼", "더비슈즈", "메리제인", "펌프스",
               "샌들", "슬리퍼", "부츠", "워커", "기타 신발", "맨발")
SHOE_CHECKPOINT_VERSION = 1


def build_shoe_heads(input_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
    """신발 crop의 정규화된 FashionSigLIP 임베딩 → 신발 종류 logits."""
    import torch.nn as nn
    model = nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(hidden_dim, len(SHOE_LABELS)),
    )
    model.input_dim, model.hidden_dim, model.dropout = input_dim, hidden_dim, dropout
    return model


def shoe_classification_loss(logits, labels):
    """labels: SHOE_LABELS 인덱스. 미라벨(-1)은 제외하며 전부 미라벨이면 실패한다."""
    import torch.nn.functional as F
    valid = labels != -1
    if not valid.any():
        raise ValueError("신발 학습 배치에 유효한 라벨이 없습니다.")
    return F.cross_entropy(logits[valid], labels[valid])


def save_shoe_checkpoint(path, heads, *, backbone_model_id: str,
                         training_examples: int, threshold: float):
    """학습 후 검증 세트에서 정한 임계값과 함께 저장한다."""
    import torch
    if training_examples <= 0 or not 0 < threshold <= 1:
        raise ValueError("학습 표본 수와 검증된 confidence 임계값이 필요합니다.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": SHOE_CHECKPOINT_VERSION, "labels": SHOE_LABELS,
        "backbone_model_id": backbone_model_id, "preprocessing": "squash",
        "input_dim": heads.input_dim, "hidden_dim": heads.hidden_dim,
        "dropout": heads.dropout, "training_examples": training_examples,
        "threshold": threshold, "state_dict": heads.state_dict(),
    }, path)


class ShoePredictor:
    def __init__(self, path, *, model_id: str, device: str = "cpu"):
        import torch
        # 본인이 학습한 신발 체크포인트만 사용한다.
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if (checkpoint.get("version") != SHOE_CHECKPOINT_VERSION
                or tuple(checkpoint.get("labels", ())) != SHOE_LABELS
                or checkpoint.get("backbone_model_id") != model_id
                or checkpoint.get("preprocessing") != "squash"
                or checkpoint.get("training_examples", 0) <= 0):
            raise ValueError("호환되는 학습 완료 신발 체크포인트가 아닙니다.")
        self.threshold = float(checkpoint["threshold"])
        if not 0 < self.threshold <= 1:
            raise ValueError("신발 confidence 임계값이 올바르지 않습니다.")
        self.heads = build_shoe_heads(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
        self.heads.load_state_dict(checkpoint["state_dict"], strict=True)
        self.heads.to(device).eval()

    def predict(self, features) -> dict:
        import torch
        with torch.inference_mode():
            probabilities = self.heads(features).softmax(dim=-1)[0]
        confidence, index = probabilities.max(dim=0)
        label = SHOE_LABELS[int(index)]
        accepted = float(confidence) >= self.threshold and label not in {"맨발", "기타 신발"}
        return {"status": "recognized" if accepted else "uncertain", "accepted": accepted,
                "item_type": label if accepted else "분석 보류",
                "confidence": float(confidence), "source": "trained_shoe_head"}


def shoe_crop(rgb, segmentation):
    """FASHN feet(15) 영역 bbox. 맨발/신발 구분은 학습 헤드가 담당한다."""
    import numpy as np
    from PIL import Image
    ys, xs = np.where(segmentation == 15)
    if len(xs) < 64 or xs.max() - xs.min() < 8 or ys.max() - ys.min() < 8:
        return None
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(rgb).crop(box)
