from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import FASHION_SIGLIP_MODEL_ID


class FashionClassifier:
    """FashionSigLIP을 필요할 때만 로드하는 zero-shot 분류 래퍼."""

    def __init__(
        self,
        enabled: bool = False,
        model_id: str = FASHION_SIGLIP_MODEL_ID,
        device: str = "auto",
    ) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.backend = None
        self.device = "cpu"
        if enabled:
            import torch

            self._torch = torch
            self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
            try:
                from transformers import AutoModel, AutoProcessor

                self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self.backend = "transformers"
            except NotImplementedError:
                # transformers 5.x는 모델을 meta 디바이스에서 초기화하는데, Marqo 원격
                # 코드가 __init__에서 open_clip.create_model()로 실제 가중치를 만들려다
                # "Cannot copy out of meta tensor"로 실패한다. 모델 카드가 안내하는
                # open_clip 직접 로딩으로 대체한다.
                import open_clip

                self.model, _, self._preprocess = open_clip.create_model_and_transforms(f"hf-hub:{model_id}")
                self._tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
                self.model = self.model.to(self.device)
                self.backend = "open_clip"
            self.model.eval()

    def _score_prompts(self, image: str | Path | Image.Image, prompts: list[str]) -> np.ndarray:
        if not self.enabled or self.model is None:
            return np.zeros(len(prompts), dtype=np.float32)
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        with self._torch.inference_mode():
            if self.backend == "open_clip":
                pixels = self._preprocess(pil).unsqueeze(0).to(self.device)
                tokens = self._tokenizer(prompts).to(self.device)
                image_features = self.model.encode_image(pixels, normalize=True)
                text_features = self.model.encode_text(tokens, normalize=True)
            else:
                processed = self.processor(text=prompts, images=[pil], padding="max_length", return_tensors="pt")
                processed = {name: tensor.to(self.device) for name, tensor in processed.items()}
                image_features = self.model.get_image_features(processed["pixel_values"], normalize=True)
                text_features = self.model.get_text_features(processed["input_ids"], normalize=True)
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        return probabilities.float().cpu().numpy()

    def classify(self, image: str | Path | Image.Image, labels: list[str]) -> dict[str, float]:
        if not self.enabled or self.model is None:
            # 무거운 모델을 끈 상태에서는 결과를 위조하지 않고 미사용 상태를 반환한다.
            return {label: 0.0 for label in labels}
        probabilities = self._score_prompts(image, labels)
        return {label: float(score) for label, score in zip(labels, probabilities)}

    def best_mapped_labels(
        self,
        image: str | Path | Image.Image,
        prompt_groups: dict[str, dict[str, str]],
        fallback: str = "분석 보류",
    ) -> dict[str, tuple[str, float]]:
        """같은 이미지의 여러 속성을 이미지 인코딩 한 번으로 분류한다."""
        if not self.enabled:
            return {group: (fallback, 0.0) for group in prompt_groups}
        flattened = [prompt for labels in prompt_groups.values() for prompt in labels.values()]
        scores = self._score_prompts(image, flattened)
        results: dict[str, tuple[str, float]] = {}
        offset = 0
        for group, label_prompts in prompt_groups.items():
            labels = list(label_prompts)
            group_scores = scores[offset:offset + len(labels)]
            group_scores = group_scores / max(float(group_scores.sum()), 1e-12)
            best_index = int(np.argmax(group_scores))
            results[group] = (labels[best_index], float(group_scores[best_index]))
            offset += len(labels)
        return results

    def best_label(self, image: str | Path | Image.Image, labels: list[str], fallback: str) -> str:
        scores = self.classify(image, labels)
        if not self.enabled:
            return fallback
        return max(scores, key=scores.get)

    def best_mapped_label(
        self,
        image: str | Path | Image.Image,
        label_prompts: dict[str, str],
        fallback: str = "분석 보류",
    ) -> tuple[str, float]:
        """사용자용 한국어 라벨과 모델용 영어 프롬프트를 분리한다."""
        if not self.enabled:
            return fallback, 0.0
        prompt_scores = self.classify(image, list(label_prompts.values()))
        best_label = max(label_prompts, key=lambda label: prompt_scores[label_prompts[label]])
        return best_label, float(prompt_scores[label_prompts[best_label]])
