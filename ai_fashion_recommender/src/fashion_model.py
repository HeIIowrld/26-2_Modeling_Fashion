from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from config import FASHION_SIGLIP_MODEL_ID
from fashion_attribute_model import PREPROCESS_SQUASH, apply_preprocess_mode


class FashionClassifier:
    """FashionSigLIP 특징을 학습 속성 헤드와 zero-shot 경로가 공유하는 래퍼."""

    def __init__(
        self,
        enabled: bool = False,
        model_id: str = FASHION_SIGLIP_MODEL_ID,
        device: str = "auto",
        attribute_checkpoint: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.attribute_predictor = None
        self._text_feature_cache = {}
        self.device = "cpu"
        if enabled:
            import torch
            import open_clip

            self._torch = torch
            self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{model_id}",
                device=self.device,
            )
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
            self.model.eval()
            if attribute_checkpoint:
                checkpoint = Path(attribute_checkpoint).expanduser().resolve()
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"학습된 의류 속성 헤드가 없습니다: {checkpoint}")
                from fashion_attribute_model import FashionAttributePredictor

                self.attribute_predictor = FashionAttributePredictor(
                    checkpoint,
                    image_encoder=self.model,
                    preprocess=self.preprocess,
                    model_id=self.model_id,
                    device=self.device,
                )

    @property
    def trained_attributes_enabled(self) -> bool:
        return self.attribute_predictor is not None

    def predict_trained_attributes(
        self,
        image: str | Path | Image.Image,
        tasks: list[str] | None = None,
    ):
        if self.attribute_predictor is None:
            return {}
        return self.attribute_predictor.predict(image, tasks=tasks)

    def _encode_image(self, image: str | Path | Image.Image, mode: str = PREPROCESS_SQUASH):
        if not self.enabled or self.model is None or self.preprocess is None:
            return None
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        image_tensor = self.preprocess(apply_preprocess_mode(pil, mode)).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            return self.model.encode_image(image_tensor, normalize=True).float()

    @property
    def head_preprocess_modes(self) -> list[str]:
        """학습 헤드가 필요로 하는 crop 처리 방식들. zero-shot 프롬프트는 항상 기본 방식을 쓴다."""
        if self.attribute_predictor is None:
            return [PREPROCESS_SQUASH]
        return list(self.attribute_predictor.required_modes)

    def _score_prompt_features(self, image_features, prompts: list[str]) -> np.ndarray:
        if image_features is None or self.model is None or self.tokenizer is None:
            return np.zeros(len(prompts), dtype=np.float32)
        cache_key = tuple(prompts)
        text_features = self._text_feature_cache.get(cache_key)
        if text_features is None:
            text_tokens = self.tokenizer(prompts).to(self.device)
            with self._torch.inference_mode():
                text_features = self.model.encode_text(text_tokens, normalize=True).float()
            self._text_feature_cache[cache_key] = text_features
        with self._torch.inference_mode():
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        return probabilities.float().cpu().numpy()

    def _score_prompts(self, image: str | Path | Image.Image, prompts: list[str]) -> np.ndarray:
        if not self.enabled or self.model is None or self.preprocess is None or self.tokenizer is None:
            return np.zeros(len(prompts), dtype=np.float32)
        return self._score_prompt_features(self._encode_image(image), prompts)

    def classify(self, image: str | Path | Image.Image, labels: list[str]) -> dict[str, float]:
        if not self.enabled or self.model is None or self.preprocess is None or self.tokenizer is None:
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
        return self._best_mapped_labels_from_features(self._encode_image(image), prompt_groups, fallback)

    def _best_mapped_labels_from_features(
        self,
        image_features,
        prompt_groups: dict[str, dict[str, str]],
        fallback: str = "분석 보류",
    ) -> dict[str, tuple[str, float]]:
        if image_features is None:
            return {group: (fallback, 0.0) for group in prompt_groups}
        flattened = [prompt for labels in prompt_groups.values() for prompt in labels.values()]
        scores = self._score_prompt_features(image_features, flattened)
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

    def analyze_crop(
        self,
        image: str | Path | Image.Image,
        *,
        tasks: list[str],
        prompt_groups: dict[str, dict[str, str]],
        fallback: str = "분석 보류",
        geometry: list[float] | None = None,
    ):
        """한 번의 이미지 인코딩으로 학습 헤드와 zero-shot fallback을 모두 계산한다."""
        if not self.enabled:
            return {}, {group: (fallback, 0.0) for group in prompt_groups}
        # zero-shot 프롬프트 점수는 FashionSigLIP 기본 전처리를 기준으로 만들어졌으므로
        # 항상 기본 방식으로 인코딩하고, 학습 헤드가 요구하는 방식만 추가로 계산한다.
        encoded = {PREPROCESS_SQUASH: self._encode_image(image, PREPROCESS_SQUASH)}
        for mode in self.head_preprocess_modes:
            if mode not in encoded:
                encoded[mode] = self._encode_image(image, mode)
        head_input = (
            encoded[self.head_preprocess_modes[0]]
            if len(self.head_preprocess_modes) == 1
            else {mode: encoded[mode] for mode in self.head_preprocess_modes}
        )
        learned = (
            self.attribute_predictor.predict_features(head_input, tasks=tasks, geometry=geometry)
            if self.attribute_predictor is not None
            else {}
        )
        zero_shot_features = encoded[PREPROCESS_SQUASH]
        zero_shot = self._best_mapped_labels_from_features(zero_shot_features, prompt_groups, fallback)
        return learned, zero_shot

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
