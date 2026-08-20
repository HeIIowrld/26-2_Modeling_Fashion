from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pose_analyzer import _to_rgb_array
from schemas import PoseAnalysis


LABELS = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}


def _point(landmarks: dict, name: str, width: int, height: int) -> tuple[int, int]:
    x, y, _ = landmarks[name]
    return int(np.clip(x * width, 0, width - 1)), int(np.clip(y * height, 0, height - 1))


class ClothingParser:
    """의류 영역을 분리한다.

    `fashn-ai/fashn-human-parser`가 설치·활성화된 경우 실제 18-class
    SegFormer 결과를 사용한다. 그렇지 않으면 Notebook이 즉시 실행되도록
    MediaPipe 관절로 상·하의의 중앙 영역을 근사한다.
    """

    def __init__(self, use_fashn: bool = True) -> None:
        self.backend = "pose-guided-fallback"
        self._parser = None
        if use_fashn:
            try:
                from fashn_human_parser import FashnHumanParser

                self._parser = FashnHumanParser(device=self._fashn_device())
                self.backend = "fashn-human-parser"
            except ImportError as exc:
                raise RuntimeError(
                    "FASHN 파서를 사용하려면 `pip install fashn-human-parser`를 실행하세요."
                ) from exc

    @staticmethod
    def _fashn_device() -> str | None:
        """ROCm Windows는 MIOpen BatchNorm JIT 버그(ROCm/ROCm#6150)로 SegFormer가
        GPU에서 실패하므로 CPU를 강제한다. 그 외 환경은 파서의 자동 선택에 맡긴다."""
        import platform

        try:
            import torch
        except ImportError:
            return None
        if platform.system() == "Windows" and getattr(torch.version, "hip", None):
            return "cpu"
        return None

    def parse(
        self,
        image: str | Path | Image.Image | np.ndarray,
        pose: PoseAnalysis,
    ) -> dict[str, np.ndarray | str | list[str]]:
        rgb = _to_rgb_array(image)
        height, width = rgb.shape[:2]

        if self._parser is not None:
            segmentation = np.asarray(self._parser.predict(rgb), dtype=np.uint8)
            # 색상·속성 분석용 순수 의류 마스크. 스카프·벨트 같은 액세서리는
            # 상의 색이나 학습 속성 분류를 오염시키므로 별도 마스크로 분리한다.
            upper_mask = np.isin(segmentation, [3, 4]).astype(np.uint8)
            lower_mask = np.isin(segmentation, [4, 5, 6]).astype(np.uint8)
            accessory_mask = np.isin(segmentation, [7, 8, 9, 10, 11, 17]).astype(np.uint8)
            # VTON용 확장 마스크: 팔/다리까지 포함해야(CatVTON AutoMasker 방식) 원래
            # 옷의 실루엣이 남지 않고 소매·기장이 다른 옷으로도 바꿀 수 있다.
            # 색상 분석 등에는 순수 의류 마스크(upper/lower_mask)를 그대로 쓴다.
            upper_style_mask = np.isin(segmentation, [3, 4, 10, 12]).astype(np.uint8)
            lower_style_mask = np.isin(segmentation, [4, 5, 6, 7, 14]).astype(np.uint8)
            present = [LABELS[index] for index in np.unique(segmentation) if index in LABELS and index != 0]
        else:
            if not pose.landmarks:
                raise ValueError("포즈 랜드마크가 없어 대체 의류 마스크를 만들 수 없습니다.")
            upper_mask = np.zeros((height, width), dtype=np.uint8)
            lower_mask = np.zeros((height, width), dtype=np.uint8)

            ls = _point(pose.landmarks, "left_shoulder", width, height)
            rs = _point(pose.landmarks, "right_shoulder", width, height)
            lh = _point(pose.landmarks, "left_hip", width, height)
            rh = _point(pose.landmarks, "right_hip", width, height)
            lk = _point(pose.landmarks, "left_knee", width, height)
            rk = _point(pose.landmarks, "right_knee", width, height)
            la = _point(pose.landmarks, "left_ankle", width, height)
            ra = _point(pose.landmarks, "right_ankle", width, height)

            # 팔과 배경색의 영향을 줄이기 위해 몸통 중앙 다각형만 사용한다.
            upper_polygon = np.array([ls, rs, rh, lh], dtype=np.int32)
            lower_polygon = np.array([lh, rh, rk, ra, la, lk], dtype=np.int32)
            cv2.fillConvexPoly(upper_mask, upper_polygon, 1)
            cv2.fillPoly(lower_mask, [lower_polygon], 1)
            upper_style_mask, lower_style_mask = upper_mask, lower_mask
            accessory_mask = np.zeros((height, width), dtype=np.uint8)
            present = ["top-region", "bottom-region"]
            segmentation = upper_mask * 3 + lower_mask * 6

        return {
            "backend": self.backend,
            "segmentation": segmentation,
            "upper_mask": upper_mask.astype(bool),
            "lower_mask": lower_mask.astype(bool),
            "upper_style_mask": upper_style_mask.astype(bool),
            "lower_style_mask": lower_style_mask.astype(bool),
            "accessory_mask": accessory_mask.astype(bool),
            "present_labels": present,
        }

    @staticmethod
    def colorize(segmentation: np.ndarray) -> Image.Image:
        palette = np.array(
            [
                [0, 0, 0], [255, 205, 180], [80, 45, 25], [60, 130, 220],
                [190, 80, 170], [230, 150, 50], [65, 90, 170], [130, 80, 40],
                [40, 150, 100], [220, 70, 70], [150, 100, 200], [70, 180, 210],
                [245, 190, 150], [255, 215, 180], [210, 170, 130], [100, 100, 100],
                [100, 170, 220], [240, 210, 40],
            ],
            dtype=np.uint8,
        )
        return Image.fromarray(palette[np.clip(segmentation, 0, len(palette) - 1)])
