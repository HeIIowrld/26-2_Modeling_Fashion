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

                self._parser = FashnHumanParser()
                self.backend = "fashn-human-parser"
            except ImportError as exc:
                raise RuntimeError(
                    "FASHN 파서를 사용하려면 `pip install fashn-human-parser`를 실행하세요."
                ) from exc

    def parse(
        self,
        image: str | Path | Image.Image | np.ndarray,
        pose: PoseAnalysis,
    ) -> dict[str, np.ndarray | str | list[str]]:
        rgb = _to_rgb_array(image)
        height, width = rgb.shape[:2]

        if self._parser is not None:
            segmentation = np.asarray(self._parser.predict(rgb), dtype=np.uint8)
            # 색·패턴 분석에서 스카프와 벨트가 본 의류 색을 오염시키지 않도록
            # 주 의류와 액세서리 마스크를 분리한다.
            upper_mask = np.isin(segmentation, [3, 4]).astype(np.uint8)
            lower_mask = np.isin(segmentation, [4, 5, 6]).astype(np.uint8)
            accessory_mask = np.isin(segmentation, [7, 8, 9, 10, 11, 17]).astype(np.uint8)
            top_mask = (segmentation == 3).astype(np.uint8)
            dress_mask = (segmentation == 4).astype(np.uint8)
            skirt_mask = (segmentation == 5).astype(np.uint8)
            pants_mask = (segmentation == 6).astype(np.uint8)
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
            top_mask = upper_mask.copy()
            dress_mask = np.zeros_like(upper_mask)
            skirt_mask = np.zeros_like(lower_mask)
            pants_mask = lower_mask.copy()
            accessory_mask = np.zeros_like(upper_mask)
            present = ["top-region", "bottom-region"]
            segmentation = upper_mask * 3 + lower_mask * 6

        return {
            "backend": self.backend,
            "segmentation": segmentation,
            "upper_mask": upper_mask.astype(bool),
            "lower_mask": lower_mask.astype(bool),
            "top_mask": top_mask.astype(bool),
            "dress_mask": dress_mask.astype(bool),
            "skirt_mask": skirt_mask.astype(bool),
            "pants_mask": pants_mask.astype(bool),
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
