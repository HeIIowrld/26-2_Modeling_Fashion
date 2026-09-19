from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pose_analyzer import PoseAnalyzer, _to_rgb_array
from schemas import PoseAnalysis


MIN_INPUT_SHORT_SIDE = 320
MIN_INPUT_SHARPNESS = 25.0


class QualityChecker:
    """입력 또는 VTON 결과의 기본 품질을 수치로 확인한다."""

    def __init__(self, pose_analyzer: PoseAnalyzer) -> None:
        self.pose_analyzer = pose_analyzer

    def check_input(self, image: str | Path | Image.Image, pose: PoseAnalysis | None = None) -> dict:
        """이미 계산한 pose를 받으면 MediaPipe를 다시 실행하지 않는다."""
        rgb = _to_rgb_array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        pose = pose or self.pose_analyzer.analyze(rgb)
        height, width = rgb.shape[:2]
        issues = list(pose.warnings)
        # 전체 포즈 점수가 높아도 발이 프레임 밖이면 전신사진으로 통과시키지 않는다.
        # 가림으로 신뢰도가 낮은 좌표는 잘림의 근거로 사용하지 않는다.
        framing_points = [pose.landmarks.get(name) for name in
                          ("left_ankle", "right_ankle", "left_foot", "right_foot")]
        confident = [p for p in framing_points if p is not None and np.isfinite(p).all() and p[2] >= 0.5]
        cropped = any(not (0 <= p[0] < 1 and 0 <= p[1] < 1) for p in confident)
        framing = "cropped" if cropped else "visible" if len(confident) == len(framing_points) else "uncertain"
        if cropped:
            issues.append("발목 또는 발끝이 사진 밖에 있습니다. 발끝까지 들어오는 전신사진으로 다시 촬영해 주세요.")
        elif framing == "uncertain":
            issues.append("발목·발끝이 가려져 전신 구도를 확정하지 못했습니다. 하의 기장 검사는 별도 확인이 필요합니다.")
        if min(width, height) < MIN_INPUT_SHORT_SIDE:
            issues.append(
                f"짧은 변이 {MIN_INPUT_SHORT_SIDE}px보다 작아 세부 의류 분석이 불안정할 수 있습니다."
            )
        if sharpness < MIN_INPUT_SHARPNESS:
            issues.append("사진이 흐릿할 가능성이 있습니다.")
        return {
            "passed": (
                pose.valid
                and not cropped
                and min(width, height) >= MIN_INPUT_SHORT_SIDE
                and sharpness >= MIN_INPUT_SHARPNESS
            ),
            "resolution": [width, height],
            "sharpness": round(sharpness, 2),
            "full_body_score": pose.full_body_score,
            "lower_body_framing": framing,
            "issues": issues,
        }

    def compare_pose(self, before: str | Path | Image.Image, after: str | Path | Image.Image) -> dict:
        first = self.pose_analyzer.analyze(before)
        second = self.pose_analyzer.analyze(after)
        shared = set(first.landmarks) & set(second.landmarks)
        if not first.valid or not second.valid or not shared:
            return {"passed": False, "pose_difference": None, "reason": "두 이미지의 전신 자세를 비교할 수 없습니다."}
        differences = []
        for name in shared:
            x1, y1, _ = first.landmarks[name]
            x2, y2, _ = second.landmarks[name]
            differences.append(float(np.hypot(x1 - x2, y1 - y2)))
        difference = float(np.mean(differences))
        return {
            "passed": difference < 0.08,
            "pose_difference": round(difference, 4),
            "reason": "정규화 좌표의 평균 관절 이동량을 비교한 참고 지표입니다.",
        }
