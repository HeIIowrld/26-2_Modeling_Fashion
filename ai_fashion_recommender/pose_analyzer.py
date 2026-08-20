from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from config import MIN_BODY_SHAPE_CONFIDENCE, MIN_FULL_BODY_SCORE, MIN_LANDMARK_VISIBILITY
from schemas import PoseAnalysis


POSE_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot"),
)


def _to_rgb_array(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """지원하는 입력 형식을 RGB uint8 NumPy 배열로 통일한다."""
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("이미지는 HxWx3 형태여야 합니다.")
    return array.astype(np.uint8)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PoseAnalyzer:
    """MediaPipe의 사전학습 Pose 모델로 체형 참고 비율을 계산한다."""

    def __init__(self, model_complexity: int = 1) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.45,
        )
        self._pose_landmark = mp.solutions.pose.PoseLandmark
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._pose.close()
            self._closed = True

    def __enter__(self) -> "PoseAnalyzer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def analyze(self, image: str | Path | Image.Image | np.ndarray) -> PoseAnalysis:
        rgb = _to_rgb_array(image)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return PoseAnalysis(
                valid=False,
                full_body_score=0.0,
                body_shape="분석 불가",
                shoulder_hip_ratio=0.0,
                upper_lower_ratio=0.0,
                leg_ratio=0.0,
                posture="분석 불가",
                body_shape_confidence=0.0,
                warnings=["사람의 자세를 찾지 못했습니다."],
            )

        raw = result.pose_landmarks.landmark
        names = {
            "nose": self._pose_landmark.NOSE,
            "left_shoulder": self._pose_landmark.LEFT_SHOULDER,
            "right_shoulder": self._pose_landmark.RIGHT_SHOULDER,
            "left_elbow": self._pose_landmark.LEFT_ELBOW,
            "right_elbow": self._pose_landmark.RIGHT_ELBOW,
            "left_wrist": self._pose_landmark.LEFT_WRIST,
            "right_wrist": self._pose_landmark.RIGHT_WRIST,
            "left_hip": self._pose_landmark.LEFT_HIP,
            "right_hip": self._pose_landmark.RIGHT_HIP,
            "left_knee": self._pose_landmark.LEFT_KNEE,
            "right_knee": self._pose_landmark.RIGHT_KNEE,
            "left_ankle": self._pose_landmark.LEFT_ANKLE,
            "right_ankle": self._pose_landmark.RIGHT_ANKLE,
            "left_heel": self._pose_landmark.LEFT_HEEL,
            "right_heel": self._pose_landmark.RIGHT_HEEL,
            "left_foot": self._pose_landmark.LEFT_FOOT_INDEX,
            "right_foot": self._pose_landmark.RIGHT_FOOT_INDEX,
        }
        landmarks = {
            name: (raw[index].x, raw[index].y, raw[index].visibility)
            for name, index in names.items()
        }

        required = list(landmarks.values())
        mean_visibility = float(np.mean([point[2] for point in required]))
        visible_fraction = float(np.mean([point[2] >= MIN_LANDMARK_VISIBILITY for point in required]))

        left_shoulder = landmarks["left_shoulder"][:2]
        right_shoulder = landmarks["right_shoulder"][:2]
        left_hip = landmarks["left_hip"][:2]
        right_hip = landmarks["right_hip"][:2]
        left_ankle = landmarks["left_ankle"][:2]
        right_ankle = landmarks["right_ankle"][:2]

        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        ankle_mid = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)

        shoulder_width = _distance(left_shoulder, right_shoulder)
        hip_width = max(_distance(left_hip, right_hip), 1e-6)
        torso_length = _distance(shoulder_mid, hip_mid)
        leg_length = max(_distance(hip_mid, ankle_mid), 1e-6)
        body_span = max(ankle_mid[1] - max(0.0, landmarks["nose"][1] - torso_length * 0.45), 1e-6)

        shoulder_hip_ratio = shoulder_width / hip_width
        upper_lower_ratio = torso_length / leg_length
        leg_ratio = leg_length / body_span

        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
        torso_tilt = abs(shoulder_mid[0] - hip_mid[0])
        posture = "정면에 가까움" if shoulder_tilt < 0.06 and torso_tilt < 0.08 else "기울어짐 또는 측면 자세"

        feet_inside = max(landmarks["left_foot"][1], landmarks["right_foot"][1]) < 0.99
        head_inside = landmarks["nose"][1] > 0.03
        full_body_score = 0.50 * visible_fraction + 0.30 * mean_visibility + 0.10 * float(feet_inside) + 0.10 * float(head_inside)

        # 관절 간격 비율은 실제 신체 폭이 아니므로 자세·가시성·경계 근접도를
        # 함께 보고 신뢰도가 낮으면 체형 라벨을 만들지 않는다.
        front_score = float(np.clip(1.0 - max(shoulder_tilt / 0.12, torso_tilt / 0.16), 0.0, 1.0))
        if 0.90 < shoulder_hip_ratio < 1.12:
            boundary_margin = min(shoulder_hip_ratio - 0.90, 1.12 - shoulder_hip_ratio) / 0.11
        else:
            boundary_margin = min(abs(shoulder_hip_ratio - 0.90), abs(shoulder_hip_ratio - 1.12)) / 0.12
        margin_score = float(np.clip(boundary_margin, 0.0, 1.0))
        body_shape_confidence = (
            0.45 * mean_visibility
            + 0.25 * front_score
            + 0.20 * full_body_score
            + 0.10 * margin_score
        )

        if body_shape_confidence < MIN_BODY_SHAPE_CONFIDENCE:
            body_shape = "분석 불확실"
        elif shoulder_hip_ratio >= 1.12:
            body_shape = "상체 강조형"
        elif shoulder_hip_ratio <= 0.90:
            body_shape = "하체 강조형"
        else:
            body_shape = "균형형"

        warnings: list[str] = []
        if not feet_inside:
            warnings.append("발끝이 사진 밖으로 잘렸을 가능성이 있습니다.")
        if posture != "정면에 가까움":
            warnings.append("정면 자세가 아니어서 가로 비율의 오차가 커질 수 있습니다.")
        if mean_visibility < 0.75:
            warnings.append("일부 관절이 옷이나 물체에 가려졌습니다.")
        if body_shape == "분석 불확실":
            warnings.append("촬영 자세 또는 경계에 가까운 비율 때문에 체형 분류를 보류했습니다.")
        warnings.append("어깨·골반 값은 관절 간격 기반 상대 추정치이며 실제 신체 치수가 아닙니다.")

        return PoseAnalysis(
            valid=full_body_score >= MIN_FULL_BODY_SCORE,
            full_body_score=round(full_body_score, 4),
            body_shape=body_shape,
            shoulder_hip_ratio=round(shoulder_hip_ratio, 4),
            upper_lower_ratio=round(upper_lower_ratio, 4),
            leg_ratio=round(leg_ratio, 4),
            posture=posture,
            body_shape_confidence=round(float(np.clip(body_shape_confidence, 0, 1)), 4),
            warnings=warnings,
            landmarks=landmarks,
        )

    def draw_landmarks(
        self,
        image: str | Path | Image.Image | np.ndarray,
        analysis: PoseAnalysis | None = None,
    ) -> Image.Image:
        """기존 분석 결과가 있으면 MediaPipe를 다시 실행하지 않고 시각화한다."""
        rgb = _to_rgb_array(image)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if analysis is None:
            result = self._pose.process(rgb)
        else:
            result = None
            height, width = rgb.shape[:2]
            points = {
                name: (int(x * width), int(y * height))
                for name, (x, y, visibility) in analysis.landmarks.items()
                if visibility >= MIN_LANDMARK_VISIBILITY
            }
            for start, end in POSE_EDGES:
                if start in points and end in points:
                    cv2.line(canvas, points[start], points[end], (80, 220, 80), 3, cv2.LINE_AA)
            for point in points.values():
                cv2.circle(canvas, point, 5, (40, 80, 240), -1, cv2.LINE_AA)
        if result is not None and result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                canvas,
                result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
