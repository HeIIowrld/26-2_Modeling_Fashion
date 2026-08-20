from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from config import MIN_BODY_SHAPE_CONFIDENCE, MIN_FULL_BODY_SCORE, MIN_LANDMARK_VISIBILITY
from schemas import (
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_TRIANGLE,
    SHAPE_UNAVAILABLE,
    SHAPE_UNCERTAIN,
    PoseAnalysis,
)


def _load_body_shape_reference() -> tuple[float, float]:
    """체형 분류 경계를 기준 분포에서 읽는다.

    `shoulder_hip_ratio`는 어깨 관절과 골반 관절 사이의 간격 비율이다. 골반 관절은
    엉덩이 바깥 폭보다 훨씬 좁아서 이 값은 사람마다 대략 1.4~2.2에 분포한다.
    반면 예전 경계값 0.90/1.12는 신체 표면 치수(어깨너비÷엉덩이너비) 기준이라
    거의 모든 사람이 '역삼각체형'으로 분류됐다.

    그래서 절대값 대신 기준 분포의 백분위를 쓴다. 기준표를 한국인 표본으로
    교체하면 코드 수정 없이 경계가 갱신된다.
    """
    import json

    path = Path(__file__).resolve().parent / "data" / "body_shape_reference.json"
    if not path.is_file():
        # 기준표가 없으면 관찰된 분포의 3분위로 대체한다.
        return 1.7286, 1.8547
    percentiles = json.loads(path.read_text(encoding="utf-8"))["percentiles"]
    return float(percentiles["33"]), float(percentiles["67"])


LOWER_BODY_RATIO, UPPER_BODY_RATIO = _load_body_shape_reference()


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


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """2D 화면 좌표와 3D world 좌표를 모두 처리한다."""
    return math.dist(a[: len(b)], b[: len(a)])


def silhouette_and_landmarks(image: str | Path | Image.Image | np.ndarray, threshold: float = 0.5):
    """인물 실루엣 마스크와 그 사진의 관절 좌표를 함께 돌려준다.

    마스크와 관절은 **같은 사진**에서 나와야 한다. 체형용 사진을 따로 받으면
    코디 사진의 관절 좌표는 위치가 어긋나기 때문이다. 그래서 한 번의 호출로 둘 다 만든다.

    기본 PoseAnalyzer는 분할을 꺼서 돌린다. 매 분석마다 계산하면 느려지는데
    체형 추정은 사진 한 장에 한 번만 필요하기 때문이다.
    """
    rgb = _to_rgb_array(image)
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.45,
    ) as pose:
        result = pose.process(rgb)
    if result.segmentation_mask is None or not result.pose_landmarks:
        return None, None
    landmark = mp.solutions.pose.PoseLandmark
    raw = result.pose_landmarks.landmark
    landmarks = {
        name: (raw[index].x, raw[index].y, raw[index].visibility)
        for name, index in {
            "left_shoulder": landmark.LEFT_SHOULDER, "right_shoulder": landmark.RIGHT_SHOULDER,
            "left_elbow": landmark.LEFT_ELBOW, "right_elbow": landmark.RIGHT_ELBOW,
            "left_wrist": landmark.LEFT_WRIST, "right_wrist": landmark.RIGHT_WRIST,
            "left_hip": landmark.LEFT_HIP, "right_hip": landmark.RIGHT_HIP,
        }.items()
    }
    return result.segmentation_mask > threshold, landmarks


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
                body_shape=SHAPE_UNAVAILABLE,
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
        # 체형 비율은 3D world 좌표로 잰다. 정규화 2D 좌표는 원근과 몸의 회전에 휘둘려
        # 골반이 정면으로 겹쳐 보이면 간격이 0에 가까워지고 비율이 폭발한다.
        # 같은 표본 1,409장에서 변동계수가 2D 3.80 vs world 0.18로 차이가 컸다.
        world = (
            {
                name: (
                    result.pose_world_landmarks.landmark[index].x,
                    result.pose_world_landmarks.landmark[index].y,
                    result.pose_world_landmarks.landmark[index].z,
                )
                for name, index in names.items()
            }
            if result.pose_world_landmarks
            else None
        )

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

        if world is not None:
            def midpoint(first: str, second: str) -> tuple[float, float, float]:
                a, b = world[first], world[second]
                return tuple((a[axis] + b[axis]) / 2 for axis in range(3))

            world_shoulder_mid = midpoint("left_shoulder", "right_shoulder")
            world_hip_mid = midpoint("left_hip", "right_hip")
            world_ankle_mid = midpoint("left_ankle", "right_ankle")
            world_shoulder_width = _distance(world["left_shoulder"], world["right_shoulder"])
            world_hip_width = max(_distance(world["left_hip"], world["right_hip"]), 1e-6)
            world_torso = _distance(world_shoulder_mid, world_hip_mid)
            world_leg = max(_distance(world_hip_mid, world_ankle_mid), 1e-6)
            shoulder_hip_ratio = world_shoulder_width / world_hip_width
            upper_lower_ratio = world_torso / world_leg
            leg_ratio = world_leg / max(world_torso + world_leg, 1e-6)
        else:
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
        band = max(UPPER_BODY_RATIO - LOWER_BODY_RATIO, 1e-6)
        if LOWER_BODY_RATIO < shoulder_hip_ratio < UPPER_BODY_RATIO:
            boundary_margin = min(
                shoulder_hip_ratio - LOWER_BODY_RATIO, UPPER_BODY_RATIO - shoulder_hip_ratio
            ) / (band / 2)
        else:
            boundary_margin = min(
                abs(shoulder_hip_ratio - LOWER_BODY_RATIO), abs(shoulder_hip_ratio - UPPER_BODY_RATIO)
            ) / band
        margin_score = float(np.clip(boundary_margin, 0.0, 1.0))
        body_shape_confidence = (
            0.45 * mean_visibility
            + 0.25 * front_score
            + 0.20 * full_body_score
            + 0.10 * margin_score
        )

        if body_shape_confidence < MIN_BODY_SHAPE_CONFIDENCE:
            body_shape = SHAPE_UNCERTAIN
        elif shoulder_hip_ratio >= UPPER_BODY_RATIO:
            body_shape = SHAPE_INVERTED_TRIANGLE
        elif shoulder_hip_ratio <= LOWER_BODY_RATIO:
            body_shape = SHAPE_TRIANGLE
        else:
            body_shape = SHAPE_RECTANGLE

        warnings: list[str] = []
        if not feet_inside:
            warnings.append("발끝이 사진 밖으로 잘렸을 가능성이 있습니다.")
        if posture != "정면에 가까움":
            warnings.append("정면 자세가 아니어서 가로 비율의 오차가 커질 수 있습니다.")
        if mean_visibility < 0.75:
            warnings.append("일부 관절이 옷이나 물체에 가려졌습니다.")
        if body_shape == SHAPE_UNCERTAIN:
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
