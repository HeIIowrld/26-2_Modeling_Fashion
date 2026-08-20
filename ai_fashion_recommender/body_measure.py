"""사진에서 3D 체형을 복원해 둘레를 추정하는 자리.

쓰리사이즈를 입력하지 않은 사용자를 위해 사진으로 둘레를 추정한다.
모델을 붙이는 곳은 `_estimate()` 한 곳이다.

## 왜 아직 비어 있나

SMPL 계열 모델 파일은 사이트에서 **회원가입과 라이선스 동의**를 거쳐야 받을 수 있고,
연구용 비상업 조건이다. 제3자가 대신 동의할 수 없으므로 파일은 직접 받아 넣어야 한다.

  1. https://smpl.is.tue.mpg.de 에서 가입하고 모델 라이선스에 동의한다.
  2. 받은 파일을 `models/smpl/`에 넣는다.
  3. `_estimate()`에 회귀 모델(HMR2.0·CLIFF 등)을 연결한다.
  4. `config.ENABLE_BODY_MEASUREMENT = True`로 바꾼다.

## 정확도를 반드시 표시할 것

정면 사진 한 장으로 둘레를 재는 것은 오차가 크다. 공개된 결과 기준으로 PIFuHD는
허리둘레가 약 10% 어긋나고, 2025년 최신 기법도 가슴 MAE가 3.3cm 수준이다.
체형 분류는 가슴과 허리의 차이로 갈리므로 두 오차가 겹치면 분류가 뒤집힐 수 있다.
그래서 추정값은 반드시 `BASIS_ESTIMATE`로 표시해 입력값과 구분한다.

옷도 문제다. 관절 위치는 옷을 뚫고 추정되지만(측정 결과 개인차의 6%), 둘레는 옷 두께가
그대로 더해진다. 추정 기능을 켤 거라면 몸이 드러나는 사진을 함께 요구해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import ENABLE_BODY_MEASUREMENT


class MeasurementNotReady(RuntimeError):
    """추정 모델이 아직 연결되지 않았을 때 화면에 그대로 보여줄 사유."""


@dataclass(frozen=True)
class BodyMeasurement:
    """추정한 몸통 치수.

    `*_cm`은 키를 알 때만 채워진다. 사진 한 장에는 절대 크기 정보가 없기 때문이다.
    `*_width`는 실루엣에서 잰 상대 폭(픽셀)이라 키가 없어도 서로 비교할 수 있다.
    체형 분류는 세 부위의 비율로 결정되므로 폭만 있어도 판정할 수 있다.
    """

    chest_width: float
    waist_width: float
    hip_width: float
    source: str
    chest_cm: float | None = None
    waist_cm: float | None = None
    hip_cm: float | None = None


# 어깨~골반 구간에서 각 부위를 재는 높이 (0=어깨, 1=골반).
MEASURE_LEVELS = {"chest": 0.25, "waist": 0.65, "hip": 1.0}


class BodyMeasurementEstimator:
    """사진의 인물 실루엣에서 몸통 폭을 재어 체형을 판정한다.

    MediaPipe(Apache 2.0)의 `segmentation_mask`만 쓰므로 별도 모델 파일이나
    라이선스 동의가 필요 없다. SMPL 계열은 가입·비상업 조건이라 쓰지 않았다.

    한계가 분명하다.
    - 정면 폭만 알 수 있고 두께(깊이)는 알 수 없다. 그래서 둘레를 정확히 낼 수 없다.
      대신 세 부위의 **폭 비율**로 체형을 판정한다. 분류는 비율로 갈리므로 폭만으로도 된다.
    - 옷 두께가 그대로 폭에 더해진다. 몸이 드러나는 사진일수록 정확하다.
    """

    NOT_READY_REASON = "사진에서 몸통 윤곽을 찾지 못했습니다."

    def __init__(self, enabled: bool = ENABLE_BODY_MEASUREMENT) -> None:
        self.enabled = enabled

    @property
    def available(self) -> bool:
        return self.enabled

    def estimate(self, person_image, pose, height_cm: float | None = None) -> BodyMeasurement:
        if not self.available:
            raise MeasurementNotReady(
                "사진 기반 체형 추정이 꺼져 있습니다. config.ENABLE_BODY_MEASUREMENT를 확인하세요."
            )
        return self._estimate(person_image, pose, height_cm)

    @staticmethod
    def _estimate(person_image, pose, height_cm: float | None) -> BodyMeasurement:
        import numpy as np

        from pose_analyzer import _to_rgb_array, silhouette_and_landmarks

        rgb = _to_rgb_array(person_image)
        # 마스크와 관절은 반드시 같은 사진에서 얻는다. 체형용 사진을 따로 받으면
        # 코디 사진의 관절 좌표는 위치가 맞지 않는다.
        mask, landmarks = silhouette_and_landmarks(rgb)
        if mask is None or not mask.any() or not landmarks:
            raise MeasurementNotReady(BodyMeasurementEstimator.NOT_READY_REASON)

        height, width = mask.shape

        torso = _without_arms(mask, landmarks, width, height)

        def y_at(name: str) -> float:
            left, right = landmarks[f"left_{name}"], landmarks[f"right_{name}"]
            return (left[1] + right[1]) / 2 * height

        shoulder_y, hip_y = y_at("shoulder"), y_at("hip")
        center_x = (
            (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2
            + (landmarks["left_hip"][0] + landmarks["right_hip"][0]) / 2
        ) / 2 * width

        widths = {}
        for name, ratio in MEASURE_LEVELS.items():
            y = shoulder_y + (hip_y - shoulder_y) * ratio
            value = _torso_width(torso, y, center_x)
            if value is None:
                raise MeasurementNotReady(f"{name} 높이에서 몸통을 찾지 못했습니다.")
            widths[name] = float(value)

        centimetres = {}
        if height_cm:
            ys = np.where(mask.any(axis=1))[0]
            person_px = float(ys.max() - ys.min() + 1)
            if person_px > 0:
                # 폭을 둘레로 바꾸려면 두께가 필요하다. 정면 사진에는 두께 정보가 없어
                # 타원 근사를 쓰며, 이 값은 참고용이다. 분류에는 폭 비율을 쓴다.
                scale = height_cm / person_px
                centimetres = {
                    f"{name}_cm": round(value * scale * 2.9, 1) for name, value in widths.items()
                }

        return BodyMeasurement(
            chest_width=widths["chest"],
            waist_width=widths["waist"],
            hip_width=widths["hip"],
            source="사진 실루엣",
            **centimetres,
        )


def _without_arms(mask, landmarks, width: int, height: int):
    """팔 픽셀을 지운다. 팔이 몸통에 붙어 있으면 허리가 넓게 측정된다."""
    import cv2
    import numpy as np

    torso = mask.astype(np.uint8).copy()
    forearm = max(3, int(min(width, height) * 0.035))
    for side in ("left", "right"):
        joints = [landmarks.get(f"{side}_{part}") for part in ("shoulder", "elbow", "wrist")]
        if any(joint is None for joint in joints):
            continue
        points = [(int(x * width), int(y * height)) for x, y, _ in joints]
        for start, end in zip(points, points[1:]):
            cv2.line(torso, start, end, color=0, thickness=forearm * 2)
    return torso.astype(bool)


def _torso_width(mask, y: float, center_x: float) -> int | None:
    """몸 중심을 포함하는 연속 구간의 폭. 몸에서 떨어진 팔은 자연히 빠진다."""
    height, width = mask.shape
    row = mask[max(0, min(height - 1, int(y)))]
    center = int(round(max(0, min(width - 1, center_x))))
    if not row[center]:
        return None
    left = center
    while left > 0 and row[left - 1]:
        left -= 1
    right = center
    while right < width - 1 and row[right + 1]:
        right += 1
    return right - left + 1


def estimator_status(estimator: BodyMeasurementEstimator | None = None) -> dict:
    """화면이 '추정 가능/준비 중'을 구분해 안내할 수 있게 한다."""
    estimator = estimator or BodyMeasurementEstimator()
    return {
        "available": estimator.available,
        "reason": "" if estimator.available else estimator.NOT_READY_REASON,
    }
