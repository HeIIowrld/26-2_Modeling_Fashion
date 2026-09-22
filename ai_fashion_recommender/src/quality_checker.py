from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pose_analyzer import PoseAnalyzer, _to_rgb_array
from schemas import PoseAnalysis


MIN_INPUT_SHORT_SIDE = 320
MIN_INPUT_SHARPNESS = 25.0

# 입력 framing thresholds. A low-visibility landmark is evidence of uncertainty,
# not evidence that the person was cropped.
FRAMING_VISIBILITY_THRESHOLD = 0.5
HEAD_CROP_NOSE_Y = 0.03
FRONT_VISIBILITY_THRESHOLD = 0.7
FRONT_VISIBILITY_DELTA_MAX = 0.25
FRONT_SHOULDER_TILT_MAX = 0.06
FRONT_TORSO_OFFSET_MAX = 0.08
FRONT_CLEAR_SHOULDER_TILT = 0.14
FRONT_CLEAR_TORSO_OFFSET = 0.18
FRONT_SEVERE_SHOULDER_TILT = 0.22
FRONT_SEVERE_TORSO_OFFSET = 0.28
FRONT_WIDTH_TORSO_RATIO_MIN = 0.35
FRONT_WIDTH_TORSO_RATIO_MAX = 2.5
ARM_CROSSING_MARGIN = 0.08
ARM_RAISED_MARGIN = 0.08
# 손목이 자기 어깨보다 바깥으로 나간 정도(어깨 폭 대비). 팔을 자연스럽게 내린 실제 정면 사진은
# 0.25~0.41이라 0.25에서는 10장 중 1장꼴로 거절됐다. 두 팔을 벌리거나 기둥을 잡은 사진은 0.75~1.43이다.
# 허리에 손을 올린 자세(0.29~0.36)는 이 값으로 자연 자세와 가를 수 없어 거절하지 않는다(2026-09-22 실측 100장).
ARM_OPEN_MARGIN = 0.6
ARM_MAX_OUTWARD_RATIO = 1.5
PERSON_DETECTION_ERROR = "사진에서 사람의 정면 전신을 확인할 수 없습니다. 머리부터 발끝까지 한 명만 나오도록 다시 촬영해 주세요."
CORE_TORSO_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _landmark(landmarks: dict, name: str) -> tuple[float, float, float] | None:
    point = landmarks.get(name)
    if point is None or len(point) < 3:
        return None
    try:
        values = tuple(float(value) for value in point[:3])
    except (TypeError, ValueError):
        return None
    return values if np.isfinite(values).all() else None


def _reliable(point: tuple[float, float, float] | None, threshold: float) -> bool:
    return point is not None and point[2] >= threshold


def _has_detected_person(pose: PoseAnalysis | None) -> bool:
    if pose is None or not getattr(pose, "valid", False):
        return False
    landmarks = getattr(pose, "landmarks", None)
    if not isinstance(landmarks, dict) or not landmarks:
        return False
    usable = {name: _landmark(landmarks, name) for name in landmarks}
    if not any(point is not None for point in usable.values()):
        return False
    return all(_landmark(landmarks, name) is not None for name in CORE_TORSO_LANDMARKS)


def assess_head_framing(
    landmarks: dict,
    *,
    visibility_threshold: float = FRAMING_VISIBILITY_THRESHOLD,
) -> dict:
    """Assess only clear upper-edge cropping; do not infer the hairline from nose."""
    nose = _landmark(landmarks, "nose")
    if not _reliable(nose, visibility_threshold):
        return {"status": "uncertain", "nose_y": None if nose is None else nose[1]}
    status = "cropped" if nose[1] <= HEAD_CROP_NOSE_Y else "visible"
    return {"status": status, "nose_y": round(nose[1], 4)}


def assess_lower_body_framing(
    landmarks: dict,
    *,
    visibility_threshold: float = FRAMING_VISIBILITY_THRESHOLD,
) -> dict:
    """Retain the team's ankle/foot rule while making its states explicit."""
    names = ("left_ankle", "right_ankle", "left_foot", "right_foot")
    points = [_landmark(landmarks, name) for name in names]
    confident = [point for point in points if _reliable(point, visibility_threshold)]
    if any(not (0 <= point[0] < 1 and 0 <= point[1] < 1) for point in confident):
        return {"status": "cropped", "confident_points": len(confident)}
    status = "visible" if len(confident) == len(points) else "uncertain"
    return {"status": status, "confident_points": len(confident)}


def assess_full_body_framing(head_framing: str, lower_body_framing: str) -> dict:
    """Combine independent head and lower-body states without upgrading uncertainty."""
    if "cropped" in (head_framing, lower_body_framing):
        status = "cropped"
    elif "uncertain" in (head_framing, lower_body_framing):
        status = "uncertain"
    else:
        status = "visible"
    return {"status": status}


def assess_front_pose(
    landmarks: dict,
    *,
    visibility_threshold: float = FRONT_VISIBILITY_THRESHOLD,
) -> dict:
    """Classify front-facing posture only when reliable signals agree clearly."""
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    points = {name: _landmark(landmarks, name) for name in names}
    if not all(_reliable(points[name], visibility_threshold) for name in names):
        return {"status": "uncertain", "metrics": {}}

    left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
    left_hip, right_hip = points["left_hip"], points["right_hip"]
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
    shoulder_width = float(np.hypot(
        left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1]
    ))
    torso_height = abs(hip_mid[1] - shoulder_mid[1])
    ratio = shoulder_width / max(torso_height, 1e-6)
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
    torso_offset = abs(shoulder_mid[0] - hip_mid[0])
    visibility_delta = max(points[name][2] for name in names) - min(points[name][2] for name in names)
    metrics = {
        "shoulder_tilt": round(shoulder_tilt, 4),
        "torso_offset": round(torso_offset, 4),
        "visibility_delta": round(visibility_delta, 4),
        "shoulder_width_torso_ratio": round(ratio, 4),
    }
    if visibility_delta > FRONT_VISIBILITY_DELTA_MAX:
        return {"status": "uncertain", "metrics": metrics}

    clear_tilt = shoulder_tilt >= FRONT_CLEAR_SHOULDER_TILT
    clear_offset = torso_offset >= FRONT_CLEAR_TORSO_OFFSET
    severe_signal = (
        shoulder_tilt >= FRONT_SEVERE_SHOULDER_TILT
        or torso_offset >= FRONT_SEVERE_TORSO_OFFSET
    )
    ratio_implausible = not FRONT_WIDTH_TORSO_RATIO_MIN <= ratio <= FRONT_WIDTH_TORSO_RATIO_MAX
    if (clear_tilt and clear_offset) or (severe_signal and ratio_implausible):
        status = "non_front"
    elif shoulder_tilt <= FRONT_SHOULDER_TILT_MAX and torso_offset <= FRONT_TORSO_OFFSET_MAX:
        status = "front"
    else:
        status = "uncertain"
    return {"status": status, "metrics": metrics}


def assess_arm_pose(
    landmarks: dict,
    *,
    visibility_threshold: float = FRONT_VISIBILITY_THRESHOLD,
) -> dict:
    """Classify only clear arm violations; hidden arms remain uncertain."""
    names = (
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
    )
    points = {name: _landmark(landmarks, name) for name in names}
    if not all(_reliable(points[name], visibility_threshold) for name in names):
        return {"status": "uncertain", "metrics": {}}

    left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
    left_wrist, right_wrist = points["left_wrist"], points["right_wrist"]
    center_x = (left_shoulder[0] + right_shoulder[0] + points["left_hip"][0] + points["right_hip"][0]) / 4
    hip_y = (points["left_hip"][1] + points["right_hip"][1]) / 2
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1e-6)
    # MediaPipe의 left_*는 사람 기준 왼쪽이라 정면 사진에서는 이미지 오른쪽에 찍힌다.
    # 이미지 좌우를 가정하면 팔을 내린 정상 정면 사진이 전부 '팔 교차'가 된다
    # (2026-09-22 운영 배포에서 실제 정면 사진 8장 중 7장 거절). 각 팔의 바깥쪽은
    # 그 팔 어깨가 몸 중심의 어느 쪽에 있는지로 정한다. 좌우 반전된 셀카도 같은 규칙으로 맞다.
    wrists_crossed = False
    arms_open = False
    for side in ("left", "right"):
        shoulder, wrist = points[f"{side}_shoulder"], points[f"{side}_wrist"]
        outward = 1.0 if shoulder[0] >= center_x else -1.0
        wrists_crossed |= (wrist[0] - center_x) * outward < -ARM_CROSSING_MARGIN
        arms_open |= (wrist[0] - shoulder[0]) * outward > shoulder_width * ARM_OPEN_MARGIN
    wrists_raised = min(left_wrist[1], right_wrist[1]) < shoulder_y - ARM_RAISED_MARGIN
    wrists_down = left_wrist[1] >= hip_y - 0.05 and right_wrist[1] >= hip_y - 0.05
    elbows_between = all(
        points[f"{side}_shoulder"][1] - 0.05
        <= points[f"{side}_elbow"][1]
        <= points[f"{side}_wrist"][1] + 0.05
        for side in ("left", "right")
    )
    wrists_close = (
        abs(left_wrist[0] - left_shoulder[0]) <= shoulder_width * ARM_MAX_OUTWARD_RATIO
        and abs(right_wrist[0] - right_shoulder[0]) <= shoulder_width * ARM_MAX_OUTWARD_RATIO
    )
    metrics = {
        "wrists_down": wrists_down,
        "elbows_between": elbows_between,
        "wrists_crossed": wrists_crossed,
        "wrists_raised": wrists_raised,
        "arms_open": arms_open,
        "wrists_close": wrists_close,
    }
    if wrists_crossed:
        status = "arms_crossed_or_occluded"
    elif wrists_raised or arms_open:
        status = "arms_raised_or_open"
    elif wrists_down and elbows_between and wrists_close:
        status = "arms_down"
    else:
        status = "uncertain"
    return {"status": status, "metrics": metrics}


class QualityChecker:
    """입력 또는 VTON 결과의 기본 품질을 수치로 확인한다."""

    def __init__(self, pose_analyzer: PoseAnalyzer) -> None:
        self.pose_analyzer = pose_analyzer

    def check_input(self, image: str | Path | Image.Image, pose: PoseAnalysis | None = None) -> dict:
        """이미 계산한 pose를 받으면 MediaPipe를 다시 실행하지 않는다."""
        rgb = _to_rgb_array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if pose is None and self.pose_analyzer is not None:
            pose = self.pose_analyzer.analyze(rgb)
        height, width = rgb.shape[:2]
        landmarks = getattr(pose, "landmarks", None) or {}
        if not _has_detected_person(pose):
            return {
                "passed": False,
                "resolution": [width, height],
                "sharpness": round(sharpness, 2),
                "full_body_score": float(getattr(pose, "full_body_score", 0.0) or 0.0),
                "head_framing": "uncertain",
                "lower_body_framing": "uncertain",
                "full_body_framing": "uncertain",
                "front_pose": {"status": "uncertain", "metrics": {}},
                "arm_pose": {"status": "uncertain", "metrics": {}},
                "issues": [PERSON_DETECTION_ERROR],
            }
        head = assess_head_framing(landmarks)
        lower = assess_lower_body_framing(landmarks)
        full = assess_full_body_framing(head["status"], lower["status"])
        front = assess_front_pose(landmarks)
        arms = assess_arm_pose(landmarks)
        issues: list[str] = []
        if head["status"] == "cropped":
            issues.append("머리부터 발끝까지 모두 나오게 전신사진으로 다시 촬영해 주세요.")
        if lower["status"] == "cropped":
            issues.append("발목 또는 발끝이 사진 밖에 있습니다. 발끝까지 들어오는 전신사진으로 다시 촬영해 주세요.")
        if front["status"] == "non_front":
            issues.append("몸을 정면으로 향하고 양쪽 어깨가 모두 보이도록 다시 촬영해 주세요.")
        if arms["status"] == "arms_raised_or_open":
            issues.append("팔을 몸 옆에 자연스럽게 내리고 몸통을 가리지 않도록 다시 촬영해 주세요.")
        if arms["status"] == "arms_crossed_or_occluded":
            issues.append("팔을 몸 옆에 자연스럽게 내리고 몸통을 가리지 않도록 다시 촬영해 주세요.")
        issues.extend(warning for warning in pose.warnings if warning not in issues)
        if full["status"] == "uncertain" and lower["status"] == "uncertain":
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
                and full["status"] != "cropped"
                and front["status"] != "non_front"
                and arms["status"] not in {"arms_raised_or_open", "arms_crossed_or_occluded"}
                and min(width, height) >= MIN_INPUT_SHORT_SIDE
                and sharpness >= MIN_INPUT_SHARPNESS
            ),
            "resolution": [width, height],
            "sharpness": round(sharpness, 2),
            "full_body_score": pose.full_body_score,
            "head_framing": head["status"],
            "lower_body_framing": lower["status"],
            "full_body_framing": full["status"],
            "front_pose": front,
            "arm_pose": arms,
            "issues": list(dict.fromkeys(issues))[:3],
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
