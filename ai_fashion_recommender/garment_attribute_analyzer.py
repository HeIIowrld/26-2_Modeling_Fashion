from __future__ import annotations

import math

import cv2
import numpy as np

from schemas import PoseAnalysis


def _pixel_point(pose: PoseAnalysis, name: str, width: int, height: int) -> np.ndarray:
    x, y, _ = pose.landmarks[name]
    return np.array([x * width, y * height], dtype=np.float32)


def _mask_area(segmentation: np.ndarray, labels: list[int]) -> int:
    return int(np.isin(segmentation, labels).sum())


class GarmentAttributeAnalyzer:
    """의류 픽셀 마스크의 끝점과 MediaPipe 관절을 결합해 길이를 추정한다.

    길이 임계값은 초기 휴리스틱이므로 실제 서비스 전에 라벨 데이터로
    보정해야 한다. 파서가 구분하지 않는 소재·패턴은 억지로 추정하지 않는다.
    """

    def analyze(self, segmentation: np.ndarray, pose: PoseAnalysis) -> dict:
        if not pose.landmarks:
            return self._empty("포즈 랜드마크 없음")

        height, width = segmentation.shape
        areas = {
            "top": _mask_area(segmentation, [3]),
            "dress": _mask_area(segmentation, [4]),
            "skirt": _mask_area(segmentation, [5]),
            "pants": _mask_area(segmentation, [6]),
        }

        upper_type = "원피스" if areas["dress"] > areas["top"] else ("상의" if areas["top"] > 0 else "분석 불가")
        if areas["dress"] > max(areas["skirt"], areas["pants"]):
            lower_type = "원피스"
            lower_labels = [4]
        elif areas["skirt"] > areas["pants"]:
            lower_type = "치마"
            lower_labels = [5]
        elif areas["pants"] > 0:
            lower_type = "바지"
            lower_labels = [6]
        else:
            lower_type = "분석 불가"
            lower_labels = []

        upper_labels = [4] if upper_type == "원피스" else [3]
        upper_mask = np.isin(segmentation, upper_labels).astype(np.uint8)
        lower_mask = np.isin(segmentation, lower_labels).astype(np.uint8)

        sleeve_ratio, sleeve_agreement = self._sleeve_coverage(upper_mask, pose)
        sleeve_length = self._sleeve_label(sleeve_ratio)
        upper_length = self._upper_length(upper_mask, pose)
        bottom_length = self._bottom_length(lower_mask, pose, lower_type)
        fit = self._upper_fit(upper_mask, pose)

        visible = [value[2] for value in pose.landmarks.values()]
        mask_ok = min(1.0, (areas["top"] + areas["dress"] + areas["skirt"] + areas["pants"]) / max(height * width * 0.10, 1))
        confidence = 0.45 * float(np.mean(visible)) + 0.35 * mask_ok + 0.20 * sleeve_agreement

        return {
            "upper_type": upper_type,
            "lower_type": lower_type,
            "sleeve_length": sleeve_length,
            "upper_length": upper_length,
            "bottom_length": bottom_length,
            "fit": fit,
            "pattern": "분석 보류",
            "material": "분석 보류",
            "attribute_confidence": round(float(np.clip(confidence, 0, 1)), 3),
            "measurements": {
                "sleeve_coverage_ratio": round(sleeve_ratio, 3),
                "mask_pixel_areas": areas,
            },
        }

    def _sleeve_coverage(self, mask: np.ndarray, pose: PoseAnalysis) -> tuple[float, float]:
        height, width = mask.shape
        ratios = []
        radius = max(3, int(min(width, height) * 0.012))
        for side in ("left", "right"):
            shoulder = _pixel_point(pose, f"{side}_shoulder", width, height)
            elbow = _pixel_point(pose, f"{side}_elbow", width, height)
            wrist = _pixel_point(pose, f"{side}_wrist", width, height)
            samples = []
            for index in range(41):
                t = index / 40
                if t <= 0.5:
                    point = shoulder + (elbow - shoulder) * (t / 0.5)
                else:
                    point = elbow + (wrist - elbow) * ((t - 0.5) / 0.5)
                x, y = int(point[0]), int(point[1])
                probe = np.zeros_like(mask, dtype=np.uint8)
                cv2.circle(probe, (x, y), radius, 1, -1)
                overlap = mask[probe.astype(bool)]
                occupancy = float(overlap.mean()) if overlap.size else 0.0
                samples.append((t, occupancy))
            covered = [t for t, occupancy in samples if occupancy >= 0.18]
            ratios.append(max(covered, default=0.0))
        agreement = 1.0 - min(abs(ratios[0] - ratios[1]), 1.0)
        return float(np.median(ratios)), agreement

    @staticmethod
    def _sleeve_label(ratio: float) -> str:
        if ratio < 0.18:
            return "민소매"
        if ratio < 0.58:
            return "반팔"
        if ratio < 0.83:
            return "7부 소매"
        return "긴팔"

    @staticmethod
    def _vertical_ratio(mask: np.ndarray, start_y: float, end_y: float) -> float | None:
        ys = np.where(mask > 0)[0]
        if ys.size == 0 or end_y <= start_y:
            return None
        return float((np.percentile(ys, 98) - start_y) / (end_y - start_y))

    def _upper_length(self, mask: np.ndarray, pose: PoseAnalysis) -> str:
        height, _ = mask.shape
        shoulder_y = np.mean([pose.landmarks["left_shoulder"][1], pose.landmarks["right_shoulder"][1]]) * height
        hip_y = np.mean([pose.landmarks["left_hip"][1], pose.landmarks["right_hip"][1]]) * height
        ratio = self._vertical_ratio(mask, shoulder_y, hip_y)
        if ratio is None:
            return "분석 불가"
        if ratio < 0.78:
            return "크롭 기장"
        if ratio < 1.18:
            return "기본 기장"
        return "롱 기장"

    def _bottom_length(self, mask: np.ndarray, pose: PoseAnalysis, garment_type: str) -> str:
        height, _ = mask.shape
        hip_y = np.mean([pose.landmarks["left_hip"][1], pose.landmarks["right_hip"][1]]) * height
        ankle_y = np.mean([pose.landmarks["left_ankle"][1], pose.landmarks["right_ankle"][1]]) * height
        ratio = self._vertical_ratio(mask, hip_y, ankle_y)
        if ratio is None:
            return "분석 불가"
        if garment_type == "바지":
            if ratio < 0.42:
                return "반바지"
            if ratio < 0.62:
                return "무릎 기장 바지"
            if ratio < 0.86:
                return "크롭·7부 바지"
            return "긴바지"
        if garment_type in {"치마", "원피스"}:
            if ratio < 0.38:
                return "미니 기장"
            if ratio < 0.62:
                return "무릎 기장"
            if ratio < 0.86:
                return "미디 기장"
            return "롱·맥시 기장"
        return "분석 불가"

    @staticmethod
    def _upper_fit(mask: np.ndarray, pose: PoseAnalysis) -> str:
        height, width = mask.shape
        shoulder_width = abs(pose.landmarks["left_shoulder"][0] - pose.landmarks["right_shoulder"][0]) * width
        shoulder_y = np.mean([pose.landmarks["left_shoulder"][1], pose.landmarks["right_shoulder"][1]]) * height
        hip_y = np.mean([pose.landmarks["left_hip"][1], pose.landmarks["right_hip"][1]]) * height
        y1, y2 = int(shoulder_y), int(hip_y)
        row_widths = []
        for row in mask[max(0, y1):min(height, y2)]:
            xs = np.where(row > 0)[0]
            if xs.size:
                row_widths.append(xs.max() - xs.min() + 1)
        if not row_widths or shoulder_width <= 1:
            return "분석 불가"
        ratio = float(np.percentile(row_widths, 70) / shoulder_width)
        if ratio < 0.92:
            return "슬림핏 추정"
        if ratio < 1.28:
            return "레귤러핏 추정"
        return "여유핏·오버핏 추정"

    @staticmethod
    def _empty(reason: str) -> dict:
        return {
            "upper_type": "분석 불가", "lower_type": "분석 불가",
            "sleeve_length": "분석 불가", "upper_length": "분석 불가",
            "bottom_length": "분석 불가", "fit": "분석 불가",
            "pattern": "분석 보류", "material": "분석 보류",
            "attribute_confidence": 0.0, "measurements": {}, "reason": reason,
        }
