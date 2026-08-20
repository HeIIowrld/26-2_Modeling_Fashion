from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SLEEVE_LABELS = {0: "민소매", 1: "반팔", 2: "7부 소매", 3: "긴팔"}
BOTTOM_LABELS = {0: "짧은 기장", 1: "무릎 기장", 2: "7부 기장", 3: "긴 기장"}
NECKLINE_LABELS = {
    0: "V넥", 1: "스퀘어넥", 2: "라운드넥", 3: "스탠드 칼라",
    4: "라펠 칼라", 5: "서스펜더·슬링",
}
MATERIAL_LABELS = {
    0: "데님", 1: "코튼", 2: "가죽", 3: "퍼·플리스",
    4: "니트", 5: "시폰", 6: "기타 소재",
}
PATTERN_LABELS = {
    0: "플로럴", 1: "그래픽", 2: "스트라이프", 3: "무지",
    4: "체크", 5: "기타 패턴", 6: "컬러 블록",
}


@dataclass(frozen=True)
class DeepFashionRecord:
    image_name: str
    image_path: Path
    sleeve_length: str | None
    bottom_length: str | None
    neckline: str | None
    upper_material: str | None
    upper_pattern: str | None


def _read_annotation(path: str | Path, minimum_values: int) -> dict[str, list[int]]:
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"DeepFashion 라벨 파일이 없습니다: {annotation_path}")
    rows: dict[str, list[int]] = {}
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts or (len(parts) == 1 and parts[0].isdigit()):
            continue
        try:
            values = [int(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{annotation_path.name}:{line_number} 라벨 형식이 올바르지 않습니다.") from exc
        if len(values) < minimum_values:
            raise ValueError(
                f"{annotation_path.name}:{line_number}에 라벨이 {len(values)}개뿐입니다. "
                f"최소 {minimum_values}개가 필요합니다."
            )
        rows[parts[0]] = values
    return rows


def load_deepfashion_multimodal(
    image_dir: str | Path,
    shape_annotations: str | Path,
    fabric_annotations: str | Path,
    pattern_annotations: str | Path,
    *,
    skip_missing_images: bool = True,
) -> list[DeepFashionRecord]:
    """공식 DeepFashion-MultiModal의 shape/fabric/color 텍스트 라벨을 결합한다."""
    image_root = Path(image_dir).expanduser().resolve()
    shape = _read_annotation(shape_annotations, 12)
    fabric = _read_annotation(fabric_annotations, 3)
    pattern = _read_annotation(pattern_annotations, 3)
    names = sorted(set(shape) & set(fabric) & set(pattern))
    records = []
    for name in names:
        image_path = (image_root / name).resolve()
        if not image_path.is_file() and skip_missing_images:
            continue
        shape_values, fabric_values, pattern_values = shape[name], fabric[name], pattern[name]
        records.append(
            DeepFashionRecord(
                image_name=name,
                image_path=image_path,
                sleeve_length=SLEEVE_LABELS.get(shape_values[0]),
                bottom_length=BOTTOM_LABELS.get(shape_values[1]),
                neckline=NECKLINE_LABELS.get(shape_values[9]),
                upper_material=MATERIAL_LABELS.get(fabric_values[0]),
                upper_pattern=PATTERN_LABELS.get(pattern_values[0]),
            )
        )
    return records


def _normalize_prediction(field: str, value: str) -> str:
    value = value.replace(" 추정", "")
    if field == "bottom_length":
        if any(word in value for word in ("반바지", "미니")):
            return "짧은 기장"
        if "무릎" in value:
            return "무릎 기장"
        if any(word in value for word in ("7부", "크롭", "미디")):
            return "7부 기장"
        if any(word in value for word in ("긴바지", "롱", "맥시")):
            return "긴 기장"
    return value


def evaluate_deepfashion_predictions(
    records: Iterable[DeepFashionRecord],
    predictor: Callable[[Path], dict],
    *,
    max_samples: int | None = None,
) -> dict:
    """predictor가 반환한 현재 모델 결과를 DeepFashion 정답과 비교한다."""
    fields = {
        "sleeve_length": "sleeve_length",
        "bottom_length": "bottom_length",
        "neckline": "neckline",
        "material": "upper_material",
        "pattern": "upper_pattern",
    }
    ground_truth = {field: 0 for field in fields}
    accepted = {field: 0 for field in fields}
    correct = {field: 0 for field in fields}
    pairs: dict[str, list[tuple[str, str]]] = {field: [] for field in fields}
    mismatches = []
    processed = 0
    for record in records:
        if max_samples is not None and processed >= max_samples:
            break
        prediction = predictor(record.image_path)
        processed += 1
        for predicted_field, truth_field in fields.items():
            expected = getattr(record, truth_field)
            predicted = prediction.get(predicted_field)
            if expected is None:
                continue
            ground_truth[predicted_field] += 1
            if not predicted or "불확실" in predicted or "보류" in predicted or "불가" in predicted:
                pairs[predicted_field].append((expected, "__abstain__"))
                if len(mismatches) < 30:
                    mismatches.append(
                        {"image": record.image_name, "field": predicted_field, "expected": expected, "predicted": predicted or "미응답"}
                    )
                continue
            accepted[predicted_field] += 1
            normalized = _normalize_prediction(predicted_field, predicted)
            pairs[predicted_field].append((expected, normalized))
            if normalized == expected:
                correct[predicted_field] += 1
            elif len(mismatches) < 30:
                mismatches.append(
                    {"image": record.image_name, "field": predicted_field, "expected": expected, "predicted": predicted}
                )
    metrics = {}
    for field in fields:
        labels = sorted({expected for expected, _ in pairs[field]})
        f1_values = []
        for label in labels:
            true_positive = sum(expected == label and predicted == label for expected, predicted in pairs[field])
            false_positive = sum(expected != label and predicted == label for expected, predicted in pairs[field])
            false_negative = sum(expected == label and predicted != label for expected, predicted in pairs[field])
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        selective_accuracy = correct[field] / accepted[field] if accepted[field] else None
        total = ground_truth[field]
        metrics[field] = {
            # 기존 키는 호환성을 위해 유지하되, 선택적으로 응답한 표본의 정확도임을 함께 명시한다.
            "accuracy": round(selective_accuracy, 4) if selective_accuracy is not None else None,
            "selective_accuracy": round(selective_accuracy, 4) if selective_accuracy is not None else None,
            "overall_accuracy": round(correct[field] / total, 4) if total else None,
            "coverage": round(accepted[field] / total, 4) if total else None,
            "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
            "ground_truth": total,
            "evaluated": accepted[field],
            "abstained": total - accepted[field],
            "correct": correct[field],
        }
    return {"processed_images": processed, "metrics": metrics, "mismatches": mismatches}
