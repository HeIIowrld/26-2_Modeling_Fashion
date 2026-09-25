"""공용 핏 모델용 CSV 계약과 identity 누수 검사."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from fit_vision_schema import FIT_VISION_TASKS


@dataclass(frozen=True)
class FitVisionRecord:
    image_path: Path
    split: str
    category: str
    source_domain: str
    group_id: str
    mask_path: Path | None = None
    labels: dict[str, str] = field(default_factory=dict)


REQUIRED_COLUMNS = {"image_path", "split", "category", "source_domain", "group_id"}


def load_fit_vision_csv(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    *,
    split: str | None = None,
    require_images: bool = True,
) -> list[FitVisionRecord]:
    annotation = Path(csv_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else annotation.parent
    records = []
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"핏 학습 CSV 필수 열이 없습니다: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            row_split = (row.get("split") or "").strip().lower()
            if row_split not in {"train", "val", "test"}:
                raise ValueError(f"{annotation.name}:{line_number} split은 train/val/test여야 합니다.")
            if split and row_split != split.lower():
                continue
            category = (row.get("category") or "").strip().lower()
            if category not in {"top", "bottom"}:
                raise ValueError(f"{annotation.name}:{line_number} category는 top/bottom이어야 합니다.")

            def path_value(column: str) -> Path | None:
                raw = (row.get(column) or "").strip()
                if not raw:
                    return None
                value = Path(raw).expanduser()
                value = value if value.is_absolute() else root / value
                value = value.resolve()
                if require_images and not value.is_file():
                    raise FileNotFoundError(f"{annotation.name}:{line_number} 파일이 없습니다: {value}")
                return value

            image_path = path_value("image_path")
            assert image_path is not None
            labels = {}
            for task_name, task in FIT_VISION_TASKS.items():
                value = (row.get(task_name) or "").strip()
                if not value:
                    continue
                if value not in task.labels:
                    raise ValueError(
                        f"{annotation.name}:{line_number} {task_name} 라벨이 잘못되었습니다: {value!r}"
                    )
                labels[task_name] = value
            allowed = {"quality"}
            allowed.update({"upper_fit", "upper_length"} if category == "top"
                           else {"bottom_silhouette", "bottom_length"})
            invalid = set(labels) - allowed
            if invalid:
                raise ValueError(f"{annotation.name}:{line_number} 카테고리와 맞지 않는 task: {sorted(invalid)}")
            if not labels:
                continue
            records.append(FitVisionRecord(
                image_path=image_path,
                mask_path=path_value("mask_path"),
                split=row_split,
                category=category,
                source_domain=(row.get("source_domain") or "").strip().lower(),
                group_id=(row.get("group_id") or "").strip(),
                labels=labels,
            ))
    validate_group_splits(records)
    return records


def validate_group_splits(records: list[FitVisionRecord]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        if not record.group_id:
            raise ValueError(f"group_id가 비었습니다: {record.image_path}")
        key = (record.source_domain, record.group_id)
        previous = seen.setdefault(key, record.split)
        if previous != record.split:
            raise ValueError(
                f"같은 identity/group이 여러 split에 있습니다: {key} -> {previous}, {record.split}"
            )


def fit_csv_header() -> list[str]:
    return [
        "image_path", "mask_path", "split", "category", "source_domain", "group_id",
        *FIT_VISION_TASKS,
    ]
