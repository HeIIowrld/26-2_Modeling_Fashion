"""핏 분류 데이터의 CPU 라벨링 저장소와 deterministic group split."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from fit_vision_dataset import fit_csv_header
from fit_vision_schema import FIT_VISION_TASKS


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def stable_group_split(group_id: str) -> str:
    """같은 group은 어느 컴퓨터에서도 같은 80/10/10 split을 받는다."""
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "val" if value < 90 else "test"


def discover_images(images_dir: str | Path) -> list[Path]:
    root = Path(images_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"핏 라벨링 이미지 폴더가 없습니다: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    )


def infer_group_id(path: Path, source_domain: str, pattern: str = "") -> str:
    if pattern:
        matched = re.search(pattern, path.as_posix())
        if not matched:
            raise ValueError(f"group 정규식과 맞지 않는 파일입니다: {path}")
        return matched.group(1) if matched.groups() else matched.group(0)
    if source_domain == "shop":
        matched = re.search(r"(?:MS)?(\d{5,})", path.stem, re.IGNORECASE)
        if matched:
            return f"product_{matched.group(1)}"
    # user 사진은 같은 사람/착장의 여러 view가 있으면 --group-regex를 반드시 쓴다.
    return path.stem


def relative_or_absolute(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


class FitAnnotationStore:
    def __init__(self, csv_path: str | Path, dataset_root: str | Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.rows: dict[tuple[str, str], dict[str, str]] = {}
        if self.csv_path.is_file():
            with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    self.rows[(row.get("image_path", ""), row.get("category", ""))] = {
                        column: row.get(column, "") for column in fit_csv_header()
                    }

    def key(self, image_path: Path, category: str) -> tuple[str, str]:
        return relative_or_absolute(image_path, self.dataset_root), category

    def get(self, image_path: Path, category: str) -> dict[str, str] | None:
        return self.rows.get(self.key(image_path, category))

    def save(
        self,
        image_path: Path,
        *,
        category: str,
        source_domain: str,
        group_id: str,
        labels: dict[str, str],
        mask_path: Path | None = None,
    ) -> dict[str, str]:
        if category not in {"top", "bottom"}:
            raise ValueError("category는 top/bottom이어야 합니다.")
        allowed = {"quality"}
        allowed.update(
            ("upper_fit", "upper_length")
            if category == "top" else ("bottom_silhouette", "bottom_length")
        )
        for task, label in labels.items():
            if task not in allowed:
                raise ValueError(f"{category}에 사용할 수 없는 task입니다: {task}")
            if label and label not in FIT_VISION_TASKS[task].labels:
                raise ValueError(f"{task} 라벨이 잘못되었습니다: {label}")
        row = {column: "" for column in fit_csv_header()}
        row.update({
            "image_path": relative_or_absolute(image_path, self.dataset_root),
            "mask_path": relative_or_absolute(mask_path, self.dataset_root),
            "split": stable_group_split(f"{source_domain}:{group_id}"),
            "category": category,
            "source_domain": source_domain,
            "group_id": group_id,
            **labels,
        })
        self.rows[(row["image_path"], category)] = row
        self.flush()
        return row

    def flush(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fit_csv_header())
            writer.writeheader()
            writer.writerows(sorted(self.rows.values(), key=lambda row: (
                row["source_domain"], row["group_id"], row["image_path"], row["category"]
            )))
        temporary.replace(self.csv_path)

    def counts(self) -> dict[str, dict[str, int]]:
        result = {}
        for task in FIT_VISION_TASKS:
            counts = Counter(row.get(task, "") for row in self.rows.values() if row.get(task, ""))
            result[task] = dict(counts)
        return result
