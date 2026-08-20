from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from fashion_attribute_schema import ATTRIBUTE_TASKS, MULTI_LABEL_TASKS, label_index


MULTI_VALUE_SEPARATOR = "|"


@dataclass(frozen=True)
class AttributeRecord:
    image_path: Path
    split: str
    bbox: tuple[float, float, float, float] | None = None
    labels: dict[str, list[str]] = field(default_factory=dict)


def _parse_bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    values = [row.get(name, "").strip() for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("bbox_x, bbox_y, bbox_w, bbox_h는 모두 입력하거나 모두 비워야 합니다.")
    bbox = tuple(float(value) for value in values)
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("bbox_w와 bbox_h는 0보다 커야 합니다.")
    return bbox


def load_attribute_csv(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    *,
    split: str | None = None,
    require_images: bool = True,
) -> list[AttributeRecord]:
    annotation_path = Path(csv_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else annotation_path.parent
    records: list[AttributeRecord] = []
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("학습 CSV에는 image_path 열이 필요합니다.")
        for line_number, row in enumerate(reader, start=2):
            row_split = (row.get("split") or "train").strip().lower()
            if split and row_split != split.lower():
                continue
            path = Path(row["image_path"].strip()).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if require_images and not path.is_file():
                raise FileNotFoundError(f"{annotation_path.name}:{line_number} 이미지가 없습니다: {path}")

            labels: dict[str, list[str]] = {}
            for task_name, task in ATTRIBUTE_TASKS.items():
                raw = (row.get(task_name) or "").strip()
                if not raw:
                    continue
                values = [value.strip() for value in raw.split(MULTI_VALUE_SEPARATOR) if value.strip()]
                if not task.multi_label and len(values) != 1:
                    raise ValueError(f"{annotation_path.name}:{line_number} {task_name}은 단일 라벨이어야 합니다.")
                unknown = set(values) - set(task.labels)
                if unknown:
                    raise ValueError(
                        f"{annotation_path.name}:{line_number} {task_name}에 정의되지 않은 라벨이 있습니다: {sorted(unknown)}"
                    )
                labels[task_name] = values
            if labels:
                records.append(AttributeRecord(path, row_split, _parse_bbox(row), labels))
    return records


def encode_record_targets(record: AttributeRecord) -> tuple[dict[str, int | list[float]], dict[str, bool]]:
    targets: dict[str, int | list[float]] = {}
    valid: dict[str, bool] = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        values = record.labels.get(task_name, [])
        valid[task_name] = bool(values)
        if task.multi_label:
            encoded = [0.0] * len(task.labels)
            indices = label_index(task_name)
            for value in values:
                encoded[indices[value]] = 1.0
            targets[task_name] = encoded
        else:
            targets[task_name] = label_index(task_name)[values[0]] if values else -1
    return targets, valid


# Fashionpedia 영문 ontology를 이 프로젝트의 비교적 작은 헤드 라벨로 합친다.
# 포함되지 않은 속성은 버리는 대신 CSV에 빈 값으로 남아 해당 loss에서 제외된다.
FASHIONPEDIA_CATEGORY_MAP = {
    "top, t-shirt, sweatshirt": "티셔츠", "shirt, blouse": "셔츠",
    "t-shirt": "티셔츠", "shirt": "셔츠", "blouse": "블라우스", "sweater": "니트",
    "cardigan": "가디건", "jacket": "재킷", "coat": "코트", "vest": "베스트",
    "top": "탑", "pants": "팬츠", "shorts": "쇼츠", "skirt": "스커트",
    "dress": "원피스", "jumpsuit": "점프수트",
}

FASHIONPEDIA_GARMENT_PART_LABELS = {
    "hood": ("detail", "후드"),
    "pocket": ("detail", "포켓"),
    "zipper": ("detail", "지퍼"),
    "bow": ("detail", "리본"),
    "ruffle": ("detail", "프릴·러플"),
    "sequin": ("detail", "스팽글"),
    "belt": ("detail", "벨트"),
    "lapel": ("collar", "라펠 칼라"),
}
FASHIONPEDIA_ASSOCIATED_PARTS = {
    "sleeve", "collar", "lapel", "neckline", *FASHIONPEDIA_GARMENT_PART_LABELS.keys()
}

ATTRIBUTE_KEYWORDS = {
    "sleeve_length": {
        "sleeveless": "민소매", "short sleeve": "반팔", "three quarter sleeve": "7부 소매",
        "three-quarter sleeve": "7부 소매", "long sleeve": "긴팔",
        "sleeve length short": "반팔", "sleeve length elbow": "반팔",
        "sleeve length three quarter": "7부 소매", "sleeve length wrist": "긴팔",
        "short (length)": "반팔", "elbow-length": "반팔",
        "three quarter (length)": "7부 소매", "wrist-length": "긴팔",
    },
    "sleeve_shape": {
        "puff sleeve": "퍼프 소매", "balloon sleeve": "벌룬 소매", "bell sleeve": "벨 소매",
        "raglan sleeve": "래글런 소매", "cap sleeve": "캡 소매", "regular sleeve": "기본 소매",
        "sleeve type puff": "퍼프 소매", "sleeve type bell": "벨 소매",
        "sleeve type raglan": "래글런 소매", "sleeve type cap": "캡 소매",
        "set-in sleeve": "기본 소매", "raglan (sleeve)": "래글런 소매",
        "cap (sleeve)": "캡 소매", "puff (sleeve)": "퍼프 소매",
        "bell (sleeve)": "벨 소매", "circular flounce (sleeve)": "벨 소매",
        "bishop (sleeve)": "벌룬 소매", "poet (sleeve)": "벌룬 소매",
    },
    "upper_length": {
        "crop": "크롭 기장", "hip length": "기본 기장", "longline": "롱 기장",
        "above-the-hip (length)": "크롭 기장", "hip (length)": "기본 기장",
    },
    "lower_length": {
        "mini": "쇼츠·미니 기장", "above the knee": "쇼츠·미니 기장", "knee length": "무릎 기장",
        "midi": "미디·7부 기장", "ankle length": "롱·긴바지 기장", "floor length": "롱·긴바지 기장",
        "micro (length)": "쇼츠·미니 기장", "above-the-knee (length)": "쇼츠·미니 기장",
        "knee (length)": "무릎 기장", "below the knee (length)": "미디·7부 기장",
        "maxi (length)": "롱·긴바지 기장", "floor (length)": "롱·긴바지 기장",
    },
    "neckline": {
        "crew neck": "라운드넥", "neckline crew": "라운드넥", "round neck": "라운드넥",
        "v-neck": "V넥", "v neck": "V넥", "neckline v": "V넥",
        "square neck": "스퀘어넥", "boat neck": "보트넥", "off-the-shoulder": "오프숄더",
        "off shoulder": "오프숄더", "halter neck": "홀터넥", "turtleneck": "터틀넥",
        "crew (neck)": "라운드넥", "round (neck)": "라운드넥", "square (neckline)": "스퀘어넥",
        "boat (neck)": "보트넥", "halter (neck)": "홀터넥", "turtle (neck)": "터틀넥",
    },
    "collar": {
        "polo collar": "폴로 칼라", "collar polo": "폴로 칼라", "polo-style collar": "폴로 칼라",
        "shirt collar": "셔츠 칼라", "stand collar": "스탠드 칼라", "lapel": "라펠 칼라",
        "peter pan collar": "피터팬 칼라", "sailor collar": "세일러 칼라", "no collar": "칼라 없음",
        "polo (collar)": "폴로 칼라", "shirt (collar)": "셔츠 칼라",
        "regular (collar)": "셔츠 칼라", "banded (collar)": "스탠드 칼라",
        "mandarin (collar)": "스탠드 칼라", "peter pan (collar)": "피터팬 칼라",
        "sailor (collar)": "세일러 칼라", "collarless": "칼라 없음",
    },
    "upper_fit": {
        "slim fit": "슬림핏", "silhouette tight (fit)": "슬림핏", "regular fit": "레귤러핏",
        "silhouette regular (fit)": "레귤러핏", "loose fit": "여유핏",
        "silhouette loose (fit)": "여유핏", "silhouette oversized": "오버핏",
    },
    "lower_fit": {
        "skinny": "슬림핏", "straight": "스트레이트핏", "tapered": "테이퍼드핏",
        "wide leg": "와이드핏", "flare": "플레어핏", "flared": "플레어핏",
        "silhouette tight (fit)": "슬림핏", "silhouette straight": "스트레이트핏",
    },
    "silhouette": {
        "silhouette a-line": "A라인", "silhouette straight": "H라인",
        "silhouette fit and flare": "X라인", "silhouette balloon": "O라인",
        "h-line": "H라인", "x-line": "X라인", "o-line": "O라인",
        "silhouette asymmetric": "비대칭", "silhouette asymmetrical": "비대칭",
    },
    "pattern": {
        "solid": "무지", "plain": "무지", "stripe": "스트라이프", "plaid": "체크", "check": "체크", "dot": "도트",
        "floral": "플로럴", "graphic": "그래픽", "cartoon": "그래픽", "letters": "그래픽",
        "numbers": "그래픽", "animal": "애니멀", "camouflage": "카모", "color block": "컬러 블록",
    },
    "material": {
        "cotton": "코튼", "denim": "데님", "knit": "니트", "wool": "울", "leather": "가죽",
        "suede": "스웨이드", "chiffon": "시폰", "silk": "실크·새틴", "satin": "실크·새틴",
        "fleece": "퍼·플리스", "fur": "퍼·플리스", "mesh": "메시", "lace fabric": "레이스",
    },
    "detail": {
        "pocket": "포켓", "ruffle": "프릴·러플", "frill": "프릴·러플", "zipper": "지퍼",
        "belt": "벨트", "button": "단추", "bow": "리본", "lace detail": "레이스", "pleat": "플리츠",
        "embroidery": "자수", "sequin": "스팽글", "hood": "후드", "no detail": "디테일 없음",
        "zip-up": "지퍼", "single breasted": "단추", "double breasted": "단추",
    },
}


def _refine_fashionpedia_category(category: str, names: list[str]) -> str:
    joined = " ".join(name.lower() for name in names)
    if "polo (shirt)" in joined:
        return "폴로 셔츠"
    if "hoodie" in joined:
        return "후드티"
    if category == "재킷" and "blazer" in joined:
        return "블레이저"
    if category == "팬츠" and "jeans" in joined:
        return "청바지"
    top_nicknames = (
        "crop (top)", "halter (top)", "camisole", "tank (top)", "peasant (top)",
        "tube (top)", "tunic (top)", "smock (top)",
    )
    if category == "티셔츠" and any(name in joined for name in top_nicknames):
        return "탑"
    return category


def _map_attribute_names(names: list[str], category: str = "") -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for task_name, keyword_map in ATTRIBUTE_KEYWORDS.items():
        values = []
        for raw_name in names:
            normalized = raw_name.lower().replace("_", "-").replace("/", " ")
            for keyword, label in keyword_map.items():
                if keyword in normalized and label not in values:
                    values.append(label)
        if values:
            # Fashionpedia annotation noise가 단일 헤드에서 충돌하면 첫 번째 ontology 값을 사용한다.
            mapped[task_name] = values if task_name in MULTI_LABEL_TASKS else values[:1]

    upper_categories = {"티셔츠", "폴로 셔츠", "셔츠", "블라우스", "니트", "가디건", "후드티", "재킷", "블레이저", "코트", "베스트", "탑", "원피스", "점프수트"}
    lower_categories = {"팬츠", "청바지", "쇼츠", "스커트"}
    upper_length_categories = upper_categories - {"코트", "원피스", "점프수트"}
    lower_length_categories = lower_categories | {"원피스", "점프수트"}
    if category not in upper_categories:
        for task_name in ("sleeve_length", "sleeve_shape", "neckline", "collar", "upper_fit"):
            mapped.pop(task_name, None)
    if category not in upper_length_categories:
        mapped.pop("upper_length", None)
    if category not in lower_categories:
        mapped.pop("lower_subtype", None)
        mapped.pop("lower_fit", None)
    if category not in lower_length_categories:
        mapped.pop("lower_length", None)

    joined = " ".join(name.lower() for name in names)
    if category == "쇼츠":
        mapped["lower_length"] = ["쇼츠·미니 기장"]
    elif category in {"팬츠", "청바지"} and any(value in joined for value in ("capri (pants)", "crop (pants)")):
        mapped["lower_length"] = ["미디·7부 기장"]
    if category in {"니트", "가디건"} or "sweater (dress)" in joined:
        mapped.setdefault("material", []).append("니트")
    if category == "청바지":
        mapped.setdefault("material", []).append("데님")
    if "hoodie" in joined:
        mapped.setdefault("detail", []).append("후드")
    for task_name in MULTI_LABEL_TASKS:
        if task_name in mapped:
            mapped[task_name] = list(dict.fromkeys(mapped[task_name]))
    return mapped


def _intersection_over_part(part_bbox: list[float], garment_bbox: list[float]) -> float:
    px, py, pw, ph = part_bbox
    gx, gy, gw, gh = garment_bbox
    left, top = max(px, gx), max(py, gy)
    right, bottom = min(px + pw, gx + gw), min(py + ph, gy + gh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / max(float(pw * ph), 1.0)


def _annotation_key(annotation: dict) -> int:
    return int(annotation["id"]) if "id" in annotation else id(annotation)


def _associate_garment_parts(
    annotations: list[dict],
    categories: dict[int, str],
) -> dict[int, list[dict]]:
    """부품 bbox를 가장 많이 감싸는 주 의류 하나에 연결한다."""
    garments = [
        annotation
        for annotation in annotations
        if categories[annotation["category_id"]].lower().strip() in FASHIONPEDIA_CATEGORY_MAP
    ]
    assignments: dict[int, list[dict]] = defaultdict(list)
    for part in annotations:
        part_category = categories[part["category_id"]].lower().strip()
        if part_category not in FASHIONPEDIA_ASSOCIATED_PARTS:
            continue
        candidates = []
        for garment in garments:
            containment = _intersection_over_part(part["bbox"], garment["bbox"])
            if containment < 0.55:
                continue
            garment_area = max(float(garment["bbox"][2] * garment["bbox"][3]), 1.0)
            # containment이 같으면 더 구체적인 작은 의류 bbox를 선택한다.
            candidates.append((containment, -garment_area, garment))
        if candidates:
            owner = max(candidates, key=lambda value: (value[0], value[1]))[2]
            assignments[_annotation_key(owner)].append(part)
    return assignments


def convert_fashionpedia_instances(
    annotation_json: str | Path,
    output_csv: str | Path,
    *,
    split: str,
    image_prefix: str | Path = "",
    image_splits: dict[int, str] | None = None,
) -> dict[str, int]:
    """Fashionpedia instances_attributes JSON을 crop 학습용 CSV로 변환한다."""
    source = Path(annotation_json).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    images = {item["id"]: item["file_name"] for item in payload["images"]}
    categories = {item["id"]: item["name"] for item in payload["categories"]}
    attributes = {
        item["id"]: " ".join(
            value for value in (str(item.get("supercategory", "")).strip(), str(item["name"]).strip()) if value
        )
        for item in payload.get("attributes", [])
    }
    prefix = Path(image_prefix) if str(image_prefix).strip() else None
    output = Path(output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *ATTRIBUTE_TASKS.keys()
    ]
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    written = skipped_category = category_only = associated_parts = 0
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_id, image_annotations in annotations_by_image.items():
            part_assignments = _associate_garment_parts(image_annotations, categories)
            for annotation in image_annotations:
                raw_category = categories[annotation["category_id"]].lower().strip()
                category = FASHIONPEDIA_CATEGORY_MAP.get(raw_category)
                if not category:
                    skipped_category += 1
                    continue
                linked_parts = part_assignments.get(_annotation_key(annotation), [])
                associated_parts += len(linked_parts)
                attribute_ids = list(annotation.get("attribute_ids", []))
                for part in linked_parts:
                    attribute_ids.extend(part.get("attribute_ids", []))
                names = [attributes[attr_id] for attr_id in attribute_ids if attr_id in attributes]
                category = _refine_fashionpedia_category(category, names)
                mapped = _map_attribute_names(names, category)
                for part in linked_parts:
                    part_category = categories[part["category_id"]].lower().strip()
                    derived = FASHIONPEDIA_GARMENT_PART_LABELS.get(part_category)
                    if derived:
                        task_name, label = derived
                        if task_name in MULTI_LABEL_TASKS:
                            mapped.setdefault(task_name, []).append(label)
                            mapped[task_name] = list(dict.fromkeys(mapped[task_name]))
                        else:
                            mapped.setdefault(task_name, [label])
                if category in {"티셔츠", "셔츠"} and "폴로 칼라" in mapped.get("collar", []):
                    category = "폴로 셔츠"
                mapped["category"] = [category]
                if len(mapped) == 1:
                    category_only += 1
                x, y, width, height = annotation["bbox"]
                image_path = Path(images[image_id])
                if prefix is not None:
                    image_path = prefix / image_path
                row = {
                    "image_path": image_path.as_posix(),
                    "split": image_splits.get(image_id, split) if image_splits else split,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": width,
                    "bbox_h": height,
                }
                for task_name in ATTRIBUTE_TASKS:
                    row[task_name] = MULTI_VALUE_SEPARATOR.join(mapped.get(task_name, []))
                writer.writerow(row)
                written += 1
    return {
        "written": written,
        "skipped_category": skipped_category,
        "category_only": category_only,
        "associated_parts": associated_parts,
    }
