"""GPU 없이 검증할 수 있는 평가 집계·상품 선택 규칙."""
from collections import defaultdict
import json
import hashlib
import re

import numpy as np


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_run_manifest(results_path, manifest):
    """재개 시 코드·입력·설정이 다른 결과를 같은 JSONL에 섞지 않는다."""
    path = results_path.with_suffix(".manifest.json")
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("평가 입력/코드/설정이 변경됐습니다. 별도 EVAL_ROOT를 사용하세요.")
    elif results_path.exists() and results_path.stat().st_size:
        raise ValueError("manifest 없는 기존 결과에는 이어 쓸 수 없습니다. 별도 EVAL_ROOT를 사용하세요.")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dedup(path):
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "error" not in record or "error" in latest.get(record["pair"], {"error": 1}):
            latest[record["pair"]] = record
    return ([r for r in latest.values() if "error" not in r],
            [r for r in latest.values() if "error" in r])


def cluster_ci(items, stat, key="person", n=2000, rng=None):
    """재추출된 각 클러스터에 별도 ID를 부여해 중복 추출의 가중치를 보존한다."""
    if not items:
        return (float("nan"),) * 3
    rng = np.random.default_rng(20260914) if rng is None else rng
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    groups = list(groups.values())
    point, samples = stat(items), []
    for _ in range(n):
        pick = rng.choice(len(groups), len(groups), replace=True)
        sample = [{**row, key: draw} for draw, i in enumerate(pick) for row in groups[i]]
        value = stat(sample)
        if np.isfinite(value):
            samples.append(value)
    lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (float("nan"),) * 2
    return point, float(lo), float(hi)


def is_short_bottom(row):
    # 이름과 판매자 카테고리만 사용한다. 모델 추정 catalog_length는 정답이 아니다.
    text = f"{row.get('name', '')} {row.get('detail_category', '')}".casefold()
    return (any(w in text for w in ("반바지", "쇼츠", "숏팬츠", "숏 팬츠", "버뮤다", "하프 팬츠"))
            or bool(re.search(r"\b(shorts?|hot\s+pants|half\s+tights)\b", text)))


def is_wide_bottom(row):
    text = f"{row.get('name', '')} {row.get('detail_category', '')}".casefold()
    return (not is_short_bottom(row)
            and "스커트" not in text and "skirt" not in text
            and (any(w in text for w in ("와이드", "벌룬", "배기", "커브드"))
                 or bool(re.search(r"\b(wide|balloon|baggy|curved)\b", text))))
