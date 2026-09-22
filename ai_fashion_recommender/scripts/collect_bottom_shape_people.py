"""하의 기장 실험용 사람 사진을 DeepFashion 밖에서 모은다 (마스터 노드에서 실행).

  fp : Fashionpedia train (Flickr 등 실제 거리 사진, 사진별 CC 라이선스). 공식 S3 zip 에서
       필요한 파일만 Range 요청으로 꺼낸다. ND·권리 미확인 라이선스는 뺀다(합성은 2차 저작물).
  ms : 무신사 공개 검색 API 의 반바지·치마 상품 착용 썸네일(_big).

평가 폴더에만 두고 저장소에는 올리지 않는다. 그룹은 폴더 이름(short / skirt / long)이다.
usage: collect_bottom_shape_people.py OUT_DIR FP_ANNOTATIONS(instances_attributes_train2020.json)
"""
import io
import json
import random
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_deepfashion_mm_sample import HttpRangeFile  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(sys.argv[1])
ANNOTATIONS = Path(sys.argv[2])
ZIP_URL = "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
ALLOWED_LICENSES = {0, 1, 3, 5, 6, 7, 8, 9, 10}
PER_GROUP = {"short": 45, "skirt": 35, "long": 20}
HEADERS = {"User-Agent": "Mozilla/5.0"}
MS_QUERIES = {"short": [("반바지", "003"), ("버뮤다 팬츠", "003"), ("데님 쇼츠", "003")],
              "skirt": [("미니 스커트", None), ("롱스커트", None), ("플리츠 스커트", None)]}
MS_PER_QUERY = 12


def fashionpedia():
    data = json.loads(ANNOTATIONS.read_text())
    labels, boxes = defaultdict(set), defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] < 24:
            labels[a["image_id"]].add(a["category_id"])
            boxes[a["image_id"]].append((a["category_id"], a["bbox"]))
    pools = defaultdict(list)
    for row in data["images"]:
        found = labels[row["id"]]
        if row["license"] not in ALLOWED_LICENSES or row["height"] < row["width"] or min(row["width"], row["height"]) < 480:
            continue
        if 23 not in found or found & {10, 11}:  # 신발이 보이고, 원피스·점프수트가 아닌 사진
            continue
        # 신발 상자가 사진 아래 끝에 붙어 있으면 발이 잘린 사진이다.
        if any(c == 23 and b[1] + b[3] >= row["height"] - 2 for c, b in boxes[row["id"]]):
            continue
        bottoms = found & {6, 7, 8}
        group = {frozenset({7}): "short", frozenset({8}): "skirt", frozenset({6}): "long"}.get(frozenset(bottoms))
        if group:
            pools[group].append(row)
    rng = random.Random(20260921)
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ZIP_URL), buffer_size=1 << 20))
    members = {Path(n).name: n for n in archive.namelist() if n.endswith(".jpg")}
    licenses = {r["id"]: r["name"] for r in data["licenses"]}
    manifest = []
    for group, rows in pools.items():
        rng.shuffle(rows)
        (OUT / group).mkdir(parents=True, exist_ok=True)
        print(f"fp {group}: 후보 {len(rows)}", flush=True)
        for row in rows[:PER_GROUP[group]]:
            if row["file_name"] not in members:
                continue
            name = f"fp_{row['file_name']}"
            (OUT / group / name).write_bytes(archive.read(members[row["file_name"]]))
            manifest.append({"image": f"{group}/{name}", "source": "fashionpedia", "license": licenses[row["license"]],
                             "original_url": row.get("original_url", "")})
    return manifest


def musinsa():
    manifest = []
    for group, queries in MS_QUERIES.items():
        (OUT / group).mkdir(parents=True, exist_ok=True)
        for keyword, category in queries:
            query = {"gf": "A", "keyword": keyword, "sortCode": "POPULAR", "page": 1, "size": 40, "caller": "SEARCH"}
            if category:
                query["category"] = category
            url = "https://api.musinsa.com/api2/dp/v2/plp/goods?" + urllib.parse.urlencode(query)
            items = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=15).read())["data"]["list"]
            taken = 0
            for item in items:
                name = f"ms_{item['goodsNo']}.jpg"
                if (OUT / group / name).exists():
                    continue
                thumb = item["thumbnail"]
                thumb = "https:" + thumb if thumb.startswith("//") else thumb
                big = re.sub(r"_\d+\.jpg$", "_big.jpg", thumb)
                try:
                    image = Image.open(BytesIO(urllib.request.urlopen(
                        urllib.request.Request(big, headers=HEADERS), timeout=15).read())).convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    print(f"  건너뜀 {name}: {exc}", flush=True)
                    continue
                if image.height < image.width:  # 가로 사진은 전신 착용컷이 아니다
                    continue
                image.save(OUT / group / name, quality=95)
                manifest.append({"image": f"{group}/{name}", "source": "musinsa", "product": item.get("goodsName", "")})
                taken += 1
                if taken >= MS_PER_QUERY:
                    break
            print(f"ms {group} '{keyword}': {taken}", flush=True)
    return manifest


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = fashionpedia() + musinsa()
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"총 {len(manifest)}장 -> {OUT}", flush=True)
