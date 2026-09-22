"""긴 바지 상품이 원래 하의 모양대로 그려지는 문제의 A/B (2026-09-21, reports/vton_quality/bottom_shape_2026-09-21.md).

usage:
  마스터(인터넷):  eval_bottom_shape.py RELEASE OUT DFMM 10 5 --fetch
  GPU(DeepFashion): LENGTH_VARIANTS=native,leghull,hull,hull_shoe eval_bottom_shape.py RELEASE OUT DFMM 7 5
  GPU(다른 출처):   eval_bottom_shape.py RELEASE OUT PEOPLE_DIR 10 5 8 --wild
                    (PEOPLE_DIR 아래 short/ skirt/ long/ — collect_bottom_shape_people.py 로 모은다)
  LENGTH_PRODUCTS=MS...,MS... 로 상품을 줄일 수 있다. hull_shoe 는 운영 long-hull 정책과 같은 마스크다.

처음 가설(아래)은 절반만 맞았다. 계단을 지워도 드러난 신발이 보호 영역이라 바짓단이 발목에서
모였고, 신발 윗부분 보호를 푼 hull_shoe 가 채택됐다.

가설: 하의 마스크는 다리까지 덮지만 원래 옷 윤곽을 그대로 쓴다. 7부 조거는 밑단에서
마스크 폭이 갑자기 좁아지는데, 모델이 이 계단을 '여기서 바지가 끝난다'로 읽는다.

변형 (하의 마스크만 바꾸고 나머지는 운영 generate() 그대로):
  native   : 현재 운영 (원래 옷 라벨 + 다리)
  leghull  : 골반 아래를 다리별로 나눠 각 다리의 볼록 껍질로 채운다 (계단 제거, 다리 사이 유지)
  hull     : 하체 전체의 볼록 껍질 (CatVTON AutoMasker 방식, 다리 사이까지 채움)
  hull_shoe: hull + 신발 윗부분(SHOE_TOP=0.4) 보호 해제 — 채택된 운영 long-hull 과 같은 마스크

지표 (합성 결과를 다시 파싱):
  reach    : 두 다리 중 짧은 쪽 바지 밑단 위치 (골반=0, 발목=1). 0.9 이상이면 발목까지 닿음
  rendered : 운영 품질 검사와 같은 측정기로 잰 기장 분류
  sim / dE : 상품 옷 조각과의 FashionSigLIP 유사도 / 주조색 색차
  flare    : 합성된 바지의 밑단/허리 폭 비율 (상품 사진 비율과 비교)
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

ROOT = Path(sys.argv[1])            # 운영 릴리스 (모델·데이터셋)
OUT = Path(sys.argv[2])             # 결과 폴더
DFMM = Path(sys.argv[3])            # deepfashion_mm 폴더 (manifest.json, images/)
N_SHORT = int(sys.argv[4]) if len(sys.argv) > 4 else 10
N_LONG = int(sys.argv[5]) if len(sys.argv) > 5 else 5

from PIL import Image

HEADERS = {"User-Agent": "Mozilla/5.0"}
QUERIES = ["와이드 데님 팬츠", "슬랙스", "스키니 진", "조거 팬츠", "카고 팬츠", "스트레이트 데님 팬츠"]
PER_QUERY = 2
SHORT_WORDS = re.compile(r"반바지|쇼츠|숏|하프|버뮤다|7부|8부|9부|크롭|short", re.I)
VARIANTS = tuple(os.environ.get("LENGTH_VARIANTS", "native,leghull,hull").split(","))
SHOE_TOP = float(os.environ.get("SHOE_TOP", "0.4"))


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=15).read()


def collect_products():
    refs = OUT / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    products = []
    # 재현 사례(7부 조거 → 조거 모양으로 그려진 와이드 슬랙스)를 반드시 넣는다.
    repro = Path(os.environ.get("BOTTOM_SHAPE_REPRO_REF", "/data1/dsl01/eval/vton_refres_20260921/refs/MS7156288_thumb.jpg"))
    if repro.is_file():
        Image.open(repro).convert("RGB").save(refs / "MS7156288.jpg", quality=95)
        products.append({"pid": "MS7156288", "name": "로웰 커브드 벨티드 슬랙스 (블랙)", "query": "재현"})
    for keyword in QUERIES:
        query = urllib.parse.urlencode({"gf": "A", "keyword": keyword, "sortCode": "POPULAR",
                                        "page": 1, "size": 20, "caller": "SEARCH", "category": "003"})
        items = json.loads(fetch("https://api.musinsa.com/api2/dp/v2/plp/goods?" + query))["data"]["list"]
        taken = 0
        for item in items:
            pid = f"MS{item['goodsNo']}"
            name = item.get("goodsName", "")
            if SHORT_WORDS.search(name) or any(p["pid"] == pid for p in products):
                continue
            thumb = item["thumbnail"]
            thumb = "https:" + thumb if thumb.startswith("//") else thumb
            try:
                Image.open(BytesIO(fetch(thumb))).convert("RGB").save(refs / f"{pid}.jpg", quality=95)
            except Exception as exc:  # noqa: BLE001
                print(f"  건너뜀 {pid}: {exc}", flush=True)
                continue
            products.append({"pid": pid, "name": name, "query": keyword})
            taken += 1
            if taken >= PER_QUERY:
                break
    return products


def leg_split(landmarks, height):
    """골반 높이와 두 다리를 가르는 중심선 x(y)를 돌려준다."""
    hip_y = (landmarks["left_hip"][1] + landmarks["right_hip"][1]) / 2
    ankle_y = (landmarks["left_ankle"][1] + landmarks["right_ankle"][1]) / 2
    hip_x = (landmarks["left_hip"][0] + landmarks["right_hip"][0]) / 2
    ankle_x = (landmarks["left_ankle"][0] + landmarks["right_ankle"][0]) / 2
    ys = np.arange(height)
    t = np.clip((ys - hip_y) / max(1.0, ankle_y - hip_y), 0, 1.2)
    return hip_y, ankle_y, hip_x + t * (ankle_x - hip_x)


def _fill_hull(points, shape):
    canvas = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points[:, None, :].astype(np.int32))
        cv2.fillPoly(canvas, [hull], 1)
    return canvas.astype(bool)


KEEP_OUT = (1, 2, 3, 8, 9, 11, 12, 13, 15)  # 얼굴·머리·상의·가방·모자·안경·팔·손·발


def leghull_mask(raw, seg, landmarks):
    height, width = raw.shape
    hip_y, _, mid = leg_split(landmarks, height)
    mask = raw.astype(bool).copy()
    ys, xs = np.where(mask)
    below = ys >= hip_y
    for side in (xs < mid[ys], xs >= mid[ys]):
        pick = below & side
        if pick.sum() >= 50:
            mask |= _fill_hull(np.stack([xs[pick], ys[pick]], axis=1), raw.shape)
    return mask & ~np.isin(seg, KEEP_OUT)


def hull_mask(raw, seg, landmarks):
    ys, xs = np.where(raw)
    mask = raw.astype(bool) | _fill_hull(np.stack([xs, ys], axis=1), raw.shape)
    return mask & ~np.isin(seg, KEEP_OUT)


def shoe_tops(seg, fraction):
    """신발(15) 덩어리마다 위쪽 fraction 만큼. 긴 바짓단이 신발 등을 덮을 자리다."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((seg == 15).astype(np.uint8), connectivity=8)
    rows = np.arange(seg.shape[0])[:, None]
    top = np.zeros(seg.shape, bool)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 200:
            continue
        cut = stats[i, cv2.CC_STAT_TOP] + fraction * stats[i, cv2.CC_STAT_HEIGHT]
        top |= (labels == i) & (rows < cut)
    return top


def leg_reach(pants, landmarks):
    """두 다리 중 짧은 쪽 밑단 위치. 골반 0, 발목 1."""
    height = pants.shape[0]
    hip_y, ankle_y, mid = leg_split(landmarks, height)
    ys, xs = np.where(pants)
    if len(ys) < 200:
        return None
    reaches = []
    for side in (xs < mid[ys], xs >= mid[ys]):
        side_ys = ys[side]
        if len(side_ys) < 50:
            return 0.0
        # 몇 픽셀짜리 오라벨 꼬리를 무시하려고 한 행에 3픽셀 이상 있는 가장 낮은 행을 쓴다.
        rows, counts = np.unique(side_ys, return_counts=True)
        rows = rows[counts >= 3]
        reaches.append((rows.max() - hip_y) / max(1.0, ankle_y - hip_y) if len(rows) else 0.0)
    return float(min(reaches))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "products.json"
    if "--fetch" in sys.argv:
        products = collect_products()
        manifest.write_text(json.dumps(products, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"상품 {len(products)}개 수집 -> {manifest}", flush=True)
        return

    global np, cv2
    import cv2
    import numpy as np
    import catvton_tryon as ct
    from catvton_tryon import GARMENT_TARGET_LABELS, CatVTONTryOn, TryOnNotReady, landmarks_to_pixels
    from clothing_parser import ClothingParser
    from fashion_attribute_model import PREPROCESS_SQUASH
    from fashion_model import FashionClassifier
    from outfit_analyzer import OutfitAnalyzer, _masked_crop
    from pose_analyzer import PoseAnalyzer
    from schemas import Product, Recommendation
    from tryon_quality import dominant_color_distance, rendered_bottom_length

    products = json.loads(manifest.read_text(encoding="utf-8"))
    only = [x for x in os.environ.get("LENGTH_PRODUCTS", "").split(",") if x]
    if only:
        products = [p for p in products if p["pid"] in only]
    pose_analyzer = PoseAnalyzer(model_complexity=1)
    parser = ClothingParser(use_fashn=True)
    heads = ROOT / "ai_fashion_recommender" / "models" / "fashion_attribute_heads_augmented.pt"
    classifier = FashionClassifier(enabled=True, attribute_checkpoint=heads if heads.is_file() else None)
    analyzer = OutfitAnalyzer(parser, classifier)
    embed = lambda img: classifier._encode_image(img, PREPROCESS_SQUASH).cpu().numpy()[0]  # noqa: E731

    tryon = CatVTONTryOn.fast(garment_cache_dir=OUT / "cache")
    tryon._garment_parser = parser
    state = {"variant": "native", "landmarks": None, "mask_area": None}
    original_policy = CatVTONTryOn._apply_mask_policy

    def patched(self, raw_mask, category, garment_path, garment, product, segmentation, landmarks_px, context):
        mask = original_policy(self, raw_mask, category, garment_path, garment, product,
                               segmentation, landmarks_px, context)
        if category == "bottom" and state["variant"] != "native":
            base = np.asarray(mask).astype(bool)
            if state["variant"] == "hull_shoe":
                # context 의 segmentation 에서 신발 윗부분을 이미 다리(14)로 바꿔 두었다.
                base = base | state["shoe_top"]
            build = leghull_mask if state["variant"] == "leghull" else hull_mask
            mask = build(base, segmentation, state["landmarks"])
        state["mask_area"] = int(np.asarray(mask).astype(bool).sum())
        return mask

    CatVTONTryOn._apply_mask_policy = patched

    # 상품: 정제 → 거절 여부 → 기준 옷 조각 → 기장 분류
    ground = {}
    for p in products:
        path = OUT / "refs" / f"{p['pid']}.jpg"
        try:
            garment = tryon._prepare_garment_reference(path, "bottom")
        except TryOnNotReady as exc:
            print(f"  상품 거절 {p['pid']}: {exc}", flush=True)
            continue
        rgb = np.asarray(garment.convert("RGB"))
        seg = parser.parse(garment, pose=None)["segmentation"]
        mask = np.isin(seg, GARMENT_TARGET_LABELS["bottom"])
        if mask.sum() < 400:
            continue
        with Image.open(path) as source:
            length = ct.classify_reference_bottom_length(classifier, garment, source_image=source.convert("RGB"))
        ground[p["pid"]] = {"rgb": rgb, "mask": mask, "vec": embed(_masked_crop(rgb, mask)),
                            "flare": ct.pants_flare_ratio(seg == 6), "length": length}
        print(f"  상품 {p['pid']} 기장={length or '보류'} flare={ground[p['pid']]['flare']}", flush=True)
    (OUT / "ground.json").write_text(json.dumps({k: {"flare": v["flare"], "length": v["length"]}
                                                 for k, v in ground.items()}, ensure_ascii=False), encoding="utf-8")

    # 사람: DeepFashion 라벨로 현재 하의 기장을 안다 (0 쇼츠, 1 무릎 위, 2 7부, 3 긴바지).
    groups = {"short": [], "long": [], "skirt": []}
    wild = "--wild" in sys.argv
    if wild:
        # DFMM 인자 자리에 short/ skirt/ long/ 폴더를 둔 사진 폴더를 받는다. 출처(fp_/ms_)를 번갈아 뽑는다.
        n_skirt = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].isdigit() else N_LONG
        limits = {"short": N_SHORT, "skirt": n_skirt, "long": N_LONG}
        candidates = []
        for group in groups:
            files = sorted((DFMM / group).glob("*.jpg"))
            by_source = [[f for f in files if f.name.startswith(prefix)] for prefix in ("fp_", "ms_")]
            for i in range(max(len(x) for x in by_source)):
                candidates += [(group, x[i], None) for x in by_source if i < len(x)]
    else:
        rows = json.loads((DFMM / "manifest.json").read_text(encoding="utf-8"))
        women = "--women" in sys.argv
        limits = {"short": N_SHORT, "long": 0 if women else N_LONG, "skirt": N_LONG if women else 0}
        candidates = []
        for row in sorted(rows, key=lambda r: r["image"]):
            category = row["image"].split("-")[1]
            if women != row["image"].startswith("WOMEN") or category in {"Dresses", "Rompers_Jumpsuits"}:
                continue
            group = "short" if row["lower_length"] in (0, 1, 2) else "long" if row["lower_length"] == 3 else None
            if group is not None:
                candidates.append((group, DFMM / "images" / row["image"], row))
    for group, person, row in candidates:
        if len(groups[group]) >= limits[group]:
            continue
        try:
            pose = pose_analyzer.analyze(person)
            outfit, parsed = analyzer.analyze(person, pose)
        except Exception as exc:  # noqa: BLE001
            print(f"  분석 실패 {person.name}: {exc}", flush=True)
            continue
        seg = parsed["segmentation"]
        lm = landmarks_to_pixels(getattr(pose, "landmarks", None), seg.shape[1], seg.shape[0])
        needed = ("left_hip", "right_hip", "left_ankle", "right_ankle")
        pants = seg == 6
        if wild:
            if group == "skirt":
                if pants.sum() >= 400:
                    continue
                pants = seg == 5
        elif row["image"].startswith("WOMEN") and (seg == 5).sum() >= 2000 and pants.sum() < 400:
            group, pants = "skirt", seg == 5
        elif row["image"].split("-")[1] == "Skirts":
            continue
        if len(groups[group]) >= limits[group]:
            continue
        # 하의를 입었고(사진 넓이의 0.4% 이상), 두 발목이 보이고, 밑단이 화면에 잘리지 않은 사진만 쓴다.
        # DeepFashion 은 처음 실행과 같은 사람이 뽑히도록 절대 픽셀 기준을 유지한다.
        min_pants, min_feet = (0.004 * seg.size, 0.0008 * seg.size) if wild else (2000, 500)
        if any(n not in lm for n in needed) or pants.sum() < min_pants or (seg == 15).sum() < min_feet:
            continue
        if np.where(pants.any(axis=1))[0].max() >= seg.shape[0] - 3:
            continue
        before_reach = leg_reach(pants, lm)
        if before_reach is None or (wild and group == "short" and before_reach >= 0.86) or (wild and group == "long" and before_reach < 0.9):
            continue
        label = row["lower_length"] if row else group
        groups[group].append({"person": person, "pose": pose, "outfit": outfit, "parsed": parsed,
                              "lm": lm, "label": label, "reach0": before_reach})
        print(f"  사람 {group} {person.name} 라벨={label} 분석={outfit.bottom_length} "
              f"현재 밑단={before_reach:.2f}", flush=True)

    results = OUT / "results.jsonl"
    done = set()
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["person"], r["pid"], r["variant"]))
    for group in ("short", "skirt", "long"):
        for entry in groups[group]:
            person, parsed = entry["person"], entry["parsed"]
            context = {k: parsed.get(k) for k in ("upper_mask", "lower_mask", "upper_style_mask",
                                                  "lower_style_mask", "segmentation")}
            context.update({"outfit": entry["outfit"], "classifier": classifier, "pose": entry["pose"],
                            "parser": parser})
            state["landmarks"] = entry["lm"]
            for p in products:
                if p["pid"] not in ground:
                    continue
                g = ground[p["pid"]]
                for variant in VARIANTS:
                    if (person.name, p["pid"], variant) in done:
                        continue
                    state["variant"] = variant
                    if variant == "hull_shoe":
                        top = shoe_tops(parsed["segmentation"], SHOE_TOP)
                        relabeled = parsed["segmentation"].copy()
                        relabeled[top] = 14
                        context["segmentation"] = relabeled
                        state["shoe_top"] = top
                    else:
                        context["segmentation"] = parsed["segmentation"]
                    product = Product(p["pid"], p["name"], "bottom", "", "", [], [], 0, "", True,
                                      image_path=str(OUT / "refs" / f"{p['pid']}.jpg"))
                    rec = Recommendation(rank=0, products=[product], total_score=0.0, score_breakdown={}, reasons=[])
                    target = OUT / "renders" / f"{person.stem}__{p['pid']}__{variant}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    try:
                        tryon.generate(person, rec, target, context=context)
                    except Exception as exc:  # noqa: BLE001
                        row = {"group": group, "person": person.name, "pid": p["pid"], "variant": variant,
                               "error": f"{type(exc).__name__}: {exc}"}
                        results.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"  실패 {row}", flush=True)
                        continue
                    after = np.asarray(Image.open(target).convert("RGB"))
                    seg = parser.parse(Image.fromarray(after), pose=None)["segmentation"]
                    region = np.isin(seg, GARMENT_TARGET_LABELS["bottom"])
                    sim = float(embed(_masked_crop(after, region)) @ g["vec"]) if region.sum() >= 400 else None
                    row = {
                        "group": group, "person": person.name, "label": entry["label"],
                        "current": entry["outfit"].bottom_length, "reach0": entry["reach0"],
                        "pid": p["pid"], "variant": variant, "target_length": g["length"],
                        "reach": leg_reach(seg == 6, entry["lm"]),
                        "rendered": rendered_bottom_length(seg, entry["lm"]),
                        "sim": sim, "dE": dominant_color_distance(g["rgb"], g["mask"], after, region),
                        "flare": ct.pants_flare_ratio(seg == 6), "ref_flare": g["flare"],
                        "mask_area": state["mask_area"],
                        "warnings": [w for w in tryon.last_warnings if "기장" in w],
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                    results.open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    pose_analyzer.close()
    print("=== 완료", flush=True)


if __name__ == "__main__":
    main()
