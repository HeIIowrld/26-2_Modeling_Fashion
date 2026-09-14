"""VTON(CatVTON) 핏 속성 보존 평가와 자동 품질 검사.

사람 사진 × 상품 레퍼런스를 교차로 합성하고, 결과를 사람 눈 대신 수치로 잰다.
운영 서버와 같은 설정(CatVTONTryOn.fast(), augmented 속성 헤드, 같은 마스크 경로)을
그대로 호출하고, 측정 코드는 합성 결과에만 붙는다.

핏의 정답은 무신사 판매자가 붙인 핏 태그(detail_fit)다. 카탈로그 `fit` 칼럼은
우리 속성 모델이 사진에서 읽은 값이라 정답으로 쓰면 순환 논리가 된다.

측정 항목 (사람·상품 한 쌍마다)
  핏  judge_ref / judge_orig / judge_res   — 같은 속성 헤드가 레퍼런스·원래 옷·결과 옷을 본 확률
      width_orig / width_res / width_mask  — 포즈 기준 옷 폭(분류기 없이 기하로)
  품질 outside_psnr / outside_ssim          — 마스크 밖(바뀌면 안 되는 곳) 보존
      face_psnr                            — 얼굴·머리 보존
      sim_res_ref / sim_orig_ref / rank    — FashionSigLIP 임베딩 충실도와 레퍼런스 식별 순위
      color_de_res / color_de_orig         — 대표색 CIEDE2000
      target_fill / skin_in_orig           — 마스크 안 옷 채움, 원래 옷 자리에 드러난 피부
      sharp_ratio                          — 결과 옷 영역 선명도 / 원래 옷 선명도
      warnings                             — 운영 품질 게이트가 낸 경고

    python eval_vton_fit.py --smoke          # 사람 2 × 상품 2 (카테고리마다)
    python eval_vton_fit.py                  # 전체
    python eval_vton_fit.py --offset 0 --limit 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ["EVAL_ROOT"])
REPO = ROOT / "repo" / "ai_fashion_recommender"
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402

from catvton_tryon import (  # noqa: E402
    GARMENT_TARGET_LABELS, PROTECT_LABELS, CatVTONTryOn, _dilate_mask,
    _largest_component, _refine_garment_mask, _solidify_mask,
)
from clothing_parser import ClothingParser  # noqa: E402
from fashion_attribute_model import PREPROCESS_SQUASH  # noqa: E402
from fashion_model import FashionClassifier  # noqa: E402
from outfit_analyzer import (  # noqa: E402
    OutfitAnalyzer, _ciede2000, _crop_geometry, _dominant_rgb, _garment_crop,
    _masked_crop, _rgb_to_cielab,
)
from pose_analyzer import PoseAnalyzer  # noqa: E402
from schemas import Product, Recommendation  # noqa: E402

CATALOG = Path(os.environ["EVAL_CATALOG"])
GARMENT_RAW = Path(os.environ["EVAL_GARMENT_RAW"])
PEOPLE = ROOT / "people"
OUT = ROOT / "out"
SEED = 20260914

FIT_ORDER = ("슬림핏", "레귤러핏", "여유핏", "오버핏")  # 판매자 태그 → 0~3
TOP_TYPES = {"티셔츠", "셔츠", "니트", "후드티", "폴로 셔츠", "블라우스", "탑"}
BOTTOM_TYPES = {"청바지", "치노 팬츠", "슬랙스", "조거·스웨트팬츠", "카고 팬츠", "트랙팬츠", "레깅스"}
SHORT_WORDS = ("반바지", "쇼츠", "숏", "하프", "버뮤다", "스커트")
TASKS = {
    "top": ["upper_fit", "sleeve_length", "upper_length", "category"],
    "bottom": ["lower_fit", "pant_leg_shape", "lower_length", "category"],
}
SKIN_LABELS = (12, 14, 16)  # arms, legs, torso

# 전신이 아닌 사진(허리까지)은 하의 평가에서 뺀다.
UPPER_ONLY_PEOPLE = {"demo_1.jpg", "demo_2.jpg", "demo_3.jpg", "demo_4.jpg"}


# ────────────────────────── 상품 선택 ──────────────────────────
def select_garments(per_class: int) -> dict[str, list[dict]]:
    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8-sig")))
    rng = random.Random(SEED)
    chosen: dict[str, list[dict]] = {"top": [], "bottom": []}
    for category, types in (("top", TOP_TYPES), ("bottom", BOTTOM_TYPES)):
        for fit in FIT_ORDER:
            pool = [
                row for row in rows
                if row["category"] == category
                and row["detail_fit"] == fit
                and row["item_type"] in types
                and not any(word in row["name"] for word in SHORT_WORDS)
                and (GARMENT_RAW / Path(row["image_path"]).name).is_file()
            ]
            # 종류가 한쪽으로 쏠리지 않게 종류별로 돌아가며 뽑는다.
            by_type: dict[str, list[dict]] = defaultdict(list)
            for row in pool:
                by_type[row["item_type"]].append(row)
            for bucket in by_type.values():
                rng.shuffle(bucket)
            order = sorted(by_type)
            rng.shuffle(order)
            picked = []
            while len(picked) < per_class and any(by_type.values()):
                for item_type in order:
                    if by_type[item_type] and len(picked) < per_class:
                        picked.append(by_type[item_type].pop())
            print(f"[선택] {category}/{fit}: 후보 {len(pool)} → {len(picked)} "
                  f"({', '.join(sorted({r['item_type'] for r in picked}))})", flush=True)
            chosen[category].extend(picked)
    return chosen


def to_product(row: dict) -> Product:
    return Product(
        product_id=row["product_id"], name=row["name"], category=row["category"],
        color=row["color"], style=row["style"], purposes=[], body_shapes=[],
        price=int(float(row["price"] or 0)), season=row["season"], stock=True,
        item_type=row["item_type"], fit=row["fit"], length=row["length"],
        neckline=row.get("neckline", ""),
        image_path=str(GARMENT_RAW / Path(row["image_path"]).name),
    )


# ────────────────────────── 측정 도구 ──────────────────────────
def judge(classifier: FashionClassifier, rgb: np.ndarray, mask: np.ndarray, tasks: list[str]) -> dict:
    """운영 분석과 같은 방식(배경을 남긴 bbox crop + 기하 특징)으로 속성 확률을 읽는다."""
    if mask.sum() < 400:
        return {}
    # analyze_crop의 학습 헤드 경로와 같다(제로샷 프롬프트는 빈 목록이면 토크나이저가 죽어 뺐다).
    crop = _garment_crop(rgb, mask)
    modes = classifier.head_preprocess_modes
    encoded = {mode: classifier._encode_image(crop, mode) for mode in modes}
    head_input = encoded[modes[0]] if len(modes) == 1 else encoded
    learned = classifier.attribute_predictor.predict_features(
        head_input, tasks=tasks, geometry=_crop_geometry(mask)
    )
    return {task: {k: round(float(v), 4) for k, v in pred.scores.items()} for task, pred in learned.items()}


def embed(classifier: FashionClassifier, image: Image.Image) -> np.ndarray:
    return classifier._encode_image(image, PREPROCESS_SQUASH).cpu().numpy()[0]


def psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float | None:
    if region.sum() < 100:
        return None
    mse = float(((a[region].astype(np.float64) - b[region].astype(np.float64)) ** 2).mean())
    return 99.0 if mse < 1e-10 else round(10 * np.log10(255.0 ** 2 / mse), 2)


def ssim_region(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float | None:
    """가우시안 창 SSIM 맵을 계산한 뒤 영역 평균을 낸다."""
    if region.sum() < 100:
        return None
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda img: cv2.GaussianBlur(img, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sx, sy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))
    return round(float(ssim[region].mean()), 4)


def sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 100:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F)[mask].var())


def pixel(pose, name: str, w: int, h: int) -> tuple[float, float]:
    x, y, _ = pose.landmarks[name]
    return x * w, y * h


def band_width(mask: np.ndarray, y0: float, y1: float, scale: float) -> float | None:
    """y0~y1 행 구간에서 행마다 마스크 픽셀 수를 세고, 평균을 기준 폭으로 나눈다."""
    h = mask.shape[0]
    top, bottom = int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))
    if bottom - top < 3 or scale < 5:
        return None
    return round(float(mask[top:bottom].sum(axis=1).mean()) / scale, 4)


def widths(category: str, seg: np.ndarray, edit: np.ndarray, pose) -> dict:
    """옷이 몸 기준으로 얼마나 넓은지. 상의=어깨폭, 하의=골반폭으로 나눈다."""
    h, w = seg.shape
    garment = np.isin(seg, GARMENT_TARGET_LABELS[category])
    out = {}
    if category == "top":
        ls, rs = pixel(pose, "left_shoulder", w, h), pixel(pose, "right_shoulder", w, h)
        lh, rh = pixel(pose, "left_hip", w, h), pixel(pose, "right_hip", w, h)
        sy, hy = (ls[1] + rs[1]) / 2, (lh[1] + rh[1]) / 2
        scale = abs(ls[0] - rs[0])
        for name, a, b in (("chest", 0.25, 0.55), ("waist", 0.60, 0.95)):
            out[name] = band_width(garment, sy + a * (hy - sy), sy + b * (hy - sy), scale)
            out[name + "_mask"] = band_width(edit, sy + a * (hy - sy), sy + b * (hy - sy), scale)
    else:
        lh, rh = pixel(pose, "left_hip", w, h), pixel(pose, "right_hip", w, h)
        lk, rk = pixel(pose, "left_knee", w, h), pixel(pose, "right_knee", w, h)
        la, ra = pixel(pose, "left_ankle", w, h), pixel(pose, "right_ankle", w, h)
        hy, ky, ay = (lh[1] + rh[1]) / 2, (lk[1] + rk[1]) / 2, (la[1] + ra[1]) / 2
        scale = abs(lh[0] - rh[0])
        for name, a, b in (("thigh", hy + 0.3 * (ky - hy), hy + 0.7 * (ky - hy)),
                           ("shin", ky + 0.3 * (ay - ky), ky + 0.8 * (ay - ky))):
            out[name] = band_width(garment, a, b, scale)
            out[name + "_mask"] = band_width(edit, a, b, scale)
    return out


def reference_mask(parser: ClothingParser, raw: Image.Image, category: str) -> np.ndarray | None:
    """_prepare_garment_reference와 같은 절차로 원본 상품 사진의 옷 마스크를 만든다."""
    seg = parser.parse(raw, pose=None)["segmentation"]
    mask = np.isin(seg, GARMENT_TARGET_LABELS[category])
    if mask.sum() < 0.02 * mask.size:
        return None
    mask = _refine_garment_mask(_largest_component(mask))
    mask = np.logical_and(mask, ~np.isin(seg, (1, 2, 8, 9, 11, 12, 13, 17)))
    return mask if mask.sum() >= 0.02 * mask.size else None


def widen_lower_mask(mask: np.ndarray, pose, pad_ratio: float = 0.25) -> np.ndarray:
    """대조 실험용: 하의 마스크를 행마다 좌우 끝까지 채우고 골반폭의 pad_ratio만큼 넓힌다.

    원래 옷 윤곽을 지운 VITON-HD식 agnostic 박스에 가깝다. 운영 경로에는 없다.
    """
    h, w = mask.shape
    lh, rh = pixel(pose, "left_hip", w, h), pixel(pose, "right_hip", w, h)
    pad = int(abs(lh[0] - rh[0]) * pad_ratio)
    wide = np.zeros_like(mask, dtype=bool)
    for y in np.where(mask.any(axis=1))[0]:
        xs = np.where(mask[y])[0]
        wide[y, max(0, xs.min() - pad):min(w, xs.max() + pad + 1)] = True
    return wide


def edit_mask(parsed: dict, category: str) -> np.ndarray:
    """generate()가 실제로 쓰는 인페인팅 마스크를 원본 좌표에서 다시 만든다."""
    key = ("upper_style_mask", "upper_mask") if category == "top" else ("lower_style_mask", "lower_mask")
    raw = next(parsed[k] for k in key if parsed.get(k) is not None and np.any(parsed[k]))
    mask = _dilate_mask(_solidify_mask(raw)) > 0
    mask[np.isin(parsed["segmentation"], PROTECT_LABELS)] = False
    return mask


# ────────────────────────── 실행 ──────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask-mode", choices=("native", "wide"), default="native",
                    help="wide: 하의만, 마스크를 넓힌 대조 실험")
    opts = ap.parse_args()
    wide = opts.mask_mode == "wide"

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    results_path = OUT / ("smoke.jsonl" if opts.smoke else "results_wide.jsonl" if wide else "results.jsonl")
    done = set()
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "error" not in record:
                done.add(record["pair"])

    t0 = time.time()
    pose_analyzer = PoseAnalyzer()
    parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(
        enabled=True, attribute_checkpoint=os.environ["FASHION_ATTRIBUTE_HEADS_PATH"]
    )
    analyzer = OutfitAnalyzer(parser, classifier)
    vton = CatVTONTryOn.fast(max_retries=0, garment_cache_dir=OUT / "ref_cache")
    vton._garment_parser = parser
    print(f"[준비] 모델 로드 {time.time() - t0:.1f}s · device={classifier.device} · "
          f"heads={classifier.trained_attributes_enabled} · parser={parser.backend} · "
          f"preset steps={vton.num_inference_steps} scheduler={vton.scheduler}", flush=True)

    garments = select_garments(opts.per_class)
    people = sorted(p for p in PEOPLE.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if opts.smoke:
        garments = {c: [g for i, g in enumerate(v) if i % (len(v) // 2) == 0][:2] for c, v in garments.items()}
        people = [p for p in people if p.name in {"model_5.png", "049713_0.jpg"}]

    # ── 레퍼런스: 정제 이미지·마스크·판정·임베딩을 한 번씩만 ──
    refs: dict[str, dict] = {}
    ref_meta = []
    for category, rows in garments.items():
        for row in rows:
            product = to_product(row)
            raw = Image.open(product.image_path).convert("RGB")
            cleaned = vton._prepare_garment_reference(Path(product.image_path), category)
            mask = reference_mask(parser, raw, category)
            raw_rgb = np.asarray(raw)
            info = {
                "product": product, "cleaned": cleaned,
                "judge": judge(classifier, raw_rgb, mask, TASKS[category]) if mask is not None else {},
                "emb": embed(classifier, cleaned),
                "color": _dominant_rgb(raw_rgb, mask) if mask is not None else None,
                "report": vton.reference_reports.get(Path(product.image_path).name, {}),
            }
            refs[product.product_id] = info
            cleaned.save(OUT / "refs" / f"{product.product_id}.jpg", quality=90)
            ref_meta.append({
                "product_id": product.product_id, "category": category, "name": row["name"],
                "item_type": row["item_type"], "seller_fit": row["detail_fit"],
                "catalog_fit": row["fit"], "catalog_length": row["length"],
                "detail_category": row["detail_category"], "judge": info["judge"],
                "reference_report": info["report"], "mask_ok": mask is not None,
            })
    (OUT / ("refs_smoke.json" if opts.smoke else "refs.json")).write_text(
        json.dumps(ref_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[레퍼런스] {len(refs)}개 준비 {time.time() - t0:.1f}s", flush=True)

    embeddings = {c: {pid: refs[pid]["emb"] for pid in (r["product_id"] for r in rows)}
                  for c, rows in garments.items()}

    pairs = [(person, category, row) for person in people
             for category, rows in garments.items()
             if not (category == "bottom" and person.name in UPPER_ONLY_PEOPLE)
             and not (wide and category != "bottom")
             for row in rows]
    end = opts.offset + opts.limit if opts.limit else len(pairs)
    pairs = pairs[opts.offset:end]
    print(f"[쌍] {len(pairs)}개 (사람 {len(people)}명)", flush=True)

    person_cache: dict[str, dict] = {}
    with results_path.open("a", encoding="utf-8") as sink:
        for index, (person_path, category, row) in enumerate(pairs, 1):
            pid = row["product_id"]
            pair_id = f"{person_path.stem}__{pid}"
            if pair_id in done:
                continue
            started = time.time()
            record: dict = {"pair": pair_id, "person": person_path.name, "product_id": pid,
                            "category": category, "seller_fit": row["detail_fit"],
                            "item_type": row["item_type"]}
            try:
                if person_path.name not in person_cache:
                    pose = pose_analyzer.analyze(person_path)
                    outfit, parsed = analyzer.analyze(person_path, pose)
                    rgb = np.asarray(Image.open(person_path).convert("RGB"))
                    person_cache.clear()  # 사람 순서대로 돌므로 한 명만 들고 있는다
                    person_cache[person_path.name] = {
                        "pose": pose, "outfit": outfit, "parsed": parsed, "rgb": rgb,
                        "judge": {c: judge(classifier, rgb,
                                           parsed["upper_mask" if c == "top" else "lower_mask"],
                                           TASKS[c]) for c in ("top", "bottom")},
                    }
                person = person_cache[person_path.name]
                pose, parsed, orig = person["pose"], person["parsed"], person["rgb"]
                if not pose.valid:
                    raise RuntimeError("포즈 무효")

                product = refs[pid]["product"]
                if wide:
                    parsed = dict(parsed)
                    for key in ("lower_style_mask", "lower_mask"):
                        parsed[key] = widen_lower_mask(parsed[key], pose)
                record["mask_mode"] = opts.mask_mode
                out_path = OUT / "images" / f"{'wide__' if wide else ''}{pair_id}.jpg"
                context = {k: parsed.get(k) for k in (
                    "upper_mask", "lower_mask", "upper_style_mask", "lower_style_mask", "segmentation")}
                context.update({"outfit": person["outfit"], "classifier": classifier, "pose": pose})
                rec = Recommendation(rank=1, products=[product], total_score=0.0,
                                     score_breakdown={}, reasons=[])
                synth_start = time.time()
                vton.generate(person_path, rec, out_path, context=context)
                record["synth_sec"] = round(time.time() - synth_start, 2)
                record["render_kind"] = vton.last_render_kind
                record["warnings"] = list(vton.last_warnings)

                res = np.asarray(Image.open(out_path).convert("RGB"))
                parsed_res = parser.parse(res, pose)
                seg_o, seg_r = parsed["segmentation"], parsed_res["segmentation"]
                edit = edit_mask(parsed, category)
                garment_key = "upper_mask" if category == "top" else "lower_mask"
                # 원래 옷은 넓힌 마스크가 아니라 실제 분할 결과로 잰다.
                g_orig, g_res = person["parsed"][garment_key], parsed_res[garment_key]

                # ── 핏 ──
                record["judge_ref"] = refs[pid]["judge"]
                record["judge_orig"] = person["judge"][category]
                record["judge_res"] = judge(classifier, res, g_res, TASKS[category])
                record["width_orig"] = widths(category, seg_o, edit, pose)
                record["width_res"] = widths(category, seg_r, edit, pose)

                # ── 품질: 보존 ──
                margin = cv2.dilate(edit.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                outside = ~margin
                record["outside_psnr"] = psnr(orig, res, outside)
                record["outside_ssim"] = ssim_region(orig, res, outside)
                record["face_psnr"] = psnr(orig, res, np.isin(seg_o, (1, 2)))
                record["changed_outside_ratio"] = round(float(
                    (np.abs(orig.astype(int) - res.astype(int)).max(axis=2)[outside] > 24).mean()), 5)

                # ── 품질: 충실도 ──
                res_emb = embed(classifier, _masked_crop(res, g_res)) if g_res.sum() > 400 else None
                orig_emb = embed(classifier, _masked_crop(orig, g_orig)) if g_orig.sum() > 400 else None
                pool = embeddings[category]
                if res_emb is not None:
                    sims = {k: float(res_emb @ v) for k, v in pool.items()}
                    record["sim_res_ref"] = round(sims[pid], 4)
                    record["rank_res"] = 1 + sum(1 for k, s in sims.items() if k != pid and s > sims[pid])
                    record["pool_size"] = len(pool)
                if orig_emb is not None:
                    record["sim_orig_ref"] = round(float(orig_emb @ pool[pid]), 4)
                ref_color = refs[pid]["color"]
                if ref_color is not None and g_res.sum() > 400:
                    lab_ref = _rgb_to_cielab(ref_color)
                    record["color_de_res"] = round(_ciede2000(lab_ref, _rgb_to_cielab(_dominant_rgb(res, g_res))), 2)
                    if g_orig.sum() > 400:
                        record["color_de_orig"] = round(_ciede2000(lab_ref, _rgb_to_cielab(_dominant_rgb(orig, g_orig))), 2)

                # ── 품질: 구조 ──
                target_res = np.isin(seg_r, GARMENT_TARGET_LABELS[category])
                record["target_fill"] = round(float(target_res[edit].mean()), 4)
                record["garment_outside_edit"] = round(float(
                    (target_res & ~margin).sum()) / max(1, int(target_res.sum())), 5)
                record["skin_in_orig"] = round(float(np.isin(seg_r, SKIN_LABELS)[g_orig].mean()), 4) if g_orig.any() else None
                n_comp, _, stats, _ = cv2.connectedComponentsWithStats(target_res.astype(np.uint8), connectivity=8)
                areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True) if n_comp > 1 else []
                record["garment_components"] = int(sum(1 for a in areas if a > 0.01 * max(1, target_res.sum())))
                so = sharpness(orig, g_orig)
                record["sharp_res"] = round(sharpness(res, g_res), 1)
                record["sharp_ratio"] = round(record["sharp_res"] / so, 3) if so > 1 else None
                record["reference_report"] = refs[pid]["report"]

                # 결과 분할 맵은 사후 분석용으로 압축 저장
                np.savez_compressed(OUT / "images" / f"{'wide__' if wide else ''}{pair_id}_seg.npz",
                                    orig=seg_o.astype(np.uint8), res=seg_r.astype(np.uint8),
                                    edit=edit)
            except Exception as error:  # 한 쌍의 실패가 배치를 멈추지 않게 기록만 한다
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()[-1500:]
            record["total_sec"] = round(time.time() - started, 2)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            elapsed = time.time() - t0
            print(f"[{index}/{len(pairs)}] {pair_id} {category}/{row['detail_fit']} "
                  f"{record.get('total_sec')}s warn={len(record.get('warnings', []))} "
                  f"{'ERR ' + record['error'] if 'error' in record else ''} (누적 {elapsed/60:.1f}분)",
                  flush=True)


if __name__ == "__main__":
    main()
