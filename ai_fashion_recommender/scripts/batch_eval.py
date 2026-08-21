"""인스타 소스 배치 합성 평가. 사용법: python scripts/batch_eval.py [성별당 장수]

성별별 N장을 돌려 인식·추천·경고·산출물을 JSON으로 남기고,
육안 검토용으로 원본|결과 대조 시트를 만든다.

장당 약 50초라 30장이면 25분이다. 이 환경에서는 **분리된 백그라운드 프로세스로
띄우면 모델 로드 직후 CPU 0%로 멈춘다**(3회 재현). 포그라운드로 돌려야 하는데
한 번에 25분을 못 쓰는 경우가 있어 --offset/--limit으로 나눠 돌릴 수 있게 했다.
results.json은 장마다 갱신되고 같은 사진은 이름으로 덮어쓴다.

    python scripts/batch_eval.py 15                  # 30장 전부
    python scripts/batch_eval.py 15 --offset 0 --limit 6   # 앞 6장만
    python scripts/batch_eval.py 15 --skip-existing        # 결과 있는 건 건너뛴다
    python scripts/batch_eval.py 15 --fast                 # 25스텝, 장당 시간 절반
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT = Path(r"c:\Users\jeff4\OneDrive - 엔시스코리아\personal\school\20.활동\DSL\30.프로젝트\2026-2_모델링_패션\ai_fashion_recommender")
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from catvton_tryon import CatVTONTryOn
from clothing_parser import ClothingParser
from config import DATA_DIR, FASHION_ATTRIBUTE_HEADS_PATH, OUTPUT_DIR, PROJECT_DIR
from fashion_model import FashionClassifier
from outfit_analyzer import OutfitAnalyzer
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import RecommendationEngine
from schemas import UserProfile
from test_tryon import _sample_images

arguments = argparse.ArgumentParser(description="인스타 소스 배치 합성 평가")
arguments.add_argument("per_gender", nargs="?", type=int, default=15, help="성별별 장수")
arguments.add_argument("--offset", type=int, default=0, help="선택된 목록에서 건너뛸 장수")
arguments.add_argument("--limit", type=int, default=0, help="이번에 돌릴 장수 (0이면 끝까지)")
arguments.add_argument("--skip-existing", action="store_true", help="결과 파일이 있으면 건너뛴다")
arguments.add_argument("--fast", action="store_true",
                       help="CatVTONTryOn.fast() 프리셋(25스텝). 장당 시간이 절반이다")
arguments.add_argument("--out", default="batch_eval",
                       help="outputs/ 아래 저장 폴더명. 전후 비교할 때 갈라 둔다")
options = arguments.parse_args()
PER_GENDER = options.per_gender
OUT_DIR = OUTPUT_DIR / options.out
SHEET_DIR = OUT_DIR / "sheets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHEET_DIR.mkdir(parents=True, exist_ok=True)

pose_analyzer = PoseAnalyzer()
quality = QualityChecker(pose_analyzer)
parser = ClothingParser(use_fashn=True)
classifier = FashionClassifier(
    enabled=True,
    attribute_checkpoint=FASHION_ATTRIBUTE_HEADS_PATH if FASHION_ATTRIBUTE_HEADS_PATH.is_file() else None,
)
analyzer = OutfitAnalyzer(parser, classifier)
catalog_csv = DATA_DIR / "products_musinsa.csv"
engine = RecommendationEngine(PROJECT_DIR / "FASHION_RULES_MASTER.md", ProductCatalog(catalog_csv))
vton = CatVTONTryOn.fast(max_retries=0) if options.fast else CatVTONTryOn(max_retries=0)
vton._garment_parser = parser

GENDER_LABEL = {"men": "남성", "women": "여성"}
all_images = _sample_images("all", PER_GENDER)
end = options.offset + options.limit if options.limit else len(all_images)
images = all_images[options.offset:end]
print(f"평가 대상 {len(images)}장 "
      f"(성별 {PER_GENDER}장씩 중 {options.offset + 1}~{options.offset + len(images)}번)",
      flush=True)

# 이전 구간의 결과를 이어받는다. 같은 사진은 이름으로 덮어쓴다.
RESULTS_PATH = OUT_DIR / "results.json"
previous = (
    {record["image"]: record for record in json.loads(RESULTS_PATH.read_text(encoding="utf-8"))}
    if RESULTS_PATH.is_file() else {}
)


def save_results() -> list[dict]:
    """전체 목록 순서로 정렬해 저장한다. 구간을 나눠 돌려도 순서가 유지된다."""
    order = {path.name: index for index, path in enumerate(all_images)}
    merged = sorted(previous.values(), key=lambda record: order.get(record["image"], 10**6))
    RESULTS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


results = []
start_all = time.time()
for index, image_path in enumerate(images, 1):
    gender = GENDER_LABEL.get(image_path.parent.name, "")
    already_done = (OUT_DIR / f"{image_path.stem}_result.jpg").is_file()
    if options.skip_existing and already_done and image_path.name in previous:
        print(f"[{index}/{len(images)}] {image_path.name}: 건너뜀(결과 있음)", flush=True)
        continue
    record = {"image": image_path.name, "gender": gender}
    started = time.time()
    try:
        pose = pose_analyzer.analyze(image_path)
        report = quality.check_input(image_path, pose)
        record["sharpness"] = round(report["sharpness"], 1)
        if not pose.valid:
            record["status"] = "포즈 실패"
            record["issues"] = report["issues"]
            results.append(record)
            print(f"[{index}/{len(images)}] {image_path.name}: 포즈 실패", flush=True)
            continue

        outfit, parsed = analyzer.analyze(image_path, pose)
        record["recognized"] = outfit.to_summary_dict()
        record["body_shape"] = pose.body_shape

        recommendations = engine.recommend(UserProfile(gender=gender), pose, outfit, top_k=1)
        record["products"] = [
            {"id": p.product_id, "category": p.category, "name": p.name[:40]}
            for p in recommendations[0].products
        ]

        output_path = OUT_DIR / f"{image_path.stem}_result.jpg"
        vton.generate(
            image_path, recommendations[0], output_path,
            context={
                **{k: parsed.get(k) for k in (
                    "upper_mask", "lower_mask", "upper_style_mask",
                    "lower_style_mask", "segmentation")},
                "outfit": outfit,
                "classifier": classifier,
            },
        )
        record["warnings"] = list(vton.last_warnings)
        record["output"] = output_path.name
        record["status"] = "완료"
    except Exception as exc:
        record["status"] = f"오류: {type(exc).__name__}: {exc}"
    record["seconds"] = round(time.time() - started, 1)
    results.append(record)
    previous[record["image"]] = record
    # 장마다 저장한다. 멈추거나 죽어도 앞의 결과는 남는다.
    save_results()
    warn = f" / 경고 {len(record.get('warnings', []))}건" if record.get("warnings") else ""
    print(f"[{index}/{len(images)}] {image_path.name}: {record['status']}{warn} ({record['seconds']}s)", flush=True)

merged = save_results()

# 육안 검토용 대조 시트: 한 장에 5쌍(원본|결과)
ROW_H, PER_SHEET = 460, 5
done = [r for r in merged if r.get("output") and (OUT_DIR / r["output"]).is_file()]
for sheet_index in range(0, len(done), PER_SHEET):
    chunk = done[sheet_index:sheet_index + PER_SHEET]
    rows = []
    for record in chunk:
        source = next(p for p in all_images if p.name == record["image"])
        original = Image.open(source).convert("RGB")
        result = Image.open(OUT_DIR / record["output"]).convert("RGB")
        pair = []
        for img in (original, result):
            width = max(1, int(img.width * ROW_H / img.height))
            pair.append(img.resize((width, ROW_H)))
        row = Image.new("RGB", (sum(p.width for p in pair) + 8, ROW_H), "white")
        row.paste(pair[0], (0, 0))
        row.paste(pair[1], (pair[0].width + 8, 0))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 8 * len(rows)), "white")
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height + 8
    number = sheet_index // PER_SHEET + 1
    sheet.save(SHEET_DIR / f"sheet_{number:02d}.jpg", quality=88)
    print(f"시트 {number}: {', '.join(r['image'] for r in chunk)}", flush=True)

ok = sum(1 for r in results if r["status"] == "완료")
print(f"\n완료 {ok}/{len(results)}장, 총 {time.time() - start_all:.0f}s")
print("결과:", OUT_DIR)
pose_analyzer.close()
