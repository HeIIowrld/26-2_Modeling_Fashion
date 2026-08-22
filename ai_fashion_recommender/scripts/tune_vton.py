"""CatVTON 추론 파라미터를 같은 사진·같은 시드로 A/B 비교한다.

인수인계 문서의 "튜닝 전에 검토할 만한 대안 1번"을 실행하는 도구다.
가중치는 건드리지 않고 `num_inference_steps` / `guidance_scale` /
`pipeline_mask_blur` / `repaint_blur_divisor`만 한 번에 하나씩 바꾼다.

비싼 앞단(포즈·파서·분류기·추천)은 사람당 한 번만 돌리고 결과를 재사용한다.
디퓨전만 변종 수만큼 반복하므로, 변종을 하나 더 보는 비용은 사람 수 × 약 1분이다.

`CatVTONTryOn`은 파라미터를 호출 시점에 읽으므로 인스턴스를 하나만 만들고 속성만
바꾼다. 변종마다 새로 만들면 모델을 매번 다시 올린다.

이 환경에서는 **분리된 백그라운드 프로세스로 띄우면 모델 로드 직후 멈춘다**(3회 재현).
포그라운드로 돌리고, 한 번에 오래 못 돌리면 --variants로 나눠 실행할 것.
결과는 변종별 폴더에 쌓이고 시트는 매번 다시 그린다.

사용법:
    python scripts/tune_vton.py --list
    python scripts/tune_vton.py --images IMG_5455,IMG_5497 --variants baseline,cfg3.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from catvton_tryon import CatVTONTryOn
from clothing_parser import ClothingParser
from config import DATA_DIR, FASHION_ATTRIBUTE_HEADS_PATH, OUTPUT_DIR, PEOPLE_DIR, PROJECT_DIR
from fashion_model import FashionClassifier
from outfit_analyzer import OutfitAnalyzer
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from recommendation_engine import RecommendationEngine
from schemas import UserProfile

from test_tryon import IMAGE_SUFFIXES, _sample_images

# 한 번에 하나씩만 바꾼다. "baseline"은 그때그때의 현재 기본값이라 고정된 값이 아니다.
# 2026-08-21에 repaint_blur_divisor 기본값이 150에서 300으로 바뀌었으므로,
# 옛 기본값과 비교하려면 repaint150을 쓴다.
VARIANTS: dict[str, dict[str, object]] = {
    "baseline": {},
    "cfg1.5": {"guidance_scale": 1.5},
    "cfg3.5": {"guidance_scale": 3.5},
    "cfg5.0": {"guidance_scale": 5.0},
    "steps25": {"num_inference_steps": 25},
    "steps75": {"num_inference_steps": 75},
    "blur3": {"pipeline_mask_blur": 3},
    "blur17": {"pipeline_mask_blur": 17},
    "repaint75": {"repaint_blur_divisor": 75},
    "repaint150": {"repaint_blur_divisor": 150},   # 2026-08-21 이전 기본값
    "repaint300": {"repaint_blur_divisor": 300},
    # 스케줄러 교체(2026-08-22): 현재 DDIM은 eta=1.0(사실상 DDPM급 확률 샘플링)이라
    # 저스텝 열화가 크다. 결정론 2차 솔버는 20~30스텝에서 50스텝급 수렴을 노린다.
    "eta0_steps25": {"eta": 0.0, "num_inference_steps": 25},
    "dpm25": {"scheduler": "dpmpp_2m_karras", "num_inference_steps": 25},
    "dpm30": {"scheduler": "dpmpp_2m_karras", "num_inference_steps": 30},
    "unipc25": {"scheduler": "unipc", "num_inference_steps": 25},
    # 보호 영역 끝단 강제(가방·모자·손 파괴 대응)
    "acc_restore": {"protect_restore": True},
    "acc_full": {"protect_restore": True, "pipeline_recarve": True},
    # 스커트 시스루 선택적 guidance 하향
    "skirtgs1.5": {"skirt_guidance_scale": 1.5},
    "skirtgs1.75": {"skirt_guidance_scale": 1.75},
    "skirtgs2.0": {"skirt_guidance_scale": 2.0},
    # 아우터 마스크 수술(힙 아래 오버행을 하의 패스로 재배정)
    "outer_reassign": {"outerwear_policy": "reassign"},
    # 2026-08-22에 reassign/protect_restore/skirtgs1.5가 기본값으로 승격됨.
    # 아래는 그 이전 동작과 비교하기 위한 오프스위치.
    "outer_warn": {"outerwear_policy": "warn"},
    "no_protect": {"protect_restore": False},
    "no_skirtgs": {"skirt_guidance_scale": None},
}

# VARIANTS에 새 인스턴스 속성을 쓰면 반드시 여기에도 추가할 것 — 없으면 기본값
# 스냅샷·복원에서 빠져 값이 다음 변종으로 새고 records의 params에도 안 남는다.
TUNABLE = (
    "num_inference_steps", "guidance_scale", "pipeline_mask_blur", "repaint_blur_divisor",
    "scheduler", "eta", "outerwear_policy", "protect_restore", "pipeline_recarve",
    "skirt_guidance_scale",
)
GENDER_LABEL = {"men": "남성", "women": "여성"}
CONTEXT_KEYS = ("upper_mask", "lower_mask", "upper_style_mask", "lower_style_mask", "segmentation")
ROW_HEIGHT = 520
LABEL_BAND = 22


def resolve_images(stems: str, per_gender: int) -> list[Path]:
    """--images가 있으면 이름으로 찾고, 없으면 성별별로 고르게 뽑는다."""
    if not stems:
        return _sample_images("all", per_gender)
    available = {
        path.stem: path
        for folder in ("men", "women")
        for path in (PEOPLE_DIR / folder).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    chosen = []
    for stem in (item.strip() for item in stems.split(",") if item.strip()):
        if stem not in available:
            raise SystemExit(f"사진을 찾지 못했습니다: {stem}")
        chosen.append(available[stem])
    return chosen


def build_sheet(source: Path, columns: list[tuple[str, Path]], destination: Path) -> None:
    """원본과 변종 결과를 한 줄로 붙인다. 이름표가 없으면 어느 쪽인지 알 수 없다."""
    tiles = []
    for label, path in [("original", source), *columns]:
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGB")
        width = max(1, int(image.width * ROW_HEIGHT / image.height))
        tile = Image.new("RGB", (width, ROW_HEIGHT + LABEL_BAND), "white")
        tile.paste(image.resize((width, ROW_HEIGHT)), (0, LABEL_BAND))
        ImageDraw.Draw(tile).text((4, 6), label, fill="black")
        tiles.append(tile)
    if not tiles:
        return
    sheet = Image.new(
        "RGB", (sum(t.width for t in tiles) + 6 * len(tiles), ROW_HEIGHT + LABEL_BAND), "white"
    )
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (offset, 0))
        offset += tile.width + 6
    sheet.save(destination, quality=90)


def main() -> None:
    arguments = argparse.ArgumentParser(description="CatVTON 추론 파라미터 A/B")
    arguments.add_argument("--variants", default="baseline", help="쉼표로 구분한 변종 이름")
    arguments.add_argument("--images", default="", help="쉼표로 구분한 사진 이름(확장자 제외)")
    arguments.add_argument("--per-gender", type=int, default=2, help="--images가 없을 때 성별별 인원")
    arguments.add_argument("--out", default="vton_tuning", help="outputs/ 아래 저장 폴더명")
    arguments.add_argument("--list", action="store_true", help="변종 목록만 출력")
    arguments.add_argument(
        "--skip-existing", action="store_true",
        help="결과 파일이 있으면 건너뛴다. 나눠 실행할 때 앞 구간을 다시 안 돌린다.",
    )
    options = arguments.parse_args()

    if options.list:
        for name, changes in VARIANTS.items():
            print(f"{name:12} {changes or '기본값'}")
        return

    names = [name.strip() for name in options.variants.split(",") if name.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"모르는 변종: {unknown}. --list로 확인하세요.")

    out_dir = OUTPUT_DIR / options.out
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.json"
    records = (
        json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else []
    )

    pose_analyzer = PoseAnalyzer()
    parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(
        enabled=True,
        attribute_checkpoint=(
            FASHION_ATTRIBUTE_HEADS_PATH if FASHION_ATTRIBUTE_HEADS_PATH.is_file() else None
        ),
    )
    analyzer = OutfitAnalyzer(parser, classifier)
    engine = RecommendationEngine(
        PROJECT_DIR / "FASHION_RULES_MASTER.md",
        ProductCatalog(DATA_DIR / "products_musinsa.csv"),
    )
    tryon = CatVTONTryOn(max_retries=0)
    tryon._garment_parser = parser

    # 앞단은 사람당 한 번만. 변종을 늘려도 여기 비용은 늘지 않는다.
    prepared = []
    for image_path in resolve_images(options.images, options.per_gender):
        gender = GENDER_LABEL.get(image_path.parent.name, "")
        pose = pose_analyzer.analyze(image_path)
        if not pose.valid:
            print(f"건너뜀(포즈 실패): {image_path.name}", flush=True)
            continue
        outfit, parsed = analyzer.analyze(image_path, pose)
        recommendation = engine.recommend(UserProfile(gender=gender), pose, outfit, top_k=1)[0]
        prepared.append({
            "image": image_path,
            "gender": gender,
            "recommendation": recommendation,
            "products": [product.product_id for product in recommendation.products],
            "context": {
                **{key: parsed.get(key) for key in CONTEXT_KEYS},
                "outfit": outfit,
                "classifier": classifier,
                "pose": pose,  # outerwear_policy="reassign" 마스크 수술에 필요
            },
        })
        print(f"준비: {image_path.name} → {'+'.join(prepared[-1]['products'])}", flush=True)

    defaults = {key: getattr(tryon, key) for key in TUNABLE}
    for name in names:
        for key, value in defaults.items():
            setattr(tryon, key, value)
        for key, value in VARIANTS[name].items():
            setattr(tryon, key, value)
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for person in prepared:
            output_path = variant_dir / f"{person['image'].stem}.jpg"
            if options.skip_existing and output_path.is_file():
                print(f"[{name}] {person['image'].name}: 건너뜀(결과 있음)", flush=True)
                continue
            record = {
                "variant": name,
                "image": person["image"].name,
                "gender": person["gender"],
                "products": person["products"],
                "params": {key: getattr(tryon, key) for key in TUNABLE},
            }
            started = time.time()
            try:
                tryon.generate(
                    person["image"], person["recommendation"], output_path,
                    context=person["context"],
                )
                record["warnings"] = list(tryon.last_warnings)
                record["output"] = f"{name}/{person['image'].stem}.jpg"
                record["status"] = "완료"
            except Exception as exc:  # 한 장이 죽어도 나머지 변종은 계속 돌린다
                record["status"] = f"오류: {type(exc).__name__}: {exc}"
            record["seconds"] = round(time.time() - started, 1)
            records = [r for r in records
                       if not (r["variant"] == name and r["image"] == record["image"])]
            records.append(record)
            records_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{name}] {person['image'].name}: {record['status']} ({record['seconds']}s)",
                  flush=True)

    # 시트는 지금까지 쌓인 모든 변종을 담는다. 나눠 실행해도 한 장에서 비교된다.
    done_variants = [name for name in VARIANTS if (out_dir / name).is_dir()]
    for person in prepared:
        build_sheet(
            person["image"],
            [(name, out_dir / name / f"{person['image'].stem}.jpg") for name in done_variants],
            out_dir / "sheets" / f"{person['image'].stem}.jpg",
        )
    print(f"\n대조 시트: {out_dir / 'sheets'} (변종 {done_variants})", flush=True)
    pose_analyzer.close()


if __name__ == "__main__":
    main()
