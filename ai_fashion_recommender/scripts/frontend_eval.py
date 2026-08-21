"""VTON 없이 앞단(체형 분류·추천)만 배치로 재측정한다.

`batch_eval.py`는 장당 50초짜리 디퓨전을 돌리므로 앞단 지표를 보려고 쓰기엔 비싸다.
이 스크립트는 합성을 건너뛰고 포즈·인식·추천까지만 돌려 다음 두 가지를 센다.

  1. 체형 분류 분포 — 2026-08-21 배치에서 29명 전원이 한 값으로 뭉쳤던 항목.
     그 뒤 `pose_analyzer`가 백분위 경계로 재설계됐으므로 다시 재야 한다.
  2. 추천 조합 다양성 — 성별당 한 조합으로 붕괴했던 항목.

사용법:
    python scripts/frontend_eval.py --pose-only            # 전원 포즈만 (빠름)
    python scripts/frontend_eval.py --count 8              # 성별당 8장 전 구간
    python scripts/frontend_eval.py --goal-sweep           # 목표별 추천 다양성 (파서 불필요)

주의: 분리된 백그라운드 프로세스로 돌리면 모델 로드 직후 멈추는 사례가 있었다.
포그라운드로 실행할 것.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from config import DATA_DIR, FASHION_ATTRIBUTE_HEADS_PATH, OUTPUT_DIR, PEOPLE_DIR, PROJECT_DIR
from pose_analyzer import PoseAnalyzer
from schemas import UserProfile

from test_tryon import IMAGE_SUFFIXES, _sample_images

GENDER_LABEL = {"men": "남성", "women": "여성"}


def all_images() -> list[Path]:
    found: list[Path] = []
    for folder in ("men", "women"):
        directory = PEOPLE_DIR / folder
        if directory.exists():
            found.extend(sorted(
                path for path in directory.iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            ))
    return found


def summarise(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p25": round(ordered[len(ordered) // 4], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[len(ordered) * 3 // 4], 4),
        "max": round(ordered[-1], 4),
    }


# change_scope="전체 변경"이면 두 슬롯이 모두 상품으로 채워져 `_garment()`가 현재 착장을
# 읽지 않고, `pose`는 `_silhouette_score`의 GOAL_BALANCE 분기에서만 읽힌다. 그래서
# 목표별 다양성은 파서·분류기 없이 포즈만으로 정확히 잴 수 있다.
NEUTRAL_OUTFIT_FIELDS = dict(
    parser_backend="probe", upper_color="", lower_color="",
    color_harmony="보통 조합", detected_items=[], style="캐주얼",
)
PER_SHAPE = 3


def goal_sweep(out_dir: Path, per_shape: int = PER_SHAPE) -> None:
    """체형별로 몇 명씩 뽑아 silhouette_goal만 바꿔가며 top-1이 갈리는지 센다."""
    from collections import defaultdict

    from product_catalog import ProductCatalog
    from recommendation_engine import RecommendationEngine
    from schemas import GOAL_BALANCE, GOAL_NONE, OutfitAnalysis

    source = OUTPUT_DIR / "frontend_eval_pose_only" / "records.json"
    if not source.is_file():
        raise SystemExit(
            f"{source}가 없습니다. 먼저 --pose-only --out frontend_eval_pose_only 를 돌리세요."
        )
    by_shape = defaultdict(list)
    for record in json.loads(source.read_text(encoding="utf-8")):
        if record.get("valid"):
            by_shape[(record["gender"], record["body_shape"])].append(record["image"])

    by_name = {
        path.name: path
        for folder in ("men", "women")
        for path in (PEOPLE_DIR / folder).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    pose_analyzer = PoseAnalyzer()
    engine = RecommendationEngine(
        PROJECT_DIR / "FASHION_RULES_MASTER.md",
        ProductCatalog(DATA_DIR / "products_musinsa.csv"),
    )
    neutral = OutfitAnalysis(**NEUTRAL_OUTFIT_FIELDS)

    rows = {}
    for (gender, shape), names in sorted(by_shape.items()):
        for name in names[:per_shape]:
            pose = pose_analyzer.analyze(by_name[name])
            row = {"confidence": pose.body_shape_confidence, "ratio": pose.shoulder_hip_ratio}
            for goal in (GOAL_NONE, GOAL_BALANCE):
                best = engine.recommend(
                    UserProfile(gender=gender, silhouette_goal=goal), pose, neutral, top_k=1
                )[0]
                row[goal] = "+".join(product.product_id for product in best.products)
            rows[f"{gender}/{shape}/{name}"] = row
            print(f"{gender} {shape} {name} "
                  f"none={row[GOAL_NONE]} balance={row[GOAL_BALANCE]}", flush=True)

    distinct = {
        goal: {
            gender: len({row[goal] for key, row in rows.items() if key.startswith(gender)})
            for gender in ("남성", "여성")
        }
        for goal in (GOAL_NONE, GOAL_BALANCE)
    }
    (out_dir / "goal_sweep.json").write_text(
        json.dumps({"rows": rows, "distinct_top1": distinct}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"distinct_top1": distinct}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="앞단(체형·추천) 배치 재측정")
    argument_parser.add_argument("--pose-only", action="store_true", help="파서·분류기 없이 포즈만")
    argument_parser.add_argument("--count", type=int, default=15, help="성별별 장수 (--pose-only면 전원)")
    argument_parser.add_argument("--top-k", type=int, default=3, help="추천 조합 개수")
    argument_parser.add_argument("--out", default="frontend_eval", help="outputs/ 아래 저장 폴더명")
    argument_parser.add_argument(
        "--goal-sweep", action="store_true",
        help="silhouette_goal만 바꿔 추천 다양성을 잰다 (--pose-only 결과가 있어야 한다)",
    )
    argument_parser.add_argument(
        "--per-shape", type=int, default=3, help="--goal-sweep에서 체형별로 볼 인원"
    )
    options = argument_parser.parse_args()

    out_dir = OUTPUT_DIR / options.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if options.goal_sweep:
        goal_sweep(out_dir, options.per_shape)
        return

    pose_analyzer = PoseAnalyzer()
    images = all_images() if options.pose_only else _sample_images("all", options.count)
    print(f"대상 {len(images)}장", flush=True)

    analyzer = engine = classifier = None
    if not options.pose_only:
        from clothing_parser import ClothingParser
        from fashion_model import FashionClassifier
        from outfit_analyzer import OutfitAnalyzer
        from product_catalog import ProductCatalog
        from recommendation_engine import RecommendationEngine

        clothing_parser = ClothingParser(use_fashn=True)
        classifier = FashionClassifier(
            enabled=True,
            attribute_checkpoint=(
                FASHION_ATTRIBUTE_HEADS_PATH if FASHION_ATTRIBUTE_HEADS_PATH.is_file() else None
            ),
        )
        analyzer = OutfitAnalyzer(clothing_parser, classifier)
        engine = RecommendationEngine(
            PROJECT_DIR / "FASHION_RULES_MASTER.md",
            ProductCatalog(DATA_DIR / "products_musinsa.csv"),
        )

    records = []
    started_all = time.time()
    for index, image_path in enumerate(images, 1):
        gender = GENDER_LABEL.get(image_path.parent.name, "")
        record = {"image": image_path.name, "gender": gender}
        started = time.time()
        try:
            pose = pose_analyzer.analyze(image_path)
            record.update(
                valid=pose.valid,
                body_shape=pose.body_shape,
                shoulder_hip_ratio=pose.shoulder_hip_ratio,
                body_shape_confidence=pose.body_shape_confidence,
                full_body_score=pose.full_body_score,
                posture=pose.posture,
            )
            if not options.pose_only and pose.valid:
                outfit, _ = analyzer.analyze(image_path, pose)
                record["recognized"] = outfit.to_summary_dict()
                recommendations = engine.recommend(
                    UserProfile(gender=gender), pose, outfit, top_k=options.top_k
                )
                record["recommendations"] = [
                    {
                        "score": round(getattr(item, "score", 0.0), 4),
                        "products": [p.product_id for p in item.products],
                    }
                    for item in recommendations
                ]
            record["status"] = "완료"
        except Exception as exc:  # 한 장이 죽어도 배치는 계속한다
            record["status"] = f"오류: {type(exc).__name__}: {exc}"
        record["seconds"] = round(time.time() - started, 2)
        records.append(record)
        if index % 10 == 0 or index == len(images) or not options.pose_only:
            print(
                f"[{index}/{len(images)}] {image_path.name}: "
                f"{record.get('body_shape', record['status'])} ({record['seconds']}s)",
                flush=True,
            )

    valid = [r for r in records if r.get("valid")]
    summary = {
        "images": len(records),
        "valid_pose": len(valid),
        "elapsed_seconds": round(time.time() - started_all, 1),
        "body_shape": Counter(r["body_shape"] for r in valid),
        "body_shape_by_gender": {
            gender: Counter(r["body_shape"] for r in valid if r["gender"] == gender)
            for gender in ("남성", "여성")
        },
        "shoulder_hip_ratio": summarise([r["shoulder_hip_ratio"] for r in valid]),
        "body_shape_confidence": summarise([r["body_shape_confidence"] for r in valid]),
    }
    recommended = [r for r in records if r.get("recommendations")]
    if recommended:
        summary["top1_combos"] = {
            gender: Counter(
                "+".join(r["recommendations"][0]["products"])
                for r in recommended if r["gender"] == gender
            )
            for gender in ("남성", "여성")
        }
        summary["distinct_products_topk"] = {
            gender: len({
                product
                for r in recommended if r["gender"] == gender
                for item in r["recommendations"] for product in item["products"]
            })
            for gender in ("남성", "여성")
        }

    (out_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict), flush=True)


if __name__ == "__main__":
    main()
