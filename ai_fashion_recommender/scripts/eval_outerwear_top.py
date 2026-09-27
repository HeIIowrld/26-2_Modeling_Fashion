#!/usr/bin/env python3
"""Run one isolated outerwear-removal/top-reference experiment on a GPU.

Example:
    python scripts/eval_outerwear_top.py \
      --person /data/eval/person_in_fur_coat.jpg \
      --top-image /data/eval/reference_top.jpg \
      --model /data/models/FLUX.2-klein-4B \
      --out outputs/outerwear_top/fur_01.png

The command writes the result, a JSON record beside it, and four masks plus an
overlay under ``<output parent>/debug``.  It never changes the production VTON
adapter or cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from catvton_tryon import CatVTONTryOn  # noqa: E402
from clothing_parser import ClothingParser  # noqa: E402
from config import FASHION_ATTRIBUTE_HEADS_PATH, LAYERING_HEADS_PATH  # noqa: E402
from fashion_model import FashionClassifier  # noqa: E402
from outerwear_top_tryon import (  # noqa: E402
    OUTERWEAR_TOP_PROMPT,
    OUTERWEAR_TOP_PROMPT_STRICT,
    OuterwearTopTryOn,
)
from outfit_analyzer import OutfitAnalyzer  # noqa: E402
from pose_analyzer import PoseAnalyzer  # noqa: E402
from schemas import Product, Recommendation  # noqa: E402
from shoe_tryon import ShoeTryOn  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="외투를 지우고 실제 상품 상의로 바꾸는 격리 FLUX 실험"
    )
    parser.add_argument("--person", type=Path, required=True, help="외투를 입은 정면 인물 사진")
    parser.add_argument("--top-image", type=Path, required=True, help="입힐 상의 상품 이미지")
    parser.add_argument(
        "--model", type=Path,
        default=Path(os.environ["FASHION_SHOE_MODEL_PATH"])
        if os.environ.get("FASHION_SHOE_MODEL_PATH") else None,
        help="로컬 FLUX.2-klein-4B 체크포인트; FASHION_SHOE_MODEL_PATH도 사용 가능",
    )
    parser.add_argument("--out", type=Path, required=True, help="결과 PNG/JPG 경로")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument(
        "--official-segmentation", type=Path, default=None,
        help="DeepFashion-MultiModal 공식 24-class PNG; 상의/외투 정답 마스크의 상한선 실험",
    )
    parser.add_argument(
        "--oracle-outer-only", action="store_true",
        help="공식 외투 라벨만 지우고 보이는 기존 이너는 보존하는 원인 분리 실험",
    )
    parser.add_argument(
        "--oracle-inner-reference", action="store_true",
        help="외투만 제거할 때 공식 이너 조각을 FLUX 참조 이미지로 전달하는 비교 실험",
    )
    parser.add_argument("--top-name", default="외투 제거 실험 상의")
    parser.add_argument("--top-fit", default="레귤러핏", help="슬림핏/레귤러핏/여유핏/오버핏")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument(
        "--fit-guidance", action="store_true",
        help="상품 핏을 FLUX 문구에도 직접 전달하는 비교 실험",
    )
    parser.add_argument(
        "--restore-studio-background", action="store_true",
        help="균일한 스튜디오 배경에서 편집 영역의 배경 분류 픽셀을 복원하는 비교 실험",
    )
    parser.add_argument(
        "--release-pocket-hands", action="store_true",
        help="주머니 근처 손을 보호 마스크에서 풀어 재생성하는 비교 실험",
    )
    parser.add_argument(
        "--coat-proposal-mask", type=Path,
        help="별도 파서의 외투 이진 PNG; FASHN 머리카락 보호 경계의 외투 픽셀만 편집하는 실험",
    )
    parser.add_argument(
        "--accessory-mask", type=Path,
        help="사람이 지정한 외투 끈·부속품 이진 PNG를 지우기 영역에 추가하는 상한선 실험",
    )
    parser.add_argument(
        "--preserve-observed-pants", action="store_true",
        help="FASHN/공식 분할의 바지 픽셀을 지우기 영역에서 제외하는 비교 실험",
    )
    parser.add_argument(
        "--prompt-variant", choices=("baseline", "strict"), default="baseline",
        help="마스크를 고정한 채 이너/잔여물 제거 문구만 바꾸는 비교 실험",
    )
    parser.add_argument(
        "--second-pass-catvton", action="store_true",
        help="넓은 FLUX 제거 뒤 좁은 목표 마스크로 CatVTON 상품 충실도를 보강",
    )
    parser.add_argument(
        "--second-pass-flux", action="store_true",
        help="넓은 FLUX 외투 제거 뒤 좁은 목표 마스크로 FLUX 상의를 재생성",
    )
    parser.add_argument(
        "--second-stage-mask", type=Path,
        help="2차 FLUX에 생성된 중립 옷 영역을 추가할 실험용 단일 채널 마스크",
    )
    parser.add_argument(
        "--coatless-shape-reference", type=Path,
        help="2단계 생성의 1차 외투 제거용 코트 없는 전신 실루엣 참조 이미지",
    )
    parser.add_argument(
        "--bilateral-sleeves", action="store_true",
        help="2차 FLUX에 상품의 양쪽 소매 길이·소재 일치 지시를 추가하는 문구 대조",
    )
    parser.add_argument(
        "--allow-unconfirmed-outerwear", action="store_true",
        help="앞단이 외투를 판정하지 못해도 강제로 실행(마스크 검토용)",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} 파일을 찾지 못했습니다: {resolved}")
    return resolved


def main() -> None:
    options = arguments()
    if options.second_pass_catvton and options.second_pass_flux:
        raise SystemExit("2차 패스는 CatVTON 또는 FLUX 중 하나만 선택할 수 있습니다.")
    if options.second_pass_catvton and (
        options.prompt_variant != "baseline" or options.fit_guidance
    ):
        raise SystemExit("문구/핏 가이드 비교는 단일 FLUX 패스에서만 지원합니다.")
    if options.second_pass_flux and options.prompt_variant != "baseline":
        raise SystemExit("2단계 FLUX는 별도 제거·상의 생성 문구를 사용하므로 strict 문구와 병행할 수 없습니다.")
    if options.second_stage_mask and not options.second_pass_flux:
        raise SystemExit("--second-stage-mask에는 --second-pass-flux가 필요합니다.")
    if options.coatless_shape_reference and not (
        options.second_pass_flux or options.second_pass_catvton
    ):
        raise SystemExit("--coatless-shape-reference에는 2단계 생성이 필요합니다.")
    if options.bilateral_sleeves and not options.second_pass_flux:
        raise SystemExit("--bilateral-sleeves에는 --second-pass-flux가 필요합니다.")
    if options.oracle_outer_only and (
        options.second_pass_flux or options.second_pass_catvton
        or options.prompt_variant != "baseline" or options.fit_guidance
        or options.release_pocket_hands or options.coat_proposal_mask
    ):
        raise SystemExit("외투만 제거하는 실험은 2차 생성·손/머리 경계 해제·상품 핏/문구 비교와 병행할 수 없습니다.")
    person_path = require_file(options.person, "인물")
    top_path = require_file(options.top_image, "상의 상품")
    official_path = (
        require_file(options.official_segmentation, "공식 분할")
        if options.official_segmentation else None
    )
    second_stage_path = (
        require_file(options.second_stage_mask, "2차 상의 마스크")
        if options.second_stage_mask else None
    )
    shape_reference_path = (
        require_file(options.coatless_shape_reference, "외투 없는 실루엣 참조")
        if options.coatless_shape_reference else None
    )
    coat_proposal_path = (
        require_file(options.coat_proposal_mask, "외투 제안 마스크")
        if options.coat_proposal_mask else None
    )
    accessory_path = (
        require_file(options.accessory_mask, "외투 부속품 마스크")
        if options.accessory_mask else None
    )
    if options.oracle_outer_only and official_path is None:
        raise SystemExit("--oracle-outer-only에는 --official-segmentation이 필요합니다.")
    if options.oracle_inner_reference and not options.oracle_outer_only:
        raise SystemExit("--oracle-inner-reference에는 --oracle-outer-only가 필요합니다.")
    if options.model is None:
        raise SystemExit("--model 또는 FASHION_SHOE_MODEL_PATH를 지정해야 합니다.")
    model_path = options.model.expanduser().resolve()
    output_path = options.out.expanduser().resolve()
    if output_path in {person_path, top_path, official_path, second_stage_path,
                       coat_proposal_path, accessory_path, shape_reference_path}:
        raise SystemExit("--out은 입력 인물/상품/분할 이미지와 다른 경로여야 합니다.")
    debug_dir = (
        options.debug_dir.expanduser().resolve()
        if options.debug_dir else output_path.parent / "debug"
    )

    pose_analyzer = PoseAnalyzer(model_complexity=1)
    try:
        parser = ClothingParser(use_fashn=True)
        classifier_enabled = not options.allow_unconfirmed_outerwear
        classifier = FashionClassifier(
            enabled=classifier_enabled,
            # Leave GPU memory to FASHN, FLUX and (optionally) CatVTON.  This
            # classifier is used only once to gate and record the source outfit.
            device="cpu",
            attribute_checkpoint=(
                FASHION_ATTRIBUTE_HEADS_PATH if FASHION_ATTRIBUTE_HEADS_PATH.is_file() else None
            ),
            layering_checkpoint=(LAYERING_HEADS_PATH if LAYERING_HEADS_PATH.is_file() else None),
        )
        analyzer = OutfitAnalyzer(parser, classifier)
        pose = pose_analyzer.analyze(person_path)
        if not pose.landmarks:
            raise SystemExit("상체 포즈를 찾지 못했습니다.")
        outfit, parsed = analyzer.analyze(person_path, pose)

        editor = ShoeTryOn(model_path)
        if not editor.available:
            raise SystemExit(f"완전한 FLUX.2-klein-4B 체크포인트가 없습니다: {model_path}")
        # This dedicated instance must keep the pose/fit-derived second-pass mask
        # as-is.  CatVTON's production `agnostic` policy deliberately expands a
        # top mask and would undo the experiment's narrow-silhouette constraint.
        clothing = CatVTONTryOn(max_retries=0, upper_mask_policy="native")
        clothing._garment_parser = parser
        adapter = OuterwearTopTryOn(
            clothing,
            editor,
            parser,
            debug_dir=debug_dir,
            require_outerwear=not options.allow_unconfirmed_outerwear,
            num_inference_steps=options.steps,
            guidance_scale=options.guidance,
            second_pass_catvton=options.second_pass_catvton,
            second_pass_flux=options.second_pass_flux,
            fit_guidance=options.fit_guidance,
            restore_studio=options.restore_studio_background,
            release_pocket_hands=options.release_pocket_hands,
            preserve_observed_pants=options.preserve_observed_pants,
            oracle_outer_only=options.oracle_outer_only,
            oracle_inner_reference=options.oracle_inner_reference,
            bilateral_sleeves=options.bilateral_sleeves,
            prompt=(
                OUTERWEAR_TOP_PROMPT_STRICT
                if options.prompt_variant == "strict" else OUTERWEAR_TOP_PROMPT
            ),
        )
        product = Product(
            product_id="OUTERWEAR_TOP_EXPERIMENT",
            name=options.top_name,
            category="top",
            color="",
            style="",
            purposes=[],
            body_shapes=[],
            price=0,
            season="사계절",
            stock=True,
            image_path=str(top_path),
            fit=options.top_fit,
        )
        recommendation = Recommendation(
            rank=1,
            products=[product],
            total_score=0.0,
            score_breakdown={},
            reasons=["격리 외투 제거 실험"],
        )
        context = {
            **{
                key: parsed.get(key)
                for key in (
                    "upper_mask", "lower_mask", "upper_style_mask", "lower_style_mask",
                    "segmentation",
                )
            },
            "outfit": outfit,
            "classifier": classifier,
            "pose": pose,
        }
        if official_path is not None:
            with Image.open(official_path) as opened:
                context["official_segmentation"] = np.asarray(opened)
        if second_stage_path is not None:
            with Image.open(second_stage_path) as opened:
                if opened.mode not in ("1", "L"):
                    raise SystemExit("2차 상의 마스크는 단일 채널 이미지여야 합니다.")
                context["second_stage_mask"] = np.asarray(opened.convert("L")) > 0
        if shape_reference_path is not None:
            with Image.open(shape_reference_path) as opened:
                context["coatless_shape_reference"] = opened.convert("RGB")
        if coat_proposal_path is not None:
            with Image.open(coat_proposal_path) as opened:
                if opened.mode not in ("1", "L"):
                    raise SystemExit("외투 제안 마스크는 단일 채널 이미지여야 합니다.")
                context["coat_proposal_mask"] = np.asarray(opened.convert("L")) > 0
        if accessory_path is not None:
            with Image.open(accessory_path) as opened:
                if opened.mode not in ("1", "L"):
                    raise SystemExit("외투 부속품 마스크는 단일 채널 이미지여야 합니다.")
                context["accessory_mask"] = np.asarray(opened.convert("L")) > 0
        started = time.time()
        result = adapter.generate(person_path, recommendation, output_path, context)
        record = {
            "status": "completed",
            "person": str(person_path),
            "top_image": str(top_path),
            "output": str(result),
            "debug_dir": str(debug_dir),
            "seconds": round(time.time() - started, 2),
            "parameters": {
                "steps": options.steps,
                "guidance": options.guidance,
                "second_pass_catvton": options.second_pass_catvton,
                "second_pass_flux": options.second_pass_flux,
                "second_stage_mask": str(second_stage_path) if second_stage_path else None,
                "coatless_shape_reference": (
                    str(shape_reference_path) if shape_reference_path else None
                ),
                "coat_proposal_mask": str(coat_proposal_path) if coat_proposal_path else None,
                "accessory_mask": str(accessory_path) if accessory_path else None,
                "bilateral_sleeves": options.bilateral_sleeves,
                "top_fit": options.top_fit,
                "require_outerwear": not options.allow_unconfirmed_outerwear,
                "source_classifier_enabled": classifier_enabled,
                "prompt_variant": options.prompt_variant,
                "fit_guidance": options.fit_guidance,
                "restore_studio_background": options.restore_studio_background,
                "release_pocket_hands": options.release_pocket_hands,
                "preserve_observed_pants": options.preserve_observed_pants,
                "official_segmentation": str(official_path) if official_path else None,
                "oracle_outer_only": options.oracle_outer_only,
                "oracle_inner_reference": options.oracle_inner_reference,
            },
            "detected_outfit": {
                key: getattr(outfit, key, "")
                for key in (
                    "upper_type", "material", "fit", "layering_state", "inner_category",
                    "outer_category",
                )
            },
            "warnings": adapter.last_warnings,
            "quality": adapter.last_quality_reports,
        }
        report_path = output_path.with_suffix(".json")
        report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
    finally:
        pose_analyzer.close()


if __name__ == "__main__":
    main()
