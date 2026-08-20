from __future__ import annotations

"""사진 업로드 → 체형·착장 분석 → 무신사 상품 추천 웹 앱.

main.ipynb의 파이프라인을 Gradio UI로 감싼 것이다.

사용법:
    python app.py                # 정식 모드 (FASHN + FashionSigLIP, 첫 실행 시 체크포인트 다운로드)
    python app.py --light        # 경량 모드 (무거운 모델 없이 UI·추천 흐름만 확인)
    python app.py --share        # 임시 공개 링크 생성 (팀원 테스트용)
"""

import argparse
import html
from pathlib import Path

import gradio as gr

from config import DATA_DIR, OUTPUT_DIR, PROJECT_DIR
from clothing_parser import ClothingParser
from fashion_model import FashionClassifier
from outfit_analyzer import OutfitAnalyzer
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import RecommendationEngine
from schemas import Recommendation, UserProfile
from virtual_tryon import VirtualTryOnAdapter

PURPOSES = ["데일리", "데이트", "출근", "여행"]
STYLES = ["캐주얼", "미니멀", "포멀", "스포티", "스트리트", "로맨틱"]
SCOPES = ["전체 변경", "상의만 변경", "하의만 변경"]
SEASONS = ["사계절", "봄", "여름", "가을", "겨울"]

_pipeline: dict | None = None
_light_mode = False
_use_vton = False


def catalog_path() -> Path:
    musinsa = DATA_DIR / "products_musinsa.csv"
    return musinsa if musinsa.exists() else DATA_DIR / "products.csv"


def get_pipeline() -> dict:
    """모델을 프로세스당 한 번만 로드한다."""
    global _pipeline
    if _pipeline is None:
        pose = PoseAnalyzer()
        parser = ClothingParser(use_fashn=not _light_mode)
        classifier = FashionClassifier(enabled=not _light_mode)
        if _use_vton:
            from catvton_tryon import CatVTONTryOn

            vton = CatVTONTryOn()
        else:
            vton = VirtualTryOnAdapter(enabled=False)
        _pipeline = {
            "pose": pose,
            "quality": QualityChecker(pose),
            "outfit": OutfitAnalyzer(parser, classifier),
            "engine": RecommendationEngine(DATA_DIR / "fashion_rules.json", ProductCatalog(catalog_path())),
            "vton": vton,
        }
    return _pipeline


def _product_image(product) -> str | None:
    if product.image_path:
        path = PROJECT_DIR / product.image_path
        if path.exists():
            return str(path)
    return None


def _product_line(product) -> str:
    name = html.escape(product.name)
    brand = f"{html.escape(product.brand)} · " if product.brand else ""
    link = f" — [무신사에서 보기]({product.url})" if product.url else ""
    return f"- {brand}**{name}** {product.price:,}원{link}"


def _recommendations_markdown(recommendations: list[Recommendation]) -> str:
    lines = []
    for rec in recommendations:
        lines.append(f"### {rec.rank}위 (점수 {rec.total_score:.1f})")
        lines.extend(_product_line(p) for p in rec.products)
        lines.append("")
        lines.extend(f"> {reason}" for reason in rec.reasons)
        lines.append("")
    return "\n".join(lines) if lines else "추천 결과가 없습니다."


def _analysis_markdown(quality: dict, pose, outfit) -> str:
    issue_text = "\n".join(f"- ⚠️ {issue}" for issue in quality["issues"]) or "- 문제 없음"
    return "\n".join([
        "#### 사진 품질",
        f"- 해상도 {quality['resolution'][0]}×{quality['resolution'][1]}, 선명도 {quality['sharpness']}, 전신 점수 {quality['full_body_score']:.2f}",
        issue_text,
        "",
        "#### 체형 분석 (사진 기반 참고값)",
        f"- 체형 분류: **{pose.body_shape}** / 자세: {pose.posture}",
        f"- 어깨·골반 비율 {pose.shoulder_hip_ratio:.2f}, 상·하체 비율 {pose.upper_lower_ratio:.2f}, 다리 비율 {pose.leg_ratio:.2f}",
        "",
        "#### 현재 착장",
        f"- 상의: {outfit.upper_type} ({outfit.upper_color}) / 하의: {outfit.lower_type} ({outfit.lower_color})",
        f"- 스타일 {outfit.style}, 색 조화 '{outfit.color_harmony}', 핏 {outfit.fit}",
        f"- 소매 {outfit.sleeve_length}, 패턴 {outfit.pattern}, 소재 {outfit.material}",
        "",
        *[f"_{note}_" for note in outfit.notes],
    ])


def recommend_outfits(image_path, purpose, style, budget, scope, season):
    empty = (None, [], "", "")
    if not image_path:
        return (*empty[:2], "사진을 먼저 업로드해주세요.", "")

    pipeline = get_pipeline()
    pose = pipeline["pose"].analyze(image_path)
    quality = pipeline["quality"].check_input(image_path, pose)
    if not pose.valid:
        issues = "\n".join(f"- {issue}" for issue in quality["issues"])
        return (*empty[:2], f"전신이 충분히 보이지 않아 분석을 중단했습니다.\n{issues}", "")

    outfit, parsed = pipeline["outfit"].analyze(image_path, pose)
    profile = UserProfile(
        purpose=purpose,
        desired_style=style,
        budget=int(budget),
        change_scope=scope,
        season=season,
    )
    recommendations = pipeline["engine"].recommend(profile, pose, outfit, top_k=3)
    if not recommendations:
        return (*empty[:2], _analysis_markdown(quality, pose, outfit), "조건에 맞는 상품이 없습니다. 예산을 높여보세요.")

    board_path = pipeline["vton"].generate(
        image_path,
        recommendations[0],
        OUTPUT_DIR / "web_result.jpg",
        context={"upper_mask": parsed.get("upper_mask"), "lower_mask": parsed.get("lower_mask")},
    )

    gallery = []
    for rec in recommendations:
        for product in rec.products:
            image = _product_image(product)
            if image:
                gallery.append((image, f"{rec.rank}위 · {product.name} · {product.price:,}원"))

    return (
        str(board_path),
        gallery,
        _analysis_markdown(quality, pose, outfit),
        _recommendations_markdown(recommendations),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI 패션 추천") as demo:
        gr.Markdown("# AI 코디 추천\n전신사진을 올리면 체형·착장을 분석해 무신사 상품을 추천합니다.")
        gr.Markdown(
            "_업로드한 사진은 추천 생성에만 사용하며 서버에 저장하지 않습니다. "
            "체형 값은 사진 기반 참고값으로 실제 신체 치수가 아닙니다._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="전신사진 업로드")
                purpose = gr.Dropdown(PURPOSES, value="데일리", label="코디 목적")
                style = gr.Dropdown(STYLES, value="캐주얼", label="원하는 스타일")
                budget = gr.Slider(30_000, 500_000, value=150_000, step=10_000, label="예산 (원)")
                scope = gr.Radio(SCOPES, value="전체 변경", label="변경 범위")
                season = gr.Dropdown(SEASONS, value="사계절", label="계절")
                submit = gr.Button("코디 추천 받기", variant="primary")
            with gr.Column(scale=2):
                board = gr.Image(label="추천 결과 (현재는 추천 보드, VTON 연결 예정)")
                gallery = gr.Gallery(label="추천 상품", columns=3, height=280)
                analysis_md = gr.Markdown()
                recs_md = gr.Markdown()

        submit.click(
            recommend_outfits,
            inputs=[image_input, purpose, style, budget, scope, season],
            outputs=[board, gallery, analysis_md, recs_md],
        )
    return demo


def main() -> None:
    global _light_mode, _use_vton
    parser = argparse.ArgumentParser(description="AI 패션 추천 웹 앱")
    parser.add_argument("--light", action="store_true", help="FASHN/SigLIP 없이 UI 흐름만 확인")
    parser.add_argument("--vton", action="store_true", help="CatVTON 실제 가상 피팅 사용 (GPU 권장)")
    parser.add_argument("--share", action="store_true", help="임시 공개 링크 생성")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    _light_mode = args.light
    _use_vton = args.vton
    if args.vton and args.light:
        parser.error("--vton은 FASHN 마스크가 필요해 --light와 함께 쓸 수 없습니다.")

    demo = build_app()
    # MediaPipe 그래프는 동시 호출에 안전하지 않아 요청을 한 번에 하나씩 처리한다.
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
