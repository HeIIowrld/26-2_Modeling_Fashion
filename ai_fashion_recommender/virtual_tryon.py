from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import FONT_PATH
from schemas import Recommendation


def _load_korean_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """사용자 지정 글꼴을 우선하고 Windows/macOS/Linux의 대표 한글 글꼴을 찾는다."""
    windows_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    candidates = [
        Path(FONT_PATH).expanduser() if FONT_PATH else None,
        Path(windows_dir) / "Fonts" / "malgun.ttf" if windows_dir else None,
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


class VirtualTryOnAdapter:
    """VTON 구현을 갈아 끼울 수 있는 공통 인터페이스."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def generate(
        self,
        person_image: str | Path,
        recommendation: Recommendation,
        output_path: str | Path,
        context: dict | None = None,
    ) -> Path:
        """context에는 구현체가 사용할 부가 정보(FASHN 마스크 등)를 담는다."""
        if self.enabled:
            raise NotImplementedError(
                "VTON 구현체(catvton_tryon.CatVTONTryOn 등)를 사용해야 합니다."
            )
        return self._make_preview(person_image, recommendation, output_path)

    @staticmethod
    def _make_preview(person_image: str | Path, recommendation: Recommendation, output_path: str | Path) -> Path:
        """실제 합성으로 오해하지 않도록 명확히 표시한 추천 보드만 만든다."""
        person = Image.open(person_image).convert("RGB")
        person.thumbnail((600, 900))
        board = Image.new("RGB", (person.width + 440, max(person.height, 500)), "white")
        board.paste(person, (0, 0))
        draw = ImageDraw.Draw(board)
        font = _load_korean_font(18)
        x = person.width + 24
        draw.text((x, 24), "VTON 미연결 - 추천 보드", fill=(180, 30, 30), font=font)
        draw.text((x, 60), f"추천 순위 #{recommendation.rank}", fill="black", font=font)
        draw.text((x, 90), f"추천 점수: {recommendation.total_score:.1f}", fill="black", font=font)
        y = 135
        for product in recommendation.products:
            draw.rectangle((x, y, x + 36, y + 36), fill=_preview_color(product.color), outline="black")
            draw.text((x + 48, y + 3), product.name, fill="black", font=font)
            draw.text((x + 48, y + 22), f"{product.price:,}원", fill="black", font=font)
            y += 58
        draw.text((x, y + 15), "이 이미지는 실제 가상 피팅 결과가 아닙니다.", fill=(90, 90, 90), font=font)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        board.save(output, quality=95)
        return output


def _preview_color(name: str) -> tuple[int, int, int]:
    return {
        "블랙": (25, 25, 25), "화이트": (235, 235, 235), "그레이": (130, 130, 130),
        "네이비": (35, 50, 90), "블루": (55, 110, 190), "브라운": (115, 75, 45),
        "베이지": (205, 185, 145), "레드": (185, 45, 45), "핑크": (220, 125, 155),
        "그린": (60, 130, 75), "옐로": (220, 185, 50), "퍼플": (115, 70, 145),
    }.get(name, (160, 160, 160))
