"""무신사 색 옵션 이름을 우리 팔레트로 바꾼다.

무신사 색 이름은 판매자가 자유롭게 쓴다: `(19)BLACK`, `코튼아이보리`, `MELANGE GRAY (기모)`.
같은 표(`catalog_derivation.json` 의 `musinsa_color`)를 카탈로그 보강과 실시간 검색이 함께 써야
"블랙을 원함"과 "블랙 옵션 보유"가 같은 기준으로 비교된다.

상품은 보통 여러 색으로 팔린다. 대표 색 하나만 저장하면 나머지 색을 원하는 사용자가
그 상품을 영영 만나지 못하고, 싫어하는 색을 걸러 달라는 요청도 헛돈다.
"""
from __future__ import annotations

import json
from pathlib import Path

_TABLE: dict[str, tuple[str, ...]] | None = None


def load_palette_table(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """팔레트 → 색 이름 후보. 파일을 읽지 못하면 빈 표를 돌려준다(매칭을 건너뛴다)."""
    global _TABLE
    default = path is None
    if default:
        if _TABLE is not None:
            return _TABLE
        from config import DATA_DIR

        path = Path(DATA_DIR) / "catalog_derivation.json"
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8")).get("musinsa_color") or {}
        table = {palette: tuple(words) for palette, words in raw.items() if isinstance(words, list)}
    except (OSError, ValueError, TypeError, AttributeError):
        table = {}
    if default:
        _TABLE = table
    return table


def palette_of(name: str, table: dict[str, tuple[str, ...]] | None = None) -> str:
    """색 이름 하나를 팔레트로. 어느 후보와도 닿지 않으면 빈 문자열."""
    table = load_palette_table() if table is None else table
    lowered = (name or "").lower()
    if not lowered:
        return ""
    # 가장 긴 후보부터 본다. '다크브라운'이 '브라운'보다 먼저 걸려야 하는 표에 대비한다.
    best, best_length = "", 0
    for palette, words in table.items():
        for word in words:
            if word and word.lower() in lowered and len(word) > best_length:
                best, best_length = palette, len(word)
    return best


def palettes_for(names, table: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """색 이름 목록을 팔레트 목록으로. 순서를 지키고 중복과 미분류는 뺀다."""
    table = load_palette_table() if table is None else table
    found: list[str] = []
    for name in names or ():
        palette = palette_of(name, table)
        if palette and palette not in found:
            found.append(palette)
    return found
