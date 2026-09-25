"""무신사 색 옵션 이름을 우리 팔레트로 바꾼다.

무신사 색 이름은 판매자가 자유롭게 쓴다: `(19)BLACK`, `코튼아이보리`, `MELANGE GRAY (기모)`.
같은 표(`catalog_derivation.json` 의 `musinsa_color`)를 카탈로그 보강과 실시간 검색이 함께 써야
"블랙을 원함"과 "블랙 옵션 보유"가 같은 기준으로 비교된다.

상품은 보통 여러 색으로 팔린다. 대표 색 하나만 저장하면 나머지 색을 원하는 사용자가
그 상품을 영영 만나지 못하고, 싫어하는 색을 걸러 달라는 요청도 헛돈다.
"""
from __future__ import annotations

import colorsys
import json
import re
from pathlib import Path

_TABLE: dict[str, tuple[str, ...]] | None = None
_VOCABULARY: dict | None = None
_PHOTO_RULES: dict | None = None


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


def load_vocabulary(path: Path | None = None) -> dict:
    """색 어휘 규격(비색 용어·경계 규칙). 읽지 못하면 규칙 없이 동작한다."""
    global _VOCABULARY
    default = path is None
    if default:
        if _VOCABULARY is not None:
            return _VOCABULARY
        from config import DATA_DIR

        path = Path(DATA_DIR) / "catalog_derivation.json"
    try:
        rules = json.loads(Path(path).read_text(encoding="utf-8")).get("color_vocabulary") or {}
    except (OSError, ValueError, TypeError, AttributeError):
        rules = {}
    if default:
        _VOCABULARY = rules
    return rules


def title_palettes(name: str, table: dict[str, tuple[str, ...]] | None = None,
                   vocabulary: dict | None = None) -> list[str]:
    """상품명에서 **색만** 읽는다. 색 단어를 품은 다른 말에 걸리지 않게 한다.

    부분 문자열만 보면 '블루종'(점퍼 종류)이 블루가 되고 'LAYERED'가 RED 가 된다.
    실제로 카탈로그 2224개에서 블루종 42건, 영어 어미 21건이 그렇게 잡혔다.
    규격은 data/catalog_derivation.json 의 `color_vocabulary` 에 있다.
    """
    table = load_palette_table() if table is None else table
    rules = load_vocabulary() if vocabulary is None else vocabulary
    text = name or ""
    # 1) 색이 아닌 패션 용어를 먼저 지운다. 지우지 않으면 그 안의 색 단어가 잡힌다.
    for term in rules.get("non_color_terms", ()):
        if term:
            text = re.sub(re.escape(term), " ", text, flags=re.IGNORECASE)
    minimum = int(rules.get("min_korean_term_length", 2))
    need_boundary = bool(rules.get("english_needs_left_boundary", True))
    found: list[tuple[int, int, str]] = []
    for palette, words in table.items():
        for word in words:
            if len(word) < minimum:
                continue  # 한 글자 색('탄')은 다른 낱말에 섞인다
            for match in re.finditer(re.escape(word), text, re.IGNORECASE):
                start = match.start()
                if need_boundary and word.isascii() and word.isalpha():
                    # 영어는 앞에 알파벳이 붙으면 다른 낱말이다(LAYERED 의 RED).
                    if start and text[start - 1].isascii() and text[start - 1].isalpha():
                        continue
                found.append((start, len(word), palette))
                break
    # 같은 자리에서 겹치면 긴 단어가 이긴다('다크그레이'에서 그레이).
    ordered, taken = [], []
    for start, length, palette in sorted(found, key=lambda item: (-item[1], item[0])):
        if any(start < end and begin < start + length for begin, end in taken):
            continue
        taken.append((start, start + length))
        ordered.append((start, palette))
    result: list[str] = []
    for _, palette in sorted(ordered):
        if palette not in result:
            result.append(palette)
    return result


def load_photo_rules(path: Path | None = None) -> dict:
    """의류 영역 색 판정 규칙. 없으면 기본값으로 동작한다."""
    global _PHOTO_RULES
    default = path is None
    if default:
        if _PHOTO_RULES is not None:
            return _PHOTO_RULES
        from config import DATA_DIR

        path = Path(DATA_DIR) / "catalog_derivation.json"
    try:
        rules = json.loads(Path(path).read_text(encoding="utf-8")).get("photo_color_rules") or {}
    except (OSError, ValueError, TypeError, AttributeError):
        rules = {}
    rules = {"min_agreement": 0.8, "min_garment_ratio": 0.02,
             "hsv": {"dark": 0.16, "gray_low": 0.22, "neutral_sat": 0.12,
                     "navy_value": 0.35, "beige_sat": 0.30}, **rules}
    if default:
        _PHOTO_RULES = rules
    return rules


def palette_from_rgb(rgb, hsv_rules: dict | None = None) -> str:
    """픽셀 하나의 색을 팔레트로. 무채색과 유채색을 먼저 가른다.

    RGB 최근접은 어두운 옷을 전부 검정으로 보낸다(브라운·카키·네이비가 블랙으로).
    명도·채도를 먼저 보고 색상환으로 고르면 같은 표본에서 58% → 69% 였다.
    """
    hsv_rules = (load_photo_rules() if hsv_rules is None else hsv_rules)
    limits = hsv_rules.get("hsv", hsv_rules)
    red, green, blue = (max(0, min(255, int(value))) / 255 for value in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    hue *= 360
    if saturation < limits["neutral_sat"]:
        return "블랙" if value < limits["gray_low"] else ("그레이" if value < 0.75 else "화이트")
    if value < limits["dark"]:
        return "블랙"
    if hue < 15 or hue >= 345:
        return "버건디" if value < 0.45 else ("핑크" if saturation < 0.35 else "레드")
    if hue < 40:
        if value < 0.4:
            return "브라운"
        return "베이지" if saturation < limits["beige_sat"] else ("브라운" if value < 0.7 else "오렌지")
    if hue < 70:
        return "카키" if value < 0.55 else ("베이지" if saturation < 0.4 else "옐로")
    if hue < 160:
        return "카키" if saturation < 0.45 and value < 0.6 else "그린"
    if hue < 250:
        return "네이비" if value < limits["navy_value"] else "블루"
    if hue < 290:
        return "퍼플"
    return "핑크"


def color_from_pixels(pixels, hsv_rules: dict | None = None) -> tuple[str, float]:
    """의류 픽셀 표본에서 색과 그 확신도를 낸다.

    확신도는 '같은 팔레트로 떨어진 픽셀의 비율'이다. 실측에서 이 값이 정확도와 함께
    올라간다(0.6 이상 77%, 0.8 이상 87%, 0.9 이상 94%). 평균색의 신뢰도와 달리 쓸모가 있다.
    """
    votes: dict[str, int] = {}
    total = 0
    for pixel in pixels or ():
        palette = palette_from_rgb(pixel, hsv_rules)
        votes[palette] = votes.get(palette, 0) + 1
        total += 1
    if not total:
        return "", 0.0
    winner = max(votes, key=lambda key: (votes[key], key))
    return winner, round(votes[winner] / total, 3)
