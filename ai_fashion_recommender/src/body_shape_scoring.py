"""Optional semantic-group scoring for body-shape recommendation rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

SEMANTIC_GROUP_KEYWORDS: dict[str, frozenset[str]] = {
    "fit_silhouette": frozenset({
        "세미핏", "레귤러", "정돈된 핏", "여유핏", "스트레이트", "세미와이드", "와이드",
    }),
    "neckline_structure": frozenset({"V넥", "넥라인 포인트", "어깨 구조"}),
    "waist_emphasis": frozenset({"허리 기준점", "허리선", "미드라이즈", "하이라이즈"}),
    "length": frozenset({"기본 기장", "풀렝스"}),
}
SEMANTIC_BODY_SHAPE_MAX_SCORE = 4.0


def _canonical_keyword(keyword: str) -> str:
    return "".join(str(keyword).lower().split())


_CANONICAL_GROUPS = {
    _canonical_keyword(keyword): group
    for group, keywords in SEMANTIC_GROUP_KEYWORDS.items()
    for keyword in keywords
}


def semantic_group_for_keyword(keyword: str) -> str | None:
    """Return the configured semantic group for a target keyword."""
    return _CANONICAL_GROUPS.get(_canonical_keyword(keyword))


def semantic_body_shape_score(
    active_r_bod_keywords: Iterable[str],
    matched_r_bod_keywords: Iterable[str],
) -> tuple[float, list[str]]:
    """Return normalized coverage and the matched distinct groups.

    Only keywords already proven to have an R-BOD rule provenance should be
    passed to this function. Unknown keywords are ignored rather than inferred.
    """
    active_groups = {
        group
        for keyword in active_r_bod_keywords
        if (group := semantic_group_for_keyword(keyword)) is not None
    }
    matched_groups = {
        group
        for keyword in matched_r_bod_keywords
        if (group := semantic_group_for_keyword(keyword)) is not None
        and group in active_groups
    }
    if not active_groups:
        return 0.0, []
    coverage = len(matched_groups) / len(active_groups)
    return SEMANTIC_BODY_SHAPE_MAX_SCORE * coverage, sorted(matched_groups)


def active_r_bod_keywords(
    keyword_rules: Mapping[str, Iterable[str]],
) -> list[str]:
    """Extract only keywords with an explicit R-BOD rule connection."""
    return sorted({
        keyword
        for keyword, rule_ids in keyword_rules.items()
        if any(str(rule_id).startswith("R-BOD-") for rule_id in rule_ids)
    })
