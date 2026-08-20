from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


RULE_HEADING = re.compile(r"^##\s+(R-[A-Z]+-\d+)\s+(.+?)\s*$")
CONFIDENCE = re.compile(r"^-\s*신뢰도:\s*\*\*(.+?)\*\*\s*$")
EVIDENCE = re.compile(r"^-\s*근거 수준:\s*\*\*(.+?)\*\*\s*$")


@dataclass(frozen=True)
class FashionRule:
    rule_id: str
    title: str
    confidence: str | None
    evidence: str | None
    text: str


class FashionRuleBook:
    """사람이 관리하는 패션 규칙 Markdown을 추천 엔진용으로 읽는다.

    규칙의 자연어 본문을 임의로 실행하지 않고, `R-*` ID가 있는 규칙만
    활성화한다. 실제 계산식은 테스트 가능한 Python 함수로 유지한다.
    """

    def __init__(self, source_path: Path, rules: dict[str, FashionRule], updated_at: str | None) -> None:
        self.source_path = source_path
        self.rules = rules
        self.updated_at = updated_at

    @classmethod
    def from_markdown(cls, path: str | Path) -> "FashionRuleBook":
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"패션 규칙 Markdown 파일이 없습니다: {source_path}")

        lines = source_path.read_text(encoding="utf-8").splitlines()
        updated_at = None
        for line in lines[:10]:
            if line.startswith("조사일:"):
                updated_at = line.split(":", 1)[1].strip() or None
                break

        headings: list[tuple[int, str, str]] = []
        for index, line in enumerate(lines):
            match = RULE_HEADING.match(line)
            if match:
                headings.append((index, match.group(1), match.group(2).strip()))

        rules: dict[str, FashionRule] = {}
        for position, (start, rule_id, title) in enumerate(headings):
            end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
            block_lines = lines[start + 1:end]
            confidence = next(
                (match.group(1).strip() for line in block_lines if (match := CONFIDENCE.match(line))),
                None,
            )
            evidence = next(
                (match.group(1).strip() for line in block_lines if (match := EVIDENCE.match(line))),
                None,
            )
            rules[rule_id] = FashionRule(
                rule_id=rule_id,
                title=title,
                confidence=confidence,
                evidence=evidence,
                text="\n".join(block_lines).strip(),
            )

        if not rules:
            raise ValueError(f"{source_path.name}에서 `## R-...` 형식의 규칙을 찾지 못했습니다.")
        return cls(source_path, rules, updated_at)

    def has(self, rule_id: str) -> bool:
        return rule_id in self.rules

    def title(self, rule_id: str) -> str:
        rule = self.rules.get(rule_id)
        return rule.title if rule else rule_id

    @property
    def active_rule_ids(self) -> list[str]:
        return sorted(self.rules)
