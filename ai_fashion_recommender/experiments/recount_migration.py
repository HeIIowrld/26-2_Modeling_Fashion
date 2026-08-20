"""이전 대상 재계산 — 포함/제외를 서로 겹치지 않는 집합으로 분류하고 전수 검증한다.

모든 용량은 바이트로 계산하고 표기만 MB(=10^6 bytes)로 환산한다.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent
TARGET_SUBDIR = "ai_fashion_recommender"

EXCLUDE_DIR_PARTS = {
    "images", "cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    "raw", "runs", "runs_relabel", ".venv", "venv", "env", "node_modules",
    ".huggingface", "huggingface", "outputs",
}
EXCLUDE_GLOBS = [
    "*.pt", "*.pth", "*.parquet", "*.log",
    "*instances_subset.json", "*train2020*.json", "*val2020*.json",
    "*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.bmp",
    "*_round3_inputs.json",
]
FORCE_INCLUDE = {
    "models/fashion_attribute_heads.pt",
    "models/fashion_attribute_heads_augmented.pt",
}
EXPLICIT_EXCLUDE_MODELS = [
    "models/fashion_attribute_heads_augmented_r3*.pt",
    "models/fashion_attribute_heads_detailnone.pt",
    "models/fashion_attribute_heads_finetuned.pt",
]


def mb(value: int) -> float:
    return value / 1_000_000


def classify(rel: str, name: str, parts: tuple[str, ...]) -> tuple[str, str]:
    """('include'|'exclude', 사유) 를 돌려준다. 두 집합은 상호배타적이다."""
    if rel in FORCE_INCLUDE:
        return "include", "배포 모델 (명시 포함)"
    for part in parts[:-1]:
        if part in EXCLUDE_DIR_PARTS:
            return "exclude", f"디렉터리 {part}/"
    for pattern in EXPLICIT_EXCLUDE_MODELS:
        if fnmatch.fnmatch(rel, pattern):
            return "exclude", "미채택 모델"
    for pattern in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
            return "exclude", f"패턴 {pattern}"
    return "include", "일반 포함"


def main() -> None:
    all_files = [p for p in sorted(SOURCE.rglob("*")) if p.is_file()]
    include: dict[str, int] = {}
    exclude: dict[str, int] = {}
    include_reason: dict[str, str] = {}
    exclude_reason: dict[str, str] = {}
    unclassified: list[str] = []

    for path in all_files:
        rel = path.relative_to(SOURCE).as_posix()
        parts = path.relative_to(SOURCE).parts
        size = path.stat().st_size
        bucket, reason = classify(rel, path.name, parts)
        if bucket == "include":
            include[rel] = size
            include_reason[rel] = reason
        elif bucket == "exclude":
            exclude[rel] = size
            exclude_reason[rel] = reason
        else:
            unclassified.append(rel)

    total_files = len(all_files)
    total_bytes = sum(p.stat().st_size for p in all_files)

    print("=" * 92)
    print("집합 검증")
    print("=" * 92)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    overlap = set(include) & set(exclude)
    check("included ∩ excluded = 0", not overlap, f"{len(overlap)}건")
    union = set(include) | set(exclude) | set(unclassified)
    all_rel = {p.relative_to(SOURCE).as_posix() for p in all_files}
    check("included ∪ excluded ∪ unclassified = 전체",
          union == all_rel, f"{len(union):,} / 전체 {len(all_rel):,}")
    check("unclassified = 0", not unclassified, f"{len(unclassified)}건")
    if unclassified:
        for rel in unclassified[:20]:
            print(f"      {rel}")

    check("개수 합계 = 전체",
          len(include) + len(exclude) + len(unclassified) == total_files,
          f"{len(include):,} + {len(exclude):,} + {len(unclassified)} = "
          f"{len(include) + len(exclude) + len(unclassified):,} / {total_files:,}")
    byte_sum = sum(include.values()) + sum(exclude.values())
    check("byte 합계 = 전체",
          byte_sum == total_bytes,
          f"{byte_sum:,} / {total_bytes:,} bytes")

    for model in sorted(FORCE_INCLUDE):
        count = sum(1 for rel in include if rel == model)
        check(f"배포 모델 정확히 1회 — {Path(model).name}", count == 1, f"{count}회")

    # ---- 포함: 그룹별
    print()
    print("=" * 92)
    print(f"포함  {len(include):,}개  {mb(sum(include.values())):.2f} MB  "
          f"({sum(include.values()):,} bytes)")
    print("=" * 92)
    groups: dict[str, list[tuple[str, int]]] = {}
    for rel, size in include.items():
        top = rel.split("/")[0] if "/" in rel else "(루트)"
        groups.setdefault(top, []).append((rel, size))
    group_count = group_bytes = 0
    print(f"  {'그룹':<16}{'개수':>8}{'bytes':>16}{'MB':>10}")
    print("  " + "-" * 50)
    for name in sorted(groups):
        entries = groups[name]
        size = sum(s for _, s in entries)
        group_count += len(entries)
        group_bytes += size
        print(f"  {name:<16}{len(entries):>8}{size:>16,}{mb(size):>10.2f}")
    print("  " + "-" * 50)
    print(f"  {'합계':<16}{group_count:>8}{group_bytes:>16,}{mb(group_bytes):>10.2f}")
    check("포함 그룹 합계 = 포함 전체",
          group_count == len(include) and group_bytes == sum(include.values()),
          f"{group_count} / {len(include)}, {group_bytes:,} / {sum(include.values()):,}")

    # ---- 제외: 사유별
    print()
    print("=" * 92)
    print(f"제외  {len(exclude):,}개  {mb(sum(exclude.values())):.2f} MB  "
          f"({sum(exclude.values()):,} bytes)   ※ 로컬에는 그대로 유지")
    print("=" * 92)
    reasons: dict[str, list[tuple[str, int]]] = {}
    for rel, size in exclude.items():
        reasons.setdefault(exclude_reason[rel], []).append((rel, size))
    reason_count = reason_bytes = 0
    print(f"  {'사유':<32}{'개수':>8}{'bytes':>16}{'MB':>10}")
    print("  " + "-" * 66)
    for name in sorted(reasons, key=lambda r: -sum(s for _, s in reasons[r])):
        entries = reasons[name]
        size = sum(s for _, s in entries)
        reason_count += len(entries)
        reason_bytes += size
        print(f"  {name:<32}{len(entries):>8}{size:>16,}{mb(size):>10.2f}")
    print("  " + "-" * 66)
    print(f"  {'합계':<32}{reason_count:>8}{reason_bytes:>16,}{mb(reason_bytes):>10.2f}")
    check("제외 사유 합계 = 제외 전체",
          reason_count == len(exclude) and reason_bytes == sum(exclude.values()),
          f"{reason_count} / {len(exclude)}, {reason_bytes:,} / {sum(exclude.values()):,}")

    print()
    print(f"  전체 조사 파일 {total_files:,}개  {mb(total_bytes):.2f} MB")

    failed = [n for n, ok, _ in checks if not ok]
    print()
    print("=" * 92)
    print(f"검증 {len(checks)}건 중 통과 {len(checks)-len(failed)}건, 실패 {len(failed)}건")
    for name in failed:
        print(f"  FAIL: {name}")
    print("=" * 92)

    (SOURCE / "reports" / "21_migration_plan.json").write_text(json.dumps({
        "target_subdir": TARGET_SUBDIR,
        "totals": {"files": total_files, "bytes": total_bytes},
        "include": {"files": len(include), "bytes": sum(include.values()),
                    "by_group": {k: {"files": len(v), "bytes": sum(s for _, s in v)}
                                 for k, v in sorted(groups.items())},
                    "paths": sorted(include)},
        "exclude": {"files": len(exclude), "bytes": sum(exclude.values()),
                    "by_reason": {k: {"files": len(v), "bytes": sum(s for _, s in v)}
                                  for k, v in sorted(reasons.items())}},
        "unclassified": unclassified,
        "checks": [{"name": n, "passed": o, "detail": d} for n, o, d in checks],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장: reports/21_migration_plan.json")
    if failed:
        raise SystemExit("검증 실패 — 복사를 진행하지 않습니다.")


if __name__ == "__main__":
    main()
