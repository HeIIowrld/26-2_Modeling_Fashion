"""저장소 이전 계획 — dry run. 파일을 복사·수정·삭제하지 않는다.

포함/제외 규칙을 적용해 옮길 파일 목록을 만들고,
기존 팀 저장소 구조와의 충돌만 보고한다.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent
import os
# 저장소 위치는 환경변수로 지정한다. 절대경로를 코드에 박지 않는다.
REPO = Path(os.environ.get("FASHION_REPO_DIR",
                           Path.home() / "26-2_Modeling_Fashion_git")).expanduser()
TARGET_SUBDIR = "ai_fashion_recommender"

EXCLUDE_DIR_PARTS = {
    "images", "cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    "raw", "runs", "runs_relabel", ".venv", "venv", "env", "node_modules",
    ".huggingface", "huggingface",
}
EXCLUDE_GLOBS = [
    "*.pt", "*.pth", "*.parquet", "*.log",
    "*instances_subset.json", "*train2020*.json", "*val2020*.json",
    "data/input_person.jpg",
    "*.jpg", "*.jpeg", "*.png", "*.webp",
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


def relative(path: Path) -> str:
    return path.relative_to(SOURCE).as_posix()


def is_excluded(path: Path) -> tuple[bool, str]:
    rel = relative(path)
    if rel in FORCE_INCLUDE:
        return False, ""
    for part in path.relative_to(SOURCE).parts[:-1]:
        if part in EXCLUDE_DIR_PARTS:
            return True, "디렉터리 " + part
    for pattern in EXPLICIT_EXCLUDE_MODELS:
        if fnmatch.fnmatch(rel, pattern):
            return True, "미채택 모델"
    for pattern in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True, "패턴 " + pattern
    return False, ""


def main() -> None:
    include: list[Path] = []
    exclude: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        skip, reason = is_excluded(path)
        if skip:
            exclude.setdefault(reason, []).append((relative(path), path.stat().st_size))
        else:
            include.append(path)

    total = sum(p.stat().st_size for p in include)
    print("=" * 90)
    print(f"이전 대상: {len(include)}개 파일 / {total / 1e6:.2f} MB")
    print(f"원본: {SOURCE.name}  ->  대상: {REPO.name}/{TARGET_SUBDIR}/")
    print("=" * 90)

    by_top: dict[str, list[tuple[str, int]]] = {}
    for path in include:
        rel = relative(path)
        group = rel.split("/")[0] if "/" in rel else "(루트)"
        by_top.setdefault(group, []).append((rel, path.stat().st_size))
    for group in sorted(by_top):
        entries = by_top[group]
        size = sum(s for _, s in entries)
        print(f"\n  [{group}]  {len(entries)}개  {size / 1e6:.2f} MB")
        for rel, s in sorted(entries, key=lambda e: -e[1])[:6]:
            print(f"      {s / 1e6:>8.3f} MB  {rel}")
        if len(entries) > 6:
            print(f"      … 외 {len(entries) - 6}개")

    print()
    print("=" * 90)
    print("제외 대상 (로컬에는 그대로 남김)")
    print("=" * 90)
    for reason in sorted(exclude, key=lambda r: -sum(s for _, s in exclude[r])):
        entries = exclude[reason]
        size = sum(s for _, s in entries)
        print(f"  {reason:<30}{len(entries):>6}개  {size / 1e6:>10.2f} MB")
    excluded_count = sum(len(v) for v in exclude.values())
    excluded_bytes = sum(s for v in exclude.values() for _, s in v)
    print(f"  {'합계':<30}{excluded_count:>6}개  {excluded_bytes / 1e6:>10.2f} MB")

    print()
    print("=" * 90)
    print("기존 저장소 구조와의 충돌 검사")
    print("=" * 90)
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                             text=True).stdout.split()
    print(f"  저장소 추적 파일 {len(tracked)}개: {tracked}")
    existing = {p.relative_to(REPO).as_posix() for p in REPO.rglob("*")
                if p.is_file() and ".git/" not in p.as_posix()}

    conflicts = [f"{TARGET_SUBDIR}/{relative(p)}" for p in include
                 if f"{TARGET_SUBDIR}/{relative(p)}" in existing]
    print(f"\n  덮어쓰게 되는 기존 파일: {len(conflicts)}건")
    for item in conflicts[:10]:
        print(f"    {item}")
    if not conflicts:
        print("    없음 - 저장소가 비어 있어 충돌 없음")

    root_readme = "README.md" in existing
    project_readme = any(relative(p) == "README.md" for p in include)
    print(f"\n  저장소 루트 README.md 존재: {root_readme}")
    print(f"  프로젝트 README.md 포함    : {project_readme}")
    if root_readme and project_readme:
        print(f"    -> 경로가 달라 충돌 아님 (README.md vs {TARGET_SUBDIR}/README.md)")

    oversized = [(relative(p), p.stat().st_size) for p in include
                 if p.stat().st_size >= 100 * 1024 * 1024]
    print(f"\n  100MB 이상 파일: {len(oversized)}건")
    print("  최대 파일 5개:")
    for rel, s in sorted(((relative(p), p.stat().st_size) for p in include),
                         key=lambda e: -e[1])[:5]:
        print(f"    {s / 1e6:>8.2f} MB  {rel}")

    has_ignore = ".gitignore" in existing
    has_attrs = ".gitattributes" in existing
    print(f"\n  저장소 .gitignore   : {'있음' if has_ignore else '없음 - 프로젝트 것을 이식해야 함'}")
    print(f"  저장소 .gitattributes: {'있음' if has_attrs else '없음 (LFS 미설정)'}")

    data_dirs = sorted({str(Path(rel).parent).replace("\\", "/")
                        for entries in exclude.values() for rel, _ in entries
                        if Path(rel).parts[0] == "data"})
    print(f"\n  제외로 비게 되는 data 하위 경로 (.gitkeep 후보): {len(data_dirs)}개")
    for item in data_dirs[:12]:
        print(f"    {item}")

    (SOURCE / "reports" / "21_migration_plan.json").write_text(json.dumps({
        "source": SOURCE.name,
        "target_repo": str(REPO),
        "target_subdir": TARGET_SUBDIR,
        "include_count": len(include),
        "include_bytes": total,
        "include_files": [relative(p) for p in include],
        "exclude_summary": {r: {"count": len(v), "bytes": sum(s for _, s in v)}
                            for r, v in exclude.items()},
        "conflicts": conflicts,
        "oversized": oversized,
        "gitkeep_candidates": data_dirs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장: reports/21_migration_plan.json  (계획만, 복사 없음)")


if __name__ == "__main__":
    main()
