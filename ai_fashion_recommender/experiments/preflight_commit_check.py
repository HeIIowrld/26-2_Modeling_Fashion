"""커밋 전 안전 점검 — 아직 Git 저장소가 없으므로 파일시스템 기준으로 검사한다.

git status / diff 대신 "커밋 예정 파일 목록"을 직접 구성해 같은 항목을 확인한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 커밋 대상 (사용자 지정 목록)
INCLUDE_GLOBS = [
    "experiments/*.py",
    "reports/*.md",
    "reports/*.json",
    "reports/manifests/*",
    "data/fashionpedia_train_r3/manifest.json",
    "data/fashionpedia_train_r3/selection_r3.json",
    "data/fashionpedia_train/manifest.json",
    "data/fashionpedia_train/manifest_r2.json",
    "data/fashionpedia_train/manifest_r2_corrected.json",
    "data/fashionpedia_train/selection.json",
    "data/fashionpedia_train/selection_r2.json",
    "data/fashionpedia_train/fashion_attribute_annotations.csv",
    "data/fashionpedia_train/fashion_attribute_annotations_r2.csv",
    "data/fashionpedia_train_r3/fashion_attribute_annotations.csv",
    "data/provenance/fashionpedia/shard_index.json",
    "models/fashion_attribute_heads_augmented_r3.metrics.json",
    ".gitignore",
]
# 명시적 제외 (커밋 목록에 잡히더라도 뺀다)
EXCLUDE_PATTERNS = [
    re.compile(r"reports/_round3_inputs\.json$"),   # 중간 산출물
    re.compile(r"reports/.*\.log$"),                # 임시 로그
]

ABSOLUTE = re.compile(
    # 실제 절대경로 형태만 잡는다. AppData 같은 단어 하나로 잡으면
    # 이 검사기 자신의 정규식 정의까지 걸린다.
    r"[A-Za-z]:[\\/]{1,2}Users[\\/]|/home/[a-z0-9_.-]+/|/Users/[a-z0-9_.-]+/")
SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|bearer\s+[A-Za-z0-9._-]{20,}"
    r"|password\s*=\s*['\"][^'\"]{3,}|passwd\s*=|BEGIN [A-Z ]*PRIVATE KEY"
    r"|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,})")
FORBIDDEN_SUFFIX = (".pt", ".pth", ".parquet", ".jpg", ".jpeg", ".png", ".webp", ".env")
TEXT_SUFFIX = (".py", ".md", ".json", ".csv", ".txt", ".gitignore", "")


def collect() -> list[Path]:
    files: set[Path] = set()
    for pattern in INCLUDE_GLOBS:
        for path in PROJECT_DIR.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(PROJECT_DIR).as_posix()
            if any(p.search(rel) for p in EXCLUDE_PATTERNS):
                continue
            files.add(path)
    return sorted(files)


def main() -> None:
    files = collect()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("=" * 88)
    print(f"커밋 예정 파일 {len(files)}개")
    print("=" * 88)
    total = 0
    by_dir: dict[str, list[tuple[str, int]]] = {}
    for path in files:
        rel = path.relative_to(PROJECT_DIR).as_posix()
        size = path.stat().st_size
        total += size
        by_dir.setdefault(rel.split("/")[0], []).append((rel, size))
    for group in sorted(by_dir):
        entries = by_dir[group]
        print(f"\n  [{group}]  {len(entries)}개  {sum(s for _, s in entries)/1e6:.2f} MB")
        for rel, size in sorted(entries, key=lambda e: -e[1])[:8]:
            print(f"    {size/1e6:>8.3f} MB  {rel}")
        if len(entries) > 8:
            print(f"    … 외 {len(entries)-8}개")
    print(f"\n  총 용량: {total/1e6:.2f} MB")

    print()
    print("=" * 88)
    print("안전 점검")
    print("=" * 88)

    oversized = [p for p in files if p.stat().st_size >= 100 * 1024 * 1024]
    check("100MB 이상 파일", not oversized, f"{len(oversized)}건")

    forbidden = [p for p in files if p.suffix.lower() in FORBIDDEN_SUFFIX]
    check("이미지·모델·parquet·.env 포함 여부", not forbidden,
          f"{len(forbidden)}건" + (f" {[p.name for p in forbidden[:3]]}" if forbidden else ""))

    abs_hits, secret_hits, unreadable = [], [], []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIX and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except Exception:
            unreadable.append(path)
            continue
        stripped = re.sub(r"https?://", "", text)
        if ABSOLUTE.search(stripped):
            # 매치 문자열은 기록하지 않는다. 그 자체가 다음 검사에서 오탐이 되기 때문이다.
            abs_hits.append(path.relative_to(PROJECT_DIR).as_posix())
        if SECRET.search(text):
            secret_hits.append(path.relative_to(PROJECT_DIR).as_posix())
    check("절대경로 흔적", not abs_hits,
          f"{len(abs_hits)}건" + (f" 파일: {abs_hits[:3]}" if abs_hits else ""))
    check("API key·token·password·개인정보", not secret_hits,
          f"{len(secret_hits)}건" + (f" {secret_hits[:3]}" if secret_hits else ""))
    check("읽기 실패 파일", not unreadable, f"{len(unreadable)}건")

    manifest_dir = PROJECT_DIR / "reports" / "manifests"
    manifest_abs = 0
    for path in manifest_dir.glob("*.csv"):
        text = re.sub(r"https?://", "", path.read_text(encoding="utf-8-sig"))
        manifest_abs += len(ABSOLUTE.findall(text))
    check("manifest CSV 절대경로", manifest_abs == 0, f"{manifest_abs}건")

    # 제외돼야 할 것들이 목록에 없는지
    should_exclude = {
        "3차 체크포인트": list(PROJECT_DIR.glob("models/fashion_attribute_heads_augmented_r3*.pt")),
        "원본 이미지": list(PROJECT_DIR.glob("data/**/images/*"))[:1],
        "instances_subset": list(PROJECT_DIR.glob("data/**/instances_subset.json")),
        "원본 annotation": list(PROJECT_DIR.glob("data/**/*val2020*.json")),
        "임베딩 캐시": list(PROJECT_DIR.glob("data/cache/*.pt")),
        "임시 로그": list(PROJECT_DIR.glob("reports/*.log")),
    }
    file_set = set(files)
    for label, paths in should_exclude.items():
        leaked = [p for p in paths if p in file_set]
        check(f"제외 확인 — {label}", not leaked, f"{len(leaked)}건 stage됨 (기대 0)")

    print()
    failed = [n for n, ok, _ in results if not ok]
    print("=" * 88)
    print(f"점검 {len(results)}건 중 통과 {len(results)-len(failed)}건, 실패 {len(failed)}건")
    for name in failed:
        print(f"  FAIL: {name}")
    print("=" * 88)

    (PROJECT_DIR / "reports" / "20_preflight_commit_check.json").write_text(json.dumps({
        "files": [{"path": p.relative_to(PROJECT_DIR).as_posix(), "bytes": p.stat().st_size}
                  for p in files],
        "total_bytes": total,
        "checks": [{"name": n, "passed": o, "detail": d} for n, o, d in results],
        "absolute_path_hits": abs_hits,
        "secret_hits": secret_hits,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장: reports/20_preflight_commit_check.json")


if __name__ == "__main__":
    main()
