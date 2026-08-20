"""OneDrive 프로젝트 → 로컬 clone 복사. 원본은 읽기만 한다 (삭제·이동 없음).

reports/21_migration_plan.json 의 include 목록만 복사하고
복사 후 개수·바이트·SHA256을 전수 대조한다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent
import os
# 저장소 위치는 환경변수로 지정한다. 절대경로를 코드에 박지 않는다.
REPO = Path(os.environ.get("FASHION_REPO_DIR",
                           Path.home() / "26-2_Modeling_Fashion_git")).expanduser()
TARGET = REPO / "ai_fashion_recommender"

# 코드가 자동 생성하지 않고, 스크립트가 읽기 대상으로 존재를 요구하는 곳만
GITKEEP_DIRS = [
    "data/provenance/fashionpedia/r1",
    "data/provenance/fashionpedia/r2",
    "data/provenance/fashionpedia/r3",
    "data/provenance/fashionpedia/seed",
]
GITKEEP_NOTE = (
    "# 이 폴더는 재현용 원본 주석을 두는 자리입니다.\n"
    "# 내용물은 .gitignore 로 제외됩니다. data/README.md 3절 참조.\n"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    plan = json.loads((SOURCE / "reports" / "21_migration_plan.json").read_text(encoding="utf-8"))
    paths = plan["include"]["paths"]
    expected_bytes = plan["include"]["bytes"]
    print(f"복사 대상 {len(paths):,}개 / {expected_bytes/1e6:.2f} MB")
    print(f"  {SOURCE}")
    print(f"  -> {TARGET}")

    copied = 0
    for rel in paths:
        src = SOURCE / rel
        dst = TARGET / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)          # 복사만. 원본 유지.
        copied += 1
        if copied % 40 == 0:
            print(f"  {copied}/{len(paths)}", flush=True)
    print(f"  {copied}/{len(paths)} 완료")

    for rel in GITKEEP_DIRS:
        directory = TARGET / rel
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").write_text(GITKEEP_NOTE, encoding="utf-8")
    print(f"\n.gitkeep {len(GITKEEP_DIRS)}개 생성 (코드가 자동 생성하지 않는 폴더만)")

    print("\n무결성 대조")
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    missing = [rel for rel in paths if not (TARGET / rel).is_file()]
    check("복사 누락", not missing, f"{len(missing)}건")

    size_mismatch = [rel for rel in paths
                     if (TARGET / rel).stat().st_size != (SOURCE / rel).stat().st_size]
    check("크기 불일치", not size_mismatch, f"{len(size_mismatch)}건")

    actual_bytes = sum((TARGET / rel).stat().st_size for rel in paths)
    check("총 바이트 일치", actual_bytes == expected_bytes,
          f"{actual_bytes:,} / {expected_bytes:,}")

    models = ["models/fashion_attribute_heads.pt",
              "models/fashion_attribute_heads_augmented.pt"]
    for rel in models:
        same = sha256(SOURCE / rel) == sha256(TARGET / rel)
        check(f"SHA256 일치 — {Path(rel).name}", same,
              sha256(TARGET / rel)[:32] + "…")

    source_still = all((SOURCE / rel).is_file() for rel in paths)
    check("원본 전부 그대로 존재", source_still, f"{len(paths)}개 확인")

    excluded_models = list((SOURCE / "models").glob("fashion_attribute_heads_augmented_r3*.pt")) \
        + [SOURCE / "models" / "fashion_attribute_heads_detailnone.pt",
           SOURCE / "models" / "fashion_attribute_heads_finetuned.pt"]
    local_kept = all(p.is_file() for p in excluded_models)
    check("제외 모델 로컬 보존", local_kept, f"{len(excluded_models)}개 원본 유지")

    leaked = [p.relative_to(TARGET).as_posix() for p in TARGET.rglob("*")
              if p.is_file() and (
                  p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".parquet", ".log")
                  or "instances_subset" in p.name
                  or "train2020" in p.name or "val2020" in p.name)]
    check("이미지·원본주석·로그 유입", not leaked,
          f"{len(leaked)}건" + (f" {leaked[:3]}" if leaked else ""))

    r3_leak = list(TARGET.glob("models/fashion_attribute_heads_augmented_r3*.pt"))
    other_leak = [TARGET / "models" / n for n in
                  ("fashion_attribute_heads_detailnone.pt", "fashion_attribute_heads_finetuned.pt")]
    leaked_models = r3_leak + [p for p in other_leak if p.is_file()]
    check("미채택 모델 유입", not leaked_models, f"{len(leaked_models)}건")

    repo_readme = (REPO / "README.md")
    check("저장소 루트 README.md 미변경", repo_readme.is_file() and repo_readme.stat().st_size == 62,
          f"{repo_readme.stat().st_size} bytes")

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\n검증 {len(checks)}건 중 통과 {len(checks)-len(failed)}건, 실패 {len(failed)}건")
    for name in failed:
        print(f"  FAIL: {name}")
    if failed:
        raise SystemExit("복사 검증 실패")
    print(f"\n복사 완료: {actual_bytes/1e6:.2f} MB")


if __name__ == "__main__":
    main()
