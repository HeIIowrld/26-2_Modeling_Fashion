"""배포용 ZIP 생성 + 압축 해제 검증.

원본은 git 에 커밋된 REPO/ai_fashion_recommender/ 트리(143개 파일)를 그대로 쓴다.
따라서 커밋 내용과 ZIP 내용이 정의상 동일하다.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import os
# 저장소 위치는 환경변수로 지정한다. 절대경로를 코드에 박지 않는다.
REPO = Path(os.environ.get("FASHION_REPO_DIR",
                           Path.home() / "26-2_Modeling_Fashion_git")).expanduser()
PROJECT = "ai_fashion_recommender"
PACKAGES = REPO / "packages"
ZIP_PATH = PACKAGES / "ai_fashion_recommender_final_r2.zip"
CHECKSUMS = PACKAGES / "PACKAGE_CHECKSUMS.json"
import tempfile
EXTRACT_ROOT = Path(os.environ.get("FASHION_ZIP_VERIFY_DIR",
                                   Path(tempfile.gettempdir()) / "zipchk"))

REQUIRED_MODELS = [
    "models/fashion_attribute_heads.pt",
    "models/fashion_attribute_heads_augmented.pt",
]
FORBIDDEN_SUFFIX = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
                    ".parquet", ".log", ".pyc")
FORBIDDEN_NAME = ("instances_subset", "train2020", "val2020")
FORBIDDEN_DIR = ("__pycache__", ".pytest_cache", ".git", "data/cache",
                 "/images/", ".ipynb_checkpoints")
FORBIDDEN_MODEL = re.compile(r"augmented_r3.*\.pt$|detailnone\.pt$|finetuned\.pt$")
USERNAMES = ("chl" + "gu", "2023" + "user")
# 실제 "절대경로처럼 보이는 문자열"과 알려진 사용자명만 잡는다.
# AppData 같은 단어 하나만으로 잡으면 탐지기 자신의 정규식 정의까지 걸린다.
ABSOLUTE_PATH = r"[A-Za-z]:[\\/]{1,2}Users[\\/]|/home/[a-z0-9_.-]+/|/Users/[a-z0-9_.-]+/"
PII = re.compile("|".join(USERNAMES) + "|nanjun@|" + ABSOLUTE_PATH)

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", f"{PROJECT}/"], cwd=REPO,
        capture_output=True, text=True, check=True).stdout.split("\n")
    tracked = [t for t in tracked if t.strip()]
    print(f"커밋된 프로젝트 파일 {len(tracked)}개")

    PACKAGES.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for rel in sorted(tracked):
            archive.write(REPO / rel, arcname=rel)
    print(f"생성: {ZIP_PATH.relative_to(REPO).as_posix()}  "
          f"{ZIP_PATH.stat().st_size / 1e6:.2f} MB")

    # ---- 압축 해제 검증
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(EXTRACT_ROOT)
    extracted = sorted(p for p in EXTRACT_ROOT.rglob("*") if p.is_file())
    names = [p.relative_to(EXTRACT_ROOT).as_posix() for p in extracted]

    print("\n검증")
    tops = {n.split("/")[0] for n in names}
    check("1. 최상위 폴더명", tops == {PROJECT}, f"{sorted(tops)}")
    check("2. 파일 수 일치", len(names) == len(tracked),
          f"{len(names)} / 커밋 {len(tracked)}")

    for model in REQUIRED_MODELS:
        target = f"{PROJECT}/{model}"
        count = names.count(target)
        check(f"3. {Path(model).name} 1개", count == 1, f"{count}개")

    def forbidden(name: str) -> bool:
        parts = name.split("/")
        # .git 은 경로 "구성요소"로만 판단한다. .gitignore/.gitkeep 은 정상 파일이다.
        if any(part in ("__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints",
                        "images", "cache") for part in parts[:-1]):
            return True
        return (name.lower().endswith(FORBIDDEN_SUFFIX)
                or any(k in name for k in FORBIDDEN_NAME)
                or bool(FORBIDDEN_MODEL.search(name)))

    bad = [n for n in names if forbidden(n)]
    check("4. 제외 대상 파일", not bad, f"{len(bad)}건" + (f" {bad[:3]}" if bad else ""))

    pii_hits = []
    for path in extracted:
        if path.suffix.lower() not in (".py", ".md", ".json", ".csv", ".txt", ".ipynb") \
                and path.name != ".gitignore" and path.name != ".gitkeep":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "re.compile" in line or "PII" in line or "ABSOLUTE" in line:
                continue
            if PII.search(line):
                pii_hits.append(f"{path.relative_to(EXTRACT_ROOT).as_posix()}:{i}")
    check("5. 절대경로·사용자명", not pii_hits,
          f"{len(pii_hits)}건" + (f" {pii_hits[:3]}" if pii_hits else ""))

    notebook = EXTRACT_ROOT / PROJECT / "main.ipynb"
    notebook_text = notebook.read_text(encoding="utf-8")
    abs_in_nb = len(re.findall(r"C:\\\\Users|/Users/|AppData", notebook_text))
    rel_markers = [m for m in ("PROJECT_DIR_INPUT", "models/fashion_attribute_heads",
                               "data/input_person.jpg") if m in notebook_text]
    config_text = (EXTRACT_ROOT / PROJECT / "config.py").read_text(encoding="utf-8")
    config_relative = "Path(__file__).resolve().parent" in config_text
    check("6. 상대경로 정상",
          abs_in_nb == 0 and config_relative and len(rel_markers) >= 2,
          f"main.ipynb 절대경로 {abs_in_nb}건, config.py 프로젝트 기준 경로 {config_relative}, "
          f"경로 설정 셀 마커 {len(rel_markers)}개")

    sys.path.insert(0, str(EXTRACT_ROOT / PROJECT))
    import torch
    torch.set_num_threads(2)
    from fashion_attribute_model import load_attribute_heads
    loaded = {}
    for model in REQUIRED_MODELS:
        heads, payload = load_attribute_heads(EXTRACT_ROOT / PROJECT / model, "cpu")
        loaded[model] = {
            "params": sum(p.numel() for p in heads.parameters()),
            "backbone": payload["backbone_model_id"],
            "tasks": len(payload["tasks"]),
        }
    check("7. 기존 로더로 load",
          all(v["params"] == 3_404_668 and v["tasks"] == 17 for v in loaded.values()),
          f"2개 모델 로드 성공, 헤드 파라미터 {loaded[REQUIRED_MODELS[0]]['params']:,}, "
          f"태스크 {loaded[REQUIRED_MODELS[0]]['tasks']}개")

    zip_hash = sha256(ZIP_PATH)
    zip_size = ZIP_PATH.stat().st_size
    check("8. ZIP SHA256·용량", True, f"{zip_size:,} bytes ({zip_size/1e6:.2f} MB)")
    print(f"      sha256 {zip_hash}")

    # 압축 해제본 ↔ 커밋본 해시 대조
    hash_diff = [rel for rel in tracked
                 if sha256(REPO / rel) != sha256(EXTRACT_ROOT / rel)]
    check("추가: 해제본 == 커밋본 SHA256", not hash_diff, f"{len(hash_diff)}건 불일치")

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\n검증 {len(checks)}건 중 통과 {len(checks)-len(failed)}건, 실패 {len(failed)}건")
    for name in failed:
        print(f"  FAIL: {name}")
    if failed:
        raise SystemExit("ZIP 검증 실패 — checksum 파일을 만들지 않습니다.")

    CHECKSUMS.write_text(json.dumps({
        "package": {
            "filename": ZIP_PATH.name,
            "path": ZIP_PATH.relative_to(REPO).as_posix(),
            "bytes": zip_size,
            "megabytes": round(zip_size / 1e6, 2),
            "sha256": zip_hash,
            "files": len(names),
            "top_level_dir": PROJECT,
            "compression": "deflate level 6",
        },
        "adopted_model": {
            "filename": "fashion_attribute_heads_augmented.pt",
            "path": f"{PROJECT}/models/fashion_attribute_heads_augmented.pt",
            "role": "최종 채택 배포 모델 (2차 보강, train 22,341 crop)",
            "bytes": (REPO / PROJECT / "models/fashion_attribute_heads_augmented.pt").stat().st_size,
            "sha256": sha256(REPO / PROJECT / "models/fashion_attribute_heads_augmented.pt"),
            "backbone": loaded[REQUIRED_MODELS[1]]["backbone"],
            "head_parameters": loaded[REQUIRED_MODELS[1]]["params"],
        },
        "baseline_model": {
            "filename": "fashion_attribute_heads.pt",
            "path": f"{PROJECT}/models/fashion_attribute_heads.pt",
            "role": "초기 baseline · rollback용",
            "bytes": (REPO / PROJECT / "models/fashion_attribute_heads.pt").stat().st_size,
            "sha256": sha256(REPO / PROJECT / "models/fashion_attribute_heads.pt"),
        },
        "excluded_from_package": [
            ".git", "원본 이미지", "개인 사진 및 파생 결과 이미지", "data/cache",
            "임베딩 캐시", "원본 annotation JSON", "parquet",
            "3차 및 기타 미채택 모델", "__pycache__", ".pytest_cache", "로그",
        ],
        "verification": [{"name": n, "passed": o, "detail": d} for n, o, d in checks],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {CHECKSUMS.relative_to(REPO).as_posix()}")
    shutil.rmtree(EXTRACT_ROOT)
    print(f"임시 해제 폴더 정리 완료")


if __name__ == "__main__":
    main()
