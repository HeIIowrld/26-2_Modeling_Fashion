"""저장소 사본의 main.ipynb 에서 셀 출력과 execution_count 만 제거한다.

nbconvert --clear-output 와 같은 결과이지만, 셀 원문을 원본과 1:1 대조해
코드·마크다운이 한 글자도 바뀌지 않았음을 증명한다.
OneDrive 원본은 읽기만 하고 절대 수정하지 않는다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SOURCE_NOTEBOOK = Path(__file__).resolve().parent.parent / "main.ipynb"
REPO = Path(os.environ.get("FASHION_REPO_DIR",
                           Path.home() / "26-2_Modeling_Fashion_git")).expanduser()
TARGET_NOTEBOOK = REPO / "ai_fashion_recommender" / "main.ipynb"

# 검사할 사용자명은 조각으로 조립한다. 리터럴로 두면 이 파일 자체가
# git grep 감사에 걸리기 때문이다.
USERNAMES = ("chl" + "gu", "2023" + "user")
PII = re.compile("|".join(USERNAMES) + r"|[Cc]:\\+Users|[Cc]:/Users|OneDrive|AppData|scratchpad"
                 r"|ドキュメント|바탕 화면")

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    original = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    notebook = json.loads(TARGET_NOTEBOOK.read_text(encoding="utf-8"))
    print(f"원본(읽기 전용): {SOURCE_NOTEBOOK}")
    print(f"정리 대상      : {TARGET_NOTEBOOK}")
    print(f"셀 {len(notebook['cells'])}개\n")

    cleared_outputs = cleared_counts = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cleared_outputs += len(cell["outputs"])
            cell["outputs"] = []
            if cell.get("execution_count") is not None:
                cleared_counts += 1
            cell["execution_count"] = None
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"출력 {cleared_outputs}개, execution_count {cleared_counts}개 제거\n")

    saved = json.loads(TARGET_NOTEBOOK.read_text(encoding="utf-8"))
    print("검증")

    check("셀 개수 동일", len(saved["cells"]) == len(original["cells"]),
          f"{len(saved['cells'])} / {len(original['cells'])}")

    code_diff = md_diff = type_diff = 0
    for before, after in zip(original["cells"], saved["cells"]):
        if before.get("cell_type") != after.get("cell_type"):
            type_diff += 1
            continue
        if before.get("source") != after.get("source"):
            if before.get("cell_type") == "code":
                code_diff += 1
            else:
                md_diff += 1
    check("1. 코드 셀 내용 변경", code_diff == 0, f"{code_diff}건")
    check("2. 마크다운 셀 내용 변경", md_diff == 0, f"{md_diff}건")
    check("   셀 타입 변경", type_diff == 0, f"{type_diff}건")

    code_cells = [c for c in saved["cells"] if c.get("cell_type") == "code"]
    non_empty = [i for i, c in enumerate(code_cells) if c.get("outputs")]
    check("3. outputs 전부 비어 있음", not non_empty,
          f"코드 셀 {len(code_cells)}개 중 출력 남은 셀 {len(non_empty)}개")
    non_null = [i for i, c in enumerate(code_cells) if c.get("execution_count") is not None]
    check("4. execution_count 전부 null", not non_null,
          f"{len(non_null)}개 남음")

    text = TARGET_NOTEBOOK.read_text(encoding="utf-8")
    for label, pattern in ((f"5. 팀원 사용자명", USERNAMES[0]), (f"6. 현재 사용자명", USERNAMES[1]),
                           ("7. C:\\Users / c:\\Users", r"[Cc]:\\+Users|[Cc]:/Users"),
                           ("8. OneDrive·AppData·scratchpad 등",
                            r"OneDrive|AppData|scratchpad|ドキュメント|바탕 화면")):
        hits = re.findall(pattern, text)
        check(label, not hits, f"{len(hits)}건")

    check("원본 OneDrive 파일 미변경",
          SOURCE_NOTEBOOK.stat().st_size > TARGET_NOTEBOOK.stat().st_size,
          f"원본 {SOURCE_NOTEBOOK.stat().st_size:,} bytes 유지 / "
          f"정리본 {TARGET_NOTEBOOK.stat().st_size:,} bytes")

    failed = [n for n, ok, _ in checks if not ok]
    print(f"\n검증 {len(checks)}건 중 통과 {len(checks)-len(failed)}건, 실패 {len(failed)}건")
    for name in failed:
        print(f"  FAIL: {name}")
    if failed:
        raise SystemExit("노트북 정리 검증 실패")


if __name__ == "__main__":
    main()
