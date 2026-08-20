"""저장소 루트에서 실행해도 웹 앱이 켜지도록 실제 진입점으로 넘겨준다.

실제 구현은 `ai_fashion_recommender/run_web.py`에 있다. 상품 CSV와 패션 규칙 문서가
그 폴더 기준 상대경로라, 루트에서 그대로 실행하면 파일을 찾지 못한다.

    python run_web.py --check    # 옵션은 그대로 전달된다
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_fashion_recommender"
ENTRY_POINT = PROJECT_DIR / "run_web.py"


def main() -> int:
    if not ENTRY_POINT.is_file():
        print(
            f"[실행할 수 없습니다]\n  진입점을 찾지 못했습니다: {ENTRY_POINT}\n"
            "  저장소를 통째로 내려받았는지 확인하세요.",
            file=sys.stderr,
        )
        return 1
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    runpy.run_path(str(ENTRY_POINT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
