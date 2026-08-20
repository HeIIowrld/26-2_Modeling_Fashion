"""웹 앱 실행 진입점. 저장소 루트에서 이 파일을 실행하면 된다.

    python run_web.py                # http://127.0.0.1:8000
    python run_web.py --check        # 서버를 켜지 않고 환경만 점검

실제 구현은 `web/run_web.py`에 있다. 웹 화면과 서버 코드는 `web/`에,
체형·의류 분석과 추천 규칙은 Notebook과 공용이라 `ai_fashion_recommender/`에 있다.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
ENTRY_POINT = WEB_DIR / "run_web.py"


def main() -> int:
    if not ENTRY_POINT.is_file():
        print(
            f"[실행할 수 없습니다]\n  진입점을 찾지 못했습니다: {ENTRY_POINT}\n"
            "  저장소를 통째로 내려받았는지 확인하세요.",
            file=sys.stderr,
        )
        return 1
    if str(WEB_DIR) not in sys.path:
        sys.path.insert(0, str(WEB_DIR))
    runpy.run_path(str(ENTRY_POINT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
