"""저장소 루트를 sys.path 에 넣는다.

web/tests 는 `import web.app` 을 한다. `python -m pytest` 는 현재 폴더를 자동으로
sys.path 에 넣어 주지만, 콘솔 스크립트 `pytest` 는 넣어 주지 않는다. 그래서 같은
저장소에서 호출 방식만 달라도 수집이 깨졌다. CI 는 어느 쪽으로 부를지 모르므로
여기서 한 번 고정한다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
