"""팀원 환경에서 웹 앱을 실행하는 진입점.

어느 폴더에서 실행해도 동작한다.

    python web/run_web.py                # http://127.0.0.1:8000
    python web/run_web.py --port 9000    # 포트 지정
    python web/run_web.py --lan          # 같은 와이파이의 다른 기기에서도 접속 허용
    python web/run_web.py --check        # 서버를 켜지 않고 환경만 점검

실행 전에 파이썬 버전·패키지·데이터 파일을 먼저 점검하고, 문제가 있으면
무엇을 어떻게 고쳐야 하는지 한국어로 알려준다.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import webbrowser
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
# 분석·규칙·모델은 웹 전용이 아니라 Notebook과 공용이라 별도 폴더에 둔다.
CORE_DIR = WEB_DIR.parent / "ai_fashion_recommender"

# mediapipe 0.10.x 휠이 제공되는 범위. 이 밖의 버전에서는 설치 자체가 실패한다.
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)

# import 이름과 pip 이름이 다른 패키지가 있어 쌍으로 관리한다.
REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe==0.10.14"),
    ("torch", "torch"),
    ("open_clip", "open-clip-torch==3.3.0"),
    ("fashn_human_parser", "fashn-human-parser==0.1.1"),
]

REQUIRED_FILES = [
    (CORE_DIR / "data" / "products.csv", "상품 카탈로그"),
    (CORE_DIR / "FASHION_RULES_MASTER.md", "패션 규칙 문서"),
]

def _model_path() -> Path:
    """실제로 로드될 체크포인트를 config 에서 가져온다.

    여기에 파일 이름을 하드코딩하면 config 의 기본값이 바뀌었을 때 점검이 엉뚱한
    파일을 보게 된다(실제로 rollback 본을 검사하고 있었다). config 를 못 읽는
    환경에서도 점검 자체는 돌아야 하므로 실패하면 배포 모델 이름으로 떨어진다.
    """
    try:
        sys.path.insert(0, str(CORE_DIR / "src"))
        from config import FASHION_ATTRIBUTE_HEADS_PATH

        return Path(FASHION_ATTRIBUTE_HEADS_PATH)
    except Exception:
        return CORE_DIR / "models" / "fashion_attribute_heads_augmented.pt"


MODEL_PATH = _model_path()


class CheckFailed(Exception):
    """사용자가 고칠 수 있는 환경 문제."""


def check_python() -> str:
    current = sys.version_info[:2]
    if current < MIN_PYTHON or current > MAX_PYTHON:
        raise CheckFailed(
            f"파이썬 {current[0]}.{current[1]}에서는 실행할 수 없습니다.\n"
            f"  이 프로젝트는 파이썬 {MIN_PYTHON[0]}.{MIN_PYTHON[1]} ~ {MAX_PYTHON[0]}.{MAX_PYTHON[1]}만 지원합니다.\n"
            "  mediapipe 0.10.14가 그 밖의 버전용 설치 파일을 제공하지 않기 때문입니다.\n"
            "  python.org에서 3.11을 설치한 뒤 그 파이썬으로 다시 실행하세요."
        )
    return f"파이썬 {current[0]}.{current[1]}"


def check_packages() -> str:
    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_PACKAGES
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise CheckFailed(
            "필요한 패키지가 설치되어 있지 않습니다: " + ", ".join(missing) + "\n"
            "  아래 명령으로 한 번에 설치하세요.\n"
            f'    "{sys.executable}" -m pip install -r "{CORE_DIR / "requirements.txt"}"'
        )
    return f"필수 패키지 {len(REQUIRED_PACKAGES)}종"


def check_files() -> str:
    missing = [f"{label}({path.name})" for path, label in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise CheckFailed(
            "필요한 파일이 없습니다: " + ", ".join(missing) + "\n"
            "  프로젝트 폴더를 통째로 복사했는지 확인하세요."
        )
    return "데이터 파일"


def check_model() -> str:
    if MODEL_PATH.is_file():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        return f"학습된 속성 헤드 ({size_mb:.0f}MB)"
    return (
        "학습된 속성 헤드 없음 → FashionSigLIP 제로샷으로 자동 대체\n"
        f"     (정확한 세부 분석을 하려면 {MODEL_PATH.relative_to(CORE_DIR.parent)} 를 받아 넣으세요)"
    )


def find_free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise CheckFailed(
        f"{preferred}번부터 20개 포트가 모두 사용 중입니다. --port 로 다른 번호를 지정하세요."
    )


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def run_checks() -> None:
    print("환경 점검")
    for check in (check_python, check_packages, check_files, check_model):
        print(f"  [OK] {check()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 코디 추천 웹 서버를 실행합니다.")
    parser.add_argument("--port", type=int, default=8000, help="사용할 포트 (기본 8000)")
    parser.add_argument("--lan", action="store_true", help="같은 네트워크의 다른 기기에서도 접속 허용")
    parser.add_argument("--check", action="store_true", help="서버를 켜지 않고 환경만 점검")
    parser.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    args = parser.parse_args()

    try:
        run_checks()
        if args.check:
            print("\n환경 점검을 통과했습니다. `python web/run_web.py` 로 서버를 켤 수 있습니다.")
            return 0
        port = find_free_port(args.port)
    except CheckFailed as error:
        print(f"\n[실행할 수 없습니다]\n  {error}", file=sys.stderr)
        return 1

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    address = f"http://{local_ip() if args.lan else '127.0.0.1'}:{port}"

    if port != args.port:
        print(f"\n  {args.port}번 포트가 사용 중이라 {port}번으로 대신 실행합니다.")
    print(f"\n주소: {address}")
    if args.lan:
        print("  같은 와이파이에 연결된 기기에서 위 주소로 접속할 수 있습니다.")
    print("  첫 분석은 모델을 내려받고 메모리에 올리느라 1분 이상 걸릴 수 있습니다.")
    print("  종료하려면 Ctrl+C 를 누르세요.\n")

    # 런타임 모듈은 CORE_DIR/src 에 있다. WEB_DIR은 app.py를 찾기 위해 필요하다.
    #
    # web/app.py(FastAPI)와 ai_fashion_recommender/app.py(Gradio)는 이름이 같다.
    # 이 스크립트를 실행하면 파이썬이 WEB_DIR을 sys.path에 미리 넣어두기 때문에
    # "없을 때만 추가" 방식으로는 WEB_DIR이 CORE_DIR보다 뒤로 밀려서
    # `import app`이 Gradio 쪽을 잡는다. 이미 있어도 지웠다가 다시 넣어
    # WEB_DIR이 반드시 맨 앞에 오도록 한다.
    for path in (CORE_DIR, CORE_DIR / "src", WEB_DIR):
        entry = str(path)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

    if not args.no_browser:
        webbrowser.open(address)

    import uvicorn

    uvicorn.run("app:app", host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
