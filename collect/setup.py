"""수집 파이프라인 준비 스크립트.

- MediaPipe PoseLandmarker 모델(.task) 다운로드
- imageio-ffmpeg 번들 바이너리를 yt-dlp가 인식하는 이름(ffmpeg.exe)으로 복사

    python collect/setup.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"

MODELS = {
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def fetch_models() -> None:
    TOOLS.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dst = TOOLS / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {name} (이미 있음, {dst.stat().st_size / 1e6:.1f}MB)")
            continue
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"[ok  ] {name} ({dst.stat().st_size / 1e6:.1f}MB)")


def stage_ffmpeg() -> None:
    """yt-dlp는 basename으로 ffmpeg 여부를 판단하므로 표준 이름으로 복사한다."""
    dst = TOOLS / "ffmpeg.exe"
    if dst.exists():
        print(f"[skip] ffmpeg.exe (이미 있음)")
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        print("[warn] imageio-ffmpeg 없음 - 영상 병합이 필요한 포맷은 건너뜁니다.")
        return
    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shutil.copy2(src, dst)
    print(f"[ok  ] ffmpeg.exe <- {src}")


def check_imports() -> bool:
    ok = True
    for mod in ("cv2", "numpy", "mediapipe", "yt_dlp"):
        try:
            __import__(mod)
            print(f"[ok  ] import {mod}")
        except ImportError as exc:
            print(f"[FAIL] import {mod}: {exc}")
            ok = False
    return ok


if __name__ == "__main__":
    if not check_imports():
        print("\n다음 명령으로 의존성을 설치하세요:")
        print("  python -m pip install -r collect/requirements.txt")
        sys.exit(1)
    fetch_models()
    stage_ffmpeg()
    print("\n준비 완료. 다음 단계:")
    print("  1) collect/sources.json 에서 수집할 유튜브 영상/검색어를 확인·수정")
    print("  2) python collect/collect.py --target 50")
