"""유튜브 룩북 영상에서 정면 전신 패션 사진을 자동 수집한다.

파이프라인:
    sources.json -> yt-dlp 다운로드 -> 프레임 샘플링 -> MediaPipe Pose 필터
    -> 중복 제거 -> 인물 크롭 저장 -> manifest.csv

README 2단계(체형·자세 분석)가 요구하는 조건 - 정면, 전신, 몸/옷 경계가 뚜렷한
프레임 - 을 그대로 통과 기준으로 사용한다.

    python collect/setup.py            # 최초 1회
    python collect/collect.py --target 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.stdout.reconfigure(line_buffering=True)  # 로그로 리다이렉트해도 진행상황이 보이게

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TOOLS = ROOT / "tools"
DATA = PROJECT / "data"
GENDERS = ("male", "female")

# 이 프로젝트 경로에는 한글이 들어 있다. OpenCV VideoCapture 와 yt-dlp 는
# 비ASCII 경로에서 불안정하므로 영상은 ASCII 임시 경로에 받아서 처리한다.
VIDEO_DIR = Path(tempfile.gettempdir()) / "fashion_collect_video"

# BlazePose 33 landmark 인덱스
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHO, R_SHO = 11, 12
L_WRI, R_WRI = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28
BODY_REQUIRED = (L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)

# 제목으로 룩북 영상을 고른다. 다운로드 전에 거르는 것이 가장 큰 시간 절약이다.
GOOD_WORDS = (
    "룩북", "lookbook", "코디", "데일리룩", "outfit", "ootd", "스타일링",
    "styling", "출근룩", "하객룩", "데이트룩", "여름옷", "겨울옷", "착장",
)
# 부분일치로 걸러지므로 다른 단어에 포함되지 않는 표현만 넣는다.
# ("리뷰"는 "프리뷰"에, "총정리"는 정상적인 룩북 모음 영상 제목에 걸린다)
BAD_WORDS = (
    "하울", "언박싱", "브이로그", "vlog", "메이크업", "makeup",
    "q&a", "먹방", "asmr", "shorts", "다이어트", "헤어스타일", "향수",
    # AI 생성 룩북은 실제 인물이 아니라 체형·착장 학습 데이터로 쓸 수 없다
    "ai lookbook", "ai룩북", "ai 룩북", "ai generated", "ai model",
    "midjourney", "stable diffusion", "가상인간", "버추얼",
)


def title_rank(title: str) -> int:
    """1=룩북 유력, 0=보통, -1=정면 전신이 거의 없는 유형."""
    t = (title or "").lower()
    if any(w in t for w in BAD_WORDS):
        return -1
    return 1 if any(w in t for w in GOOD_WORDS) else 0


# --------------------------------------------------------------------------
# 필터 기준
# --------------------------------------------------------------------------
@dataclass
class Criteria:
    min_visibility: float = 0.55       # 전신 관절 최소 신뢰도
    min_person_height: float = 0.55    # 인물 높이 / 프레임 높이
    min_shoulder_ratio: float = 0.26   # 어깨너비 / 상체길이 (측면일수록 작아짐)
    max_shoulder_tilt: float = 0.28    # 어깨 기울기 (|dy|/dx)
    min_sharpness: float = 55.0        # Laplacian 분산
    min_face_margin: float = 0.01      # 머리 위 여백
    max_foot_y: float = 0.995          # 발이 프레임 안에 있어야 함
    dedup_distance: int = 12           # dHash 해밍거리 하한
    min_out_height: int = 480          # 크롭 결과 최소 높이(px)
    max_second_person: float = 0.38    # 배경 인물 허용 크기 (높이 비율)
    subtitle_check: bool = True        # 하단 박힌 자막 프레임 제외
    prescreen_samples: int = 16        # 영상 사전 선별 샘플 수 (0이면 끔)
    prescreen_hits: int = 2            # 사전 선별 통과에 필요한 전신 프레임 수


@dataclass
class Stats:
    scanned: int = 0
    reasons: dict = field(default_factory=dict)

    def reject(self, why: str) -> None:
        self.reasons[why] = self.reasons.get(why, 0) + 1

    def summary(self) -> str:
        if not self.reasons:
            return "-"
        items = sorted(self.reasons.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{k}:{v}" for k, v in items)


# --------------------------------------------------------------------------
# 한글 경로 대응 입출력 (cv2.imread/imwrite 는 비ASCII 경로를 못 다룬다)
# --------------------------------------------------------------------------
def imread_u(path: Path) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# --------------------------------------------------------------------------
# 프레임 판정
# --------------------------------------------------------------------------
def dhash(img: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = small[:, 1:] > small[:, :-1]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def has_subtitle(crop: np.ndarray) -> bool:
    """하단에 박힌 자막(흰 글자)을 글자 뭉치의 베이스라인 정렬로 검출한다.

    유튜브 패션 영상은 자막이 구워진 경우가 많고, 자막은 하의·신발 위에
    겹쳐 의류 파싱을 방해하므로 해당 프레임은 버린다.
    """
    h, w = crop.shape[:2]
    band = crop[int(h * 0.74):, :]
    bh = band.shape[0]
    if bh < 40 or w < 40:
        return False

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 226, 255, cv2.THRESH_BINARY)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    blobs = []
    for i in range(1, n):
        x, y, bw, bhh, area = stats[i]
        if not (0.07 * bh <= bhh <= 0.45 * bh):      # 글자 크기대
            continue
        if not (0.2 <= bw / max(bhh, 1) <= 3.0):     # 글자 비율
            continue
        if area < 0.12 * bw * bhh:                   # 속이 빈 잡음 제거
            continue
        blobs.append((x, x + bw, y + bhh, bhh))

    # 같은 베이스라인에 글자 4개 이상이 가로로 넓게 늘어서면 자막으로 본다
    for _, _, base, ref_h in blobs:
        line = [b for b in blobs if abs(b[2] - base) <= max(4, 0.12 * ref_h)]
        if len(line) >= 4:
            span = max(b[1] for b in line) - min(b[0] for b in line)
            if span >= 0.30 * w:
                return True
    return False


def landmark_bbox(lms) -> tuple[float, float, float, float]:
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    return min(xs), min(ys), max(xs), max(ys)


def in_frame(lm, margin: float = 0.004) -> bool:
    """관절이 화면 안에 실제로 있는지 확인한다.

    MediaPipe 는 화면 밖 관절도 위치를 추정해 visibility 를 높게 돌려준다.
    (실측: 미디엄 샷에서 인물 박스 높이가 1.2를 넘고 y가 음수로 나온다)
    따라서 visibility 만으로는 전신 여부를 판정할 수 없고 좌표를 함께 봐야 한다.
    """
    return -margin <= lm.x <= 1 + margin and margin <= lm.y <= 1 - margin


def quick_fullbody(lms, c: Criteria) -> bool:
    """사전 선별용 완화 판정 - 머리와 발이 화면 안에 있는가."""
    return (
        lms[NOSE].y > c.min_face_margin
        and max(lms[L_ANK].y, lms[R_ANK].y) < c.max_foot_y
        and min(lms[i].visibility for i in BODY_REQUIRED) >= 0.5
    )


def judge(lms, frame: np.ndarray, c: Criteria) -> tuple[bool, str]:
    """정면 전신 여부를 판정한다. (통과여부, 탈락사유)"""
    # 1) 전신 관절이 모두 잡히고, 추정이 아니라 실제로 화면 안에 있어야 한다
    for idx in BODY_REQUIRED:
        if lms[idx].visibility < c.min_visibility:
            return False, "전신아님"
        if not in_frame(lms[idx]):
            return False, "프레임밖"
    if lms[NOSE].visibility < c.min_visibility:
        return False, "얼굴가림"

    x0, y0, x1, y1 = landmark_bbox(lms)

    # 2) 인물이 충분히 크게 나와야 옷 디테일이 남는다
    if (y1 - y0) < c.min_person_height:
        return False, "인물작음"

    # 3) 머리와 발이 프레임 안에 있어야 한다 (잘린 전신 제외)
    if lms[NOSE].y < c.min_face_margin:
        return False, "머리잘림"
    if max(lms[L_ANK].y, lms[R_ANK].y) > c.max_foot_y:
        return False, "발잘림"

    # 4) 정면 판정: 어깨너비 대비 상체길이, 코 위치, 양쪽 귀 노출
    sho_w = abs(lms[L_SHO].x - lms[R_SHO].x)
    sho_cy = (lms[L_SHO].y + lms[R_SHO].y) / 2
    hip_cy = (lms[L_HIP].y + lms[R_HIP].y) / 2
    torso = abs(hip_cy - sho_cy)
    if torso < 1e-6 or sho_w / torso < c.min_shoulder_ratio:
        return False, "측면/뒷면"
    if min(lms[L_EAR].visibility, lms[R_EAR].visibility) < 0.45:
        return False, "측면/뒷면"

    lo, hi = sorted((lms[L_SHO].x, lms[R_SHO].x))
    span = hi - lo
    if not (lo - 0.25 * span) <= lms[NOSE].x <= (hi + 0.25 * span):
        return False, "몸틀어짐"

    # 5) 어깨가 심하게 기울면 앉거나 누운 자세 - 비율 계산이 망가진다
    if sho_w > 1e-6 and abs(lms[L_SHO].y - lms[R_SHO].y) / sho_w > c.max_shoulder_tilt:
        return False, "자세불량"

    # 6) 무릎이 골반 아래 - 서 있는 자세
    if min(lms[L_KNEE].y, lms[R_KNEE].y) < hip_cy:
        return False, "자세불량"

    # 7) 손을 어깨 위로 올린 프레임 제외.
    #    옷을 들어 보이는 장면이 많아 상의가 가려지고, 어깨너비 기준의
    #    체형 비율 계산도 왜곡된다.
    if min(lms[L_WRI].y, lms[R_WRI].y) < sho_cy:
        return False, "손올림"

    # 8) 흔들린 프레임 제거
    h, w = frame.shape[:2]
    cx0, cy0 = int(max(x0, 0) * w), int(max(y0, 0) * h)
    cx1, cy1 = int(min(x1, 1) * w), int(min(y1, 1) * h)
    if cx1 - cx0 < 20 or cy1 - cy0 < 20:
        return False, "인물작음"
    gray = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < c.min_sharpness:
        return False, "흔들림"

    return True, ""


def crop_person(frame: np.ndarray, lms, c: Criteria) -> np.ndarray | None:
    """인물 바운딩박스를 3:4 비율로 확장해 잘라낸다."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = landmark_bbox(lms)
    px, py = 0.10 * (x1 - x0), 0.09 * (y1 - y0)
    x0, x1 = (x0 - px) * w, (x1 + px) * w
    y0, y1 = (y0 - py) * h, (y1 + py) * h

    box_h = y1 - y0
    want_w = box_h * 0.75
    if want_w > (x1 - x0):
        cx = (x0 + x1) / 2
        x0, x1 = cx - want_w / 2, cx + want_w / 2

    x0, y0 = int(max(x0, 0)), int(max(y0, 0))
    x1, y1 = int(min(x1, w)), int(min(y1, h))
    if y1 - y0 < c.min_out_height or x1 - x0 < 40:
        return None
    return frame[y0:y1, x0:x1].copy()


# --------------------------------------------------------------------------
# 유튜브
# --------------------------------------------------------------------------
def ydl_common() -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "retries": 3}
    ffmpeg = TOOLS / "ffmpeg.exe"
    if ffmpeg.exists():
        opts["ffmpeg_location"] = str(ffmpeg)
    return opts


def resolve_videos(source: dict) -> list[dict]:
    """sources.json 항목 하나를 영상 목록으로 펼친다.

    제목이 룩북 계열이 아닌 영상은 아예 받지 않는다. 채널 목록은 넉넉히
    가져와 제목으로 추린 뒤 max_videos 만큼만 남긴다.
    """
    from yt_dlp import YoutubeDL

    limit = int(source.get("max_videos", 5))
    if source.get("type") == "search":
        target = f"ytsearch{limit * 3}:{source['query']}"
        fetch = limit * 3
    else:
        target = source["url"]
        fetch = limit * 4  # 채널은 넉넉히 훑어 제목으로 고른다

    opts = ydl_common() | {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "playlistend": fetch,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=False)
    except Exception as exc:
        print(f"    [warn] 소스 해석 실패 ({target}): {exc}")
        return []

    picked, skipped = [], 0
    for e in info.get("entries") or [info]:
        if not e:
            continue
        vid = e.get("id")
        if not vid or len(vid) != 11:
            continue
        title = e.get("title", "")
        rank = title_rank(title)
        if rank < 0:
            skipped += 1
            continue
        picked.append(
            {"url": f"https://www.youtube.com/watch?v={vid}", "title": title, "rank": rank}
        )

    picked.sort(key=lambda v: -v["rank"])  # 룩북 유력 영상부터
    if skipped:
        print(f"    제목 기준 {skipped}편 제외, {len(picked)}편 중 {limit}편 사용")
    return picked[:limit]


def download(url: str, max_height: int, max_minutes: int) -> Path | None:
    """오디오 없이 영상만 받는다(병합 불필요). 이미 받았으면 재사용한다."""
    from yt_dlp import YoutubeDL

    vid = url.rsplit("=", 1)[-1]
    for existing in VIDEO_DIR.glob(f"{vid}.*"):
        return existing

    opts = ydl_common() | {
        "format": (
            f"bv*[ext=mp4][height<={max_height}]/"
            f"bv*[height<={max_height}]/b[ext=mp4]/b"
        ),
        "outtmpl": str(VIDEO_DIR / "%(id)s.%(ext)s"),
        "match_filter": lambda i, *, incomplete=False: (
            None
            if 20 < (i.get("duration") or 0) <= max_minutes * 60
            else "길이 조건 미충족"
        ),
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        print(f"    [warn] 다운로드 실패: {exc}")
        return None

    files = list(VIDEO_DIR.glob(f"{vid}.*"))
    return files[0] if files else None


# --------------------------------------------------------------------------
# 영상 1편 처리
# --------------------------------------------------------------------------
def prescreen(cap, landmarker, total: int, c: Criteria, samples: int, min_hits: int) -> int:
    """영상 전체를 훑기 전에 균등 샘플로 전신 등장 빈도를 본다.

    토크·리뷰 위주 영상은 400프레임을 검사해도 0장이 나온다. 16프레임만 먼저
    확인해 전신이 거의 없는 영상은 건너뛰는 편이 훨씬 빠르다.
    """
    hits = 0
    start, end = int(total * 0.06), int(total * 0.95)
    step = max((end - start) // max(samples, 1), 1)
    for pos in range(start, end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.pose_landmarks and any(quick_fullbody(p, c) for p in res.pose_landmarks):
            hits += 1
            if hits >= min_hits:
                return hits
    return hits


def scan_video(
    path: Path,
    landmarker,
    out_dir: Path,
    gender: str,
    url: str,
    quota: int,
    interval: float,
    c: Criteria,
    hashes: list[int],
    rows: list[dict],
    reject_dir: Path | None,
) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"    [warn] 영상 열기 실패: {path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 0

    if c.prescreen_samples:
        hits = prescreen(cap, landmarker, total, c, c.prescreen_samples, c.prescreen_hits)
        if hits < c.prescreen_hits:
            cap.release()
            print(f"    사전선별 통과 실패 ({hits}/{c.prescreen_samples}) - 건너뜀")
            return 0

    step = max(int(fps * interval), 1)
    start, end = int(total * 0.04), int(total * 0.97)  # 인트로/아웃트로 제외
    stats = Stats()
    saved = 0

    for pos in range(start, end, step):
        if saved >= quota:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        stats.scanned += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        poses = result.pose_landmarks
        if not poses:
            stats.reject("인물없음")
            continue

        # 큰 인물이 둘 이상이면 누구의 코디인지 모호하다
        big = [
            p for p in poses
            if (landmark_bbox(p)[3] - landmark_bbox(p)[1]) > c.max_second_person
        ]
        if len(big) > 1:
            stats.reject("다중인물")
            continue

        lms = big[0] if big else poses[0]
        ok_frame, why = judge(lms, frame, c)
        if not ok_frame:
            stats.reject(why)
            continue

        crop = crop_person(frame, lms, c)
        if crop is None:
            stats.reject("크롭작음")
            continue

        if c.subtitle_check and has_subtitle(crop):
            stats.reject("자막")
            continue

        h = dhash(crop)
        if any(hamming(h, prev) < c.dedup_distance for prev in hashes):
            stats.reject("중복")
            continue

        idx = len(hashes) + 1
        name = f"{gender}_{idx:03d}_{path.stem}_{pos}.jpg"
        if not imwrite_u(out_dir / name, crop):
            stats.reject("저장실패")
            continue
        hashes.append(h)
        rows.append(
            {
                "file": f"{gender}/{name}",
                "gender": gender,
                "video_id": path.stem,
                "source_url": url,
                "timestamp_sec": round(pos / fps, 2),
                "width": crop.shape[1],
                "height": crop.shape[0],
            }
        )
        saved += 1

    cap.release()
    print(f"    프레임 {stats.scanned}개 검사 -> {saved}장 채택 | 탈락: {stats.summary()}")
    return saved


# --------------------------------------------------------------------------
def used_video_ids(images_root: Path) -> set[str]:
    """이미 수집에 쓴 영상 ID. 파일명 `<성별>_<번호>_<영상ID>_<프레임>.jpg` 에서 뽑는다.

    탈락 이미지도 포함해야 같은 영상을 다시 받지 않는다.
    """
    ids: set[str] = set()
    for folder in (images_root, DATA / "rejected"):
        for path in folder.rglob("*.jpg"):
            parts = path.stem.split("_")
            if len(parts) >= 4 and parts[-1].isdigit():
                ids.add("_".join(parts[2:-1]))
    return ids


def append_manifest(path: Path, rows: list[dict]) -> None:
    """수집 기록을 즉시 덧붙인다.

    마지막에 한 번만 쓰면 실행을 중단했을 때 출처 기록이 통째로 사라진다.
    """
    if not rows:
        return
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_landmarker(model: Path):
    # 경로 대신 버퍼로 넘긴다 - 한글 경로에서 model_asset_path 는 열리지 않는다.
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model.read_bytes()),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main() -> int:
    global VIDEO_DIR

    ap = argparse.ArgumentParser(description="유튜브 정면 전신 패션 사진 자동 수집")
    ap.add_argument("--target", type=int, default=50, help="성별당 목표 장수")
    ap.add_argument("--per-video", type=int, default=6, help="영상 1편당 최대 장수")
    ap.add_argument("--interval", type=float, default=1.5, help="프레임 샘플링 간격(초)")
    ap.add_argument("--max-height", type=int, default=720, help="다운로드 최대 해상도")
    ap.add_argument("--max-minutes", type=int, default=15, help="영상 최대 길이(분)")
    ap.add_argument("--gender", choices=GENDERS, help="한쪽 성별만 수집")
    ap.add_argument("--model", default="heavy", choices=("heavy", "lite"))
    ap.add_argument("--sources", default=str(ROOT / "sources.json"))
    ap.add_argument("--out", default=str(DATA / "images"))
    ap.add_argument(
        "--video-dir",
        default=str(VIDEO_DIR),
        help="영상 임시 저장 경로 (한글이 없는 경로여야 한다)",
    )
    ap.add_argument("--keep-video", action="store_true", help="원본 영상 유지")
    ap.add_argument(
        "--allow-subtitle", action="store_true", help="박힌 자막이 있는 프레임도 허용"
    )
    ap.add_argument(
        "--new-videos-only",
        action="store_true",
        help="이미 수집에 쓴 영상은 건너뛴다 (착장 다양성 확보)",
    )
    ap.add_argument(
        "--min-person-height",
        type=float,
        help="인물 높이 / 프레임 높이 하한 (기본 0.55, 낮출수록 수집량 증가)",
    )
    args = ap.parse_args()

    VIDEO_DIR = Path(args.video_dir)
    if not str(VIDEO_DIR).isascii():
        print(f"[warn] 영상 경로에 비ASCII 문자가 있어 영상 읽기가 실패할 수 있습니다: {VIDEO_DIR}")

    model = TOOLS / f"pose_landmarker_{args.model}.task"
    if not model.exists():
        print(f"모델이 없습니다: {model}\n먼저 `python collect/setup.py` 를 실행하세요.")
        return 1

    # utf-8-sig: 메모장·PowerShell 로 편집하면 BOM 이 붙는다
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8-sig"))
    out_root = Path(args.out)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.gender}" if args.gender else ""
    manifest = DATA / f"manifest{suffix}.csv"
    used = used_video_ids(out_root) if args.new_videos_only else set()
    if used:
        print(f"기존에 쓴 영상 {len(used)}편은 건너뜁니다.")
    c = Criteria(subtitle_check=not args.allow_subtitle)
    if args.min_person_height:
        c.min_person_height = args.min_person_height
    landmarker = build_landmarker(model)
    targets = [args.gender] if args.gender else list(GENDERS)
    rows: list[dict] = []
    started = time.time()

    for gender in targets:
        out_dir = out_root / gender
        out_dir.mkdir(parents=True, exist_ok=True)
        hashes: list[int] = []

        # 이어서 수집할 수 있도록 기존 결과를 해시에 반영한다
        for old in sorted(out_dir.glob("*.jpg")):
            img = imread_u(old)
            if img is not None:
                hashes.append(dhash(img))
        kept = len(hashes)

        # qa.py 로 걸러낸 이미지도 해시에 넣어 같은 프레임을 다시 받지 않게 한다.
        # 목표 장수 계산에는 넣지 않으므로 별도로 센다.
        rejected_hashes: list[int] = []
        for old in sorted((DATA / "rejected" / gender).glob("*.jpg")):
            img = imread_u(old)
            if img is not None:
                rejected_hashes.append(dhash(img))
        hashes.extend(rejected_hashes)
        saved_total = kept  # 목표 대비 진행도는 실제 보관 장수로 센다

        if kept or rejected_hashes:
            print(
                f"[{gender}] 기존 {kept}장, 탈락 {len(rejected_hashes)}장 - "
                f"중복 없이 이어서 수집합니다."
            )

        print(f"\n=== {gender} : 목표 {args.target}장 ===")
        for source in sources.get(gender, []):
            if saved_total >= args.target:
                break
            label = source.get("query") or source.get("url", "?")
            print(f"  [소스] {label}")
            for video in resolve_videos(source):
                if saved_total >= args.target:
                    break
                url = video["url"]
                if url.rsplit("=", 1)[-1] in used:
                    continue
                path = download(url, args.max_height, args.max_minutes)
                if path is None:
                    continue
                print(f"    {video['title'][:55]}")
                quota = min(args.per_video, args.target - saved_total)
                fresh: list[dict] = []
                saved_total += scan_video(
                    path, landmarker, out_dir, gender, url, quota,
                    args.interval, c, hashes, fresh, None,
                )
                append_manifest(manifest, fresh)  # 영상 단위로 즉시 기록
                rows.extend(fresh)
                if not args.keep_video:
                    path.unlink(missing_ok=True)
                print(f"    누적 {saved_total}/{args.target}장")

        if saved_total < args.target:
            print(
                f"[{gender}] {saved_total}/{args.target}장에서 소스가 소진되었습니다. "
                f"sources.json 에 영상을 추가하거나 --per-video 를 올려 다시 실행하세요."
            )

    if rows:
        print(f"\nmanifest: {manifest} (+{len(rows)}행)")

    print(f"이번 실행 {len(rows)}장 수집, {time.time() - started:.0f}초 소요")
    print(f"저장 위치: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
