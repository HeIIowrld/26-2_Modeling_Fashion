"""수집된 이미지를 전수 재검사한다.

수집 기준을 바꾼 뒤 이전 결과가 섞여 있을 수 있으므로, 저장된 이미지를 직접
다시 판정한다. 크롭된 이미지가 대상이므로 원본 프레임 기준과는 다른, 크롭에
맞춘 기준을 쓴다.

- 머리·발이 크롭 안에 있는가 (크롭 시 상하 9% 여백을 주므로 잘렸다면 0에 붙는다)
- 정면·서 있는 자세인가
- 인물이 한 명인가
- 출처 영상 제목이 제외 대상(AI 룩북 등)은 아닌가

    python collect/qa.py                 # 검사만
    python collect/qa.py --move          # 불합격을 data/rejected/ 로 이동
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

from collect import (  # noqa: E402
    BODY_REQUIRED, DATA, L_ANK, L_EAR, L_SHO, L_HIP, NOSE, R_ANK, R_EAR, R_SHO, R_HIP,
    build_landmarker, imread_u, landmark_bbox, title_rank,
)


def check(lms_list) -> str:
    """크롭 이미지 기준 판정. 통과면 빈 문자열."""
    if not lms_list:
        return "인물없음"
    heights = [landmark_bbox(p)[3] - landmark_bbox(p)[1] for p in lms_list]
    if sum(1 for h in heights if h > 0.5) > 1:
        return "다중인물"
    lms = lms_list[heights.index(max(heights))]

    for idx in BODY_REQUIRED:
        if lms[idx].visibility < 0.5:
            return "전신아님"

    # 크롭 시 위아래로 9% 여백을 주므로, 원본에서 잘렸다면 여백이 사라져 0에 붙는다
    if lms[NOSE].y < 0.02:
        return "머리잘림"
    if max(lms[L_ANK].y, lms[R_ANK].y) > 0.98:
        return "발잘림"

    sho_w = abs(lms[L_SHO].x - lms[R_SHO].x)
    sho_cy = (lms[L_SHO].y + lms[R_SHO].y) / 2
    hip_cy = (lms[L_HIP].y + lms[R_HIP].y) / 2
    torso = abs(hip_cy - sho_cy)
    if torso < 1e-6 or sho_w / torso < 0.24:
        return "측면/뒷면"
    if min(lms[L_EAR].visibility, lms[R_EAR].visibility) < 0.4:
        return "측면/뒷면"
    return ""


def source_titles(video_ids: set[str]) -> dict[str, str]:
    """출처 영상 제목을 받아 제외 대상인지 확인한다."""
    from yt_dlp import YoutubeDL

    titles = {}
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": True}
    with YoutubeDL(opts) as ydl:
        for vid in sorted(video_ids):
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                titles[vid] = info.get("title", "")
            except Exception:
                titles[vid] = ""
    return titles


def main() -> int:
    ap = argparse.ArgumentParser(description="수집 이미지 전수 재검사")
    ap.add_argument("--move", action="store_true", help="불합격을 data/rejected/ 로 이동")
    ap.add_argument("--skip-titles", action="store_true", help="출처 제목 확인 생략")
    ap.add_argument("--images", default=str(DATA / "images"))
    args = ap.parse_args()

    landmarker = build_landmarker(ROOT / "tools" / "pose_landmarker_heavy.task")
    root = Path(args.images)
    failures: list[tuple[Path, str]] = []
    video_ids: set[str] = set()
    total = 0

    for gender in ("male", "female"):
        folder = root / gender
        files = sorted(folder.glob("*.jpg"))
        bad = 0
        for path in files:
            total += 1
            # 파일명: <성별>_<번호>_<영상ID>_<프레임>.jpg
            parts = path.stem.split("_")
            if len(parts) >= 4:
                video_ids.add("_".join(parts[2:-1]))
            img = imread_u(path)
            if img is None:
                failures.append((path, "읽기실패"))
                bad += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            why = check(res.pose_landmarks)
            if why:
                failures.append((path, why))
                bad += 1
        print(f"{gender}: {len(files)}장 중 {len(files) - bad}장 합격, {bad}장 불합격")

    if not args.skip_titles and video_ids:
        print(f"\n출처 영상 {len(video_ids)}편 제목 확인 중...")
        titles = source_titles(video_ids)
        flagged = {v: t for v, t in titles.items() if title_rank(t) < 0}
        if flagged:
            print("제외 대상 영상에서 나온 이미지가 있습니다:")
            for vid, title in flagged.items():
                print(f"  [{vid}] {title[:60]}")
                for path in root.rglob(f"*_{vid}_*.jpg"):
                    failures.append((path, "제외대상영상"))
        else:
            print("모든 출처 영상이 정상입니다 (AI 룩북 등 제외 대상 없음).")

    print(f"\n합계 {total}장 중 불합격 {len(failures)}장")
    reasons: dict[str, int] = {}
    for _, why in failures:
        reasons[why] = reasons.get(why, 0) + 1
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {why}: {n}")

    if failures:
        report = DATA / "qa_failures.csv"
        with report.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "reason"])
            for path, why in failures:
                w.writerow([path.name, why])
        print(f"목록: {report}")

    if args.move and failures:
        for path, why in failures:
            dst = DATA / "rejected" / path.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            path.rename(dst / path.name)
        print(f"{len(failures)}장을 data/rejected/ 로 옮겼습니다.")
        print("부족분은 다시 실행해 채우세요: python collect/collect.py --target 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
