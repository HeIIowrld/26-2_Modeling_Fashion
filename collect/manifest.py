"""이미지 파일명에서 출처 기록(manifest)을 다시 만든다.

파일명이 `<성별>_<번호>_<영상ID>_<프레임>.jpg` 형식이라 출처를 복원할 수 있다.
수집 도중 실행을 중단해 기록이 빠졌거나, 여러 번 나눠 돌려 manifest 가 흩어졌을
때 하나로 정리한다.

    python collect/manifest.py            # data/manifest.csv 재생성
    python collect/manifest.py --titles   # 영상 제목까지 채움 (네트워크 필요)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

from collect import DATA, imread_u  # noqa: E402


def parse_name(path: Path) -> dict | None:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    gender, _, frame = parts[0], parts[1], parts[-1]
    video_id = "_".join(parts[2:-1])
    if not frame.isdigit():
        return None
    return {"gender": gender, "video_id": video_id, "frame": int(frame)}


def fetch_titles(video_ids: list[str]) -> dict[str, str]:
    from yt_dlp import YoutubeDL

    titles: dict[str, str] = {}
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": True}
    with YoutubeDL(opts) as ydl:
        for i, vid in enumerate(video_ids, 1):
            try:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={vid}", download=False
                )
                titles[vid] = info.get("title", "")
            except Exception:
                titles[vid] = ""
            print(f"  {i}/{len(video_ids)} {vid}: {titles[vid][:45]}")
    return titles


def main() -> int:
    ap = argparse.ArgumentParser(description="이미지에서 manifest 재생성")
    ap.add_argument("--titles", action="store_true", help="영상 제목도 채운다")
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--out", default=str(DATA / "manifest.csv"))
    args = ap.parse_args()

    root = Path(args.images)
    rows: list[dict] = []
    for gender in ("male", "female"):
        for path in sorted((root / gender).glob("*.jpg")):
            meta = parse_name(path)
            if meta is None:
                print(f"[warn] 파일명 형식이 다릅니다: {path.name}")
                continue
            img = imread_u(path)
            h, w = (img.shape[:2] if img is not None else (0, 0))
            rows.append(
                {
                    "file": f"{gender}/{path.name}",
                    "gender": gender,
                    "video_id": meta["video_id"],
                    "source_url": f"https://www.youtube.com/watch?v={meta['video_id']}",
                    "title": "",
                    "frame": meta["frame"],
                    "width": w,
                    "height": h,
                }
            )

    if args.titles and rows:
        ids = sorted({r["video_id"] for r in rows})
        print(f"영상 {len(ids)}편 제목 조회 중...")
        titles = fetch_titles(ids)
        for r in rows:
            r["title"] = titles.get(r["video_id"], "")

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    videos = len({r["video_id"] for r in rows})
    print(f"\n{out}: {len(rows)}장 / 출처 영상 {videos}편")
    for gender in ("male", "female"):
        n = sum(1 for r in rows if r["gender"] == gender)
        v = len({r["video_id"] for r in rows if r["gender"] == gender})
        print(f"  {gender}: {n}장 (영상 {v}편)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
