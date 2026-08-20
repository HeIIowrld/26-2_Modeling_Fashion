"""패션 유튜브 채널을 찾아 sources.json 에 넣을 형태로 정리한다.

검색어로 영상을 훑어 어떤 채널이 반복해서 잡히는지 세고, 채널별 업로드 수와
영상 제목을 확인해 룩북 채널인지 검증한다. 채널명을 추측하지 않고 실제 검색
결과에서만 뽑기 때문에 없는 채널이 들어갈 일이 없다.

    python collect/discover_channels.py                 # 후보 출력만
    python collect/discover_channels.py --write         # sources.json 에 반영
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

from collect import title_rank  # noqa: E402  수집기와 같은 기준으로 제목을 평가

# 채널을 찾기 위한 탐색용 검색어 (수집용 sources.json 과는 별개)
PROBE = {
    "male": [
        "남자 룩북",
        "남자 룩북 4K",
        "남자 데일리룩 코디",
        "남자 여름 룩북 lookbook",
        "남성 캐주얼 코디 추천",
        "men lookbook korean fashion",
        "mens lookbook outfit ideas studio",
    ],
    "female": [
        "여자 룩북",
        "여자 데일리룩 코디",
        "여자 여름 룩북 lookbook",
        "여성 캐주얼 코디 추천",
        "women lookbook korean fashion",
    ],
}

def probe(query: str, limit: int) -> list[dict]:
    from yt_dlp import YoutubeDL

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as exc:
        print(f"  [warn] 검색 실패 ({query}): {exc}")
        return []
    return [e for e in (info.get("entries") or []) if e]


def discover(gender: str, per_query: int, min_hits: int) -> list[dict]:
    channels: dict[str, dict] = defaultdict(
        lambda: {"hits": 0, "score": 0, "titles": [], "url": "", "name": ""}
    )

    for query in PROBE[gender]:
        print(f"  검색: {query}")
        for e in probe(query, per_query):
            url = e.get("channel_url") or e.get("uploader_url")
            name = e.get("channel") or e.get("uploader")
            if not url or not name:
                continue
            ch = channels[url]
            ch["url"] = url
            ch["name"] = name
            ch["hits"] += 1
            ch["score"] += title_rank(e.get("title", ""))
            if len(ch["titles"]) < 3:
                ch["titles"].append(e.get("title", ""))

    ranked = [c for c in channels.values() if c["hits"] >= min_hits and c["score"] > 0]
    ranked.sort(key=lambda c: (c["score"], c["hits"]), reverse=True)
    return ranked


def verify(channels: list[dict], per_channel: int, max_height: int) -> None:
    """후보 채널의 영상을 실제로 받아 전신이 나오는지 확인한다.

    제목만으로는 스튜디오 룩북과 토크·팁 영상을 구분할 수 없다. 저해상도로
    1편씩만 받아 사전선별을 돌리면 채널당 1~2분에 실측할 수 있다.
    """
    import cv2
    import collect as C

    lm = C.build_landmarker(ROOT / "tools" / "pose_landmarker_heavy.task")
    crit = C.Criteria()

    for ch in channels:
        hits = 0
        videos = C.resolve_videos(
            {"type": "url", "url": f"{ch['url']}/videos", "max_videos": per_channel}
        )
        for v in videos:
            path = C.download(v["url"], max_height, 15)
            if path is None:
                continue
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                hits += C.prescreen(
                    cap, lm, total, crit, crit.prescreen_samples, crit.prescreen_hits
                )
            cap.release()
            path.unlink(missing_ok=True)
        ch["verified"] = hits
        mark = "통과" if hits >= crit.prescreen_hits else "탈락"
        print(f"  [{mark}] {ch['name']}: 전신 {hits}프레임 / 영상 {len(videos)}편")


def main() -> int:
    ap = argparse.ArgumentParser(description="패션 유튜브 채널 탐색")
    ap.add_argument("--per-query", type=int, default=15, help="검색어당 확인할 영상 수")
    ap.add_argument("--min-hits", type=int, default=2, help="채널 최소 등장 횟수")
    ap.add_argument("--top", type=int, default=6, help="성별당 채택할 채널 수")
    ap.add_argument("--videos-per-channel", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="sources.json 에 반영")
    ap.add_argument("--sources", default=str(ROOT / "sources.json"))
    ap.add_argument(
        "--verify",
        action="store_true",
        help="후보 채널 영상을 실제로 받아 전신 등장 여부를 확인 (권장, 채널당 1~2분)",
    )
    ap.add_argument("--verify-videos", type=int, default=1)
    ap.add_argument("--gender", choices=("male", "female"), help="한쪽 성별만 탐색")
    args = ap.parse_args()

    found: dict[str, list[dict]] = {}
    for gender in [args.gender] if args.gender else ["male", "female"]:
        print(f"\n=== {gender} 채널 탐색 ===")
        ranked = discover(gender, args.per_query, args.min_hits)
        picked = ranked[: args.top]
        if args.verify and picked:
            print(f"  -- 실측 검증 ({len(picked)}개 채널) --")
            verify(picked, args.verify_videos, 360)
            picked = [c for c in picked if c.get("verified", 0) >= 2]
            print(f"  검증 통과 {len(picked)}개")
        found[gender] = picked
        if not picked:
            print("  조건을 만족하는 채널이 없습니다. --min-hits 를 낮춰보세요.")
        for c in picked:
            print(f"  [{c['score']:+d}점 / {c['hits']}회] {c['name']}")
            print(f"      {c['url']}")
            for t in c["titles"]:
                print(f"      - {t[:60]}")

    if not args.write:
        print("\n반영하려면 --write 옵션을 붙여 다시 실행하세요.")
        return 0

    path = Path(args.sources)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for gender, picked in found.items():
        entries = [
            {
                "type": "url",
                "url": f"{c['url']}/videos",
                "max_videos": args.videos_per_channel,
                "channel": c["name"],
            }
            for c in picked
        ]
        # 기존 채널은 지우지 않고 병합한다. 실측으로 수율이 확인된 채널을
        # 이번 탐색 결과가 비었다는 이유로 잃어버리면 안 된다.
        existing = [s for s in data.get(gender, []) if s.get("type") == "url"]
        new_urls = {e["url"] for e in entries}
        merged = existing + [e for e in entries if e["url"] not in {s["url"] for s in existing}]
        added = len([e for e in entries if e["url"] not in {s["url"] for s in existing}])
        searches = [s for s in data.get(gender, []) if s.get("type") == "search"]
        data[gender] = merged + searches
        print(f"  {gender}: 기존 {len(existing)}개 유지, {added}개 추가 (중복 {len(new_urls) - added}개)")

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nsources.json 갱신 완료: 채널 {sum(len(v) for v in found.values())}개 추가")
    print("이어서 실행하면 중복 없이 누적됩니다:")
    print("  python collect/collect.py --target 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
