"""수집 로그를 읽어 소스(채널)별 수율을 집계한다.

어떤 채널이 실제로 쓸 만했는지 보고 sources.json 을 정리하는 데 쓴다.
제목 점수는 룩북 채널을 완벽히 가려내지 못하므로, 한 번 돌린 뒤 이 집계로
수율이 0인 채널을 빼는 것이 가장 확실하다.

    python collect/report.py collect/run_male.log collect/run_female.log
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(line_buffering=True)

RE_SOURCE = re.compile(r"^\s*\[소스\]\s*(.+?)\s*$")
RE_ACCEPT = re.compile(r"프레임\s*(\d+)개 검사 -> (\d+)장 채택")
RE_SKIP = re.compile(r"사전선별 통과 실패")


def channel_names() -> dict[str, str]:
    """sources.json 의 url -> 채널명 대응표."""
    try:
        data = json.loads((ROOT / "sources.json").read_text(encoding="utf-8-sig"))
    except OSError:
        return {}
    names = {}
    for gender in ("male", "female"):
        for s in data.get(gender, []):
            if s.get("type") == "url" and s.get("channel"):
                names[s["url"]] = s["channel"]
    return names


def parse(path: Path) -> list[dict]:
    names = channel_names()
    sources: list[dict] = []
    cur: dict | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_SOURCE.match(line)
        if m:
            label = m.group(1)
            cur = {
                "label": names.get(label, label),
                "videos": 0,
                "skipped": 0,
                "frames": 0,
                "images": 0,
            }
            sources.append(cur)
            continue
        if cur is None:
            continue
        if RE_SKIP.search(line):
            cur["videos"] += 1
            cur["skipped"] += 1
            continue
        m = RE_ACCEPT.search(line)
        if m:
            cur["videos"] += 1
            cur["frames"] += int(m.group(1))
            cur["images"] += int(m.group(2))
    return sources


def main() -> int:
    logs = [Path(p) for p in sys.argv[1:]] or [
        ROOT / "run_male.log",
        ROOT / "run_female.log",
    ]
    for log in logs:
        if not log.exists():
            print(f"[skip] 로그 없음: {log}")
            continue
        rows = parse(log)
        print(f"\n=== {log.name} ===")
        print(f"{'소스':38s} {'영상':>4s} {'건너뜀':>5s} {'프레임':>6s} {'수집':>4s}")
        print("-" * 62)
        for r in sorted(rows, key=lambda r: -r["images"]):
            label = r["label"]
            if len(label) > 36:
                label = label[:35] + "…"
            print(
                f"{label:38s} {r['videos']:4d} {r['skipped']:5d} "
                f"{r['frames']:6d} {r['images']:4d}"
            )
        dead = [r["label"] for r in rows if r["images"] == 0 and r["videos"]]
        if dead:
            print("\n수집 0장 - sources.json 에서 빼는 것을 권함:")
            for d in dead:
                print(f"  - {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
