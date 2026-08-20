from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class FeedbackStore:
    """초기에는 JSONL에 피드백을 저장하고, 이후 DB 저장소로 교체한다."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, recommendation_rank: int, action: str, note: str = "") -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendation_rank": recommendation_rank,
            "action": action,
            "note": note,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
