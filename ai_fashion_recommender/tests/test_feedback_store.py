import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # 런타임 모듈은 src/에 있다

from feedback_store import FeedbackStore


class FeedbackStoreTests(unittest.TestCase):
    def test_append_writes_one_jsonl_record_with_given_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            store = FeedbackStore(path)

            record = store.append(1, "마음에 들어요", note="핏이 좋아요")

            self.assertEqual(record["recommendation_rank"], 1)
            self.assertEqual(record["action"], "마음에 들어요")
            self.assertEqual(record["note"], "핏이 좋아요")
            self.assertIn("timestamp", record)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), record)

    def test_append_defaults_note_to_empty_string(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.jsonl")

            record = store.append(2, "별로예요")

            self.assertEqual(record["note"], "")

    def test_multiple_appends_accumulate_as_separate_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            store = FeedbackStore(path)

            store.append(1, "저장")
            store.append(2, "다른 스타일 요청")

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["action"], "저장")
            self.assertEqual(json.loads(lines[1])["action"], "다른 스타일 요청")

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "feedback.jsonl"
            FeedbackStore(path)

            self.assertTrue(path.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
