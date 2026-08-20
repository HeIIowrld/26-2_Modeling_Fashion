import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fashion_rules import FashionRuleBook


class FashionRuleBookTests(unittest.TestCase):
    def test_loads_rule_metadata_from_markdown(self):
        book = FashionRuleBook.from_markdown(ROOT / "FASHION_RULES_RESEARCH.md")
        self.assertGreaterEqual(len(book.rules), 20)
        self.assertEqual(book.rules["R-SIL-01"].confidence, "높음")
        self.assertEqual(book.rules["R-KOR-01"].evidence, "A + D")

    def test_rejects_markdown_without_rule_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.md"
            path.write_text("# 설명만 있는 문서\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "규칙을 찾지 못했습니다"):
                FashionRuleBook.from_markdown(path)


if __name__ == "__main__":
    unittest.main()
