import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fashion_rules import FashionRuleBook


class FashionRuleBookTests(unittest.TestCase):
    def test_loads_rule_metadata_from_markdown(self):
        # 조사 원본 문서는 저장소마다 docs/ 아래에 있기도 하고 루트에 있기도 하다.
        candidates = [ROOT / "docs" / "FASHION_RULES_RESEARCH.md", ROOT / "FASHION_RULES_RESEARCH.md"]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            self.skipTest("FASHION_RULES_RESEARCH.md가 없습니다. 웹 실행에는 필요하지 않습니다.")
        book = FashionRuleBook.from_markdown(source)
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
