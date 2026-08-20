import json
import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PROJECT_DIR, resolve_path


class PathConfigurationTests(unittest.TestCase):
    def test_relative_path_uses_given_base_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = resolve_path("images/person.jpg", "unused", directory)
            self.assertEqual(result, (Path(directory) / "images" / "person.jpg").resolve())

    def test_absolute_path_is_not_rebased(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = (Path(directory) / "person.jpg").resolve()
            self.assertEqual(resolve_path(absolute, "unused", PROJECT_DIR), absolute)

    def test_default_path_uses_project_directory(self):
        self.assertEqual(resolve_path(None, "data"), (PROJECT_DIR / "data").resolve())

    def test_main_notebook_uses_master_fashion_rules(self):
        notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("RULES_PATH_INPUT = r'FASHION_RULES_MASTER.md'", source)
        self.assertIn(
            "resolve_local_path(RULES_PATH_INPUT, 'FASHION_RULES_MASTER.md'",
            source,
        )
        self.assertNotIn("FASHION_RULES_RESEARCH.md", source)

    def test_main_notebook_exposes_trained_attribute_head_path(self):
        notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("ATTRIBUTE_HEADS_PATH_INPUT = r'models/fashion_attribute_heads.pt'", source)
        self.assertIn("attribute_checkpoint=ATTRIBUTE_HEADS_PATH", source)


if __name__ == "__main__":
    unittest.main()
