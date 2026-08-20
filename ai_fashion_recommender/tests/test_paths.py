import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
