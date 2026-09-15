from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))

from run_web import load_server_env


class ServerEnvTests(unittest.TestCase):
    def test_loads_allowlisted_values_without_overwriting_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "GEMINI_API_KEY=\"file-key\"\n"
                "FASHION_LLM_REASONS=1\n"
                "IGNORED_SECRET=do-not-load\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GEMINI_API_KEY": "shell-key"}, clear=True):
                loaded = load_server_env(path)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "shell-key")
                self.assertEqual(os.environ["FASHION_LLM_REASONS"], "1")
                self.assertNotIn("IGNORED_SECRET", os.environ)
                self.assertEqual(loaded, ["FASHION_LLM_REASONS"])

    def test_rejects_malformed_active_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GEMINI_API_KEY\n", encoding="utf-8")
            with self.assertRaises(Exception):
                load_server_env(path)


if __name__ == "__main__":
    unittest.main()
