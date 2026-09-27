"""Manifest validation for the isolated multi-case LVTON runner."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_layering_vton_batch import load_cases  # noqa: E402


class LVTONBatchManifestTests(unittest.TestCase):
    def test_resolves_three_distinct_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "person.jpg").touch()
            (root / "garment.jpg").touch()
            manifest = root / "cases.json"
            entries = [
                {"person": "person.jpg", "garment": "garment.jpg",
                 "output": f"out_{index}.png", "description": "Remove the old coat; apply only the top."}
                for index in range(3)
            ]
            manifest.write_text(json.dumps(entries), encoding="utf-8")
            cases = load_cases(manifest)
            self.assertEqual(len(cases), 3)
            self.assertEqual(cases[0]["output"], root / "out_0.png")

    def test_rejects_colliding_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "person.jpg").touch()
            (root / "garment.jpg").touch()
            manifest = root / "cases.json"
            item = {"person": "person.jpg", "garment": "garment.jpg",
                    "output": "same.png", "description": "Remove the old coat; apply only the top."}
            manifest.write_text(json.dumps([item, item]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(manifest)


if __name__ == "__main__":
    unittest.main()
