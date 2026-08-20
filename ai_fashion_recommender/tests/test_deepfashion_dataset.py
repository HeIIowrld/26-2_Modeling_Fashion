import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepfashion_dataset import evaluate_deepfashion_predictions, load_deepfashion_multimodal


class DeepFashionDatasetTests(unittest.TestCase):
    def test_load_and_evaluate_official_text_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "image"
            images.mkdir()
            Image.new("RGB", (8, 8), "white").save(images / "person.jpg")
            shape = root / "shape.txt"
            fabric = root / "fabric.txt"
            pattern = root / "color.txt"
            shape.write_text("person.jpg 1 3 0 0 0 0 0 0 0 2 0 1\n", encoding="utf-8")
            fabric.write_text("person.jpg 1 6 6\n", encoding="utf-8")
            pattern.write_text("person.jpg 4 3 3\n", encoding="utf-8")

            records = load_deepfashion_multimodal(images, shape, fabric, pattern)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sleeve_length, "반팔")
            self.assertEqual(records[0].bottom_length, "긴 기장")
            self.assertEqual(records[0].neckline, "라운드넥")
            self.assertEqual(records[0].upper_material, "코튼")
            self.assertEqual(records[0].upper_pattern, "체크")

            result = evaluate_deepfashion_predictions(
                records,
                lambda _: {
                    "sleeve_length": "반팔",
                    "bottom_length": "긴바지",
                    "neckline": "라운드넥",
                    "material": "코튼 추정",
                    "pattern": "체크",
                },
            )
            self.assertEqual(result["processed_images"], 1)
            self.assertTrue(all(metric["accuracy"] == 1.0 for metric in result["metrics"].values()))
            self.assertTrue(all(metric["coverage"] == 1.0 for metric in result["metrics"].values()))
            self.assertTrue(all(metric["macro_f1"] == 1.0 for metric in result["metrics"].values()))

            abstained = evaluate_deepfashion_predictions(
                records,
                lambda _: {
                    "sleeve_length": "분석 보류",
                    "bottom_length": "분석 보류",
                    "neckline": "분석 보류",
                    "material": "분석 보류",
                    "pattern": "분석 보류",
                },
            )
            self.assertTrue(all(metric["coverage"] == 0.0 for metric in abstained["metrics"].values()))
            self.assertTrue(all(metric["overall_accuracy"] == 0.0 for metric in abstained["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
