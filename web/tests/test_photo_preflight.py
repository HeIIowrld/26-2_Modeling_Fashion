import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from fastapi import UploadFile

import web.app as web_app


class FakeEngine:
    def __init__(self, passed):
        self.pose_analyzer = SimpleNamespace(analyze=lambda path: SimpleNamespace(valid=passed))
        self.quality_checker = SimpleNamespace(
            check_input=lambda path, pose=None: {
                "passed": passed,
                "issues": [] if passed else ["사진에서 사람의 정면 전신을 확인할 수 없습니다."],
            }
        )


def image_upload():
    buffer = io.BytesIO()
    Image.new("RGB", (320, 640), (220, 220, 220)).save(buffer, format="JPEG")
    buffer.seek(0)
    return UploadFile(file=buffer, filename="person.jpg")


class PhotoPreflightTests(unittest.TestCase):
    def run_validation(self, passed):
        with tempfile.TemporaryDirectory() as temporary:
            old_root = web_app.SESSION_ROOT
            old_engine = web_app.get_engine
            web_app.SESSION_ROOT = Path(temporary)
            web_app.get_engine = lambda: FakeEngine(passed)
            try:
                response = asyncio.run(web_app.validate_photo(image_upload()))
                return json.loads(response.body), list(Path(temporary).iterdir())
            finally:
                web_app.SESSION_ROOT = old_root
                web_app.get_engine = old_engine

    def test_valid_photo_preflight_succeeds_and_cleans_temp_file(self):
        payload, remaining = self.run_validation(True)
        self.assertTrue(payload["valid"])
        self.assertEqual(remaining, [])

    def test_invalid_photo_preflight_fails_and_cleans_temp_file(self):
        payload, remaining = self.run_validation(False)
        self.assertFalse(payload["valid"])
        self.assertIn("정면 전신", payload["issues"][0])
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
