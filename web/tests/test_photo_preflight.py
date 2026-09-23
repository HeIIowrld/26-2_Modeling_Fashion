import asyncio
import io
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from fastapi import UploadFile

import web.app as web_app


class FakeEngine:
    def __init__(self, passed):
        import numpy as np
        self.outfit_analyzer = SimpleNamespace(analyze=lambda path, pose: (
            SimpleNamespace(fit="레귤러핏", lower_fit="스트레이트핏", attribute_sources={}),
            {"backend": "fashn-human-parser", "segmentation": np.full((10, 10), 6)}))
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
    def test_skirt_is_rejected_before_conditions_and_temp_photo_is_deleted(self):
        engine = FakeEngine(True)
        outfit, parsed = engine.outfit_analyzer.analyze(None, None)
        parsed['segmentation'][:] = 5
        engine.outfit_analyzer.analyze = Mock(return_value=(outfit, parsed))
        with tempfile.TemporaryDirectory() as temporary, patch.object(web_app, 'SESSION_ROOT', Path(temporary)), patch.object(web_app, 'get_engine', return_value=engine):
            response = asyncio.run(web_app.validate_photo(image_upload()))
            payload = json.loads(response.body)
            self.assertFalse(payload['valid'])
            self.assertEqual(payload['quality']['body_visibility']['status'], 'occluded')
            self.assertIn('치마', payload['issues'][0])
            self.assertEqual(list(Path(temporary).iterdir()), [])

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
