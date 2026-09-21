"""휴대폰 사진 업로드 검사.

세로로 찍은 사진은 대개 센서 방향 그대로 저장되고 "돌려서 보라"는 EXIF 태그만
붙는다. 그 태그를 버리면 사진이 누운 채로 분석에 들어가 어깨·골반 위치가
어긋나고 체형 판정이 조용히 틀린다. 에러가 나지 않아 더 위험하다.
"""

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException
from PIL import Image

import web.app as web_app


def _portrait_stored_sideways() -> bytes:
    """가로로 저장하고 '왼쪽으로 90도 돌려서 보라'(Orientation=6)를 붙인 사진."""
    image = Image.new("RGB", (120, 60), "white")
    image.paste(Image.new("RGB", (30, 60), "red"), (0, 0))  # 왼쪽 끝 표시
    exif = image.getexif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


class PhotoUploadFormatTests(unittest.TestCase):
    def test_exif_rotation_is_applied_to_the_pixels(self):
        with TemporaryDirectory() as raw:
            target = Path(raw) / "original.jpg"
            web_app._save_upload(_portrait_stored_sideways(), target, "전신 사진")
            with Image.open(target) as saved:
                # 회전이 실제로 적용됐다면 세로 사진이 된다.
                self.assertEqual((saved.width, saved.height), (60, 120))
                self.assertIsNone(saved.getexif().get(274))

    def test_unsupported_format_is_rejected_with_the_listed_formats(self):
        buffer = BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, "BMP")
        with TemporaryDirectory() as raw:
            with self.assertRaises(HTTPException) as caught:
                web_app._save_upload(buffer.getvalue(), Path(raw) / "x.jpg", "전신 사진")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn(web_app.ALLOWED_FORMATS_LABEL, caught.exception.detail)

    def test_heic_upload_is_stored_as_jpeg(self):
        if not web_app.HEIF_READY:
            self.skipTest("pillow-heif 가 설치되지 않았습니다")
        self.assertIn("HEIF", web_app.ALLOWED_FORMATS)
        self.assertIn("HEIC", web_app.ALLOWED_FORMATS_LABEL)
        buffer = BytesIO()
        Image.new("RGB", (40, 90), "white").save(buffer, "HEIF")
        with TemporaryDirectory() as raw:
            target = Path(raw) / "original.jpg"
            web_app._save_upload(buffer.getvalue(), target, "전신 사진")
            with Image.open(target) as saved:
                self.assertEqual(saved.format, "JPEG")
                self.assertEqual((saved.width, saved.height), (40, 90))

    # HEIC 의 회전은 pillow-heif 가 디코딩 단계에서 적용하고 EXIF 태그를 1 로 정리한다.
    # 그래서 합성 파일로는 재현되지 않는다. 실제 기기 사진으로 한 번 확인해야 한다.

    def test_no_upload_path_bypasses_the_shared_saver(self):
        # 경로마다 따로 열면 HEIC 허용과 회전 보정을 빠뜨린다. 2026-09-21 에 조건 입력 전
        # 사진 검사(/api/validate-photo)가 따로 저장해 아이폰 사진을 첫 단계에서 막았다.
        source = Path(web_app.__file__).read_text(encoding="utf-8")
        self.assertNotIn('opened.convert("RGB").save(', source)
        self.assertGreaterEqual(source.count("_save_upload("), 5)  # 정의 1 + 전신·체형·보유 옷·사진 검사

    def test_photo_validation_stores_rotated_photo_upright(self):
        import asyncio

        class _Upload:
            async def read(self):
                return _portrait_stored_sideways()

        with TemporaryDirectory() as raw:
            target = Path(raw) / "photo.jpg"
            asyncio.run(web_app._store_uploaded_image(_Upload(), target))
            with Image.open(target) as saved:
                self.assertEqual((saved.width, saved.height), (60, 120))


if __name__ == "__main__":
    unittest.main()
