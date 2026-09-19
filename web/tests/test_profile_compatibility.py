import asyncio
import io
import json

import pytest
from fastapi import HTTPException, UploadFile
import web.app as web_app


@pytest.mark.parametrize('profile', [
    {'change_scope': '전체 변경'},
    {'change_categories': ['shoes'], 'gender': '남성'},
])
def test_legacy_and_valid_modern_profiles_reach_image_validation(profile):
    with pytest.raises(HTTPException, match='이미지 파일이 비어'):
        asyncio.run(web_app.analyze(image=UploadFile(file=io.BytesIO(b''), filename='test.jpg'),
                    profile=json.dumps(profile), body_image=None, wardrobe_images=[]))


@pytest.mark.parametrize('profile', [
    {}, {'change_categories': ['shoes']}, {'change_categories': ['top'], 'gender': ''},
    {'change_scope': '전체 변경', 'gender': 'invalid'},
])
def test_modern_or_explicit_invalid_gender_is_rejected(profile):
    with pytest.raises(HTTPException, match='성별'):
        asyncio.run(web_app.analyze(image=UploadFile(file=io.BytesIO(b''), filename='test.jpg'),
                    profile=json.dumps(profile), body_image=None, wardrobe_images=[]))
