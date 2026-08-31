import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import web.app as web_app


class TryOnPreviewCacheTests(unittest.TestCase):
    def test_rank_one_reuses_the_preview_generated_during_analysis(self):
        session = Path(tempfile.mkdtemp())
        (session / "preview.jpg").write_bytes(b"preview")
        job = {
            "status": "done",
            "recommendations": [SimpleNamespace(rank=1)],
            "result": {
                "tryon": {"available": True, "warnings": ["목 부분 확인 필요"]},
                "images": {"preview": "preview.jpg"},
            },
        }

        with (
            patch.object(web_app, "_read_job", return_value=job),
            patch.object(web_app, "_session_dir", return_value=session),
            patch.object(web_app, "generate_tryon") as generate,
        ):
            response = web_app.create_tryon("a" * 32, 1)

        self.assertEqual(
            response,
            {"image": "preview.jpg", "cached": True, "warnings": ["목 부분 확인 필요"]},
        )
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
