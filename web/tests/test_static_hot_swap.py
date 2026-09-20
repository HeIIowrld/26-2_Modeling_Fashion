"""화면 파일이 심볼릭 링크를 따라가는지 검사한다.

운영은 releases/fitta_current 링크를 바꿔 배포한다. 정적 경로를 기동 시점에
풀어 버리면 링크를 바꿔도 옛 화면이 나가고, 반영하려면 Slurm 작업을 놓고
다시 줄을 서야 한다. 그 회귀를 막는다.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from starlette.staticfiles import StaticFiles


def _can_symlink(root: Path) -> bool:
    try:
        (root / "probe").symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    (root / "probe").unlink()
    return True


class StaticHotSwapTests(unittest.TestCase):
    def test_app_does_not_pin_the_static_directory_to_a_resolved_path(self):
        import web.app as web_app

        # WEB_DIR 은 sys.path 와 옛 세션 폴더에 쓰이므로 풀어도 된다.
        # 정적 경로만은 링크를 유지해야 한다.
        self.assertEqual(web_app.STATIC_DIR, Path(web_app.__file__).parent / "static")
        mounted = next(route for route in web_app.app.routes if getattr(route, "name", "") == "static")
        self.assertEqual(Path(mounted.app.directory), web_app.STATIC_DIR)

    def test_lookup_follows_the_link_after_it_is_repointed(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            if not _can_symlink(root):
                self.skipTest("이 환경에서는 심볼릭 링크를 만들 수 없습니다")
            for name, body in (("old", "이전 화면"), ("new", "새 화면")):
                (root / name / "static").mkdir(parents=True)
                (root / name / "static" / "app.js").write_text(body, encoding="utf-8")
            link = root / "current"
            link.symlink_to(root / "old", target_is_directory=True)

            files = StaticFiles(directory=link / "static")
            first, _ = files.lookup_path("app.js")
            self.assertEqual(Path(first).read_text(encoding="utf-8"), "이전 화면")

            os.remove(link) if link.is_file() else link.unlink()
            link.symlink_to(root / "new", target_is_directory=True)
            second, _ = files.lookup_path("app.js")
            self.assertEqual(Path(second).read_text(encoding="utf-8"), "새 화면")


if __name__ == "__main__":
    unittest.main()
