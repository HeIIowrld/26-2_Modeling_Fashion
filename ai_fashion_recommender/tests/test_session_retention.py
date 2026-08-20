import os
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # 런타임 모듈은 src/에 있다
sys.path.insert(0, str(ROOT.parent / "web"))

import app as web_app


def make_session(root: Path, name: str, age_minutes: float = 0.0) -> Path:
    session = root / name
    session.mkdir(parents=True)
    (session / "original.jpg").write_bytes(b"fake-photo")
    if age_minutes:
        old = time.time() - age_minutes * 60
        os.utime(session, (old, old))
    return session


class PruneSessionTests(unittest.TestCase):
    def test_expired_session_photo_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = make_session(root, "a" * 32, age_minutes=45)

            removed = web_app.prune_sessions(root, ttl=timedelta(minutes=30), max_sessions=20)

            self.assertEqual(removed, ["a" * 32])
            self.assertFalse(stale.exists())

    def test_recent_session_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh = make_session(root, "b" * 32, age_minutes=5)

            removed = web_app.prune_sessions(root, ttl=timedelta(minutes=30), max_sessions=20)

            self.assertEqual(removed, [])
            self.assertTrue((fresh / "original.jpg").is_file())

    def test_running_session_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            running = make_session(root, "c" * 32, age_minutes=90)

            removed = web_app.prune_sessions(
                root,
                ttl=timedelta(minutes=30),
                max_sessions=20,
                protected=frozenset({"c" * 32}),
            )

            self.assertEqual(removed, [])
            self.assertTrue(running.exists())

    def test_oldest_sessions_drop_once_the_cap_is_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(5):
                make_session(root, f"{index:032x}", age_minutes=index)

            removed = web_app.prune_sessions(root, ttl=timedelta(hours=24), max_sessions=3)

            self.assertEqual(len(removed), 2)
            self.assertEqual(len(list(root.iterdir())), 3)
            # 남는 것은 가장 최근 3개다.
            self.assertNotIn(f"{4:032x}", [path.name for path in root.iterdir()])

    def test_missing_root_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(web_app.prune_sessions(Path(directory) / "없음"), [])

    def test_loose_files_beside_sessions_are_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            note = root / "README.txt"
            note.write_text("keep me", encoding="utf-8")
            make_session(root, "d" * 32, age_minutes=90)

            web_app.prune_sessions(root, ttl=timedelta(minutes=30), max_sessions=20)

            self.assertTrue(note.is_file())


class PurgeSessionTests(unittest.TestCase):
    def test_photo_files_are_gone_after_purge(self):
        with tempfile.TemporaryDirectory() as directory:
            session = make_session(Path(directory), "e" * 32)
            self.assertTrue(web_app.purge_session(session))
            self.assertFalse(session.exists())

    def test_empty_leftover_folder_still_counts_as_purged(self):
        """동기화 클라이언트가 폴더를 붙잡아도 사진이 사라졌으면 성공이다."""
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / ("f" * 32)
            session.mkdir(parents=True)
            self.assertTrue(web_app.purge_session(session))

    def test_remaining_photo_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            session = make_session(Path(directory), "g" * 32)
            with mock.patch.object(web_app.shutil, "rmtree"):  # 삭제가 실패한 상황
                self.assertFalse(web_app.purge_session(session))

    def test_prune_does_not_report_sessions_it_failed_to_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_session(root, "h" * 32, age_minutes=45)
            with mock.patch.object(web_app.shutil, "rmtree"):
                removed = web_app.prune_sessions(root, ttl=timedelta(minutes=30), max_sessions=20)
            self.assertEqual(removed, [])


class StorageLocationTests(unittest.TestCase):
    def test_photos_are_not_stored_inside_the_project_folder(self):
        """프로젝트가 OneDrive 안에 있으면 사진이 클라우드로 동기화되기 때문이다."""
        project = Path(web_app.WEB_DIR).parent.resolve()
        self.assertNotIn(project, web_app.SESSION_ROOT.resolve().parents)

    def test_legacy_folder_photos_are_purged(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "web_sessions"
            make_session(legacy, "i" * 32)
            make_session(legacy, "j" * 32)

            self.assertEqual(web_app.purge_legacy_sessions(legacy), 2)
            leftover = [f for f in legacy.rglob("*") if f.is_file()] if legacy.exists() else []
            self.assertEqual(leftover, [])

    def test_missing_legacy_folder_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(web_app.purge_legacy_sessions(Path(directory) / "없음"), 0)


class RetentionPolicyTests(unittest.TestCase):
    def test_policy_endpoint_reports_the_configured_window(self):
        policy = web_app.retention()
        self.assertEqual(policy["ttl_minutes"], int(web_app.SESSION_TTL.total_seconds() // 60))
        self.assertEqual(policy["max_sessions"], web_app.MAX_SESSIONS)

    def test_delete_rejects_a_malformed_job_id(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            web_app.delete_job("../../etc")
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
