import socket
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # 런타임 모듈은 src/에 있다
sys.path.insert(0, str(ROOT.parent / "web"))

import run_web


class PythonVersionCheckTests(unittest.TestCase):
    def test_supported_version_passes(self):
        with mock.patch.object(run_web.sys, "version_info", (3, 11, 6)):
            self.assertIn("3.11", run_web.check_python())

    def test_too_new_version_names_the_supported_range(self):
        with mock.patch.object(run_web.sys, "version_info", (3, 13, 0)):
            with self.assertRaises(run_web.CheckFailed) as caught:
                run_web.check_python()
        message = str(caught.exception)
        self.assertIn("3.13", message)
        self.assertIn("mediapipe", message)

    def test_too_old_version_is_rejected(self):
        with mock.patch.object(run_web.sys, "version_info", (3, 8, 0)):
            with self.assertRaises(run_web.CheckFailed):
                run_web.check_python()


class PackageCheckTests(unittest.TestCase):
    def test_missing_package_reports_pip_name_not_import_name(self):
        with mock.patch.object(run_web.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(run_web.CheckFailed) as caught:
                run_web.check_packages()
        message = str(caught.exception)
        self.assertIn("python-multipart", message)
        self.assertNotIn(" multipart,", message)
        self.assertIn("pip install", message)


class RequiredFileTests(unittest.TestCase):
    def test_shipped_files_exist(self):
        self.assertEqual(run_web.check_files(), "데이터 파일")

    def test_missing_file_is_named(self):
        with mock.patch.object(run_web, "REQUIRED_FILES", [(ROOT / "없는파일.csv", "상품 카탈로그")]):
            with self.assertRaises(run_web.CheckFailed) as caught:
                run_web.check_files()
        self.assertIn("상품 카탈로그", str(caught.exception))


class ModelCheckTests(unittest.TestCase):
    def test_missing_model_degrades_instead_of_failing(self):
        with mock.patch.object(run_web, "MODEL_PATH", ROOT / "models" / "없는모델.pt"):
            message = run_web.check_model()
        self.assertIn("제로샷", message)


class PortSelectionTests(unittest.TestCase):
    def test_busy_port_falls_back_to_the_next_free_one(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            busy_port = taken.getsockname()[1]
            self.assertNotEqual(run_web.find_free_port(busy_port), busy_port)

    def test_free_port_is_returned_unchanged(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        self.assertEqual(run_web.find_free_port(free_port), free_port)


if __name__ == "__main__":
    unittest.main()
