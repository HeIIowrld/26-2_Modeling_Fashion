import json
import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import FASHION_ATTRIBUTE_HEADS_PATH, PROJECT_DIR, resolve_path


class PathConfigurationTests(unittest.TestCase):
    def test_relative_path_uses_given_base_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = resolve_path("images/person.jpg", "unused", directory)
            self.assertEqual(result, (Path(directory) / "images" / "person.jpg").resolve())

    def test_absolute_path_is_not_rebased(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = (Path(directory) / "person.jpg").resolve()
            self.assertEqual(resolve_path(absolute, "unused", PROJECT_DIR), absolute)

    def test_default_path_uses_project_directory(self):
        self.assertEqual(resolve_path(None, "data"), (PROJECT_DIR / "data").resolve())

    def test_main_notebook_uses_master_fashion_rules(self):
        notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("RULES_PATH_INPUT = r'FASHION_RULES_MASTER.md'", source)
        self.assertIn(
            "resolve_local_path(RULES_PATH_INPUT, 'FASHION_RULES_MASTER.md'",
            source,
        )
        self.assertNotIn("FASHION_RULES_RESEARCH.md", source)

    def test_main_notebook_exposes_trained_attribute_head_path(self):
        notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("ATTRIBUTE_HEADS_PATH_INPUT = r'models/fashion_attribute_heads.pt'", source)
        self.assertIn("attribute_checkpoint=ATTRIBUTE_HEADS_PATH", source)


if __name__ == "__main__":
    unittest.main()


class DeployedCheckpointTests(unittest.TestCase):
    """어떤 체크포인트를 배포로 볼지 문서와 코드가 어긋나지 않게 한다.

    예전에는 CHECKSUMS.json 이 augmented 를 배포 모델로 적어 놓았는데 config.py 는
    baseline 을 읽었다. 오류가 안 나고 결과만 조용히 달라져서, 클론한 사람은 왜 결과가
    다른지 알 방법이 없었다. GPU 서버에서도 환경변수로 우회해야 했다.
    """

    @classmethod
    def setUpClass(cls):
        cls.checksums = json.loads(
            (PROJECT_DIR / "models" / "CHECKSUMS.json").read_text(encoding="utf-8")
        )

    def deployed_name(self) -> str:
        """설명 문장을 파싱하지 않는다. 문구가 바뀌면 검사가 조용히 무력해진다."""
        name = self.checksums.get("deployed")
        self.assertTrue(name, "CHECKSUMS.json 에 deployed 항목이 없습니다")
        return name

    def test_config_loads_the_checkpoint_the_document_calls_deployed(self):
        self.assertEqual(FASHION_ATTRIBUTE_HEADS_PATH.name, self.deployed_name())

    def test_the_deployed_checkpoint_exists(self):
        self.assertTrue(
            FASHION_ATTRIBUTE_HEADS_PATH.is_file(),
            f"{FASHION_ATTRIBUTE_HEADS_PATH} 가 없습니다",
        )

    def test_the_deployed_checkpoint_matches_its_recorded_hash(self):
        """다른 파일을 같은 이름으로 덮어써도 조용히 넘어가지 않게 한다."""
        import hashlib

        recorded = self.checksums["files"][FASHION_ATTRIBUTE_HEADS_PATH.name]["sha256"]
        actual = hashlib.sha256(FASHION_ATTRIBUTE_HEADS_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, recorded)

    def test_the_baseline_is_still_available_for_rollback(self):
        rollback = PROJECT_DIR / "models" / self.checksums["rollback"]
        self.assertTrue(rollback.is_file())
        self.assertNotEqual(rollback.name, FASHION_ATTRIBUTE_HEADS_PATH.name)
