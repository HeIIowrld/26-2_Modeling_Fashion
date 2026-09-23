"""목 서버로 프런트를 만질 때 1단계를 넘어갈 수 있는지 확인한다.

실제 서버는 /api/validate-photo 를 통과해야 조건 입력을 열어 준다. 목에 이 경로가
없으면 화면 작업이 업로드 단계에서 막히고, 그 사실을 사람이 눌러 보고서야 안다.
"""
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mock_server import MockHandler  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64
BOUNDARY = "fittaboundary"


def multipart(filename: str, fields: dict | None = None) -> bytes:
    parts = []
    for name, value in (fields or {}).items():
        parts.append(f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                     .encode("utf-8"))
    parts.append(f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
                 "Content-Type: image/jpeg\r\n\r\n".encode("utf-8") + JPEG + b"\r\n")
    parts.append(f"--{BOUNDARY}--\r\n".encode("utf-8"))
    return b"".join(parts)


class MockPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def post(self, path: str, body: bytes):
        request = urllib.request.Request(
            self.base + path, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def test_ordinary_photo_opens_the_next_step(self):
        payload = self.post("/api/validate-photo", multipart("person.jpg"))
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["issues"], [])
        self.assertEqual(payload["warnings"], [])

    def test_uncertain_photo_passes_with_a_warning(self):
        payload = self.post("/api/validate-photo", multipart("loose-person.jpg"))
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["warnings"])
        self.assertEqual(payload["quality"]["body_visibility"]["status"], "uncertain")

    def test_occluding_photo_is_blocked_with_a_retake_message(self):
        payload = self.post("/api/validate-photo", multipart("skirt-person.jpg"))
        self.assertFalse(payload["valid"])
        self.assertIn("치마", payload["issues"][0])
        self.assertEqual(payload["warnings"], [])

    def test_analyze_still_reads_the_profile_after_sharing_the_upload_reader(self):
        payload = self.post("/api/analyze", multipart("person.jpg", {"profile": json.dumps({"gender": "여성"})}))
        self.assertEqual(len(payload["job_id"]), 32)
        self.assertEqual(payload["status"], "running")


if __name__ == "__main__":
    unittest.main()
