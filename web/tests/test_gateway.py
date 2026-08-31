from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from gateway import create_app


class GatewayTests(unittest.TestCase):
    def test_proxies_api_method_query_body_and_content_type(self) -> None:
        def upstream(request: httpx.Request) -> httpx.Response:
            payload = {
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query.decode(),
                "body": request.content.decode(),
                "forwarded_host": request.headers["x-forwarded-host"],
            }
            return httpx.Response(201, json=payload, headers={"cache-control": "no-store"})

        app = create_app("http://gpu.invalid", transport=httpx.MockTransport(upstream))
        with TestClient(app) as client:
            response = client.post("/api/jobs/abc/tryon/1?quality=high", content=b"payload")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(
            response.json(),
            {
                "method": "POST",
                "path": "/api/jobs/abc/tryon/1",
                "query": "quality=high",
                "body": "payload",
                "forwarded_host": "testserver",
            },
        )

    def test_preserves_binary_image_response(self) -> None:
        jpeg = b"\xff\xd8test-image\xff\xd9"

        def upstream(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=jpeg, headers={"content-type": "image/jpeg"})

        app = create_app("http://gpu.invalid", transport=httpx.MockTransport(upstream))
        with TestClient(app) as client:
            response = client.get("/api/jobs/abc/images/tryon_1.jpg")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.content, jpeg)

    def test_returns_503_when_gpu_worker_is_unreachable(self) -> None:
        def upstream(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        app = create_app("http://gpu.invalid", transport=httpx.MockTransport(upstream))
        with TestClient(app) as client:
            response = client.get("/api/health")

        self.assertEqual(response.status_code, 503)
        self.assertIn("GPU", response.json()["detail"])

    def test_serves_frontend_locally(self) -> None:
        app = create_app(
            "http://gpu.invalid",
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        )
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("no-cache", response.headers["cache-control"])


if __name__ == "__main__":
    unittest.main()
