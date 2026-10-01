"""시연 업로드·조회가 GPU를 호출하지 않고 대기 뒤 같은 조합을 반환하는지 검사한다."""

import io
import json
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from web.demo_api import SESSION_TTL_SECONDS, STAGE_SECONDS, is_demo_person
from web.gateway import create_app


def photo(color="white"):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 90), color).save(buffer, "JPEG")
    return buffer.getvalue()


class DemoAPITests(unittest.TestCase):
    def setUp(self):
        self.upstream = Mock(side_effect=AssertionError("시연 요청은 GPU 서버에 보내면 안 됩니다."))
        self.app = create_app("http://gpu.invalid", transport=httpx.MockTransport(self.upstream))
        self.demo = self.app.state.demo_api
        self.demo.preflight_seconds = 0
        self.now = 1000.0
        self.demo.clock = lambda: self.now
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def analyze(self, *, filename="input_person.jpg", raw=None, profile=None):
        return self.client.post("/api/analyze", files={
            "image": (filename, raw if raw is not None else photo(), "image/jpeg"),
        }, data={"profile": json.dumps(profile or {"gender": "남성", "change_categories": ["top"]})})

    def test_preflight_and_all_followup_requests_stay_local(self):
        options = self.client.get("/api/options")
        self.assertEqual(options.status_code, 200)
        self.assertEqual([stage["key"] for stage in options.json()["stages"]],
                         [stage for stage, _seconds in STAGE_SECONDS])
        self.assertEqual(self.client.get("/api/retention").json()["ttl_minutes"], 30)
        preflight = self.client.post("/api/validate-photo", files={
            "image": ("input_person.jpg", photo(), "image/jpeg"),
        })
        self.assertTrue(preflight.json()["valid"])
        response = self.analyze()
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        path = f"/api/jobs/{job_id}"
        self.now += 30
        result = self.client.get(path).json()["result"]
        original = self.client.get(path + "/images/original.jpg")
        self.assertEqual(original.status_code, 200)
        self.assertEqual(original.headers["content-type"], "image/jpeg")
        self.assertEqual(Image.open(io.BytesIO(original.content)).size, (40, 90))
        for method in (self.client.get, self.client.post):
            batch = method(path + "/shopping-tryon-batch")
            self.assertEqual(batch.json()["total"], 3)
            self.assertEqual(batch.json()["ready"], 3)
            self.assertEqual(batch.json()["status"], "done")
        self.assertTrue(result["tryon"]["available"])
        self.assertEqual(result["outfit"]["shoes"]["status"], "barefoot")
        self.assertEqual(result["outfit_summary"]["상의"], "그레이 체크 셔츠 (긴소매)")
        self.assertEqual(result["outfit_summary"]["하의"], "버건디 와이드 팬츠 (풀렝스)")
        self.assertEqual(result["outfit_summary"]["신발"],
                         "맨발이어서 신발 패션 분석을 진행하지 않았습니다.")
        self.assertNotIn("mock", result)  # 기존 화면의 데모 배지가 켜지지 않는다.
        for index, item in enumerate(batch.json()["items"], start=1):
            self.assertEqual(item["product_ids"], result["shopping_outfits"][index - 1]["product_ids"])
            self.assertEqual(item["image"], f"input-person-look-{index}.png")
            image = self.client.get(path + "/images/" + item["image"])
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image.headers["content-type"], "image/png")
            self.assertEqual(Image.open(io.BytesIO(image.content)).size, (941, 1672))
            selected = self.client.post(path + "/tryon-products", json={"product_ids": item["product_ids"]})
            self.assertEqual(selected.json()["image"], item["image"])
        self.assertEqual(self.client.post(path + "/tryon-products", json={}).status_code, 400)
        self.assertEqual(self.client.delete(path).json(), {"deleted": True})
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.get(path + "/images/original.jpg").status_code, 404)
        self.upstream.assert_not_called()

    def test_jpg_upload_works_when_the_gpu_worker_is_unreachable(self):
        def offline(request):
            raise httpx.ConnectError("offline", request=request)

        app = create_app("http://gpu.invalid", transport=httpx.MockTransport(offline))
        app.state.demo_api.preflight_seconds = 0
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/options").status_code, 200)
            self.assertEqual(client.get("/api/retention").status_code, 200)
            preflight = client.post("/api/validate-photo", files={
                "image": ("input_person.jpg", photo(), "image/jpeg"),
            })
            self.assertTrue(preflight.json()["valid"])
            created = client.post("/api/analyze", files={
                "image": ("input_person.jpg", photo(), "image/jpeg"),
            }, data={"profile": "{}"})
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["stage"], "prepare")
            self.assertEqual(client.get("/api/jobs/" + created.json()["job_id"]).status_code, 200)
            client.delete("/api/jobs/" + created.json()["job_id"])

    def test_progress_hides_result_until_the_full_delay(self):
        job_id = self.analyze().json()["job_id"]
        start = self.now
        elapsed = 0.0
        expected_history = []
        for stage, seconds in STAGE_SECONDS:
            self.now = start + elapsed + seconds / 2
            state = self.client.get(f"/api/jobs/{job_id}").json()
            expected_history.append(stage)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["stage"], stage)
            self.assertEqual(state["stage_history"], expected_history)
            self.assertIsNone(state["result"])
            elapsed += seconds
        self.now = start + 29.999
        self.assertIsNone(self.client.get(f"/api/jobs/{job_id}").json()["result"])
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/shopping-tryon-batch").json()["ready"], 0)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/images/input-person-look-1.png").status_code, 404)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/shopping-tryon-batch").status_code, 409)
        self.now = start + 30
        done = self.client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(done["status"], "done")
        self.assertIsNone(done["stage"])
        self.assertEqual(done["stage_history"], expected_history)
        self.assertIsNotNone(done["result"])
        self.upstream.assert_not_called()

    def test_conflicting_conditions_still_return_three_fixed_looks(self):
        for profile in (
            {"gender": "남성", "change_categories": ["top"], "max_budget": 1000},
            {"gender": "여성", "change_categories": ["shoes"], "desired_style": "스포티"},
        ):
            job_id = self.analyze(profile=profile).json()["job_id"]
            self.now += 30
            result = self.client.get(f"/api/jobs/{job_id}").json()["result"]
            products = result["shopping_results"]
            self.assertEqual([p["product_id"] for p in products[:3]], ["MS5215504", "MS6961115", "MS1021359"])
            self.assertEqual([p["price"] for p in products[:3]], [54890, 33330, 81000])
            self.assertEqual([p["review_score"] for p in products[:3]], [100, 100, 96])
            self.assertEqual([p["review_count"] for p in products[:3]], [25, 44, 5849])
            self.assertEqual(len(result["shopping_outfits"]), 3)
            self.assertEqual(result["shopping_outfits"][0]["products"], products[:3])
            self.assertEqual(result["shopping_outfits"][1]["product_ids"],
                             ["MS7038178", "MS1312128", "MS1021359"])
            self.assertEqual(result["shopping_outfits"][2]["product_ids"],
                             ["MS7033441", "MS6365296", "MS1021359"])
            self.assertEqual(result["request"], profile)
        self.upstream.assert_not_called()

    def test_unrelated_upload_is_proxied_with_its_original_multipart_body(self):
        received = []
        self.upstream.side_effect = lambda request: (
            received.append(request) or httpx.Response(202, json={"normal": True})
        )
        boundary = "TEST-DEMO-BOUNDARY"
        raw = photo()
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input_person2.jpeg"'
                '\r\nContent-Type: image/jpeg\r\n\r\n').encode() + raw + f'\r\n--{boundary}--\r\n'.encode()
        response = self.client.post("/api/analyze", content=body,
                                    headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(received[0].content, body)
        self.assertEqual(received[0].headers["content-type"], f"multipart/form-data; boundary={boundary}")
        self.assertEqual(self.demo.jobs, {})

    def test_invalid_image_and_profile_are_rejected_locally(self):
        for raw, status in ((b"", 400), (b"not an image", 400), (b"x" * (12 * 1024 * 1024 + 1), 413)):
            with self.subTest(status=status):
                self.assertEqual(self.analyze(raw=raw).status_code, status)
        response = self.client.post("/api/analyze", files={
            "image": ("input_person.jpg", photo(), "image/jpeg"),
        }, data={"profile": "[]"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.demo.jobs, {})
        self.upstream.assert_not_called()

    def test_expiration_and_running_job_deletion_never_fall_through_to_gpu(self):
        job_id = self.analyze().json()["job_id"]
        self.now += SESSION_TTL_SECONDS
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 404)
        self.assertNotIn(job_id, self.demo.jobs)
        self.assertNotIn(job_id, self.demo.expirations)
        job_id = self.analyze().json()["job_id"]
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}").status_code, 200)
        self.now += 30
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 404)
        self.upstream.assert_not_called()

    def test_matching_uses_only_exact_filename_and_accepts_client_paths(self):
        for filename in ("input_person.jpg", "INPUT_PERSON.JPG", "C:\\fakepath\\input_person.jpg"):
            self.assertTrue(is_demo_person(filename))
        for filename in (None, "input_person.jpeg", "input_person2.jpg", "other.jpg"):
            self.assertFalse(is_demo_person(filename))

    def test_direct_api_also_skips_model_warmup_and_inference(self):
        import web.app as server
        with (
            patch.object(server, "_model_readiness", {"status": "warming"}),
            patch.object(server, "get_engine", side_effect=AssertionError("GPU must stay idle")) as engine,
            patch.object(server, "run_pipeline") as pipeline,
            patch.object(server._demo_api, "duration_seconds", 0),
            patch.object(server._demo_api, "preflight_seconds", 0),
        ):
            client = TestClient(server.app)
            response = client.post("/api/analyze", files={
                "image": ("input_person.jpg", photo(), "image/jpeg"),
            }, data={"profile": "{}"})
            job_id = response.json()["job_id"]
            self.assertEqual(client.get(f"/api/jobs/{job_id}").json()["status"], "done")
            client.delete(f"/api/jobs/{job_id}")
            engine.assert_not_called()
            pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
