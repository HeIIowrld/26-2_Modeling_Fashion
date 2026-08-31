import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import web.app as web_app


def recommendation(rank: int):
    return SimpleNamespace(rank=rank, products=[SimpleNamespace(product_id=f"P{rank}")])


class TryOnBatchTests(unittest.TestCase):
    def setUp(self):
        self.job_id = "b" * 32
        self.tempdir = tempfile.TemporaryDirectory()
        self.session = Path(self.tempdir.name)
        (self.session / "person.jpg").write_bytes(b"person")
        (self.session / "preview.jpg").write_bytes(b"preview")
        self.job = {
            "id": self.job_id,
            "status": "done",
            "recommendations": [recommendation(rank) for rank in (1, 2, 3)],
            "result": {
                "tryon": {
                    "available": True,
                    "reason": "",
                    "warnings": ["rank one warning"],
                    "preview_kind": "tryon",
                },
                "images": {"preview": "preview.jpg"},
            },
            "person_image": self.session / "person.jpg",
            "tryon_context": {"mask": object()},
            "work_lock": threading.Lock(),
            "cancelled": False,
        }
        with web_app._jobs_lock:
            web_app._jobs[self.job_id] = self.job

    def tearDown(self):
        with web_app._jobs_lock:
            web_app._jobs.pop(self.job_id, None)
        self.tempdir.cleanup()

    def test_batch_reuses_rank_one_and_generates_remaining_ranks(self):
        calls = []

        def fake_generate(_person, reco, output, *, context):
            self.assertIs(context, self.job["tryon_context"])
            calls.append(reco.rank)
            output.write_bytes(f"rank-{reco.rank}".encode())
            return output, [f"warning-{reco.rank}"]

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            initialized = web_app._initialize_tryon_batch(self.job_id)
            self.assertEqual(initialized["ready"], 1)
            self.assertEqual(initialized["items"][0]["image"], "preview.jpg")
            web_app._start_tryon_batch(self.job_id)
            for _ in range(200):
                snapshot = web_app._read_tryon_batch(self.job_id)
                if snapshot["status"] == "done":
                    break
                time.sleep(0.005)
            else:
                self.fail("try-on batch did not finish")

        self.assertEqual(calls, [2, 3])
        self.assertEqual(snapshot["ready"], 3)
        self.assertEqual(
            [item["image"] for item in snapshot["items"]],
            ["preview.jpg", "tryon_2.jpg", "tryon_3.jpg"],
        )

    def test_preview_board_is_not_reused_as_a_real_tryon(self):
        self.job["recommendations"] = [recommendation(1)]
        self.job["result"]["tryon"]["preview_kind"] = "preview"

        def fake_generate(_person, _reco, output, *, context):
            self.assertTrue(context is self.job["tryon_context"])
            output.write_bytes(b"generated")
            return output, []

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            response = web_app.create_tryon(self.job_id, 1)

        self.assertEqual(response["image"], "tryon_1.jpg")
        self.assertFalse(response["cached"])

    def test_unavailable_adapter_does_not_start_a_worker(self):
        self.job["result"]["tryon"] = {
            "available": False,
            "reason": "GPU unavailable",
            "warnings": [],
            "preview_kind": "preview",
        }
        initialized = web_app._initialize_tryon_batch(self.job_id)
        started = web_app._start_tryon_batch(self.job_id)
        self.assertEqual(initialized["status"], "unavailable")
        self.assertEqual(started["status"], "unavailable")
        self.assertEqual(started["ready"], 0)

    def test_one_failed_rank_does_not_stop_the_remaining_batch(self):
        calls = []

        def fake_generate(_person, reco, output, *, context):
            calls.append(reco.rank)
            if reco.rank == 2:
                raise RuntimeError("rank two failed")
            output.write_bytes(f"rank-{reco.rank}".encode())
            return output, []

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            web_app._initialize_tryon_batch(self.job_id)
            web_app._start_tryon_batch(self.job_id)
            for _ in range(200):
                snapshot = web_app._read_tryon_batch(self.job_id)
                if snapshot["status"] == "partial":
                    break
                time.sleep(0.005)
            else:
                self.fail("try-on batch did not finish after a partial failure")

        self.assertEqual(calls, [2, 3])
        self.assertEqual(snapshot["ready"], 2)
        self.assertEqual(snapshot["finished"], 3)
        self.assertEqual(snapshot["items"][1]["status"], "failed")
        self.assertEqual(snapshot["items"][2]["status"], "done")

    def test_manual_cached_result_does_not_finish_an_active_batch_early(self):
        web_app._initialize_tryon_batch(self.job_id)
        web_app._set_tryon_batch_item(self.job_id, 1, status="done", image="preview.jpg")
        web_app._set_tryon_batch_item(self.job_id, 2, status="done", image="tryon_2.jpg")
        web_app._set_tryon_batch_item(self.job_id, 3, status="running")

        web_app._finish_tryon_batch(self.job_id)

        snapshot = web_app._read_tryon_batch(self.job_id)
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["ready"], 2)
        self.assertEqual(snapshot["finished"], 2)


if __name__ == "__main__":
    unittest.main()
