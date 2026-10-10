import unittest
import os
from unittest.mock import patch
from urllib.error import HTTPError
from audiobook_generator.core.gpu_reservation import session


class TestGpuReservation(unittest.TestCase):

    def setUp(self):
        env = patch.dict(os.environ, {"GPU_COORDINATOR_URL": "http://x", "GPU_COORDINATOR_TOKEN": "t"}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_waits_on_busy_gpu_and_cancels_the_same_token_on_release(self):
        calls = []
        def post(path, body, timeout=600):
            calls.append((path, body["token"]))
            if len(calls) == 1:
                raise HTTPError("http://x", 409, "busy", {}, None)
            return {"token": body["token"]}
        with patch("audiobook_generator.core.gpu_reservation._post", post), \
                patch("audiobook_generator.core.gpu_reservation.time.sleep") as sleep:
            with session("strata"):
                pass
        self.assertEqual([c[0] for c in calls], ["/acquire", "/acquire", "/release"])
        self.assertEqual(len({c[1] for c in calls}), 1)
        sleep.assert_called_once_with(1)

    def test_interrupted_acquire_cancels_pending_token_and_preserves_interrupt(self):
        with patch("audiobook_generator.core.gpu_reservation._post", side_effect=[KeyboardInterrupt(), {}]) as post:
            with self.assertRaises(KeyboardInterrupt):
                with session("strata"):
                    self.fail("Should not acquire")
        self.assertEqual([c.args[0] for c in post.call_args_list], ["/acquire", "/release"])
        self.assertEqual(post.call_args_list[0].args[1]["token"], post.call_args_list[1].args[1]["token"])

    def test_nested_llm_requests_reuse_the_cast_reservation_without_waiting(self):
        def post(path, body, timeout=600):
            return {"token": body["token"]}
        with patch("audiobook_generator.core.gpu_reservation._post", side_effect=post) as mocked:
            with session("ollama") as outer:
                with session("ollama") as inner:
                    self.assertEqual(inner, outer)
                with self.assertRaisesRegex(RuntimeError, "Cannot switch"):
                    with session("breeze"):
                        pass
        self.assertEqual([c.args[0] for c in mocked.call_args_list], ["/acquire", "/release"])

    def test_a_failed_release_is_reported_after_an_otherwise_successful_job(self):
        def post(path, body, timeout=600):
            if path == "/release":
                raise OSError("down")
            return {"token": body["token"]}
        with patch("audiobook_generator.core.gpu_reservation._post", post):
            with self.assertRaisesRegex(RuntimeError, "GPU release failed"):
                with session("strata"):
                    pass

    def test_unconfigured_bypass(self):
        with patch.dict(os.environ, {}, clear=True):
            with session("strata") as gpu:
                self.assertIsNone(gpu)

    def test_unsupported_engine_bypass(self):
        with patch.dict(os.environ, {"GPU_COORDINATOR_URL": "http://x", "GPU_COORDINATOR_TOKEN": "t"}, clear=True):
            with session("unsupported_engine") as gpu:
                self.assertIsNone(gpu)

    def test_missing_token_refuses_before_network(self):
        with patch.dict(os.environ, {"GPU_COORDINATOR_URL": "http://x"}, clear=True):
            with patch("audiobook_generator.core.gpu_reservation._post") as mock_post:
                with self.assertRaises(RuntimeError):
                    with session("strata") as gpu:
                        pass
                mock_post.assert_not_called()

    def test_normal_acquire_yield_release(self):
        with patch.dict(os.environ, {"GPU_COORDINATOR_URL": "http://x", "GPU_COORDINATOR_TOKEN": "t"}, clear=True):
            with patch("audiobook_generator.core.gpu_reservation.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value.hex = "abc"
                with patch("audiobook_generator.core.gpu_reservation._post") as mock_post:
                    mock_post.side_effect = [{"token": "abc"}, {}]
                    with session("strata") as gpu:
                        self.assertEqual(gpu, "abc")
                    self.assertEqual([c[0][0] for c in mock_post.call_args_list], ["/acquire", "/release"])

    def test_training_reserves_the_gpu_and_blocks_nested_cast_requests(self):
        def post(path, body, timeout=600):
            return {"token": body["token"]}
        with patch("audiobook_generator.core.gpu_reservation._post", side_effect=post) as mocked:
            with session("training"):
                with self.assertRaisesRegex(RuntimeError, "Cannot switch"):
                    with session("ollama"):
                        pass
        self.assertEqual([c.args[0] for c in mocked.call_args_list], ["/acquire", "/release"])
        self.assertEqual(mocked.call_args_list[0].args[1]["engine"], "training")

    def test_caller_exception_releases_and_preserves_original(self):
        with patch.dict(os.environ, {"GPU_COORDINATOR_URL": "http://x", "GPU_COORDINATOR_TOKEN": "t"}, clear=True):
            with patch("audiobook_generator.core.gpu_reservation.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value.hex = "abc"
                with patch("audiobook_generator.core.gpu_reservation._post") as mock_post:
                    mock_post.side_effect = [{"token": "abc"}, RuntimeError("release failed")]
                    with self.assertRaises(ValueError) as cm:
                        with session("strata") as gpu:
                            raise ValueError("original error")
                    self.assertEqual(str(cm.exception), "original error")


if __name__ == "__main__":
    unittest.main()
