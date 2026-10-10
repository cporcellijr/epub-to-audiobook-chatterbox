from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from audiobook_generator.core import cast_analysis, cast_llm, engine_gpu, gpu_reservation
from audiobook_generator.ui import chatterbox_ui, job_queue
import main


class IntegrationTests(unittest.TestCase):
    def test_actual_chat_client_reuses_the_surrounding_cast_lease(self):
        client = cast_llm.ChatClient.__new__(cast_llm.ChatClient)
        client.client, client.model, client.json_mode = Mock(), "qwen2.5:14b", True
        client.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
        def post(path, body, timeout=600):
            return {"token": body["token"]}
        with patch.dict(os.environ, {"GPU_COORDINATOR_URL": "http://x", "GPU_COORDINATOR_TOKEN": "t"}), \
                patch.object(gpu_reservation, "_post", side_effect=post) as request:
            with gpu_reservation.session("ollama"):
                self.assertEqual(client([{"role": "user", "content": "test"}]), "{}")
        self.assertEqual([c.args[0] for c in request.call_args_list], ["/acquire", "/release"])
        self.assertEqual(client.client.chat.completions.create.call_args.kwargs["model"], "qwen2.5:14b")

    def setUp(self):
        self.events = []

    @contextmanager
    def reserve(self, engine):
        self.events.append(("acquire", engine))
        try:
            yield "reservation"
        finally:
            self.events.append(("release", engine))

    def test_whole_book_holds_reservation_until_generation_finishes(self):
        config = SimpleNamespace(tts="openai", model_name="breeze", preview=False, log="INFO")
        with patch.object(gpu_reservation, "session", self.reserve), \
                patch.object(main, "setup_logging"), patch.object(main, "AudiobookGenerator") as generator:
            generator.return_value.run.side_effect = lambda: self.events.append(("speak", "breeze")) or True
            self.assertTrue(main.main(config, "log.txt"))
        self.assertEqual(self.events, [("acquire", "breeze"), ("speak", "breeze"), ("release", "breeze")])

    def test_failed_book_releases_reservation(self):
        config = SimpleNamespace(tts="openai", model_name="breeze", preview=False, log="INFO")
        with patch.object(gpu_reservation, "session", self.reserve), \
                patch.object(main, "setup_logging"), patch.object(main, "AudiobookGenerator") as generator:
            generator.return_value.run.side_effect = ValueError("failed book")
            with self.assertRaisesRegex(ValueError, "failed book"):
                main.main(config, "log.txt")
        self.assertEqual(self.events[-1], ("release", "breeze"))

    def test_coordinated_queue_never_preloads_breeze_outside_its_job(self):
        with patch.object(gpu_reservation, "configured", return_value=True), \
                patch.object(engine_gpu, "_start") as start, patch.object(engine_gpu.breeze_client, "loaded") as loaded:
            self.assertTrue(engine_gpu.ready_for_book("breeze"))
        loaded.assert_not_called()
        start.assert_not_called()

    def test_cast_uses_reservation_and_never_legacy_unloads_or_reloads_speech(self):
        legacy = Mock()
        result = []
        with patch.object(gpu_reservation, "configured", return_value=True), \
                patch.object(gpu_reservation, "session", self.reserve), patch.object(cast_analysis, "setup_logging"):
            cast_analysis.run_cast_analysis({}, "log", unload_allowed=legacy, unload=legacy, reload=legacy,
                                            unload_breeze=legacy, analyse=lambda _: self.events.append(("cast", "ollama")),
                                            exit=result.append)
        legacy.assert_not_called()
        self.assertEqual(result, [0])
        self.assertEqual(self.events, [("acquire", "ollama"), ("cast", "ollama"), ("release", "ollama")])

    def test_breeze_preview_uses_shared_reservation_and_releases_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = job_queue.JobQueue(str(Path(folder) / "queue.json"), lambda **s: s, lambda: "log")
            with patch.object(gpu_reservation, "configured", return_value=True), \
                    patch.object(gpu_reservation, "session", self.reserve), \
                    patch.object(engine_gpu, "prepare_breeze") as prepare:
                with self.assertRaisesRegex(ValueError, "preview"):
                    with queue.breeze_session():
                        self.assertEqual(queue._breeze_requests, 1)
                        raise ValueError("preview")
                self.assertEqual(queue._breeze_requests, 0)
            prepare.assert_not_called()
        self.assertEqual(self.events, [("acquire", "breeze"), ("release", "breeze")])


if __name__ == "__main__":
    unittest.main()
