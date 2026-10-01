import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


RUNNER = (Path(__file__).resolve().parents[2] / "docs" / "chatterbox-edition" /
          "experiments" / "multivoice" / "evaluate_cast.py")
spec = importlib.util.spec_from_file_location("private_cast_evaluator", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ScriptedChat:
    def __init__(self, reply):
        self.reply = reply
        self.messages = []

    def __call__(self, messages):
        self.messages.append(messages)
        return self.reply


class TestPrivateRunner(unittest.TestCase):
    def _args(self, tmp):
        root = Path(tmp)
        input_path = root / "book.epub"
        input_path.write_bytes(b"offline fixture")
        return SimpleNamespace(
            input_epub=input_path,
            cast_file=root / "new-cast.json",
            request_log=root / "requests.jsonl",
            model="local-test-model",
            base_url="http://offline.invalid/v1",
            chapter_selection=[2, 4],
        )

    def test_records_exact_requests_and_replies_without_constructing_a_network_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            messages = [{"role": "user", "content": "exact prompt"}]
            scripted = ScriptedChat('{"ok":true}')
            factory = Mock(return_value=scripted)
            observed = {}

            def analyze(settings, chat):
                observed.update(settings)
                reply = chat(messages)
                self.assertEqual(reply, '{"ok":true}')
                Path(settings["cast_file"]).write_text('{"status":"done"}', encoding="utf-8")
                return {"status": "done", "chapters_done": 2, "characters": {}}

            result, calls = runner.run(args, analyze=analyze, chat_factory=factory, api_key=lambda: "")
            record = json.loads(args.request_log.read_text(encoding="utf-8").splitlines()[0])
            factory.assert_called_once_with(args.base_url, args.model, "")
            self.assertEqual(scripted.messages, [messages])
            self.assertEqual(record, {"call": 1, "messages": messages, "reply": '{"ok":true}'})
            self.assertEqual((result["status"], calls), ("done", 1))
            self.assertEqual(observed["chapter_selection"], [2, 4])
            self.assertFalse(observed["auto_pick_voices"])

    def test_refuses_existing_outputs_before_constructing_the_chat_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            args.cast_file.write_text("preserve", encoding="utf-8")
            factory = Mock()
            with self.assertRaises(FileExistsError):
                runner.run(args, chat_factory=factory)
            factory.assert_not_called()
            self.assertEqual(args.cast_file.read_text(encoding="utf-8"), "preserve")
            self.assertFalse(args.request_log.exists())


if __name__ == "__main__":
    unittest.main()
