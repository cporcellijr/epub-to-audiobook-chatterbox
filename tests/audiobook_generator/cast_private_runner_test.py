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
spec = importlib.util.spec_from_file_location("private_cast_replay", RUNNER.with_name("replay_cast.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


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

    def test_a_manifest_says_which_code_model_input_and_settings_made_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)

            def analyze(settings, chat):
                chat([{"role": "user", "content": "x"}])
                return {"status": "done"}

            runner.run(args, analyze=analyze, chat_factory=lambda *_: ScriptedChat("{}"), api_key=lambda: "",
                       model_info=lambda base_url, model: {"digest": "sha256:abc"})
            manifest = json.loads(args.cast_file.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["code"]["tree_sha256"]), 64)
            self.assertEqual(manifest["model"], {"name": "local-test-model", "base_url": "http://offline.invalid/v1",
                                                 "digest": "sha256:abc"})
            self.assertEqual(manifest["input"]["file"], "book.epub")
            self.assertEqual(manifest["settings"]["chapter_selection"], [2, 4])
            self.assertEqual((manifest["chat_calls"], manifest["status"]), (1, "done"))

            def fails(settings, chat):
                raise RuntimeError("model went away")

            (Path(tmp) / "again").mkdir()
            args = self._args(Path(tmp) / "again")
            with self.assertRaises(RuntimeError):
                runner.run(args, analyze=fails, chat_factory=lambda *_: ScriptedChat("{}"), api_key=lambda: "",
                           model_info=lambda *_: {})
            manifest = json.loads(args.cast_file.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")

    def test_a_replay_answers_each_request_with_the_saved_reply_and_counts_new_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = root / "saved"
            saved.mkdir()
            asked = [{"role": "user", "content": "who speaks line 1?"}]
            (saved / "requests.jsonl").write_text(json.dumps({"call": 1, "messages": asked, "reply": "Wren"}) + "\n",
                                                  encoding="utf-8")
            (saved / "cast.json").write_text(json.dumps({"chapter_selection": [3]}), encoding="utf-8")
            (root / "book.epub").write_bytes(b"offline fixture")
            replies = []

            def analyze(settings, chat):
                replies.append(chat(asked))
                try:
                    chat([{"role": "user", "content": "a question the saved run never asked"}])
                except RuntimeError:
                    pass
                return {"status": "done"}

            result = replay.replay(root / "book.epub", saved, root / "out", analyze=analyze)
            self.assertEqual(replies, ["Wren"])
            self.assertEqual((result["status"], result["unanswered"]), ("done", 1))
            manifest = json.loads((root / "out" / "cast.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["replayed_from"], manifest["unanswered"]), (str(saved), 1))
            self.assertEqual(manifest["settings"]["chapter_selection"], [3])

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
