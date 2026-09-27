import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

from audiobook_generator.ui import chatterbox_ui, web_ui


def _fake_response(body: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


def _sample_with_pauses(path: str) -> None:
    """1 s tone, 2 s silence, 1 s tone."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=1",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono:d=2",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=1",
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", "-ar", "24000", path],
        check=True,
    )


class TestChatterboxUrl(unittest.TestCase):

    def test_derived_from_openai_base_url(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "", "OPENAI_BASE_URL": "http://chatterbox:8004/v1"}):
            self.assertEqual(chatterbox_ui.chatterbox_url(), "http://chatterbox:8004")

    def test_explicit_url_wins(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:9/", "OPENAI_BASE_URL": "http://x/v1"}):
            self.assertEqual(chatterbox_ui.chatterbox_url(), "http://cb:9")


class TestDeliverySettings(unittest.TestCase):

    def test_reads_saved_settings_from_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("generation_defaults:\n  temperature: 0.61\n  exaggeration: 0.73\n  cfg_weight: 0.45\n")
        try:
            with patch.dict(os.environ, {"CHATTERBOX_CONFIG": f.name}):
                self.assertEqual(chatterbox_ui.load_saved_settings(), (0.73, 0.45, 0.61))
        finally:
            os.remove(f.name)

    def test_missing_config_falls_back(self):
        with patch.dict(os.environ, {"CHATTERBOX_CONFIG": "/nonexistent.yaml"}):
            self.assertEqual(chatterbox_ui.read_saved_settings(), chatterbox_ui.FALLBACK_SETTINGS)

    def test_save_posts_generation_defaults(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"{}")) as urlopen:
            message = chatterbox_ui.save_settings(0.8, 0.45, 0.61)
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://cb:8004/save_settings")
        self.assertEqual(json.loads(request.data),
                         {"generation_defaults": {"exaggeration": 0.8, "cfg_weight": 0.45, "temperature": 0.61}})
        self.assertIn("Saved", message)


class TestPreviewVoice(unittest.TestCase):

    def test_preview_sends_sliders_and_returns_audio_file(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"mp3-bytes")) as urlopen:
            path = chatterbox_ui.preview_voice("love poem.wav", "Hello there.", 0.8, 0.45, 0.61, 1.25)
        try:
            request = urlopen.call_args[0][0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, "http://cb:8004/tts")
            self.assertEqual(payload["predefined_voice_id"], "love poem.wav")
            self.assertEqual(payload["text"], "Hello there.")
            self.assertEqual((payload["exaggeration"], payload["cfg_weight"], payload["temperature"]),
                             (0.8, 0.45, 0.61))
            self.assertEqual(payload["speed_factor"], 1.25)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"mp3-bytes")
        finally:
            os.remove(path)

    def test_blank_phrase_uses_built_in_phrase(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"x")) as urlopen:
            path = chatterbox_ui.preview_voice("Elena.wav", "  ", 0.5, 0.5, 0.8, 1.0)
        os.remove(path)
        self.assertEqual(json.loads(urlopen.call_args[0][0].data)["text"], chatterbox_ui.PREVIEW_PHRASE)

    def test_no_voice_is_an_error(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.preview_voice("", "Hi", 0.5, 0.5, 0.8, 1.0)


class TestAddVoice(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.mkdir(self.voices)
        self.sample = os.path.join(self.tmp.name, "My Narrator.mp3")
        _sample_with_pauses(self.sample)
        self.env = patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_adds_wav_with_pauses_removed(self):
        message, lab_update, _ = chatterbox_ui.add_voice(self.sample, "Narrator: One", True, False)
        saved = os.path.join(self.voices, "Narrator One.wav")
        self.assertTrue(os.path.isfile(saved))
        self.assertLess(chatterbox_ui._audio_seconds(saved), 3.0)
        self.assertEqual(lab_update["value"], "Narrator One.wav")
        self.assertIn(("Narrator One", "Narrator One.wav"), lab_update["choices"])
        self.assertIn("pauses removed", message)

    def test_keeps_pauses_when_asked(self):
        chatterbox_ui.add_voice(self.sample, "Raw", False, False)
        self.assertGreater(chatterbox_ui._audio_seconds(os.path.join(self.voices, "Raw.wav")), 3.5)

    def test_name_defaults_to_file_name(self):
        chatterbox_ui.add_voice(self.sample, "", True, False)
        self.assertTrue(os.path.isfile(os.path.join(self.voices, "My Narrator.wav")))

    def test_existing_voice_needs_replace(self):
        chatterbox_ui.add_voice(self.sample, "Dup", True, False)
        with self.assertRaises(gr.Error):
            chatterbox_ui.add_voice(self.sample, "Dup", True, False)
        chatterbox_ui.add_voice(self.sample, "Dup", True, True)

    def test_short_sample_warns(self):
        message, _, _ = chatterbox_ui.add_voice(self.sample, "Short", True, False)
        self.assertIn("short", message)


class TestBuildConfig(unittest.TestCase):

    def test_config_targets_chatterbox_with_one_worker(self):
        config = chatterbox_ui.build_config(
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, 3.0, -1.0, True, False,
            "auto", "double", False, False, None, "INFO", preview=False)
        self.assertEqual((config.tts, config.output_format, config.worker_count), ("openai", "mp3", 1))
        self.assertEqual((config.voice_name, config.speed), ("Elena.wav", 1.0))
        self.assertEqual((config.chapter_start, config.chapter_end), (3, -1))
        self.assertIsNone(config.instructions)
        self.assertTrue(config.skip_existing)

    SETTINGS = ("out", "Elena.wav", 1.0, 1, -1, False, False, "auto", "double", False, False, None, "INFO")

    def test_start_refuses_while_a_book_is_running(self):
        running = MagicMock()
        running.is_alive.return_value = True
        with patch.object(web_ui, "running_process", running), \
                patch.object(web_ui, "launch_audiobook_generator") as launch:
            with self.assertRaises(gr.Error):
                chatterbox_ui.start_generation("/library/book.epub", None, *self.SETTINGS)
        launch.assert_not_called()

    def test_library_pick_is_used_over_upload(self):
        with patch.object(web_ui, "running_process", None), \
                patch.object(web_ui, "launch_audiobook_generator") as launch, patch.object(gr, "Info"):
            chatterbox_ui.start_generation("/library/picked.epub", "/tmp/uploaded.epub", *self.SETTINGS)
        self.assertEqual(launch.call_args[0][0].input_file, "/library/picked.epub")

    def test_upload_used_when_nothing_picked(self):
        with patch.object(web_ui, "running_process", None), \
                patch.object(web_ui, "launch_audiobook_generator") as launch, patch.object(gr, "Info"):
            chatterbox_ui.preview_chapters(None, "/tmp/uploaded.epub", *self.SETTINGS)
        config = launch.call_args[0][0]
        self.assertEqual((config.input_file, config.preview), ("/tmp/uploaded.epub", True))

    def test_no_book_is_an_error(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.start_generation(None, None, *self.SETTINGS)


class TestBookSelection(unittest.TestCase):

    def test_library_pick_names_output_folder_from_its_title(self):
        index = {"/library/01. Detour (1988).epub": {"title": "Detour: A Novel", "author": "Jane Doe"}}
        with patch.object(chatterbox_ui.library_index, "load_index", return_value=index):
            update = chatterbox_ui.library_output_dir("/library/01. Detour (1988).epub")
        self.assertEqual(update["value"], os.path.join("audiobook_output", "Detour A Novel"))

    def test_upload_clears_library_pick(self):
        with patch.object(chatterbox_ui, "suggest_output_dir", return_value={"value": "audiobook_output/X"}):
            output_dir, library_book = chatterbox_ui.uploaded_book_selected("/tmp/x.epub")
        self.assertEqual(output_dir, {"value": "audiobook_output/X"})
        self.assertIsNone(library_book["value"])


class TestLayout(unittest.TestCase):

    def test_only_chatterbox_controls(self):
        ui = chatterbox_ui.build_ui()
        labels = {getattr(block, "label", None) for block in ui.blocks.values()}
        self.assertIn("Voice", labels)
        self.assertIn("Book", labels)
        self.assertIn("Exaggeration", labels)
        self.assertIn("Voice sample: 10-15 s of one person speaking clearly", labels)
        for removed in ("Azure", "Edge", "Piper", "Model", "Voice Instructions", "Worker Count", "Output Format"):
            self.assertNotIn(removed, labels)


if __name__ == "__main__":
    unittest.main()
