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
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [3, 4], True, False,
            "auto", "double", False, False, None, "INFO")
        self.assertEqual((config.tts, config.output_format, config.worker_count), ("openai", "mp3", 1))
        self.assertEqual((config.voice_name, config.speed), ("Elena.wav", 1.0))
        self.assertEqual((config.chapter_start, config.chapter_end, config.chapter_selection), (1, -1, [3, 4]))
        self.assertIsNone(config.instructions)
        self.assertTrue(config.skip_existing)
        self.assertFalse(config.preview)


TABLE = [[1, False, "Title page", "Title", "under 1 min"], [2, True, "One", "It was", "26 min"],
         [3, True, "Two", "It was", "25 min"]]
SETTINGS = ("out", "Elena.wav", 1.0, False, False, "auto", "double", False, False, None, "INFO")


class TestStartGeneration(unittest.TestCase):

    def _start(self, library_book, input_file, table=TABLE):
        with patch.object(web_ui, "running_process", None), patch("os.path.isfile", return_value=True), \
                patch.object(web_ui, "launch_audiobook_generator") as launch, patch.object(gr, "Info"):
            chatterbox_ui.start_generation(library_book, input_file, table, *SETTINGS)
        return launch.call_args[0][0]

    def test_ticked_chapters_are_sent(self):
        self.assertEqual(self._start("/library/book.epub", None).chapter_selection, [2, 3])

    def test_dataframe_table_is_read(self):
        import pandas as pd
        table = pd.DataFrame(TABLE, columns=chatterbox_ui.CHAPTER_COLUMNS)
        self.assertEqual(self._start("/library/book.epub", None, table).chapter_selection, [2, 3])

    def test_library_pick_is_used_over_upload(self):
        self.assertEqual(self._start("/library/picked.epub", "/tmp/uploaded.epub").input_file, "/library/picked.epub")

    def test_upload_used_when_nothing_picked(self):
        self.assertEqual(self._start(None, "/tmp/uploaded.epub").input_file, "/tmp/uploaded.epub")

    def test_nothing_ticked_is_an_error(self):
        table = [[1, False, "A", "", ""], [2, False, "B", "", ""]]
        with self.assertRaises(gr.Error):
            self._start("/library/book.epub", None, table)

    def test_start_refuses_while_a_book_is_running(self):
        running = MagicMock()
        running.is_alive.return_value = True
        with patch.object(web_ui, "running_process", running), patch("os.path.isfile", return_value=True), \
                patch.object(web_ui, "launch_audiobook_generator") as launch:
            with self.assertRaises(gr.Error):
                chatterbox_ui.start_generation("/library/book.epub", None, TABLE, *SETTINGS)
        launch.assert_not_called()

    def test_typed_text_that_is_not_a_book_is_an_error(self):
        with patch.object(web_ui, "launch_audiobook_generator") as launch:
            with self.assertRaises(gr.Error):
                chatterbox_ui.start_generation("detour", None, TABLE, *SETTINGS)
        launch.assert_not_called()

    def test_no_book_is_an_error(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.start_generation(None, None, TABLE, *SETTINGS)


class TestChapterList(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.book = os.path.join(self.tmp.name, "collection.epub")
        _write_collection_epub(self.book)

    def tearDown(self):
        self.tmp.cleanup()

    def test_front_matter_is_unticked_and_story_ticked(self):
        table, lengths, summary = chatterbox_ui.chapter_overview(
            self.book, None, 1.0, "auto", "double", False, False, None)
        rows = table["value"]
        self.assertEqual([row[1] for row in rows], [False, False, True, True])
        self.assertEqual(len(lengths), 4)
        self.assertIn("2 of 4 chapters", summary)
        self.assertIn("numbered 1–2", summary)

    def test_no_book_hides_the_table(self):
        table, lengths, summary = chatterbox_ui.chapter_overview(None, None, 1.0, "auto", "double", False, False, None)
        self.assertFalse(table["visible"])
        self.assertEqual((lengths, summary), ([], ""))

    def test_summary_follows_ticks_and_speed(self):
        lengths = [20, 30300, 30300]  # 30,300 chars = 25 min at 1.0x
        rows = [[1, False, "T", "", ""], [2, True, "A", "", ""], [3, True, "B", "", ""]]
        self.assertIn("50 min", chatterbox_ui.chapter_summary(rows, lengths, 1.0))
        self.assertIn("40 min", chatterbox_ui.chapter_summary(rows, lengths, 1.25))
        rows[2][1] = False
        self.assertIn("1 of 3 chapters", chatterbox_ui.chapter_summary(rows, lengths, 1.0))

    def test_speed_change_retimes_but_keeps_ticks(self):
        rows = [[1, False, "T", "", "x"], [2, True, "A", "", "x"]]
        updated = chatterbox_ui.retime_chapters(rows, [20, 30300], 2.0)["value"]
        self.assertEqual([row[1] for row in updated], [False, True])
        self.assertEqual(updated[1][4], "12 min")

    def test_tick_all(self):
        rows = [[1, False, "T", "", ""], [2, True, "A", "", ""]]
        self.assertEqual([row[1] for row in chatterbox_ui.tick_all_chapters(rows)["value"]], [True, True])


def _write_collection_epub(path: str) -> None:
    import zipfile
    story = "<p>" + "It was a dark and stormy night. " * 150 + "</p>"
    docs = {
        "title": "<h1>Story Collection 2</h1>",
        "blurb": "<p>Five Book Story Collection by A. N. Author</p>",
        "c1": "<h1>The First Story</h1>" + story,
        "c2": "<h1>The Second Story</h1>" + story,
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        manifest = "".join(f'<item id="{k}" href="{k}.xhtml" media-type="application/xhtml+xml"/>' for k in docs)
        spine = "".join(f'<itemref idref="{k}"/>' for k in docs)
        z.writestr("content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
                   'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:identifier id="id">t</dc:identifier><dc:title>Story Collection 2</dc:title>'
                   f'<dc:language>en</dc:language></metadata><manifest>{manifest}</manifest>'
                   f'<spine>{spine}</spine></package>')
        for key, body in docs.items():
            z.writestr(f"{key}.xhtml", '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                                       f'<body>{body}</body></html>')


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
