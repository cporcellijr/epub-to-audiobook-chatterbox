import io
import json
import multiprocessing
import os
import subprocess
import tempfile
import unittest
import urllib.error
import zipfile
from unittest.mock import MagicMock, patch

import gradio as gr
from pydub import AudioSegment

from audiobook_generator.ui import chatterbox_ui, web_ui
from audiobook_generator.ui.job_queue import CAST, DONE, QUEUED, RUNNING


def _silent_mp3_bytes(ms: int = 200) -> bytes:
    buffer = io.BytesIO()
    AudioSegment.silent(duration=ms, frame_rate=24000).export(buffer, format="mp3")
    return buffer.getvalue()


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


class TestDeliverySettingsRobustness(unittest.TestCase):
    """F-52: build_ui() calls read_saved_settings() at start-up, so a bad read must never raise.
    The Chatterbox server rewrites config.yaml non-atomically (copy, then fill), so a read can
    briefly see the path missing, a directory, empty or unparsable."""

    def setUp(self):
        self.delay = patch.object(chatterbox_ui, "SETTINGS_READ_RETRY_DELAY_SECONDS", 0.0)
        self.delay.start()

    def tearDown(self):
        self.delay.stop()

    def test_directory_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CHATTERBOX_CONFIG": tmp}):
            self.assertEqual(chatterbox_ui.read_saved_settings(), chatterbox_ui.FALLBACK_SETTINGS)

    def test_missing_file_falls_back_to_defaults(self):
        with patch.dict(os.environ, {"CHATTERBOX_CONFIG": "/nonexistent/config.yaml"}):
            self.assertEqual(chatterbox_ui.read_saved_settings(), chatterbox_ui.FALLBACK_SETTINGS)

    def test_empty_file_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            pass  # zero bytes, as a reader can briefly see mid-rewrite
        try:
            with patch.dict(os.environ, {"CHATTERBOX_CONFIG": f.name}):
                self.assertEqual(chatterbox_ui.read_saved_settings(), chatterbox_ui.FALLBACK_SETTINGS)
        finally:
            os.remove(f.name)

    def test_unparsable_file_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("{")  # guaranteed to raise yaml.YAMLError, not just parse oddly
        try:
            with patch.dict(os.environ, {"CHATTERBOX_CONFIG": f.name}):
                self.assertEqual(chatterbox_ui.read_saved_settings(), chatterbox_ui.FALLBACK_SETTINGS)
        finally:
            os.remove(f.name)

    def test_a_transient_glitch_recovers_on_retry(self):
        good = {"exaggeration": 0.9, "cfg_weight": 0.4, "temperature": 0.5}
        with patch.object(chatterbox_ui, "_read_generation_defaults", side_effect=[None, good]), \
                patch.dict(os.environ, {"CHATTERBOX_CONFIG": "/whatever.yaml"}):
            self.assertEqual(chatterbox_ui.read_saved_settings(), good)

    def test_a_permanent_failure_is_retried_exactly_once(self):
        with patch.object(chatterbox_ui, "_read_generation_defaults", return_value=None) as read, \
                patch.dict(os.environ, {"CHATTERBOX_CONFIG": "/whatever.yaml"}):
            chatterbox_ui.read_saved_settings()
        self.assertEqual(read.call_count, 2)


class TestPreviewVoice(unittest.TestCase):

    def test_preview_sends_sliders_and_returns_audio_file(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"mp3-bytes")) as urlopen:
            path = chatterbox_ui.preview_voice("narrator two.wav", "Hello there.", 0.8, 0.45, 0.61, 1.25)
        try:
            request = urlopen.call_args[0][0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, "http://cb:8004/tts")
            self.assertEqual(payload["predefined_voice_id"], "narrator two.wav")
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


class TestPreviewCleanup(unittest.TestCase):
    """F-30: preview MP3s must not accumulate in the container's writable layer forever."""

    def tearDown(self):
        chatterbox_ui._delete_if_exists(chatterbox_ui._current_preview_path)
        chatterbox_ui._current_preview_path = None

    def test_previous_preview_is_deleted_when_a_new_one_is_made(self):
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"one")):
            first = chatterbox_ui.preview_voice("Elena.wav", "Hi", 0.5, 0.5, 0.8, 1.0)
        self.assertTrue(os.path.isfile(first))
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"two")):
            second = chatterbox_ui.preview_voice("Elena.wav", "Hi", 0.5, 0.5, 0.8, 1.0)
        self.assertFalse(os.path.isfile(first))
        self.assertTrue(os.path.isfile(second))

    def test_sweep_removes_leftover_previews(self):
        handle, path = tempfile.mkstemp(prefix="voice_preview_", suffix=".mp3")
        os.close(handle)
        try:
            chatterbox_ui.sweep_voice_previews()
            self.assertFalse(os.path.isfile(path))
        finally:
            if os.path.isfile(path):
                os.remove(path)


class TestDeliveryBaselineAndPreview(unittest.TestCase):

    def tearDown(self):
        chatterbox_ui._delete_if_exists(chatterbox_ui._current_preview_path)
        chatterbox_ui._current_preview_path = None

    def test_delivery_baseline_text_formats_the_sliders(self):
        text = chatterbox_ui.delivery_baseline_text(0.73, 0.5, 0.61)
        self.assertIn("exaggeration 0.73", text)
        self.assertIn("CFG 0.5", text)
        self.assertIn("temperature 0.61", text)
        self.assertIn("Voice lab", text)

    def test_engine_changed_shows_delivery_controls_only_for_chatterbox(self):
        _, _, _, adaptive, baseline_info = chatterbox_ui.engine_changed("chatterbox")
        self.assertTrue(adaptive["visible"])
        self.assertTrue(baseline_info["visible"])
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(json.dumps({"voices": []}).encode())):
            _, _, _, adaptive, baseline_info = chatterbox_ui.engine_changed("kokoro")
        self.assertFalse(adaptive["visible"])
        self.assertFalse(baseline_info["visible"])

    def test_preview_delivery_range_sends_the_three_mood_presets_in_order(self):
        clip = _silent_mp3_bytes(200)
        with patch.dict(os.environ, {"CHATTERBOX_URL": "http://cb:8004"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(clip)) as urlopen:
            path = chatterbox_ui.preview_delivery_range("Elena.wav", "Hello.", 0.73, 0.5, 0.61, 1.0)
        try:
            payloads = [json.loads(call.args[0].data) for call in urlopen.call_args_list]
            self.assertEqual(len(payloads), 3)
            self.assertEqual([(p["exaggeration"], p["cfg_weight"], p["temperature"]) for p in payloads],
                             [(0.35, 0.35, 0.5), (0.73, 0.5, 0.61), (0.91, 0.45, 0.7)])
            self.assertTrue(all(p["predefined_voice_id"] == "Elena.wav" and p["text"] == "Hello." for p in payloads))
            combined = AudioSegment.from_file(path)
            self.assertGreater(len(combined), 1500)  # 3 clips + 2 gaps of ~1 s: much longer than one clip alone
        finally:
            os.remove(path)

    def test_preview_delivery_range_needs_a_voice(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.preview_delivery_range("", "Hi", 0.73, 0.5, 0.61, 1.0)


class TestAddVoice(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.mkdir(self.voices)
        self.sample = os.path.join(self.tmp.name, "My Narrator.mp3")
        _sample_with_pauses(self.sample)
        self.env = patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices})
        self.env.start()
        # No Chatterbox in tests: measuring the new voice fails, which must never fail the add.
        self.measure = patch.object(chatterbox_ui, "measure_voice", side_effect=OSError("no Chatterbox here"))
        self.measure.start()

    def tearDown(self):
        self.measure.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_a_new_voice_is_measured_and_a_failure_only_defers_it(self):
        message, _, _, _ = chatterbox_ui.add_voice(self.sample, "Unmeasured", True, False)
        self.assertTrue(os.path.isfile(os.path.join(self.voices, "Unmeasured.wav")))
        self.assertIn("Not measured yet", message)
        with patch.object(chatterbox_ui, "measure_voice", return_value={}) as measure, \
                patch.object(chatterbox_ui, "voice_sound_words", return_value="low for a woman, husky"):
            message, _, _, _ = chatterbox_ui.add_voice(self.sample, "Measured", True, False)
        measure.assert_called_once_with("Measured.wav")
        self.assertIn("Measured: sounds low for a woman, husky.", message)

    def test_adds_wav_with_pauses_removed(self):
        message, lab_update, _, _ = chatterbox_ui.add_voice(self.sample, "Narrator: One", True, False)
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
        message, _, _, _ = chatterbox_ui.add_voice(self.sample, "Short", True, False)
        self.assertIn("short", message)


class TestVoiceMeasuring(unittest.TestCase):
    """Measure voices (Voice lab): Chatterbox speaks a fixed sentence per voice; Praat is replaced by
    a stand-in so each voice gets a known pitch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.mkdir(self.voices)
        for name in ("Ada.wav", "Bea.wav", "Cal.wav"):
            with open(os.path.join(self.voices, name), "wb") as f:
                f.write(name.encode())
        self.patches = [
            patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices}),
            patch.object(chatterbox_ui.voice_measure, "VOICE_FEATURES_FILE", os.path.join(self.tmp.name, "features.json")),
            patch("audiobook_generator.core.cast.VOICE_GENDERS_FILE", os.path.join(self.tmp.name, "genders.json")),
        ]
        for p in self.patches:
            p.start()
        self.pitches = iter([150.0, 250.0, 120.0])
        self.fake_praat = patch.object(chatterbox_ui.voice_measure, "measure_audio",
                                       side_effect=lambda audio: {"f0_median": next(self.pitches), "f0_range": 8.0,
                                                                  "hnr": 11.0})
        self.fake_praat.start()

    def tearDown(self):
        self.fake_praat.stop()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_every_unmeasured_voice_is_measured_once(self):
        with patch.object(chatterbox_ui, "_post_json", return_value=b"wav") as post:
            self.assertEqual(chatterbox_ui.measure_voices(), "Measured 3 voices.")
            self.assertEqual(chatterbox_ui.measure_voices(), "All 3 voices are measured.")
        self.assertEqual(post.call_count, 3)
        path, payload = post.call_args_list[0].args
        self.assertEqual((path, payload["predefined_voice_id"], payload["text"]),
                         ("/tts", "Ada.wav", chatterbox_ui.voice_measure.MEASURE_TEXT))
        saved = chatterbox_ui.voice_measure.load_features()
        self.assertEqual({v: m["f0_median"] for v, m in saved.items()}, {"Ada.wav": 150.0, "Bea.wav": 250.0, "Cal.wav": 120.0})

    def test_an_unreachable_chatterbox_stops_the_run_and_a_bad_voice_is_skipped(self):
        with patch.object(chatterbox_ui, "_post_json", side_effect=urllib.error.URLError("refused")) as post:
            self.assertIn("Measured 0 of 3, then Chatterbox couldn't be reached", chatterbox_ui.measure_voices())
        self.assertEqual(post.call_count, 1)
        bad = urllib.error.HTTPError("http://cb/tts", 500, "boom", {}, io.BytesIO(b'{"detail": "voice broken"}'))
        with patch.object(chatterbox_ui, "_post_json", side_effect=[bad, b"wav", b"wav"]):
            self.assertEqual(chatterbox_ui.measure_voices(), "Measured 2 voices. Couldn't measure Ada (voice broken).")

    def test_the_voice_lab_describes_a_measured_voice_within_its_gender(self):
        self.assertIn("Not measured yet", chatterbox_ui.voice_sound_text("Ada.wav"))
        from audiobook_generator.core import cast as cast_store
        for voice in ("Ada.wav", "Bea.wav"):
            cast_store.save_voice_gender(voice, "female")
        with patch.object(chatterbox_ui, "_post_json", return_value=b"wav"):
            chatterbox_ui.measure_voices()
        self.assertIn("Sounds **low for a woman** (150 Hz)", chatterbox_ui.voice_sound_text("Ada.wav"))
        self.assertIn("high for a woman", chatterbox_ui.voice_sound_text("Bea.wav"))

    def test_deleting_a_voice_forgets_its_measurement_and_gender(self):
        from audiobook_generator.core import cast as cast_store
        cast_store.save_voice_gender("Ada.wav", "female")
        cast_store.save_voice_gender("Bea.wav", "female")
        with patch.object(chatterbox_ui, "_post_json", return_value=b"wav"):
            chatterbox_ui.measure_voices()
        chatterbox_ui.delete_own_voice("Ada.wav", [])
        self.assertNotIn("Ada.wav", chatterbox_ui.voice_measure.load_features())
        self.assertEqual(cast_store.load_voice_genders(), {"Bea.wav": "female"})


class TestBuildConfig(unittest.TestCase):

    def test_config_targets_chatterbox_with_one_worker(self):
        config = chatterbox_ui.build_config(
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [3, 4], 0.35, 0.9, True, True, False,
            "auto", "double", False, False, None, "INFO")
        self.assertEqual((config.tts, config.output_format, config.worker_count), ("openai", "mp3", 1))
        self.assertEqual((config.voice_name, config.speed), ("Elena.wav", 1.0))
        self.assertEqual((config.chapter_start, config.chapter_end, config.chapter_selection), (1, -1, [3, 4]))
        self.assertIsNone(config.instructions)
        self.assertTrue(config.skip_existing)
        self.assertFalse(config.preview)
        self.assertEqual((config.sentence_pause_ms, config.paragraph_pause_ms, config.output_m4b), (350, 900, True))
        self.assertEqual(config.language, "en")
        self.assertEqual(config.paced_unit_mode, "sentence")

    def test_paragraph_narration_units_pass_through(self):
        config = chatterbox_ui.build_config(
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [3], 0.35, 0.9, True, False, False,
            "auto", "double", False, False, None, "INFO", "paragraph")
        self.assertEqual(config.paced_unit_mode, "paragraph")


TABLE = [[1, False, "Title page", "Title", "under 1 min"], [2, True, "One", "It was", "26 min"],
         [3, True, "Two", "It was", "25 min"]]
SETTINGS = ("audiobook_output/out", "Elena.wav", 1.0, 0.35, 0.9, True, False, False, "auto", "double", False,
            False, None, "INFO")


class TestQueueSettings(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.uploads = patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(self.tmp.name, "uploads"))
        self.uploads.start()
        # These tests use "/library/book.epub" as a stand-in library pick (F-21 requires it be
        # inside the configured library folder).
        self.env = patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": "/library"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.uploads.stop()
        self.tmp.cleanup()

    def _settings(self, library_book, input_file, table=TABLE):
        with patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub" or os.path.exists(p)):
            return chatterbox_ui.queue_settings(library_book, input_file, table, *SETTINGS)

    def test_library_book_settings(self):
        settings = self._settings("/library/book.epub", None)
        self.assertEqual(settings["input_file"], "/library/book.epub")
        self.assertEqual(settings["chapter_selection"], [2, 3])
        self.assertEqual((settings["voice"], settings["sentence_pause"], settings["paragraph_pause"],
                          settings["output_m4b"]), ("Elena.wav", 0.35, 0.9, True))
        config = chatterbox_ui.build_config(**settings)  # queued settings rebuild a full config
        self.assertEqual((config.chapter_selection, config.paragraph_pause_ms), ([2, 3], 900))

    def test_narration_units_are_queued_and_old_jobs_default_to_sentences(self):
        with patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub"):
            settings = chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS, "paragraph")
        self.assertEqual(settings["paced_unit_mode"], "paragraph")
        self.assertEqual(chatterbox_ui.build_config(**settings).paced_unit_mode, "paragraph")
        del settings["paced_unit_mode"]  # a job queued before the option existed
        self.assertEqual(chatterbox_ui.build_config(**settings).paced_unit_mode, "sentence")

    def test_dataframe_table_is_read(self):
        import pandas as pd
        table = pd.DataFrame(TABLE, columns=chatterbox_ui.CHAPTER_COLUMNS)
        self.assertEqual(self._settings("/library/book.epub", None, table)["chapter_selection"], [2, 3])

    def test_upload_is_copied_so_it_outlives_the_browser_upload(self):
        upload = os.path.join(self.tmp.name, "mine.epub")
        with open(upload, "w") as f:
            f.write("epub")
        settings = self._settings(None, upload)
        self.assertNotEqual(settings["input_file"], upload)
        self.assertTrue(settings["input_file"].startswith(os.path.abspath(chatterbox_ui.QUEUE_UPLOADS)))
        with open(settings["input_file"]) as f:
            self.assertEqual(f.read(), "epub")

    def test_library_pick_wins_over_upload(self):
        upload = os.path.join(self.tmp.name, "mine.epub")
        open(upload, "w").close()
        self.assertEqual(self._settings("/library/book.epub", upload)["input_file"], "/library/book.epub")

    def test_nothing_ticked_is_an_error(self):
        with self.assertRaises(gr.Error):
            self._settings("/library/book.epub", None, [[1, False, "A", "", ""]])

    def test_typed_text_that_is_not_a_book_is_an_error(self):
        with self.assertRaises(gr.Error):
            self._settings("detour", None)

    def test_no_book_is_an_error(self):
        with self.assertRaises(gr.Error):
            self._settings(None, None)

    def test_generation_estimate_counts_ticked_chapters_only(self):
        stats = [[20, 1, 1], [36360, 1, 1], [36360, 1, 1]]  # 36,360 chars = 30 min of speech
        seconds = chatterbox_ui.generation_estimate(TABLE, stats)
        self.assertAlmostEqual(seconds, 2 * 1800 / chatterbox_ui.GENERATION_SPEED
                               * chatterbox_ui.PACED_GENERATION_OVERHEAD, delta=1)


class TestOutputDirSafety(unittest.TestCase):
    """F-12 (folder collisions) and F-21 (path containment)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "audiobook_output")
        os.makedirs(self.root)
        self.library = os.path.join(self.tmp.name, "library")
        os.makedirs(self.library)
        self.book = os.path.join(self.library, "book.epub")
        open(self.book, "w").close()
        self.patches = [
            patch.object(chatterbox_ui, "OUTPUT_ROOT", self.root),
            patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": self.library}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _settings(self, output_dir, library_book=None, skip_existing=False, active_jobs=None):
        library_book = library_book if library_book is not None else self.book
        return chatterbox_ui.queue_settings(library_book, None, TABLE, output_dir, "Elena.wav", 1.0, 0.35, 0.9,
                                            True, skip_existing, False, "auto", "double", False, False, None,
                                            "INFO", active_jobs=active_jobs)

    def test_output_dir_must_be_inside_output_root(self):
        with self.assertRaises(gr.Error):
            self._settings(os.path.join(self.tmp.name, "elsewhere"))

    def test_output_dir_inside_output_root_is_accepted(self):
        settings = self._settings(os.path.join(self.root, "Book"))
        self.assertEqual(settings["output_dir"], os.path.join(self.root, "Book"))

    def test_library_book_must_be_inside_library_dir(self):
        outside = os.path.join(self.tmp.name, "outside.epub")
        open(outside, "w").close()
        with self.assertRaises(gr.Error):
            self._settings(os.path.join(self.root, "Book"), library_book=outside)

    def test_refuses_a_folder_already_queued(self):
        target = os.path.join(self.root, "Book")
        active = [{"status": QUEUED, "title": "Book (already queued)", "settings": {"output_dir": target}}]
        with self.assertRaises(gr.Error):
            self._settings(target, active_jobs=active)

    def test_a_different_folder_is_unaffected_by_other_queued_jobs(self):
        active = [{"status": QUEUED, "title": "Other book",
                  "settings": {"output_dir": os.path.join(self.root, "Other")}}]
        settings = self._settings(os.path.join(self.root, "Book"), active_jobs=active)
        self.assertEqual(settings["output_dir"], os.path.join(self.root, "Book"))

    def test_refuses_a_folder_with_an_existing_m4b_unless_skip_existing(self):
        target = os.path.join(self.root, "Book")
        os.makedirs(target)
        open(os.path.join(target, "Book.m4b"), "w").close()
        with self.assertRaises(gr.Error):
            self._settings(target)
        # ticking "Skip chapters already made" is the documented way to resume into the same folder
        settings = self._settings(target, skip_existing=True)
        self.assertEqual(settings["output_dir"], target)

    def test_refuses_a_folder_with_a_leftover_chapters_work_folder(self):
        target = os.path.join(self.root, "Book")
        os.makedirs(os.path.join(target, chatterbox_ui.CHAPTER_WORK_FOLDER))
        with self.assertRaises(gr.Error):
            self._settings(target)


class TestUploadValidationOrder(unittest.TestCase):
    """F-31: an upload copy must never be orphaned by a validation failure discovered later."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.uploads = os.path.join(self.tmp.name, "uploads")
        self.patch = patch.object(chatterbox_ui, "QUEUE_UPLOADS", self.uploads)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_a_missing_replace_file_leaves_no_orphaned_epub_copy(self):
        upload = os.path.join(self.tmp.name, "mine.epub")
        with open(upload, "w") as f:
            f.write("epub")
        missing_replace_file = os.path.join(self.tmp.name, "gone.txt")  # never created
        with self.assertRaises(Exception):
            chatterbox_ui.queue_settings(None, upload, TABLE, "audiobook_output/out", "Elena.wav", 1.0, 0.35,
                                         0.9, True, False, False, "auto", "double", False, False,
                                         missing_replace_file, "INFO")
        self.assertEqual(os.listdir(self.uploads) if os.path.isdir(self.uploads) else [], [])

    def test_sweep_deletes_files_no_job_references(self):
        os.makedirs(self.uploads, exist_ok=True)
        orphan = os.path.join(self.uploads, "upload_orphan.epub")
        open(orphan, "w").close()
        referenced = os.path.join(self.uploads, "upload_kept.epub")
        open(referenced, "w").close()
        queue = MagicMock()
        queue.jobs.return_value = [{"settings": {"input_file": referenced, "search_and_replace_file": None}}]
        chatterbox_ui.sweep_orphaned_uploads(queue)
        self.assertFalse(os.path.isfile(orphan))
        self.assertTrue(os.path.isfile(referenced))

    def test_sweep_is_a_no_op_without_an_uploads_folder(self):
        queue = MagicMock()
        queue.jobs.return_value = []
        chatterbox_ui.sweep_orphaned_uploads(queue)  # must not raise just because nothing was ever uploaded


class TestQueueView(unittest.TestCase):

    def test_rows_and_status(self):
        queue = MagicMock()
        queue.paused = False
        queue.preparing = False
        queue.jobs.return_value = [
            {"id": "a", "title": "Book A", "voice": "Elena.wav", "chapters": 4, "estimate_seconds": 3600,
             "status": "done", "finished": "2026-09-27 10:00", "note": "", "settings": {}},
            {"id": "b", "title": "Book B", "voice": "narrator two.wav", "chapters": 5, "estimate_seconds": 1800,
             "status": "queued", "note": "", "settings": {}},
        ]
        rows, ids, status = chatterbox_ui.queue_view(queue)
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(rows[1][:4], [2, "Book B", "narrator two", 5])
        self.assertIn("done", rows[0][4])
        self.assertIn("1 book(s) to go", status)
        self.assertIn("30 min", status)

    def test_empty_and_paused(self):
        queue = MagicMock()
        queue.paused, queue.preparing, queue.jobs.return_value = True, False, []
        self.assertIn("empty", chatterbox_ui.queue_view(queue)[2])

    def test_preparing_explains_next_step(self):
        queue = MagicMock()
        queue.paused, queue.preparing = False, True
        queue.jobs.return_value = []
        self.assertIn("Add this book to queue", chatterbox_ui.queue_view(queue)[2])
        queue.jobs.return_value = [
            {"id": "a", "title": "Book A", "voice": "Elena.wav", "chapters": 4,
             "estimate_seconds": 3600, "status": "queued", "note": "", "settings": {}},
        ]
        self.assertIn("Start queued books", chatterbox_ui.queue_view(queue)[2])


class TestChapterList(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.book = os.path.join(self.tmp.name, "collection.epub")
        _write_collection_epub(self.book)

    def tearDown(self):
        self.tmp.cleanup()

    def test_front_matter_is_unticked_and_story_ticked(self):
        table, stats, summary = chatterbox_ui.chapter_overview(
            self.book, None, 1.0, 0.35, 0.9, "chatterbox", "auto", "double", False, False, None)
        rows = table["value"]
        self.assertEqual([row[1] for row in rows], [False, False, True, True])
        self.assertEqual(len(stats), 4)
        self.assertIn("2 of 4 chapters", summary)
        self.assertIn("numbered 1–2", summary)

    def test_no_book_hides_the_table(self):
        table, stats, summary = chatterbox_ui.chapter_overview(
            None, None, 1.0, 0.35, 0.9, "chatterbox", "auto", "double", False, False, None)
        self.assertFalse(table["visible"])
        self.assertEqual((stats, summary), ([], ""))

    def test_typed_text_that_is_not_a_book_explains_itself(self):
        # F-42c: typed text with no upload used to hide the table with an empty message.
        table, stats, summary = chatterbox_ui.chapter_overview(
            "detour", None, 1.0, 0.35, 0.9, "chatterbox", "auto", "double", False, False, None)
        self.assertFalse(table["visible"])
        self.assertEqual(stats, [])
        self.assertNotEqual(summary, "")
        self.assertIn("Pick the book from the list", summary)

    def test_summary_follows_ticks_speed_and_pauses(self):
        stats = [[20, 1, 1], [30300, 1, 1], [30300, 1, 1]]  # 30,300 chars = 25 min of speech at 1.0x
        rows = [[1, False, "T", "", ""], [2, True, "A", "", ""], [3, True, "B", "", ""]]
        self.assertIn("50 min", chatterbox_ui.chapter_summary(rows, stats, 1.0, 0.35, 0.9))
        self.assertIn("40 min", chatterbox_ui.chapter_summary(rows, stats, 1.25, 0.35, 0.9))
        rows[2][1] = False
        self.assertIn("1 of 3 chapters", chatterbox_ui.chapter_summary(rows, stats, 1.0, 0.35, 0.9))

    def test_pauses_add_to_listening_time(self):
        # 400 sentences in 100 paragraphs: 300 sentence pauses + 99 paragraph pauses
        self.assertAlmostEqual(chatterbox_ui.listening_seconds([30300, 400, 100], 1.0, 0.5, 1.0),
                               1500 + 150 + 99, places=3)
        self.assertAlmostEqual(chatterbox_ui.listening_seconds([30300, 400, 100], 2.0, 0.5, 1.0),
                               (1500 + 150 + 99) / 2, places=3)

    def test_chapter_stats(self):
        text = f"One. Two! Three?{chatterbox_ui.PARAGRAPH_MARK} Four... \u201cFive.\u201d"
        self.assertEqual(chatterbox_ui.chapter_stats(text)[1:], [5, 2, 1])  # 5 sentences, 2 paragraphs, 1 quoted line

    def test_speed_change_retimes_but_keeps_ticks(self):
        rows = [[1, False, "T", "", "x"], [2, True, "A", "", "x"]]
        updated = chatterbox_ui.retime_chapters(rows, [[20, 1, 1], [30300, 1, 1]], 2.0, 0.35, 0.9)["value"]
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
        self.assertIn("Voice sample: ~10 s of one person speaking clearly", labels)
        for removed in ("Azure", "Edge", "Piper", "Model", "Voice Instructions", "Worker Count", "Output Format"):
            self.assertNotIn(removed, labels)

    def test_voice_lab_explains_that_save_also_waits(self):
        # F-42d: only a preview waiting for the current chunk used to be mentioned.
        ui = chatterbox_ui.build_ui()
        texts = [block.value for block in ui.blocks.values() if isinstance(getattr(block, "value", None), str)]
        joined = " ".join(texts)
        self.assertIn("waits for the current chunk", joined)
        self.assertIn("Save", joined)


class TestUploadTableSequencing(unittest.TestCase):
    """F-42b: uploading a book must build the chapter table once, from the upload -- not once (briefly,
    wrongly) from the previous library pick and again from the upload."""

    def test_input_file_change_does_not_directly_trigger_chapter_overview(self):
        ui = chatterbox_ui.build_ui()
        input_file_id = next(block._id for block in ui.blocks.values()
                             if getattr(block, "label", None) == "EPUB file")
        direct_triggers = {
            getattr(fn.fn, "__name__", None)
            for fn in ui.fns.values()
            for target_id, event in getattr(fn, "targets", [])
            if target_id == input_file_id and event == "change"
        }
        self.assertIn("uploaded_book_selected", direct_triggers)
        self.assertNotIn("chapter_overview", direct_triggers)

    def test_chapter_overview_is_still_reachable_after_an_upload(self):
        # The .then() chain must exist somewhere, or an upload would never refresh the table at all.
        ui = chatterbox_ui.build_ui()
        names = {getattr(fn.fn, "__name__", None) for fn in ui.fns.values()}
        self.assertIn("chapter_overview", names)


class TestKokoroVoiceChoices(unittest.TestCase):

    VOICES_PAYLOAD = {
        "default_voice": "af_heart",
        "voices": [
            {"id": "af_heart", "name": "af_heart", "overall_grade": "A"},
            {"id": "af_v0bella", "name": "af_v0bella"},
            {"id": "bm_george", "name": "bm_george", "overall_grade": "C+"},
            {"id": "am_liam", "name": "am_liam", "overall_grade": "B-"},
            {"id": "jf_alpha", "name": "jf_alpha", "overall_grade": "A"},  # Japanese: excluded
        ],
    }

    def _choices(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1"}), \
                patch("urllib.request.urlopen",
                     return_value=_fake_response(json.dumps(self.VOICES_PAYLOAD).encode())):
            return chatterbox_ui.kokoro_voices_and_default()

    def test_only_english_prefixes_are_offered(self):
        choices, default = self._choices()
        values = [v for _, v in choices]
        self.assertNotIn("jf_alpha", values)
        self.assertEqual(set(values), {"af_heart", "af_v0bella", "bm_george", "am_liam"})
        self.assertEqual(default, "af_heart")

    def test_best_grade_first_ungraded_last(self):
        choices, _ = self._choices()
        values = [v for _, v in choices]
        self.assertEqual(values, ["af_heart", "am_liam", "bm_george", "af_v0bella"])

    def test_label_format_matches_id_plus_grade(self):
        choices, _ = self._choices()
        labels = {value: label for label, value in choices}
        self.assertEqual(labels["af_heart"], "Heart · American female · A")

    def test_configured_default_wins_when_offered(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1", "KOKORO_DEFAULT_VOICE": "bm_george"}), \
                patch("urllib.request.urlopen",
                     return_value=_fake_response(json.dumps(self.VOICES_PAYLOAD).encode())):
            _, default = chatterbox_ui.kokoro_voices_and_default()
        self.assertEqual(default, "bm_george")

    def test_configured_default_not_offered_falls_back_to_server_default(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1", "KOKORO_DEFAULT_VOICE": "zz_missing"}), \
                patch("urllib.request.urlopen",
                     return_value=_fake_response(json.dumps(self.VOICES_PAYLOAD).encode())):
            _, default = chatterbox_ui.kokoro_voices_and_default()
        self.assertEqual(default, "af_heart")

    def test_unreachable_server_warns_and_falls_back_to_built_in_default(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1", "KOKORO_DEFAULT_VOICE": ""}), \
                patch("urllib.request.urlopen", side_effect=OSError("no route to host")), \
                patch("gradio.Warning") as warning:
            choices, default = chatterbox_ui.kokoro_voices_and_default()
        warning.assert_called_once()
        self.assertEqual(choices, [("af_heart", "af_heart")])
        self.assertEqual(default, "af_heart")

    def test_unreachable_server_falls_back_to_configured_default(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1", "KOKORO_DEFAULT_VOICE": "am_liam"}), \
                patch("urllib.request.urlopen", side_effect=OSError("no route to host")), \
                patch("gradio.Warning"):
            choices, default = chatterbox_ui.kokoro_voices_and_default()
        self.assertEqual(choices, [("am_liam", "am_liam")])
        self.assertEqual(default, "am_liam")


class TestBuildConfigEngine(unittest.TestCase):

    def test_kokoro_engine_sets_model_and_base_url(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1"}):
            config = chatterbox_ui.build_config(
                "/tmp/book.epub", "audiobook_output/Book", "af_heart", 1.0, [1], 0.35, 0.9, True, True, False,
                "auto", "double", False, False, None, "INFO", "paragraph", "kokoro")
        self.assertEqual(config.model_name, "kokoro")
        self.assertEqual(config.openai_base_url, "http://kokoro:8880/v1")
        self.assertEqual(config.voice_name, "af_heart")
        # Kokoro always narrates by sentence, even though "paragraph" was requested.
        self.assertEqual(config.paced_unit_mode, "sentence")

    def test_chatterbox_is_the_default_engine(self):
        config = chatterbox_ui.build_config(
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [1], 0.35, 0.9, True, True, False,
            "auto", "double", False, False, None, "INFO")
        self.assertEqual((config.model_name, config.openai_base_url), ("chatterbox", None))
        self.assertEqual(config.paced_unit_mode, "sentence")

    def test_chatterbox_engine_keeps_requested_paced_unit_mode(self):
        config = chatterbox_ui.build_config(
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [1], 0.35, 0.9, True, True, False,
            "auto", "double", False, False, None, "INFO", "paragraph", "chatterbox")
        self.assertEqual(config.paced_unit_mode, "paragraph")

    def test_kokoro_without_base_url_raises_value_error(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": ""}):
            with self.assertRaises(ValueError):
                chatterbox_ui.build_config(
                    "/tmp/book.epub", "audiobook_output/Book", "af_heart", 1.0, [1], 0.35, 0.9, True, True, False,
                    "auto", "double", False, False, None, "INFO", "sentence", "kokoro")


class TestQueueSettingsEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.uploads = patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(self.tmp.name, "uploads"))
        self.uploads.start()
        self.env = patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": "/library",
                                           "KOKORO_BASE_URL": "http://kokoro:8880/v1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.uploads.stop()
        self.tmp.cleanup()

    def test_engine_is_persisted_and_flows_into_build_config(self):
        with patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub"):
            settings = chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS,
                                                    "sentence", "kokoro")
        self.assertEqual(settings["engine"], "kokoro")
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.model_name, config.openai_base_url), ("kokoro", "http://kokoro:8880/v1"))

    def test_missing_engine_key_builds_as_chatterbox(self):
        # A job queued before Kokoro existed has no "engine" key at all.
        with patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub"):
            settings = chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS)
        self.assertEqual(settings["engine"], "chatterbox")
        del settings["engine"]
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual(config.model_name, "chatterbox")

    def test_kokoro_without_base_url_is_refused_at_enqueue_time(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": ""}), \
                patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub"):
            with self.assertRaises(gr.Error):
                chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS, "sentence", "kokoro")


class TestEngineAwareEstimates(unittest.TestCase):

    def test_generation_estimate_uses_kokoro_constants(self):
        stats = [[1690, 1, 1]]
        table = [[1, True, "A", "", ""]]
        expected = (1690 / chatterbox_ui.KOKORO_CHARS_PER_AUDIO_SECOND / chatterbox_ui.KOKORO_GENERATION_SPEED
                   * chatterbox_ui.KOKORO_PACED_GENERATION_OVERHEAD)
        self.assertAlmostEqual(chatterbox_ui.generation_estimate(table, stats, "kokoro"), expected, delta=0.01)

    def test_generation_estimate_defaults_to_chatterbox(self):
        stats, table = [[1000, 1, 1]], [[1, True, "A", "", ""]]
        self.assertEqual(chatterbox_ui.generation_estimate(table, stats),
                         chatterbox_ui.generation_estimate(table, stats, "chatterbox"))
        self.assertNotAlmostEqual(chatterbox_ui.generation_estimate(table, stats, "chatterbox"),
                                  chatterbox_ui.generation_estimate(table, stats, "kokoro"), delta=0.001)

    def test_listening_seconds_uses_kokoro_chars_per_second(self):
        self.assertAlmostEqual(chatterbox_ui.listening_seconds([1000, 1, 1], 1.0, 0, 0, "chatterbox"),
                               1000 / chatterbox_ui.CHARS_PER_AUDIO_SECOND, places=3)
        self.assertAlmostEqual(chatterbox_ui.listening_seconds([1000, 1, 1], 1.0, 0, 0, "kokoro"),
                               1000 / chatterbox_ui.KOKORO_CHARS_PER_AUDIO_SECOND, places=3)

    def test_chapter_summary_differs_by_engine(self):
        stats, rows = [[1690, 1, 1]], [[1, True, "A", "", ""]]
        chatterbox_summary = chatterbox_ui.chapter_summary(rows, stats, 1.0, 0, 0, "chatterbox")
        kokoro_summary = chatterbox_ui.chapter_summary(rows, stats, 1.0, 0, 0, "kokoro")
        self.assertNotEqual(chatterbox_summary, kokoro_summary)

    def test_retime_chapters_differs_by_engine(self):
        rows, stats = [[1, True, "A", "", "x"]], [[169000, 1, 1]]
        chatterbox_row = chatterbox_ui.retime_chapters(rows, stats, 1.0, 0, 0, "chatterbox")["value"][0][4]
        kokoro_row = chatterbox_ui.retime_chapters([[1, True, "A", "", "x"]], stats, 1.0, 0, 0, "kokoro")["value"][0][4]
        self.assertNotEqual(chatterbox_row, kokoro_row)


class TestQueueViewEngine(unittest.TestCase):

    def test_kokoro_voice_shown_with_engine_prefix(self):
        queue = MagicMock()
        queue.paused = False
        queue.jobs.return_value = [
            {"id": "a", "title": "Book A", "voice": "af_heart", "chapters": 4, "estimate_seconds": 60,
             "status": "queued", "note": "", "settings": {"engine": "kokoro"}},
        ]
        rows, _, _ = chatterbox_ui.queue_view(queue)
        self.assertEqual(rows[0][2], "Kokoro · af_heart")

    def test_chatterbox_voice_still_strips_the_extension(self):
        queue = MagicMock()
        queue.paused = False
        queue.jobs.return_value = [
            {"id": "a", "title": "Book A", "voice": "Elena.wav", "chapters": 4, "estimate_seconds": 60,
             "status": "queued", "note": "", "settings": {"engine": "chatterbox"}},
        ]
        rows, _, _ = chatterbox_ui.queue_view(queue)
        self.assertEqual(rows[0][2], "Elena")


class TestEngineGatedVoiceSync(unittest.TestCase):

    def test_sync_passes_through_while_chatterbox(self):
        update = chatterbox_ui._sync_if_chatterbox("Elena.wav", "chatterbox")
        self.assertEqual(update["value"], "Elena.wav")

    def test_sync_is_a_no_op_while_kokoro(self):
        update = chatterbox_ui._sync_if_chatterbox("af_heart", "kokoro")
        self.assertNotIn("value", update)


class TestAddVoiceEngineGating(unittest.TestCase):
    """A Chatterbox file name must never land in the Make-tab dropdown while Kokoro is selected
    there; the Voice lab (always Chatterbox) is updated regardless."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.mkdir(self.voices)
        self.sample = os.path.join(self.tmp.name, "sample.wav")
        _sample_with_pauses(self.sample)
        self.env = patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_chatterbox_engine_updates_make_tab_dropdown(self):
        _, _, make_tab_update, _ = chatterbox_ui.add_voice(self.sample, "One", True, False, "chatterbox")
        self.assertEqual(make_tab_update["value"], "One.wav")

    def test_kokoro_engine_leaves_make_tab_dropdown_untouched(self):
        _, lab_update, make_tab_update, delete_update = chatterbox_ui.add_voice(
            self.sample, "Two", True, False, "kokoro")
        self.assertEqual(lab_update["value"], "Two.wav")
        self.assertNotIn("value", make_tab_update)
        self.assertNotIn("choices", make_tab_update)
        self.assertIn(("Two", "Two.wav"), delete_update["choices"])

    def test_default_engine_argument_is_chatterbox(self):
        # Existing callers that predate the engine parameter must keep updating the Make tab.
        _, _, make_tab_update, _ = chatterbox_ui.add_voice(self.sample, "Three", True, False)
        self.assertEqual(make_tab_update["value"], "Three.wav")


class TestSampleVoice(unittest.TestCase):

    def tearDown(self):
        chatterbox_ui._delete_if_exists(chatterbox_ui._current_preview_path)
        chatterbox_ui._current_preview_path = None

    def test_no_voice_is_an_error(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.sample_voice("chatterbox", "", 1.0)

    def test_chatterbox_uses_saved_settings_through_preview_voice(self):
        saved = {"exaggeration": 0.61, "cfg_weight": 0.4, "temperature": 0.9}
        with patch.object(chatterbox_ui, "read_saved_settings", return_value=saved), \
                patch.object(chatterbox_ui, "preview_voice", return_value="/tmp/x.mp3") as preview:
            path = chatterbox_ui.sample_voice("chatterbox", "Elena.wav", 1.25)
        self.assertEqual(path, "/tmp/x.mp3")
        preview.assert_called_once_with("Elena.wav", chatterbox_ui.PREVIEW_PHRASE, 0.61, 0.4, 0.9, 1.25)

    def test_kokoro_posts_to_its_own_speech_endpoint(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1"}), \
                patch("urllib.request.urlopen", return_value=_fake_response(b"mp3-bytes")) as urlopen:
            path = chatterbox_ui.sample_voice("kokoro", "af_heart", 1.0)
        request = urlopen.call_args[0][0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://kokoro:8880/v1/audio/speech")
        self.assertEqual(payload["model"], "kokoro")
        self.assertEqual(payload["voice"], "af_heart")
        self.assertEqual(payload["input"], chatterbox_ui.PREVIEW_PHRASE)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"mp3-bytes")

    def test_kokoro_not_configured_is_an_error(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": ""}):
            with self.assertRaises(gr.Error):
                chatterbox_ui.sample_voice("kokoro", "af_heart", 1.0)


class TestBuiltInVoicesMatchChatterboxFolder(unittest.TestCase):
    """BUILT_IN_VOICES must list exactly what's shipped in chatterbox/voices/, so a real
    built-in voice can never be mistakenly offered for deletion (or a custom one wrongly
    protected)."""

    def test_constant_matches_the_voices_folder(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        voices_dir = os.path.join(repo_root, "chatterbox", "voices")
        if not os.path.isdir(voices_dir):
            self.skipTest("chatterbox/voices/ is not present in this checkout")
        on_disk = {name for name in os.listdir(voices_dir) if name.lower().endswith(".wav")}
        self.assertEqual(chatterbox_ui.BUILT_IN_VOICES, on_disk)


class TestOwnVoiceChoices(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.mkdir(self.voices)
        for name in ("MyVoice.wav", "Elena.wav", "notes.txt"):
            open(os.path.join(self.voices, name), "wb").close()
        self.env = patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_excludes_built_ins_and_non_voice_files(self):
        self.assertEqual(chatterbox_ui.own_voice_names(), ["MyVoice.wav"])

    def test_choices_are_label_value_pairs(self):
        self.assertEqual(chatterbox_ui.own_voice_choices(), [("MyVoice", "MyVoice.wav")])

    def test_empty_when_folder_not_mounted(self):
        with patch.dict(os.environ, {"TTS_VOICES_DIR": ""}):
            self.assertEqual(chatterbox_ui.own_voice_names(), [])


class TestVoiceDropdownAfterDelete(unittest.TestCase):

    def test_reselects_default_when_the_deleted_voice_was_selected(self):
        update = chatterbox_ui._voice_dropdown_after_delete(["a", "b"], "MyVoice.wav", "MyVoice.wav", "Elena.wav")
        self.assertEqual(update["value"], "Elena.wav")

    def test_keeps_current_selection_when_a_different_voice_was_deleted(self):
        update = chatterbox_ui._voice_dropdown_after_delete(["a", "b"], "Elena.wav", "MyVoice.wav", "Elena.wav")
        self.assertNotIn("value", update)


class TestDeleteOwnVoice(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.mkdir(self.voices)
        open(os.path.join(self.voices, "MyVoice.wav"), "wb").close()
        open(os.path.join(self.voices, "Elena.wav"), "wb").close()  # a built-in name, present on disk
        self.env = patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_deletes_an_own_voice(self):
        message = chatterbox_ui.delete_own_voice("MyVoice.wav", [])
        self.assertIn("Deleted", message)
        self.assertFalse(os.path.isfile(os.path.join(self.voices, "MyVoice.wav")))

    def test_refuses_a_built_in_voice_even_if_present_on_disk(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice("Elena.wav", [])
        self.assertTrue(os.path.isfile(os.path.join(self.voices, "Elena.wav")))

    def test_refuses_path_traversal(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice("../MyVoice.wav", [])
        self.assertTrue(os.path.isfile(os.path.join(self.voices, "MyVoice.wav")))

    def test_refuses_a_name_with_a_forward_slash(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice("sub/MyVoice.wav", [])

    def test_refuses_a_name_with_a_backslash(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice("sub\\MyVoice.wav", [])

    def test_refuses_a_missing_file(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice("NoSuchVoice.wav", [])

    def test_refuses_none(self):
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice(None, [])

    def test_refuses_a_voice_used_by_a_queued_chatterbox_job_and_names_the_book(self):
        jobs = [{"status": QUEUED, "title": "Invented Story",
                "settings": {"engine": "chatterbox", "voice": "MyVoice.wav"}}]
        with self.assertRaises(gr.Error) as ctx:
            chatterbox_ui.delete_own_voice("MyVoice.wav", jobs)
        self.assertIn("Invented Story", str(ctx.exception))
        self.assertTrue(os.path.isfile(os.path.join(self.voices, "MyVoice.wav")))

    def test_refuses_a_voice_used_by_a_running_chatterbox_job(self):
        jobs = [{"status": RUNNING, "title": "Invented Story",
                "settings": {"engine": "chatterbox", "voice": "MyVoice.wav"}}]
        with self.assertRaises(gr.Error):
            chatterbox_ui.delete_own_voice("MyVoice.wav", jobs)

    def test_a_finished_job_does_not_block_deletion(self):
        jobs = [{"status": DONE, "title": "Invented Story",
                "settings": {"engine": "chatterbox", "voice": "MyVoice.wav"}}]
        chatterbox_ui.delete_own_voice("MyVoice.wav", jobs)  # must not raise

    def test_a_kokoro_job_with_the_same_looking_voice_name_does_not_block_deletion(self):
        jobs = [{"status": QUEUED, "title": "Invented Story",
                "settings": {"engine": "kokoro", "voice": "MyVoice.wav"}}]
        chatterbox_ui.delete_own_voice("MyVoice.wav", jobs)  # must not raise

    def test_voices_dir_not_mounted_is_an_error(self):
        with patch.dict(os.environ, {"TTS_VOICES_DIR": ""}):
            with self.assertRaises(gr.Error):
                chatterbox_ui.delete_own_voice("MyVoice.wav", [])


class TestDeleteVoiceWiring(unittest.TestCase):
    """The delete button must ask for browser confirmation before any server call, and treat a
    cancelled confirm (the js= returning null) as a complete no-op."""

    def test_delete_button_click_confirms_in_the_browser_first(self):
        ui = chatterbox_ui.build_ui()
        delete_button_id = next(block._id for block in ui.blocks.values()
                                if getattr(block, "value", None) == "Delete voice")
        triggers = [fn for fn in ui.fns.values()
                   for target_id, event in getattr(fn, "targets", [])
                   if target_id == delete_button_id and event == "click"]
        self.assertEqual(len(triggers), 1)
        self.assertIn("confirm(", triggers[0].js)
        self.assertEqual(getattr(triggers[0].fn, "__name__", None), "delete_voice")

    def test_cancelled_confirm_changes_nothing(self):
        ui = chatterbox_ui.build_ui()
        delete_button_id = next(block._id for block in ui.blocks.values()
                                if getattr(block, "value", None) == "Delete voice")
        handler = next(fn.fn for fn in ui.fns.values()
                      for target_id, event in getattr(fn, "targets", [])
                      if target_id == delete_button_id and event == "click")
        result = handler(None, "Elena.wav", "Elena.wav", "chatterbox")
        self.assertEqual(len(result), 4)
        for update in result:
            self.assertNotIn("value", update)
            self.assertNotIn("choices", update)

    def test_nothing_picked_skips_the_dialog_and_says_so(self):
        ui = chatterbox_ui.build_ui()
        delete_button_id = next(block._id for block in ui.blocks.values()
                                if getattr(block, "value", None) == "Delete voice")
        trigger = next(fn for fn in ui.fns.values()
                       for target_id, event in getattr(fn, "targets", [])
                       if target_id == delete_button_id and event == "click")
        self.assertIn("!name ? ['', lab, mk, eng]", trigger.js)
        with tempfile.TemporaryDirectory() as voices, patch.dict(os.environ, {"TTS_VOICES_DIR": voices}):
            with self.assertRaises(gr.Error) as ctx:
                trigger.fn("", "Elena.wav", "Elena.wav", "chatterbox")
        self.assertIn("Pick a voice", str(ctx.exception))


class TestHostUiProcessFactory(unittest.TestCase):
    """F-19: job processes are forked from a multithreaded server; host_ui must use the spawn
    context instead of the platform default (fork on Linux)."""

    def test_job_queue_is_built_with_a_spawn_process_factory(self):
        fake_queue = MagicMock()
        with patch.object(chatterbox_ui.library_index, "load_index", return_value={}), \
                patch.object(chatterbox_ui.library_index, "refresh_index"), \
                patch.object(chatterbox_ui, "sweep_voice_previews"), \
                patch.object(chatterbox_ui, "sweep_orphaned_uploads"), \
                patch.object(chatterbox_ui, "build_ui", return_value=MagicMock()), \
                patch.object(chatterbox_ui, "JobQueue", return_value=fake_queue) as job_queue_cls:
            chatterbox_ui.host_ui(MagicMock(host="127.0.0.1", port=7860))
        self.assertEqual(job_queue_cls.call_args.kwargs["process_factory"],
                         multiprocessing.get_context("spawn").Process)


if __name__ == "__main__":
    unittest.main()


class TestVoiceModes(unittest.TestCase):
    """Multi-voice settings travel with a job; old jobs (no voice-mode keys) build as single voice."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(self.tmp.name, "uploads")),
                        patch.object(chatterbox_ui, "CASTS_DIR", os.path.join(self.tmp.name, "casts")),
                        patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": "/library"}),
                        # The book here is a stand-in path and the voices invented; TestQueueTimeCastChecks
                        # covers the coverage and voice-existence checks with a real EPUB and voice folder.
                        patch.object(chatterbox_ui, "cast_coverage_gaps", return_value=[]),
                        patch.object(chatterbox_ui, "engine_voice_ids", return_value=None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _queue(self, *extra, **kwargs):
        with patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub" or os.path.exists(p)):
            return chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS, "sentence", "chatterbox",
                                                *extra, **kwargs)

    def _finished_cast(self, key="k1", voice="Ada.wav"):
        from audiobook_generator.core import cast as cast_store
        cast = cast_store.new_cast(key, "/library/book.epub", "T", "A", "chatterbox", "Elena.wav", [2, 3])
        cast["characters"] = {"ada": {"name": "Ada", "aliases": [], "gender": "female", "age": "adult", "lines": 4,
                                      "voice": voice}}
        cast["status"] = cast_store.STATUS_DONE
        cast_store.save_cast(chatterbox_ui.cast_file_for(key), cast)
        return cast

    def test_old_jobs_build_as_single_voice(self):
        settings = self._queue()
        for key in ("voice_mode", "dialogue_voice", "cast_file"):
            del settings[key]  # a job queued before multi-voice existed
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.voice_mode, config.dialogue_voice, config.cast_file), ("single", None, None))

    def test_delivery_defaults_to_off_and_none(self):
        settings = self._queue()
        self.assertEqual((settings["adaptive_delivery"], settings["delivery_exaggeration"],
                          settings["delivery_cfg_weight"], settings["delivery_temperature"]),
                         (False, None, None, None))
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.adaptive_delivery, config.delivery_exaggeration, config.delivery_cfg_weight,
                          config.delivery_temperature), (False, None, None, None))

    def test_delivery_settings_travel_with_the_job(self):
        settings = self._queue("dialogue", "Tom.wav", None, True, 0.8, 0.45, 0.5)
        self.assertEqual((settings["adaptive_delivery"], settings["delivery_exaggeration"],
                          settings["delivery_cfg_weight"], settings["delivery_temperature"]),
                         (True, 0.8, 0.45, 0.5))
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.adaptive_delivery, config.delivery_exaggeration, config.delivery_cfg_weight,
                          config.delivery_temperature), (True, 0.8, 0.45, 0.5))

    def test_kokoro_engine_zeroes_out_delivery_in_queue_settings_and_build_config(self):
        with patch.dict(os.environ, {"KOKORO_BASE_URL": "http://kokoro:8880/v1"}):
            with patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub" or os.path.exists(p)):
                settings = chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS, "sentence",
                                                        "kokoro", "single", None, None, True, 0.8, 0.45, 0.5)
            self.assertEqual((settings["adaptive_delivery"], settings["delivery_exaggeration"],
                              settings["delivery_cfg_weight"], settings["delivery_temperature"]),
                             (False, None, None, None))
            config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.adaptive_delivery, config.delivery_exaggeration, config.delivery_cfg_weight,
                          config.delivery_temperature), (False, None, None, None))

    def test_old_jobs_without_delivery_keys_build_with_delivery_off(self):
        settings = self._queue()
        for key in ("adaptive_delivery", "delivery_exaggeration", "delivery_cfg_weight", "delivery_temperature"):
            del settings[key]  # a job queued before adaptive delivery existed
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.adaptive_delivery, config.delivery_exaggeration, config.delivery_cfg_weight,
                          config.delivery_temperature), (False, None, None, None))

    def test_default_is_single_voice_and_carries_no_cast(self):
        settings = self._queue()
        self.assertEqual((settings["voice_mode"], settings["dialogue_voice"], settings["cast_file"]), ("single", None, None))

    def test_dialogue_mode_needs_a_dialogue_voice_and_passes_it_through(self):
        with self.assertRaises(gr.Error):
            self._queue("dialogue", None)
        settings = self._queue("dialogue", "Tom.wav")
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.voice_mode, config.dialogue_voice, config.voice_name), ("dialogue", "Tom.wav", "Elena.wav"))

    def test_cast_mode_needs_a_finished_cast(self):
        with self.assertRaises(gr.Error):
            self._queue("cast", "Tom.wav", None)
        with self.assertRaises(gr.Error):
            self._queue("cast", "Tom.wav", "nocast")
        from audiobook_generator.core import cast as cast_store
        running = self._finished_cast("k2")
        running["status"] = cast_store.STATUS_RUNNING
        cast_store.save_cast(chatterbox_ui.cast_file_for("k2"), running)
        with self.assertRaises(gr.Error):
            self._queue("cast", "Tom.wav", "k2")

    def test_cast_mode_snapshots_the_cast_with_the_narrator_and_builds_a_cast_config(self):
        from audiobook_generator.core import cast as cast_store
        self._finished_cast("k1")
        settings = self._queue("cast", "Tom.wav", "k1")
        self.assertTrue(settings["cast_file"].startswith(os.path.abspath(chatterbox_ui.QUEUE_UPLOADS)))
        snapshot = cast_store.load_cast(settings["cast_file"])
        self.assertEqual((snapshot["narrator_voice"], snapshot["characters"]["ada"]["voice"]), ("Elena.wav", "Ada.wav"))
        config = chatterbox_ui.build_config(**settings)
        self.assertEqual((config.voice_mode, config.cast_file), ("cast", settings["cast_file"]))
        # Editing the cast afterwards does not touch the queued snapshot.
        cast = cast_store.load_cast(chatterbox_ui.cast_file_for("k1"))
        cast["characters"]["ada"]["voice"] = "Other.wav"
        cast_store.save_cast(chatterbox_ui.cast_file_for("k1"), cast)
        self.assertEqual(cast_store.load_cast(settings["cast_file"])["characters"]["ada"]["voice"], "Ada.wav")

    def test_cast_voices_must_belong_to_the_engine(self):
        self._finished_cast("k3", voice="af_heart")
        with self.assertRaises(gr.Error):
            self._queue("cast", "Tom.wav", "k3")

    def test_build_config_ignores_a_cast_file_outside_cast_mode(self):
        config = chatterbox_ui.build_config(
            "/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [3], 0.35, 0.9, True, False, False,
            "auto", "double", False, False, None, "INFO", "sentence", "chatterbox", "dialogue", "Tom.wav", "/x.json")
        self.assertEqual((config.voice_mode, config.cast_file), ("dialogue", None))

    def test_cast_mode_choice_needs_the_llm_configured(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": ""}):
            self.assertEqual([v for _, v in chatterbox_ui.voice_mode_choices()], ["single", "dialogue"])
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://llm:11434/v1", "LLM_MODEL": ""}):
            self.assertEqual([v for _, v in chatterbox_ui.voice_mode_choices()], ["single", "dialogue"])
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://llm:11434/v1", "LLM_MODEL": "m"}):
            self.assertEqual([v for _, v in chatterbox_ui.voice_mode_choices()], ["single", "dialogue", "cast"])

    def test_voice_mode_changed_shows_the_right_controls(self):
        self.assertEqual([u["visible"] for u in chatterbox_ui.voice_mode_changed("single")], [False, False])
        self.assertEqual([u["visible"] for u in chatterbox_ui.voice_mode_changed("dialogue")], [True, False])
        self.assertEqual([u["visible"] for u in chatterbox_ui.voice_mode_changed("cast")], [True, True])


class TestCastPanel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.makedirs(self.voices)
        for name in ("Ada.wav", "Bea.wav", "Cal.wav", "Elena.wav"):
            open(os.path.join(self.voices, name), "w").close()
        self.patches = [patch.object(chatterbox_ui, "CASTS_DIR", os.path.join(self.tmp.name, "casts")),
                        patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(self.tmp.name, "uploads")),
                        patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices, "EBOOK_LIBRARY_DIR": "/library",
                                                "LLM_BASE_URL": "http://llm:11434/v1", "LLM_MODEL": "m"}),
                        patch("audiobook_generator.core.cast.VOICE_GENDERS_FILE",
                              os.path.join(self.tmp.name, "voice_genders.json")),
                        patch.object(chatterbox_ui.voice_measure, "VOICE_FEATURES_FILE",
                                     os.path.join(self.tmp.name, "voice_features.json"))]
        for p in self.patches:
            p.start()
        from audiobook_generator.core import cast as cast_store
        self.cast_store = cast_store
        cast_store.save_voice_gender("Ada.wav", "female")
        cast_store.save_voice_gender("Bea.wav", "female")
        cast_store.save_voice_gender("Cal.wav", "male")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _save(self, key, status, characters):
        cast = self.cast_store.new_cast(key, "/library/book.epub", "T", "A", "chatterbox", "Elena.wav", [1])
        cast["characters"] = characters
        cast["status"] = status
        cast["stats"].update(lines=10, unknown_lines=1)
        self.cast_store.save_cast(chatterbox_ui.cast_file_for(key), cast)
        return cast

    def test_no_cast_yet_hides_the_table(self):
        table, keys, status, seen = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertFalse(table["visible"])
        self.assertIn("Analyse selected chapters", status)
        self.assertIsNone(chatterbox_ui.cast_overview(None, "chatterbox", "Elena.wav", None)[3])

    def test_running_analysis_shows_progress(self):
        cast = self._save("k", self.cast_store.STATUS_RUNNING, {})
        cast["chapters_done"] = 1
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        table, _, status, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertFalse(table["visible"])
        self.assertIn("1 of 1", status)

    def test_finished_cast_gets_suggested_voices_saved_and_shown(self):
        self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": ["Annie"], "gender": "female", "age": "adult", "lines": 9, "voice": None},
            "bob": {"name": "Bob", "aliases": [], "gender": "male", "age": "adult", "lines": 3, "voice": None},
        })
        table, keys, status, seen = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertTrue(table["visible"])
        self.assertEqual(keys, ["anne", "bob"])
        rows = table["value"]
        self.assertEqual([r[0] for r in rows], ["Anne", "Bob"])
        self.assertEqual(rows[0][5], "Ada")     # first female voice; Elena (the narrator) is never suggested
        self.assertEqual(rows[1][5], "Cal")
        self.assertEqual(rows[0][7], "Annie")
        self.assertEqual((rows[0][1], rows[0][6]), ("", ""))  # no profile: blank role and "sounds like"
        saved = self.cast_store.load_cast(chatterbox_ui.cast_file_for("k"))
        self.assertEqual((saved["characters"]["anne"]["voice"], saved["characters"]["bob"]["voice"]), ("Ada.wav", "Cal.wav"))
        self.assertIn("9 of 10 lines attributed", status)
        # Unchanged file: the table is left alone on the next refresh.
        again = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", seen)
        self.assertEqual(again[0], gr.update())
        self.assertEqual(again[3], seen)

    def test_summary_shows_mood_counts_when_present(self):
        cast = self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Ada.wav"},
        })
        cast["chapters"]["h1"] = {"number": 1, "title": "One", "lines": {"1": "anne"},
                                  "moods": {"1": "soft", "2": "excited", "3": "excited", "4": "emphatic"}}
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        _, _, status, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertIn("1 soft, 1 emphasized, 2 excited", status)

    def test_summary_omits_mood_counts_when_every_line_is_normal(self):
        cast = self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Ada.wav"},
        })
        cast["chapters"]["h1"] = {"number": 1, "title": "One", "lines": {"1": "anne"}, "moods": {"1": "normal"}}
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        _, _, status, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertNotIn("soft", status)
        self.assertNotIn("excited", status)

    def test_profiles_fill_the_role_and_sounds_like_columns_and_the_summary(self):
        cast = self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Ada.wav",
                     "profile": {"role": "protagonist", "description": "A ferry pilot.", "voice": "calm, low, wry",
                                 "first_line": {"chapter": 1, "text": '"Hold on."'}}},
            "bob": {"name": "Bob", "aliases": [], "gender": "male", "age": "adult", "lines": 3, "voice": "Cal.wav",
                    "profile": {"role": "unknown", "description": "Her deckhand."}},
        })
        cast["stats"]["profiles"] = 2
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        table, _, status, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        rows = table["value"]
        self.assertEqual((rows[0][1], rows[0][6]), ("protagonist", "calm, low, wry"))
        self.assertEqual((rows[1][1], rows[1][6]), ("", ""))
        self.assertIn("2 character profiles", status)
        cast["profile_error"] = "LLM down"
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        self.assertIn("profiles stopped early (LLM down)", chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)[2])

    def test_clicking_a_row_shows_the_profile(self):
        self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Ada.wav",
                     "profile": {"role": "protagonist", "description": "A ferry pilot.", "voice": "calm, low, wry",
                                 "relationships": "Bob's captain", "first_line": {"chapter": 2, "text": '"Hold on."'}}},
            "bob": {"name": "Bob", "aliases": [], "gender": "male", "age": "adult", "lines": 1, "voice": "Cal.wav"},
        })
        evt = MagicMock(index=[0, 0])
        key, editing, _, _, profile, _ = chatterbox_ui.select_cast_row("k", ["anne", "bob"], "chatterbox", evt)
        self.assertEqual(key, "anne")
        self.assertIn("**Anne** · protagonist · female, adult · 9 lines", profile)
        for text in ("A ferry pilot.", "**Sounds like:** calm, low, wry", "**Relationships:** Bob's captain",
                     '**First line** (chapter 2): "Hold on."'):
            self.assertIn(text, profile)
        _, _, _, _, profile, _ = chatterbox_ui.select_cast_row("k", ["anne", "bob"], "chatterbox", MagicMock(index=[1, 0]))
        self.assertIn("**Bob** · male, adult · 1 line", profile)
        self.assertIn("No profile", profile)
        self.assertEqual(chatterbox_ui.select_cast_row("k", ["anne"], "chatterbox", MagicMock(index=[5, 0]))[4], "")

    def test_running_analysis_shows_profile_progress_once_the_chapters_are_done(self):
        cast = self._save("k", self.cast_store.STATUS_RUNNING, {})
        cast.update(chapters_done=1, profiles_total=4, profiles_done=3)
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        _, _, status, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertIn("Writing character profiles**: 3 of 4", status)

    def _measure(self, pitches):
        for voice, f0 in pitches.items():
            chatterbox_ui.voice_measure.save_features(voice, {"f0_median": f0, "f0_range": 8.0, "hnr": 11.0})

    def test_suggestions_match_measured_voices_to_the_profile(self):
        self._measure({"Ada.wav": 160.0, "Bea.wav": 240.0})
        self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": None,
                     "profile": {"voice_targets": {"pitch": "high"}}},
        })
        table, _, _, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertEqual(table["value"][0][5], "Bea")  # without a profile it would be Ada, the first in the list

    def test_suggest_again_rematches_all_but_the_voices_the_owner_saved(self):
        self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Ada.wav",
                     "profile": {"voice_targets": {"pitch": "high"}}},
            "bob": {"name": "Bob", "aliases": [], "gender": "male", "age": "adult", "lines": 3, "voice": "Ada.wav"},
        })
        chatterbox_ui.apply_cast_edit("k", "bob", "male", "Cal.wav", "chatterbox")  # the owner's own pick
        self.assertEqual(chatterbox_ui.resuggest_cast_voices("k", "chatterbox", "Elena.wav", False),
                         (gr.update(),) * 7)  # cancelled in the browser
        _, _, message, *_ = chatterbox_ui.resuggest_cast_voices("k", "chatterbox", "Elena.wav")
        self.assertIn("No voices are measured yet", message)
        self._measure({"Ada.wav": 160.0, "Bea.wav": 240.0})
        table, keys, message, *narrator = chatterbox_ui.resuggest_cast_voices("k", "chatterbox", "Elena.wav", True)
        self.assertEqual(narrator, [gr.update()] * 4)  # no book tone: the narrator stays as it is
        self.assertEqual(message, "Suggested voices again for 1 character; kept the 1 you picked.")
        saved = self.cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]
        self.assertEqual((saved["anne"]["voice"], saved["bob"]["voice"], saved["bob"]["voice_picked"]),
                         ("Bea.wav", "Cal.wav", True))

    def test_the_profile_shows_what_the_character_wants_and_how_its_voice_measures(self):
        self.cast_store.save_voice_gender("Elena.wav", "female")
        self._measure({"Ada.wav": 160.0, "Bea.wav": 240.0, "Elena.wav": 200.0})
        self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Bea.wav",
                     "profile": {"description": "A lark.", "voice_targets": {"pitch": "high", "delivery": "expressive"}}},
        })
        profile = chatterbox_ui.select_cast_row("k", ["anne"], "chatterbox", MagicMock(index=[0, 0]))[4]
        self.assertIn("**Voice match:** wants high pitch, expressive · Bea is high for a woman", profile)

    def _toned_cast(self):
        self.cast_store.save_voice_gender("Elena.wav", "female")
        self._measure({"Ada.wav": 160.0, "Bea.wav": 240.0, "Elena.wav": 200.0, "Cal.wav": 120.0})
        cast = self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": None,
                     "profile": {"voice_targets": {"pitch": "low"}, "first_line": {"chapter": 1, "text": '"Hold on."'}}},
        })
        cast["book_tone"] = {"point_of_view": "third", "pov_character": "", "tone": "wry and warm", "pace": "brisk",
                             "intensity": "dramatic",
                             "narrator": {"gender": "female", "pitch": "high", "quality": None, "delivery": None}}
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)

    def test_auto_pick_sets_the_narrator_from_the_books_tone_once(self):
        self._toned_cast()
        with patch.object(chatterbox_ui, "read_saved_settings",
                          return_value={"exaggeration": 0.73, "cfg_weight": 0.5, "temperature": 0.61}):
            table, keys, status, seen, voice, exaggeration, cfg, temperature = chatterbox_ui.cast_panel_update(
                "k", "chatterbox", "Elena.wav", None, True)
        self.assertEqual((voice["value"], exaggeration["value"], cfg["value"], temperature["value"]),
                         ("Bea.wav", 0.83, 0.55, 0.61))
        self.assertEqual(table["value"][0][5], "Ada")  # Anne wants low; Bea is the narrator's now
        self.assertIn("**Narrator:** Bea, exaggeration 0.83 · CFG 0.55 · temperature 0.61", status)
        self.assertIn("**Book:** third person · wry and warm · brisk pace · dramatic narration", status)
        # The next refresh (the owner may have changed the Voice since) leaves the narrator alone.
        again = chatterbox_ui.cast_panel_update("k", "chatterbox", "Cal.wav", seen, True)
        self.assertEqual(list(again[4:]), [gr.update()] * 4)
        # Auto-pick off: never touched.
        off = chatterbox_ui.cast_panel_update("k", "chatterbox", "Elena.wav", None, False)
        self.assertEqual(list(off[4:]), [gr.update()] * 4)

    def _analysis_job(self, **later):
        options = {"from_library": True, "output_dir": os.path.join(chatterbox_ui.OUTPUT_ROOT, "Book"),
                   "voice": "Elena.wav", "speed": 1.0, "sentence_pause": 0.35, "paragraph_pause": 0.9,
                   "output_m4b": True, "skip_existing": False, "output_text": False, "paced_unit_mode": "sentence",
                   "dialogue_voice": "Cal.wav", "adaptive_delivery": True, "exaggeration": 0.5, "cfg_weight": 0.5,
                   "temperature": 0.8, "estimate_seconds": 600}
        options.update(later)
        return {"kind": "cast", "title": "Cast: T", "settings": {
            "input_file": "/library/book.epub", "chapter_selection": [1], "title_mode": "auto",
            "newline_mode": "double", "remove_endnotes": False, "remove_reference_numbers": False,
            "search_and_replace_file": None, "engine": "chatterbox", "log_level": "INFO", "cast_key": "k",
            "then_queue": options}}

    def _after_cast(self, job, active_jobs=()):
        queue = MagicMock()
        queue.jobs.return_value = list(active_jobs)
        with patch.object(chatterbox_ui, "read_saved_settings",
                          return_value={"exaggeration": 0.73, "cfg_weight": 0.5, "temperature": 0.61}), \
                patch.object(chatterbox_ui, "cast_coverage_gaps", return_value=[]), \
                patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub" or os.path.exists(p)):
            return chatterbox_ui.queue_book_after_cast(queue, job), queue

    def test_auto_pick_queues_the_book_once_its_cast_is_ready_with_the_tones_narrator(self):
        self._toned_cast()
        note, queue = self._after_cast(self._analysis_job())
        self.assertEqual(note, "book added to the queue")
        title, settings, chapters, estimate, voice = queue.add.call_args.args
        self.assertEqual((title, chapters, estimate, voice), ("Book", 1, 600, "Bea.wav"))
        self.assertEqual((settings["voice_mode"], settings["voice"], settings["dialogue_voice"],
                          settings["chapter_selection"], settings["input_file"]),
                         ("cast", "Bea.wav", "Cal.wav", [1], "/library/book.epub"))
        self.assertEqual((settings["delivery_exaggeration"], settings["delivery_cfg_weight"],
                          settings["delivery_temperature"]), (0.83, 0.55, 0.61))
        snapshot = self.cast_store.load_cast(settings["cast_file"])
        self.assertEqual(snapshot["characters"]["anne"]["voice"], "Ada.wav")
        chatterbox_ui.build_config(**settings)  # a book the queue can start

    def test_without_auto_pick_or_with_a_clash_the_book_is_not_queued(self):
        self._toned_cast()
        job = self._analysis_job()
        del job["settings"]["then_queue"]
        note, queue = self._after_cast(job)
        self.assertIsNone(note)
        queue.add.assert_not_called()
        taken = {"status": "queued", "title": "Other", "settings": {"output_dir": os.path.join(chatterbox_ui.OUTPUT_ROOT, "Book")}}
        note, queue = self._after_cast(self._analysis_job(), [taken])
        self.assertTrue(note.startswith("book not queued: "))
        queue.add.assert_not_called()

    def test_start_shows_while_an_auto_pick_analysis_will_bring_its_book(self):
        queue = MagicMock()
        queue.preparing = True
        pending = {"kind": "cast", "status": "running", "settings": {"then_queue": {"voice": "Elena.wav"}}}
        queue.jobs.return_value = [pending]
        self.assertTrue(chatterbox_ui.start_available(queue))
        queue.jobs.return_value = [dict(pending, settings={})]  # Auto-pick off: nothing to start yet
        self.assertFalse(chatterbox_ui.start_available(queue))
        queue.jobs.return_value = [{"kind": "book", "status": "queued", "settings": {}}]
        self.assertTrue(chatterbox_ui.start_available(queue))
        queue.preparing = False
        self.assertFalse(chatterbox_ui.start_available(queue))

    def test_add_to_queue_is_hidden_in_cast_mode_with_auto_pick(self):
        self.assertFalse(chatterbox_ui.enqueue_button_update("cast", True)["visible"])
        self.assertTrue(chatterbox_ui.enqueue_button_update("cast", False)["visible"])
        self.assertTrue(chatterbox_ui.enqueue_button_update("single", True)["visible"])
        job = {"id": "a", "kind": "cast", "title": "Cast: T", "voice": "Elena.wav", "chapters": 1,
               "estimate_seconds": 1, "status": "done", "finished": "2026-09-29 10:00",
               "note": "book added to the queue", "settings": {}}
        self.assertIn("book added to the queue", chatterbox_ui._status_label(job))

    def test_sample_speaks_the_characters_first_line_with_their_delivery(self):
        self._toned_cast()
        with patch.object(chatterbox_ui, "preview_voice", return_value="/tmp/x.mp3") as preview:
            chatterbox_ui.sample_character("k", "anne", "chatterbox", "Ada.wav", "expressive", 1.0, 0.73, 0.5, 0.61)
        self.assertEqual(preview.call_args.args[:3], ("Ada.wav", '"Hold on."', 0.85))
        chatterbox_ui.apply_cast_edit("k", "anne", "female", "Ada.wav", "chatterbox", "even")
        saved = self.cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]["anne"]
        self.assertEqual(saved["delivery"], "even")
        self.assertIn("a little more even than the book (exaggeration -0.12) (your setting)",
                      chatterbox_ui.character_profile_text(saved))

    def test_a_first_person_narrator_shares_the_narrators_voice_unless_given_their_own(self):
        cast = self._save("k", self.cast_store.STATUS_DONE, {
            "me": {"name": "Nora", "aliases": [], "gender": "female", "age": "adult", "lines": 9, "voice": "Ada.wav",
                   "profile": {"description": "Tells the story.", "first_line": {"chapter": 1, "text": '"Hi."'}}},
            "bob": {"name": "Bob", "aliases": [], "gender": "male", "age": "adult", "lines": 3, "voice": None},
        })
        cast["book_tone"] = {"point_of_view": "first", "pov_character": "Nora", "pov_key": "me", "tone": "wry"}
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        table, keys, status, _ = chatterbox_ui.cast_overview("k", "chatterbox", "Elena.wav", None)
        self.assertEqual(table["value"][keys.index("me")][5], "(narrator's voice)")
        self.assertIsNone(self.cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]["me"]["voice"])
        self.assertIn("Nora tells the story, so their lines are read in the narrator's voice.", status)
        _, _, _, voice, profile, _ = chatterbox_ui.select_cast_row("k", keys, "chatterbox", MagicMock(index=[0, 0]))
        self.assertEqual(voice["value"], chatterbox_ui.NARRATOR_VOICE)
        self.assertEqual(voice["choices"][0], chatterbox_ui.NARRATOR_VOICE_CHOICE)
        self.assertIn("**Voice:** the narrator's", profile)
        with patch.object(chatterbox_ui, "preview_voice", return_value="/tmp/x.mp3") as preview:
            chatterbox_ui.sample_character("k", "me", "chatterbox", chatterbox_ui.NARRATOR_VOICE, "auto", 1.0,
                                           0.73, 0.5, 0.61, "Elena.wav")
        self.assertEqual(preview.call_args.args[:3], ("Elena.wav", '"Hi."', 0.73))
        # Advanced: a voice of her own, then back to the narrator's.
        table, _, _ = chatterbox_ui.apply_cast_edit("k", "me", "female", "Bea.wav", "chatterbox")
        self.assertEqual(table["value"][keys.index("me")][5], "Bea")
        table, _, message = chatterbox_ui.apply_cast_edit("k", "me", "female", chatterbox_ui.NARRATOR_VOICE, "chatterbox")
        self.assertEqual(table["value"][keys.index("me")][5], "(narrator's voice)")
        self.assertIn("narrator's voice", message)
        with self.assertRaises(gr.Error):  # only the narrating character can follow the narrator
            chatterbox_ui.apply_cast_edit("k", "bob", "male", chatterbox_ui.NARRATOR_VOICE, "chatterbox")
        # Seen live: a narrator the book never names is attributed as "I".
        unnamed = {"characters": {"i": {"name": "I", "lines": 45}},
                   "book_tone": {"point_of_view": "first", "pov_character": "I", "pov_key": "i", "tone": "wry"}}
        text = chatterbox_ui.book_tone_text(unnamed)
        self.assertNotIn("narrated by I", text)
        self.assertIn('The unnamed first-person narrator ("I") speaks their own lines', text)

    def test_the_panel_names_each_first_person_storys_teller_and_voice(self):
        cast = {"characters": {"nate": {"name": "Nate", "voice": "Cal.wav"}, "irene": {"name": "Irene", "voice": None}},
                "chapters": {"a": {"number": 3, "narrator": "nate"}, "b": {"number": 4, "narrator": "irene"},
                             "c": {"number": 5, "narrator": None}}}
        self.assertEqual(chatterbox_ui.story_tellers_text(cast),
                         "**First-person stories, each narrated by its teller's voice:** Nate (chapter 3, Cal); "
                         "Irene (chapter 4, the narrator voice)")
        self.assertEqual(chatterbox_ui.story_tellers_text({"characters": {}, "chapters": {}}), "")

    def test_analysis_can_keep_every_earlier_voice_or_only_the_owners_picks(self):
        book = os.path.join(self.tmp.name, "mine.epub")
        with open(book, "wb") as f:
            f.write(b"epub bytes")
        args = (None, book, TABLE, "chatterbox", "Elena.wav", "auto", "double", False, False, None)
        self.assertTrue(chatterbox_ui.analysis_settings(*args)["auto_pick_voices"])
        self.assertFalse(chatterbox_ui.analysis_settings(*args, "INFO", False)["auto_pick_voices"])

    def test_editing_a_character_saves_and_refreshes(self):
        self._save("k", self.cast_store.STATUS_DONE, {
            "anne": {"name": "Anne", "aliases": [], "gender": "unknown", "age": "adult", "lines": 9, "voice": "Ada.wav"}})
        table, keys, message = chatterbox_ui.apply_cast_edit("k", "anne", "female", "Bea.wav", "chatterbox")
        self.assertEqual(table["value"][0][3:6], ["female", "adult", "Bea"])
        self.assertIn("Bea.wav", message)
        with self.assertRaises(gr.Error):
            chatterbox_ui.apply_cast_edit("k", "anne", "female", "af_heart", "chatterbox")
        with self.assertRaises(gr.Error):
            chatterbox_ui.apply_cast_edit("k", None, "female", "Bea.wav", "chatterbox")
        with self.assertRaises(gr.Error):  # the right shape, but no such voice file
            chatterbox_ui.apply_cast_edit("k", "anne", "female", "Missing.wav", "chatterbox")

    def test_analysis_settings_copy_the_book_and_name_the_cast_file(self):
        book = os.path.join(self.tmp.name, "mine.epub")
        with open(book, "wb") as f:
            f.write(b"epub bytes")
        settings = chatterbox_ui.analysis_settings(None, book, TABLE, "chatterbox", "Elena.wav", "auto", "double",
                                                   False, False, None)
        self.assertEqual(settings["chapter_selection"], [2, 3])
        self.assertEqual(settings["cast_key"], self.cast_store.cast_key(book))
        self.assertEqual(settings["cast_file"], chatterbox_ui.cast_file_for(settings["cast_key"]))
        self.assertTrue(settings["input_file"].startswith(os.path.abspath(chatterbox_ui.QUEUE_UPLOADS)))
        self.assertEqual((settings["engine"], settings["voice"]), ("chatterbox", "Elena.wav"))
        with patch.dict(os.environ, {"LLM_BASE_URL": ""}):
            with self.assertRaises(gr.Error):
                chatterbox_ui.analysis_settings(None, book, TABLE, "chatterbox", "Elena.wav", "auto", "double",
                                                False, False, None)
        with self.assertRaises(gr.Error):
            chatterbox_ui.analysis_settings(None, book, [[1, False, "A", "", ""]], "chatterbox", "Elena.wav", "auto",
                                            "double", False, False, None)

    def test_analysis_estimate_counts_dialogue_lines_of_ticked_chapters(self):
        stats = [[100, 5, 2, 40], [200, 9, 3, 30], [300, 9, 3, 20]]
        self.assertAlmostEqual(chatterbox_ui.analysis_estimate(TABLE, stats), 50 * chatterbox_ui.ANALYSIS_SECONDS_PER_LINE)
        self.assertEqual(chatterbox_ui.analysis_estimate(TABLE, [[100, 5, 2]]), 0)  # stats from before the count existed

    def test_voice_gender_is_saved_from_the_voice_lab(self):
        self.assertEqual(chatterbox_ui.voice_gender_of("Elena.wav")["value"], "")
        message = chatterbox_ui.save_voice_gender("Elena.wav", "female")
        self.assertIn("Elena", message)
        self.assertEqual(chatterbox_ui.voice_gender_of("Elena.wav")["value"], "female")
        self.assertEqual(chatterbox_ui.engine_voices_with_gender("chatterbox"),
                         [("Ada.wav", "female"), ("Bea.wav", "female"), ("Cal.wav", "male"), ("Elena.wav", "female")])
        with self.assertRaises(gr.Error):
            chatterbox_ui.save_voice_gender(None, "female")

    def test_queue_rows_name_cast_jobs_and_multi_voice_books(self):
        cast_job = {"kind": CAST, "voice": "Elena.wav", "settings": {"engine": "chatterbox"}, "status": RUNNING,
                    "chapters": 3}
        self.assertEqual(chatterbox_ui._voice_column(cast_job), "cast analysis (LLM)")
        book = {"voice": "Elena.wav", "settings": {"engine": "chatterbox", "voice_mode": "cast"}}
        self.assertEqual(chatterbox_ui._voice_column(book), "Elena + cast")
        book["settings"]["voice_mode"] = "dialogue"
        self.assertEqual(chatterbox_ui._voice_column(book), "Elena + dialogue voice")
        del book["settings"]["voice_mode"]
        self.assertEqual(chatterbox_ui._voice_column(book), "Elena")
        with patch.object(chatterbox_ui.JobQueue, "chapters_done", return_value=1):
            self.assertIn("analysing cast · 1 of 3", chatterbox_ui._status_label(cast_job))


class TestQueueTimeCastChecks(unittest.TestCase):
    """A cast must cover the ticked chapters as the book reads now, and every chosen voice must exist."""

    CHAPTERS = [("One", '"Hello," said Ada. "Who is there?"'), ("Two", '"It is me," said Tom.'),
                ("Three", '"Go away," said Ada.')]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voices = os.path.join(self.tmp.name, "voices")
        os.makedirs(self.voices)
        for name in ("Elena.wav", "Tom.wav", "Ada.wav"):
            open(os.path.join(self.voices, name), "w").close()
        self.book = os.path.join(self.tmp.name, "library", "book.epub")
        os.makedirs(os.path.dirname(self.book))
        with zipfile.ZipFile(self.book, "w") as z:
            z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            z.writestr("META-INF/container.xml",
                       '<?xml version="1.0"?><container version="1.0" '
                       'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                       '<rootfile full-path="c.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
            items = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>'
                            for i in range(1, 4))
            refs = "".join(f'<itemref idref="c{i}"/>' for i in range(1, 4))
            z.writestr("c.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                                '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
                                f'</metadata><manifest>{items}</manifest><spine>{refs}</spine></package>')
            for i, (title, body) in enumerate(self.CHAPTERS, 1):
                z.writestr(f"c{i}.xhtml", '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>'
                                          f"<h1>{title}</h1><p>{body}</p></body></html>")
        self.patches = [patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(self.tmp.name, "uploads")),
                        patch.object(chatterbox_ui, "CASTS_DIR", os.path.join(self.tmp.name, "casts")),
                        patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": os.path.dirname(self.book),
                                                "TTS_VOICES_DIR": self.voices})]
        for p in self.patches:
            p.start()
        from audiobook_generator.core import cast as cast_store
        self.cast_store = cast_store

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _cast(self, analysed_chapters, voice="Ada.wav"):
        """A finished cast holding attributions for the given chapter numbers (as the book reads now)."""
        chapters = chatterbox_ui.book_chapters(self.book, "auto", "double", False, False, None)
        cast = self.cast_store.new_cast("k", self.book, "T", "A", "chatterbox", "Elena.wav", analysed_chapters)
        for n in analysed_chapters:
            cast["chapters"][self.cast_store.text_hash(chapters[n - 1][1])] = {"number": n, "title": "", "lines": {}}
        cast["characters"] = {"ada": {"name": "Ada", "aliases": [], "gender": "female", "age": "adult",
                                      "lines": 2, "voice": voice}}
        cast["status"] = self.cast_store.STATUS_DONE
        self.cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)

    def _queue(self, ticked, voice="Elena.wav", dialogue_voice="Tom.wav", newline_mode="double"):
        table = [[n, n in ticked, f"Chapter {n}", "", ""] for n in range(1, 4)]
        return chatterbox_ui.queue_settings(self.book, None, table, os.path.join(chatterbox_ui.OUTPUT_ROOT, "out"),
                                            voice, 1.0, 0.35, 0.9, True, False, False, "auto", newline_mode, False,
                                            False, None, "INFO", "sentence", "chatterbox", "cast", dialogue_voice, "k")

    def test_a_cast_covering_the_ticked_chapters_is_accepted(self):
        self._cast([1, 2, 3])
        self.assertEqual(self._queue([1, 3])["voice_mode"], "cast")

    def test_a_ticked_chapter_the_cast_never_analysed_is_refused_by_number(self):
        self._cast([1, 2])
        with self.assertRaises(gr.Error) as ctx:
            self._queue([1, 3])
        self.assertIn("chapter 3", str(ctx.exception))

    def test_a_voice_the_cast_uses_that_no_longer_exists_is_refused(self):
        self._cast([1, 2, 3], voice="Deleted.wav")
        with self.assertRaises(gr.Error) as ctx:
            self._queue([1])
        self.assertIn("Deleted.wav", str(ctx.exception))

    def test_a_narrator_or_dialogue_voice_that_no_longer_exists_is_refused(self):
        self._cast([1, 2, 3])
        with self.assertRaises(gr.Error):
            self._queue([1], voice="Gone.wav")
        with self.assertRaises(gr.Error):
            self._queue([1], dialogue_voice="Gone.wav")

    def test_no_voices_folder_means_the_voice_check_is_skipped(self):
        self._cast([1, 2, 3], voice="Anything.wav")
        with patch.dict(os.environ, {"TTS_VOICES_DIR": ""}):
            self.assertEqual(self._queue([1])["voice_mode"], "cast")

