"""Breeze around the provider: the HTTP client, the voice transcripts, the GPU handover between
engines (and the cast analysis freeing Breeze for the LLM), and the Breeze choice in the UI and queue.
Every HTTP call is mocked; no real server is contacted."""
import base64
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import gradio as gr
import requests
from pydub import AudioSegment

from audiobook_generator.core import breeze_client, chatterbox_control, engine_gpu, speech_check, voice_transcripts
from audiobook_generator.core import cast_analysis
from audiobook_generator.ui import chatterbox_ui, job_queue

BASE = {"BREEZE_BASE_URL": "http://breeze:8005/"}


def _wav_b64(ms: int = 300) -> str:
    buffer = io.BytesIO()
    AudioSegment.silent(ms, frame_rate=24000).set_channels(1).set_sample_width(2).export(buffer, format="wav")
    return base64.b64encode(buffer.getvalue()).decode()


def _response(payload=None, status=200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status.side_effect = None if status < 400 else requests.HTTPError(f"HTTP {status}")
    return response


class TestBreezeClient(unittest.TestCase):

    def setUp(self):
        patcher = patch.dict(os.environ, BASE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_configured_follows_the_environment(self):
        self.assertTrue(breeze_client.configured())
        self.assertEqual(breeze_client.base_url(), "http://breeze:8005")
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}):
            self.assertFalse(breeze_client.configured())
            self.assertIsNone(breeze_client.health())

    def test_health_load_and_unload(self):
        with patch.object(breeze_client.requests, "get",
                          return_value=_response({"status": "ok", "loaded": True, "max_batch": 32})) as get:
            self.assertEqual(breeze_client.health()["max_batch"], 32)
            self.assertTrue(breeze_client.loaded())
        self.assertEqual(get.call_args.args[0], "http://breeze:8005/health")
        with patch.object(breeze_client.requests, "post", return_value=_response({"loaded": True})) as post:
            self.assertTrue(breeze_client.load())
        self.assertEqual(post.call_args.args[0], "http://breeze:8005/api/load")
        with patch.object(breeze_client.requests, "post", return_value=_response({"loaded": False})) as post:
            self.assertTrue(breeze_client.unload())
        self.assertEqual(post.call_args.args[0], "http://breeze:8005/api/unload")

    def test_an_unreachable_server_is_reported_not_raised(self):
        with patch.object(breeze_client.requests, "get", side_effect=requests.ConnectionError("down")), \
                patch.object(breeze_client.requests, "post", side_effect=requests.ConnectionError("down")):
            self.assertIsNone(breeze_client.health())
            self.assertIsNone(breeze_client.loaded())
            self.assertFalse(breeze_client.load())
            self.assertFalse(breeze_client.unload())

    def test_a_batch_request_has_the_contract_shape_and_results_are_decoded_in_order(self):
        items = [{"id": "a", "text": "One.", "voice": "Elena.wav", "ref_text": "words", "instruction": None,
                  "cfg_scale": None},
                 {"id": "b", "text": "Two.", "voice": "Elena.wav", "ref_text": "words", "instruction": None,
                  "cfg_scale": None},
                 {"id": "c", "text": "Three.", "voice": "Elena.wav", "ref_text": "words", "instruction": None,
                  "cfg_scale": None}]
        answer = {"items": [{"id": "a", "wav_b64": _wav_b64(300), "seconds": 0.3, "error": None},
                            {"id": "b", "wav_b64": None, "seconds": 0, "error": "CUDA out of memory"},
                            {"id": "c", "wav_b64": _wav_b64(500), "seconds": 0.5, "error": None}],
                  "elapsed_s": 1.0}
        with patch.object(breeze_client.requests, "post", return_value=_response(answer)) as post:
            results = breeze_client.synthesize_batch(items, 1234)
        self.assertEqual(post.call_args.args[0], "http://breeze:8005/v1/batch")
        self.assertEqual(post.call_args.kwargs["json"], {"items": items, "seed": 1234})
        self.assertGreater(post.call_args.kwargs["timeout"][1], 600)  # a batch can take minutes
        self.assertEqual(len(results[0]), 300)
        self.assertEqual(results[1], "CUDA out of memory")
        self.assertEqual(len(results[2]), 500)

    def test_an_item_without_audio_or_error_is_an_error_string(self):
        answer = {"items": [{"id": "a", "wav_b64": None, "seconds": 0, "error": None}], "elapsed_s": 0}
        with patch.object(breeze_client.requests, "post", return_value=_response(answer)):
            self.assertIsInstance(breeze_client.synthesize_batch([{"id": "a"}], None)[0], str)

    def test_a_wrong_number_of_results_is_an_error(self):
        with patch.object(breeze_client.requests, "post", return_value=_response({"items": []})):
            with self.assertRaises(RuntimeError):
                breeze_client.synthesize_batch([{"id": "a"}], None)

    def test_a_server_that_is_still_starting_is_waited_for(self):
        answer = {"items": [{"id": "a", "wav_b64": _wav_b64(), "seconds": 0.3, "error": None}]}
        outcomes = iter([requests.ConnectionError("refused"), _response(status=503), _response(answer)])

        def post(*args, **kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        sleeps = []
        with patch.object(breeze_client.requests, "post", side_effect=post):
            results = breeze_client.synthesize_batch([{"id": "a"}], 1, sleep=sleeps.append, clock=lambda: 0)
        self.assertEqual(len(results), 1)
        self.assertEqual(sleeps, [breeze_client.SERVER_WAIT_INITIAL_DELAY_SECONDS,
                                  breeze_client.SERVER_WAIT_INITIAL_DELAY_SECONDS * 2])

    def test_it_gives_up_after_the_wait_budget(self):
        clock = iter([0, 100, 700])
        with patch.object(breeze_client.requests, "post", side_effect=requests.ConnectionError("down")):
            with self.assertRaises(requests.ConnectionError):
                breeze_client.synthesize_batch([{"id": "a"}], 1, sleep=lambda s: None, clock=lambda: next(clock))

    def test_a_client_error_is_not_retried(self):
        with patch.object(breeze_client.requests, "post", return_value=_response(status=422)) as post:
            with self.assertRaises(requests.HTTPError):
                breeze_client.synthesize_batch([{"id": "a"}], 1, sleep=lambda s: None)
        self.assertEqual(post.call_count, 1)


class _Checker:
    def __init__(self, text="  the exact words  "):
        self.text, self.heard = text, 0

    def transcribe(self, audio):
        self.heard += 1
        return speech_check.Heard(self.text, [])


class TestVoiceTranscripts(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.voices = os.path.join(self.tmp.name, "voices")
        os.makedirs(self.voices)
        self.cache = os.path.join(self.tmp.name, "voice_transcripts.json")
        self.clip = os.path.join(self.voices, "Elena.wav")
        AudioSegment.silent(1500, frame_rate=24000).export(self.clip, format="wav")

    def _transcript(self, checker, voice="Elena.wav"):
        with patch.object(speech_check, "get", return_value=checker):
            return voice_transcripts.transcript(voice, self.voices, self.cache)

    def test_a_clip_is_transcribed_once_and_cached(self):
        checker = _Checker()
        self.assertEqual(self._transcript(checker), "the exact words")
        self.assertEqual(self._transcript(checker), "the exact words")
        self.assertEqual(checker.heard, 1)
        with open(self.cache, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["voices"]["Elena.wav"]["text"], "the exact words")
        self.assertFalse(os.path.exists(f"{self.cache}.tmp"))

    def test_the_cache_is_used_without_a_checker(self):
        self._transcript(_Checker())
        self.assertEqual(self._transcript(None), "the exact words")

    def test_a_replaced_clip_is_transcribed_again(self):
        checker = _Checker()
        self._transcript(checker)
        AudioSegment.silent(2500, frame_rate=24000).export(self.clip, format="wav")
        os.utime(self.clip, (time.time() + 10, time.time() + 10))
        checker.text = "new words"
        self.assertEqual(self._transcript(checker), "new words")
        self.assertEqual(checker.heard, 2)

    def test_without_a_checker_a_new_clip_has_no_transcript(self):
        self.assertIsNone(self._transcript(None))
        self.assertFalse(os.path.exists(self.cache))

    def test_a_missing_clip_or_voices_folder_has_no_transcript(self):
        self.assertIsNone(self._transcript(_Checker(), "Gone.wav"))
        with patch.dict(os.environ, {"TTS_VOICES_DIR": ""}):
            self.assertIsNone(voice_transcripts.transcript("Elena.wav", None, self.cache))

    def test_a_failing_or_empty_transcription_is_not_cached(self):
        broken = _Checker()
        broken.transcribe = MagicMock(side_effect=RuntimeError("whisper broke"))
        self.assertIsNone(self._transcript(broken))
        self.assertIsNone(self._transcript(_Checker("   ")))
        self.assertFalse(os.path.exists(self.cache))

    def test_the_first_transcript_is_logged_at_info(self):
        with self.assertLogs(voice_transcripts.logger, "INFO") as logged:
            self._transcript(_Checker())
        self.assertIn("the exact words", logged.output[0])


class _InlineThread:
    """threading.Thread stand-in that runs its target at start()."""

    def __init__(self, target=None, name=None, daemon=None):
        self.target, self.started = target, False

    def start(self):
        self.started = True
        self.target()

    def is_alive(self):
        return False


class TestEngineHandover(unittest.TestCase):

    def setUp(self):
        patcher = patch.dict(os.environ, {**BASE, "OPENAI_BASE_URL": "http://cb:8004/v1", "LLM_BASE_URL": ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        engine_gpu._thread = None
        chatterbox_control._reload_thread = None
        self.addCleanup(setattr, engine_gpu, "_thread", None)

    def test_kokoro_never_waits(self):
        with patch.object(breeze_client, "loaded") as loaded:
            self.assertTrue(engine_gpu.ready_for_book("kokoro"))
        loaded.assert_not_called()

    def test_standalone_breeze_waits_for_handover_and_refuses_a_failed_unload(self):
        handover = MagicMock()
        handover.is_alive.return_value = True
        engine_gpu._thread = handover
        calls = []
        with patch.object(chatterbox_control, "model_loaded", return_value=True), \
                patch.object(chatterbox_control, "unload", side_effect=lambda: calls.append("unload") or True), \
                patch.object(breeze_client, "load", side_effect=lambda: calls.append("load") or True):
            engine_gpu.prepare_breeze()
        handover.join.assert_called_once()
        self.assertEqual(calls, ["unload", "load"])
        with patch.object(chatterbox_control, "model_loaded", return_value=True), \
                patch.object(chatterbox_control, "unload", return_value=False), \
                patch.object(breeze_client, "load") as load:
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                engine_gpu.prepare_breeze()
        load.assert_not_called()

    def test_a_breeze_book_waits_while_chatterbox_holds_the_gpu_and_never_blocks(self):
        release = threading.Event()
        calls = []
        state = {"breeze": False, "chatterbox": True}

        def unload():
            calls.append("chatterbox unload")
            state["chatterbox"] = False
            return True

        def load():
            calls.append("breeze load")
            release.wait(10)
            state["breeze"] = True
            return True
        with patch.object(chatterbox_control, "model_loaded", side_effect=lambda: state["chatterbox"]), \
                patch.object(chatterbox_control, "unload", side_effect=unload), \
                patch.object(breeze_client, "loaded", side_effect=lambda: state["breeze"]), \
                patch.object(breeze_client, "load", side_effect=load):
            started = time.monotonic()
            self.assertFalse(engine_gpu.ready_for_book("breeze"))
            self.assertFalse(engine_gpu.ready_for_book("breeze"))  # still loading: no second handover
            self.assertLess(time.monotonic() - started, 5)
            release.set()
            engine_gpu._thread.join(10)
            self.assertTrue(engine_gpu.ready_for_book("breeze"))
        self.assertEqual(calls, ["chatterbox unload", "breeze load"])

    def test_a_breeze_book_waits_for_chatterbox_to_unload_even_when_breeze_is_loaded(self):
        with patch.object(engine_gpu.threading, "Thread", _InlineThread), \
                patch.object(chatterbox_control, "model_loaded", return_value=True), \
                patch.object(chatterbox_control, "unload", return_value=True) as unload, \
                patch.object(breeze_client, "loaded", return_value=True), \
                patch.object(breeze_client, "load", return_value=True):
            self.assertFalse(engine_gpu.ready_for_book("breeze"))
        unload.assert_called_once()

    def test_a_breeze_book_starts_when_breeze_is_loaded_and_chatterbox_is_not(self):
        for chatterbox in (False, None):
            with patch.object(chatterbox_control, "model_loaded", return_value=chatterbox), \
                    patch.object(breeze_client, "loaded", return_value=True):
                self.assertTrue(engine_gpu.ready_for_book("breeze"))

    def test_an_unreachable_breeze_holds_its_book(self):
        with patch.object(engine_gpu.threading, "Thread", _InlineThread), \
                patch.object(chatterbox_control, "model_loaded", return_value=False), \
                patch.object(breeze_client, "loaded", return_value=None), \
                patch.object(breeze_client, "load", return_value=False):
            self.assertFalse(engine_gpu.ready_for_book("breeze"))

    def test_a_chatterbox_book_unloads_breeze_first(self):
        with patch.object(engine_gpu.threading, "Thread", _InlineThread), \
                patch.object(breeze_client, "loaded", return_value=True), \
                patch.object(breeze_client, "unload", return_value=True) as unload:
            self.assertFalse(engine_gpu.ready_for_book("chatterbox"))
        unload.assert_called_once()

    def test_a_chatterbox_book_uses_the_existing_readiness_once_breeze_is_off_the_gpu(self):
        with patch.object(breeze_client, "loaded", return_value=False), \
                patch.object(chatterbox_control, "ready_for_book", return_value=True) as ready:
            self.assertTrue(engine_gpu.ready_for_book("chatterbox"))
        ready.assert_called_once()
        with patch.object(breeze_client, "loaded", return_value=False), \
                patch.object(chatterbox_control, "ready_for_book", return_value=False):
            self.assertFalse(engine_gpu.ready_for_book("chatterbox"))

    def test_without_breeze_configured_chatterbox_behaves_as_before(self):
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}), \
                patch.object(breeze_client, "loaded") as loaded, \
                patch.object(chatterbox_control, "ready_for_book", return_value=True):
            self.assertTrue(engine_gpu.ready_for_book("chatterbox"))
        loaded.assert_not_called()

    def test_the_queue_asks_for_the_books_engine(self):
        with patch.object(engine_gpu, "ready_for_book", return_value=False) as ready:
            self.assertFalse(job_queue._chatterbox_ready({"engine": "breeze"}))
            self.assertFalse(job_queue._chatterbox_ready({}))
        self.assertEqual([c.args[0] for c in ready.call_args_list], ["breeze", "chatterbox"])

    def test_unload_breeze_if_loaded(self):
        with patch.object(breeze_client, "loaded", return_value=True), \
                patch.object(breeze_client, "unload", return_value=True) as unload:
            self.assertTrue(engine_gpu.unload_breeze_if_loaded())
        unload.assert_called_once()
        with patch.object(breeze_client, "loaded", return_value=False), \
                patch.object(breeze_client, "unload") as unload:
            self.assertFalse(engine_gpu.unload_breeze_if_loaded())
        unload.assert_not_called()
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}), patch.object(breeze_client, "loaded") as loaded:
            self.assertFalse(engine_gpu.unload_breeze_if_loaded())
        loaded.assert_not_called()

    def test_breeze_loads_only_after_ollama_is_asked_to_free_the_gpu(self):
        # Ollama keeps the cast LLM loaded for minutes after the analysis; Breeze beside it overfills the card
        calls = []
        ps = _response({"models": [{"name": "qwen2.5:14b"}]})
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://ollama:11434/v1/"}), \
                patch.object(engine_gpu.requests, "get",
                             side_effect=lambda url, timeout: calls.append(("get", url)) or ps), \
                patch.object(engine_gpu.requests, "post",
                             side_effect=lambda url, json, timeout: calls.append(("post", url, json)) or _response({})), \
                patch.object(chatterbox_control, "model_loaded", return_value=False), \
                patch.object(breeze_client, "load", side_effect=lambda: calls.append(("load",)) or True):
            engine_gpu.prepare_breeze()
        self.assertEqual(calls, [("get", "http://ollama:11434/api/ps"),
                                 ("post", "http://ollama:11434/api/generate", {"model": "qwen2.5:14b", "keep_alive": 0}),
                                 ("load",)])

    def test_an_llm_that_cannot_be_unloaded_never_blocks_breeze(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://llm:8080/v1"}):
            with patch.object(engine_gpu.requests, "get", return_value=_response(status=404)), \
                    patch.object(engine_gpu.requests, "post") as post:
                self.assertTrue(engine_gpu.unload_llm())  # not Ollama: nothing to ask
            post.assert_not_called()
            with patch.object(engine_gpu.requests, "get", side_effect=requests.ConnectionError("down")), \
                    patch.object(chatterbox_control, "model_loaded", return_value=False), \
                    patch.object(breeze_client, "load", return_value=True) as load:
                self.assertFalse(engine_gpu.unload_llm())
                engine_gpu.prepare_breeze()
            load.assert_called_once()
        with patch.object(engine_gpu.requests, "get") as get:  # no LLM configured (setUp)
            self.assertTrue(engine_gpu.unload_llm())
        get.assert_not_called()


class TestCastAnalysisFreesBreeze(unittest.TestCase):

    def _run(self, allowed=True):
        events = []
        with patch.object(cast_analysis, "setup_logging"):
            cast_analysis.run_cast_analysis(
                {"log_level": "INFO"}, "log.txt", unload_allowed=lambda: allowed,
                unload=lambda: events.append("chatterbox unload") or True,
                reload=lambda: events.append("chatterbox reload") or True,
                unload_breeze=lambda: events.append("breeze unload") or True,
                analyse=lambda settings: events.append("analyse"), exit=lambda code: events.append(code))
        return events

    def test_breeze_is_unloaded_before_the_llm_and_only_chatterbox_comes_back(self):
        self.assertEqual(self._run(), ["chatterbox unload", "breeze unload", "analyse", "chatterbox reload", 0])

    def test_nothing_is_unloaded_when_unloading_is_off(self):
        self.assertEqual(self._run(allowed=False), ["analyse", 0])

    def test_a_cast_for_breeze_keeps_the_chatterbox_voice_picks(self):
        self.assertEqual(cast_analysis.cast_store.voice_library("breeze"), "chatterbox")
        self.assertEqual(cast_analysis.cast_store.voice_library("kokoro"), "kokoro")


BOOK_ARGS = ("/tmp/book.epub", "audiobook_output/Book", "Elena.wav", 1.0, [1], 0.35, 0.9, True, True, False,
             "auto", "double", False, False, None, "INFO")
TABLE = [[1, False, "Title page", "Title", "under 1 min"], [2, True, "One", "It was", "26 min"],
         [3, True, "Two", "It was", "25 min"]]
SETTINGS = ("audiobook_output/out", "Elena.wav", 1.0, 0.35, 0.9, True, False, False, "auto", "double", False,
            False, None, "INFO")


class TestBreezeInTheUi(unittest.TestCase):

    def test_standalone_ui_actions_reserve_breeze_before_making_speech(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {**BASE, "TTS_VOICES_DIR": folder, "KOKORO_BASE_URL": ""}), \
                patch.object(engine_gpu, "prepare_breeze") as prepare:
            queue = job_queue.JobQueue(os.path.join(folder, "queue.json"), lambda **s: s, lambda: "log",
                                       process_factory=MagicMock)
            ui = chatterbox_ui.build_ui(queue)
            handlers = {f.fn.__name__: f.fn for f in ui.fns.values() if f.fn}

            def speech(*args):
                self.assertEqual(queue._breeze_requests, 1)
                return "speech"

            with patch.object(chatterbox_ui, "sample_voice", side_effect=speech), \
                    patch.object(chatterbox_ui, "sample_character", side_effect=speech), \
                    patch.object(chatterbox_ui, "measure_voice", side_effect=speech), \
                    patch.object(chatterbox_ui, "design_character_voice", side_effect=speech), \
                    patch.object(chatterbox_ui.voice_design, "design_voice", side_effect=speech), \
                    patch.object(chatterbox_ui.voice_design, "STARTER_VOICES", [("Ada", "female", "adult", "A woman.")]), \
                    patch.object(chatterbox_ui, "measure_voices", side_effect=lambda measure: measure("Ada.wav")), \
                    patch.object(chatterbox_ui, "add_voice", side_effect=lambda *a, measurer: measurer("Ada.wav")):
                self.assertEqual(handlers["make_sample"]("breeze", "Ada.wav", 1), "speech")
                self.assertEqual(handlers["cast_sample"]("k", "ada", "breeze", "Ada.wav", "auto", 1, .5, .5, .8, None), "speech")
                self.assertEqual(handlers["measure_all"](), "speech")
                self.assertEqual(handlers["add_one"]("sample", "Ada", False, False, "breeze"), "speech")
                self.assertEqual(handlers["design_for_character"]("k", "ada", "A woman.", "breeze"), "speech")
                self.assertIn("Designed 1 starter voice", list(handlers["design_starters"]())[-1])
                self.assertEqual(prepare.call_count, 6)
                self.assertEqual(queue._breeze_requests, 0)
                queue.add("Cast", {}, 1, 1, "Ada.wav", kind=job_queue.CAST)
                queue.tick()
                with self.assertRaisesRegex(gr.Error, "cast analysis"):
                    handlers["make_sample"]("breeze", "Ada.wav", 1)
                self.assertEqual(prepare.call_count, 6)

    def test_build_config_for_breeze(self):
        with patch.dict(os.environ, BASE):
            config = chatterbox_ui.build_config(*BOOK_ARGS, "paragraph", "breeze", adaptive_delivery=True,
                                                delivery_exaggeration=0.7, delivery_cfg_weight=0.5,
                                                delivery_temperature=0.8, tone_match=True)
        self.assertEqual((config.model_name, config.openai_base_url), ("breeze", None))
        self.assertEqual(config.paced_unit_mode, "sentence")
        self.assertFalse(config.adaptive_delivery)  # Breeze reads every line plain, never directed
        self.assertEqual((config.delivery_exaggeration, config.delivery_cfg_weight, config.delivery_temperature),
                         (None, None, None))
        self.assertTrue(config.tone_match)
        with patch.dict(os.environ, BASE):
            self.assertFalse(chatterbox_ui.build_config(*BOOK_ARGS, "sentence", "breeze", tone_match=False).tone_match)
            self.assertFalse(chatterbox_ui.build_config(*BOOK_ARGS, "sentence", "breeze").adaptive_delivery)

    def test_build_config_for_breeze_without_a_url_raises(self):
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}):
            with self.assertRaises(ValueError):
                chatterbox_ui.build_config(*BOOK_ARGS, "sentence", "breeze")

    def _queue_settings(self, **extra):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(folder, "uploads")), \
                patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": "/library"}), \
                patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub"):
            return chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS, "sentence",
                                                "breeze", **extra)

    def test_queue_settings_for_breeze(self):
        with patch.dict(os.environ, BASE):
            settings = self._queue_settings(adaptive_delivery=True, delivery_exaggeration=0.7,
                                            delivery_cfg_weight=0.5, delivery_temperature=0.8)
        self.assertEqual(settings["engine"], "breeze")
        self.assertFalse(settings["adaptive_delivery"])
        self.assertEqual((settings["delivery_exaggeration"], settings["delivery_cfg_weight"],
                          settings["delivery_temperature"]), (None, None, None))
        self.assertTrue(settings["tone_match"])
        with patch.dict(os.environ, BASE):
            config = chatterbox_ui.build_config(**settings)
            self.assertEqual((config.model_name, config.adaptive_delivery), ("breeze", False))
            # a Breeze book queued while lines were still directed is read plain too
            config = chatterbox_ui.build_config(**{**settings, "adaptive_delivery": True})
            self.assertFalse(config.adaptive_delivery)

    def test_queue_settings_refuses_breeze_when_not_configured(self):
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}):
            with self.assertRaises(gr.Error):
                self._queue_settings()

    def test_chatterbox_still_keeps_its_delivery_settings(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(chatterbox_ui, "QUEUE_UPLOADS", os.path.join(folder, "uploads")), \
                patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": "/library"}), \
                patch("os.path.isfile", side_effect=lambda p: p == "/library/book.epub"):
            settings = chatterbox_ui.queue_settings("/library/book.epub", None, TABLE, *SETTINGS, "sentence",
                                                    "chatterbox", adaptive_delivery=True, delivery_exaggeration=0.7)
        self.assertTrue(settings["adaptive_delivery"])
        self.assertEqual(settings["delivery_exaggeration"], 0.7)

    def test_engine_changed_gives_breeze_chatterbox_voices_and_no_delivery_controls(self):
        files = [("Elena", "Elena.wav")]
        with patch.object(chatterbox_ui, "openai_voice_choices", return_value=files), \
                patch.object(chatterbox_ui, "default_openai_voice", return_value="Elena.wav"), \
                patch.object(chatterbox_ui, "kokoro_voices_and_default") as kokoro:
            voice, _, _, adaptive, baseline = chatterbox_ui.engine_changed("breeze")
        kokoro.assert_not_called()
        self.assertEqual((voice["choices"], voice["value"]), (files, "Elena.wav"))
        self.assertFalse(adaptive["visible"])  # Breeze reads every line plain
        self.assertFalse(baseline["visible"])  # the baseline line is Chatterbox's sliders

    def test_breeze_shares_the_voice_lab_and_the_voice_files(self):
        self.assertEqual(chatterbox_ui._sync_if_chatterbox("Elena.wav", "breeze")["value"], "Elena.wav")
        self.assertNotIn("value", chatterbox_ui._sync_if_chatterbox("af_heart", "kokoro"))
        jobs = [{"status": "queued", "title": "Book", "settings": {"engine": "breeze", "voice": "Elena.wav"}}]
        self.assertEqual(chatterbox_ui._voice_in_use("Elena.wav", jobs), "Book")

    def test_the_estimate_uses_chatterbox_speech_pace_and_four_times_real_time(self):
        self.assertEqual(chatterbox_ui._engine_estimate_constants("breeze"),
                         (chatterbox_ui.CHARS_PER_AUDIO_SECOND, 4.0, 1.0))
        self.assertEqual(chatterbox_ui._engine_estimate_constants("chatterbox")[0],
                         chatterbox_ui.CHARS_PER_AUDIO_SECOND)

    def test_the_queue_shows_the_engine_for_breeze(self):
        job = {"kind": "book", "voice": "Elena.wav", "settings": {"engine": "breeze"}}
        self.assertEqual(chatterbox_ui._voice_column(job), "Breeze · Elena")
        job["settings"]["engine"] = "chatterbox"
        self.assertEqual(chatterbox_ui._voice_column(job), "Elena")

    def test_the_breeze_sample_goes_through_one_batch_item(self):
        take = AudioSegment.silent(400, frame_rate=24000)
        with patch.dict(os.environ, BASE), \
                patch.object(voice_transcripts, "transcript", return_value="clip words"), \
                patch.object(breeze_client, "synthesize_batch", return_value=[take]) as batch:
            path = chatterbox_ui.sample_voice("breeze", "Elena.wav", 1.0)
        self.addCleanup(os.remove, path)
        item = batch.call_args.args[0][0]
        self.assertEqual((item["voice"], item["ref_text"], item["text"]),
                         ("Elena.wav", "clip words", chatterbox_ui.PREVIEW_PHRASE))
        with patch.dict(os.environ, BASE), \
                patch.object(voice_transcripts, "transcript", return_value=None):
            with self.assertRaises(gr.Error):
                chatterbox_ui.sample_voice("breeze", "Elena.wav", 1.0)

    def test_breeze_make_and_cast_samples_use_selected_speed(self):
        take = AudioSegment.silent(1000, frame_rate=24000)
        make = lambda speed: chatterbox_ui.sample_voice("breeze", "Elena.wav", speed)
        cast = lambda speed: chatterbox_ui.sample_character(None, None, "breeze", "Elena.wav", "auto", speed,
                                                             0.7, 0.5, 0.61)
        with patch.dict(os.environ, BASE), \
                patch.object(voice_transcripts, "transcript", return_value="clip words"), \
                patch.object(breeze_client, "synthesize_batch", return_value=[take]) as batch:
            for sample in (make, cast):
                for speed in (0.5, 1.0, 2.0):
                    path = sample(speed)
                    try:
                        preview = AudioSegment.from_file(path)
                        self.assertEqual(preview.frame_rate, 24000)
                        self.assertAlmostEqual(len(preview) / 1000, 1 / speed, delta=0.08)
                    finally:
                        if os.path.exists(path):
                            os.remove(path)
        self.assertEqual(batch.call_count, 6)


if __name__ == "__main__":
    unittest.main()
