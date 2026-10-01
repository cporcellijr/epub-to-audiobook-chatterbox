"""Designed voices: the description built from a character's profile, designing and saving a voice
(checks, retries, what gets recorded), which characters of a Breeze cast are marked for one, the
book-start step that designs them into both casts, measuring voices through Breeze, and the cast
editor's and Voice lab's handlers. Breeze, Whisper and Praat are fakes; nothing talks to a server."""
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
from pydub import AudioSegment
from pydub.generators import Sine

from audiobook_generator.core import cast as cast_store
from audiobook_generator.core import speech_check, voice_design, voice_measure, voice_transcripts
from audiobook_generator.core.audiobook_generator import AudiobookGenerator
from audiobook_generator.ui import chatterbox_ui

RATE = 24000


def _tone(ms: int = 10000) -> AudioSegment:
    return Sine(200).to_audio_segment(duration=ms, volume=-12).set_frame_rate(RATE).set_channels(1).set_sample_width(2)


def _write_clip(path: str) -> None:
    buffer = io.BytesIO()
    _tone(1000).export(buffer, format="wav")
    with open(path, "wb") as f:
        f.write(buffer.getvalue())


def _silence(ms: int = 10000) -> AudioSegment:
    return AudioSegment.silent(ms, frame_rate=RATE)


def _measurement(f0: float = 300.0, hnr: float = 12.0) -> dict:
    return {"f0_median": f0, "f0_range": 8.0, "hnr": hnr}


def _character(name, gender="female", age="adult", lines=20, voice=None, picked=False, pitch=None, quality=None,
               delivery=None, note="", described=True) -> dict:
    character = {"name": name, "aliases": [], "gender": gender, "age": age, "lines": lines, "voice": voice}
    if picked:
        character["voice_picked"] = True
    if described:
        character["profile"] = {"description": f"{name} is a person.", "voice": note,
                                "voice_targets": {"pitch": pitch, "quality": quality, "delivery": delivery}}
    return character


def _cast(characters: dict, key: str = "k") -> dict:
    cast = cast_store.new_cast(key, "/library/book.epub", "T", "A", "breeze", "Narrator.wav", [1])
    cast["characters"] = characters
    cast["status"] = cast_store.STATUS_DONE
    return cast


class Workspace(unittest.TestCase):
    """A temp voices folder and temp data files for transcripts, measurements and genders."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.voices = os.path.join(self.tmp.name, "voices")
        os.makedirs(self.voices)
        self.path = lambda name: os.path.join(self.tmp.name, name)
        for p in (patch.dict(os.environ, {"TTS_VOICES_DIR": self.voices, "BREEZE_BASE_URL": "http://breeze:8005",
                                          "LLM_BASE_URL": "http://llm:11434/v1", "LLM_MODEL": "m"}),
                  patch.object(voice_measure, "VOICE_FEATURES_FILE", self.path("features.json")),
                  patch.object(voice_transcripts, "VOICE_TRANSCRIPTS_FILE", self.path("transcripts.json")),
                  patch("audiobook_generator.core.cast.VOICE_GENDERS_FILE", self.path("genders.json")),
                  patch.object(chatterbox_ui, "CASTS_DIR", self.path("casts")),
                  patch.object(speech_check, "get", return_value=None)):
            p.start()
            self.addCleanup(p.stop)

    def add_voice(self, name: str, gender: str = None, f0: float = None) -> str:
        _write_clip(os.path.join(self.voices, name))
        if gender:
            cast_store.save_voice_gender(name, gender)
        if f0:
            voice_measure.save_features(name, _measurement(f0), voice_measure.file_signature(os.path.join(self.voices, name)))
        return name


class TestDescribe(unittest.TestCase):

    def test_a_young_girl(self):
        girl = _character("Emera", "female", "child", pitch="high", quality="clear", delivery="expressive",
                          note="youthful, playful and sharp")
        self.assertEqual(voice_design.describe(girl), "A young girl with a high, clear voice. "
                                                     "Youthful, playful, sharp, with a lively, expressive delivery.")

    def test_a_woman_and_a_man(self):
        woman = _character("Marielle", "female", "adult", pitch="medium", quality="clear", delivery="even",
                           note="Formal and cautious.")
        self.assertEqual(voice_design.describe(woman), "An adult woman with a medium-pitched, clear voice. "
                                                      "Formal, cautious, with an even, measured delivery.")
        man = _character("Hale", "male", "elderly", pitch="low", quality="husky")
        self.assertEqual(voice_design.describe(man), "An elderly man with a low, husky voice.")

    def test_unknown_fields_are_left_out(self):
        self.assertEqual(voice_design.describe(_character("X", "unknown", "unknown", described=False)), "")
        self.assertEqual(voice_design.describe(_character("X", "male", "unknown", described=False)), "A man.")
        # No pitch target: a child still wants a high voice (core.cast.voice_targets).
        self.assertEqual(voice_design.describe(_character("X", "unknown", "child", described=False)),
                         "A young child with a high voice.")
        self.assertEqual(voice_design.describe(_character("X", "unknown", "unknown", note="gruff")), "Gruff.")

    def test_it_is_deterministic(self):
        character = _character("Emera", "female", "child", pitch="high", note="playful")
        self.assertEqual(voice_design.describe(character), voice_design.describe(dict(character)))


class TestDesignVoice(Workspace):

    def setUp(self):
        super().setUp()
        self.takes = []
        self.measures = []
        self.requests = []

        def fake_batch(items, seed=None, **kwargs):
            self.requests.append((items, seed))
            return [self.takes.pop(0)]

        for p in (patch.object(voice_design.breeze_client, "synthesize_batch", side_effect=fake_batch),
                  patch.object(voice_design.voice_measure, "measure_audio",
                               side_effect=lambda audio: self.measures.pop(0) if self.measures else _measurement())):
            p.start()
            self.addCleanup(p.stop)

    def test_a_design_is_saved_as_a_voice_and_recorded(self):
        self.takes = [_tone()]
        self.measures = [_measurement(354.0)]
        name, measurement = voice_design.design_voice("Emera", "A young girl with a high, clear voice.", "female", "child")
        self.assertEqual((name, measurement["f0_median"]), ("Emera (designed).wav", 354.0))
        clip = AudioSegment.from_wav(os.path.join(self.voices, name))
        self.assertEqual((clip.frame_rate, clip.channels, clip.sample_width), (24000, 1, 2))
        items, seed = self.requests[0]
        self.assertEqual(items, [{"id": "design", "text": voice_design.DESIGN_SAMPLE_TEXT, "voice": None,
                                  "ref_text": None, "instruction": "A young girl with a high, clear voice.",
                                  "cfg_scale": None}])
        self.assertIsInstance(seed, int)
        # Recorded like any voice: the transcript (no Whisper needed), the gender, the measurement.
        self.assertEqual(voice_transcripts.transcript(name), voice_design.DESIGN_SAMPLE_TEXT)
        self.assertEqual(cast_store.load_voice_genders(), {name: "female"})
        self.assertEqual(voice_measure.load_features()[name]["f0_median"], 354.0)
        self.assertEqual(voice_measure.load_features()[name]["signature"],
                         voice_measure.file_signature(os.path.join(self.voices, name)))
        self.assertFalse([f for f in os.listdir(self.voices) if f.endswith(".tmp")])

    def test_a_taken_name_gets_a_number_and_the_name_is_made_safe(self):
        self.add_voice("Emera (designed).wav")
        self.takes = [_tone(), _tone(), _tone()]
        second, _ = voice_design.design_voice("Emera", "A girl.", "female", "child")
        third, _ = voice_design.design_voice("Emera", "A girl.", "female", "child")
        self.assertEqual((second, third), ("Emera (designed) 2.wav", "Emera (designed) 3.wav"))
        self.takes, self.measures = [_tone()], [_measurement(120.0)]
        odd, _ = voice_design.design_voice('Dr. "Who" / <x>', "A man.", "male", "adult")
        self.assertEqual(odd, "Dr. Who x (designed).wav")

    def test_an_unknown_gender_is_not_recorded(self):
        self.takes = [_tone()]
        name, _ = voice_design.design_voice("Sam", "A voice.", "unknown", "unknown")
        self.assertEqual(cast_store.load_voice_genders(), {})
        self.assertIn(name, voice_measure.load_features())

    def test_a_silent_take_is_retried_with_a_new_seed(self):
        self.takes = [_silence(), _tone()]
        name, _ = voice_design.design_voice("Emera", "A girl.", "female", "child")
        self.assertEqual(len(self.requests), 2)
        self.assertNotEqual(self.requests[0][1], self.requests[1][1])
        self.assertEqual(os.listdir(self.voices), [name])

    def test_the_wrong_words_and_the_wrong_pitch_are_rejected(self):
        heard = iter(["a story about cats and dogs on the moon", voice_design.DESIGN_SAMPLE_TEXT,
                      voice_design.DESIGN_SAMPLE_TEXT])
        checker = SimpleNamespace(transcribe=lambda audio: SimpleNamespace(text=next(heard)))
        self.takes = [_tone(), _tone(), _tone()]
        self.measures = [_measurement(300.0), _measurement(120.0)]  # a man at 300 Hz is not a man
        with patch.object(speech_check, "get", return_value=checker):
            name, measurement = voice_design.design_voice("Hale", "An old man.", "male", "elderly")
        self.assertEqual((len(self.requests), measurement["f0_median"]), (3, 120.0))

    def test_pitch_limits_by_who_the_voice_is_for(self):
        problem = voice_design._pitch_problem
        self.assertEqual(problem(120, "male", "adult"), "")
        self.assertNotEqual(problem(200, "male", "adult"), "")
        self.assertEqual(problem(200, "female", "adult"), "")
        self.assertNotEqual(problem(120, "female", "adult"), "")
        self.assertEqual(problem(300, "male", "child"), "")  # a boy is judged as a child
        self.assertNotEqual(problem(200, "female", "child"), "")
        self.assertEqual(problem(500, "unknown", "unknown"), "")

    def test_three_failed_attempts_raise_with_every_reason_and_leave_no_file(self):
        self.takes = [_silence(), "server said no", _silence(2000)]
        with self.assertRaises(voice_design.DesignError) as raised:
            voice_design.design_voice("Emera", "A girl.", "female", "child")
        message = str(raised.exception)
        self.assertIn("attempt 1: 10.0 s of near-silence", message)
        self.assertIn("attempt 2: Breeze made no audio (server said no)", message)
        self.assertIn("attempt 3: only 2.0 s of audio", message)
        self.assertEqual((len(self.requests), os.listdir(self.voices), voice_measure.load_features()), (3, [], {}))

    def test_an_unreachable_server_stops_at_once(self):
        with patch.object(voice_design.breeze_client, "synthesize_batch", side_effect=OSError("refused")) as batch:
            with self.assertRaisesRegex(voice_design.DesignError, "could not be reached"):
                voice_design.design_voice("Emera", "A girl.", "female", "child")
        self.assertEqual(batch.call_count, 1)

    def test_nothing_to_design_from(self):
        with self.assertRaises(voice_design.DesignError):
            voice_design.design_voice("Emera", "   ", "female", "child")
        with patch.dict(os.environ, {"TTS_VOICES_DIR": ""}), self.assertRaises(voice_design.DesignError):
            voice_design.design_voice("Emera", "A girl.", "female", "child")
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}), self.assertRaises(voice_design.DesignError):
            voice_design.design_voice("Emera", "A girl.", "female", "child")


class TestWhoGetsADesign(Workspace):

    def _mark(self, characters, keys=None, traits=None, genders=None):
        cast = _cast(characters)
        marked = voice_design.mark_pending(cast, keys or list(characters), genders or {}, traits or {})
        return cast, marked

    def test_a_child_with_only_adult_voices_gets_one(self):
        # The matcher gave the child the lowest voice of the women: wrong pitch band for a high voice.
        child = _character("Emera", "female", "child", lines=10, voice="Ada.wav", pitch="high")
        traits = {"Ada.wav": {"pitch": 0.0, "husky": 0.5, "expressive": 0.5}}
        cast, marked = self._mark({"emera": child}, traits=traits, genders={"Ada.wav": "female"})
        self.assertEqual(marked, ["emera"])
        design = cast["characters"]["emera"]["voice_design"]
        self.assertEqual(design["status"], "pending")
        self.assertEqual(design["description"], voice_design.describe(child))
        self.assertEqual(cast["characters"]["emera"]["voice"], "Ada.wav")  # the fallback stays

    def test_a_voice_that_fits_needs_no_design(self):
        child = _character("Emera", "female", "child", voice="Bea.wav", pitch="high")
        traits = {"Bea.wav": {"pitch": 1.0, "husky": 0.5, "expressive": 0.5}}
        self.assertEqual(self._mark({"emera": child}, traits=traits)[1], [])

    def test_an_unmeasured_voice_cannot_be_judged_by_pitch(self):
        child = _character("Emera", "female", "child", voice="Ada.wav", pitch="high")
        self.assertEqual(self._mark({"emera": child})[1], [])

    def test_a_voice_shared_with_a_character_with_more_lines(self):
        chars = {"anna": _character("Anna", lines=50, voice="Ada.wav"),
                 "bea": _character("Bea", lines=30, voice="Ada.wav"),
                 "cy": _character("Cy", "male", lines=20, voice="Cal.wav")}
        cast, marked = self._mark(chars)
        self.assertEqual(marked, ["bea"])
        self.assertEqual(cast["characters"]["bea"]["voice_design"]["status"], "pending")
        self.assertNotIn("voice_design", cast["characters"]["anna"])

    def test_a_voice_shared_with_one_the_owner_picked(self):
        chars = {"anna": _character("Anna", lines=5, voice="Ada.wav", picked=True),
                 "bea": _character("Bea", lines=30, voice="Ada.wav")}
        self.assertEqual(self._mark(chars)[1], ["bea"])

    def test_the_wrong_gender_voice_is_a_poor_fit(self):
        chars = {"anna": _character("Anna", voice="Cal.wav")}
        self.assertEqual(self._mark(chars, genders={"Cal.wav": "male"})[1], ["anna"])
        self.assertEqual(self._mark(chars, genders={"Cal.wav": "neutral"})[1], [])

    def test_the_owner_s_picks_the_narrating_character_and_minor_characters_are_excluded(self):
        # Every one of them has a voice of the wrong gender; each is excluded for its own reason but Anna.
        wrong = {"anna": _character("Anna", lines=50, voice="Cal.wav"),
                 "picked": _character("Pat", lines=40, voice="Cal.wav", picked=True),
                 "narrator": _character("Nell", lines=30, voice="Cal.wav"),
                 "minor": _character("Min", lines=2, voice="Cal.wav"),
                 "plain": _character("Plain", lines=20, voice="Cal.wav", described=False)}
        cast = _cast(wrong)
        cast["book_tone"] = {"point_of_view": "first", "pov_key": "narrator"}
        marked = voice_design.mark_pending(cast, list(wrong), {"Cal.wav": "male"}, {})
        self.assertEqual(marked, ["anna"])

    def test_only_the_characters_just_given_a_voice_are_considered(self):
        chars = {"anna": _character("Anna", lines=50, voice="Ada.wav"),
                 "bea": _character("Bea", lines=30, voice="Ada.wav")}
        self.assertEqual(self._mark(chars, keys=["anna"])[1], [])

    def test_at_most_a_few_per_cast_most_lines_first(self):
        chars = {f"c{i}": _character(f"C{i}", lines=100 - i, voice="Ada.wav") for i in range(10)}
        cast, marked = self._mark(chars)
        self.assertEqual(marked, ["c1", "c2", "c3", "c4", "c5", "c6"])
        self.assertEqual(voice_design.mark_pending(cast, list(chars), {}, {}), [])  # none left to give

    def test_a_character_with_a_design_is_not_marked_again(self):
        chars = {"anna": _character("Anna", lines=50, voice="Ada.wav"),
                 "bea": _character("Bea", lines=30, voice="Ada.wav")}
        chars["bea"]["voice_design"] = {"status": "done", "description": "x", "file": "Ada.wav"}
        self.assertEqual(self._mark(chars)[1], [])


class TestFillMissingVoices(Workspace):
    """_fill_missing_voices with the real matcher: Breeze casts get designs marked, others none."""

    def setUp(self):
        super().setUp()
        self.add_voice("Ada.wav", "female", 120.0)
        self.add_voice("Bea.wav", "female", 300.0)
        self.add_voice("Cal.wav", "male", 100.0)
        self.add_voice("Narrator.wav", "male", 110.0)

    def _run(self, engine, characters):
        cast = _cast(characters)
        path = chatterbox_ui.cast_file_for("k")
        cast_store.save_cast(path, cast)
        return chatterbox_ui._fill_missing_voices(cast, path, engine, "Narrator.wav"), path

    def _two_who_want_high(self):
        return {"lena": _character("Lena", lines=50, pitch="high"),
                "emera": _character("Emera", "female", "child", lines=10, pitch="high")}

    def test_the_child_who_gets_the_low_voice_is_marked_and_the_other_is_not(self):
        cast, path = self._run("breeze", self._two_who_want_high())
        lena, emera = cast["characters"]["lena"], cast["characters"]["emera"]
        self.assertEqual((lena["voice"], emera["voice"]), ("Bea.wav", "Ada.wav"))
        self.assertNotIn("voice_design", lena)
        self.assertEqual(emera["voice_design"]["status"], "pending")
        self.assertIn("young girl", emera["voice_design"]["description"])
        self.assertEqual(cast_store.load_cast(path)["characters"]["emera"]["voice_design"]["status"], "pending")

    def test_other_engines_are_untouched(self):
        for engine in ("chatterbox", "kokoro"):
            with patch.object(chatterbox_ui, "kokoro_voices_and_default",
                              return_value=([("af_heart", "af_heart"), ("am_adam", "am_adam")], "af_heart")):
                cast, _ = self._run(engine, self._two_who_want_high())
            self.assertFalse([c for c in cast["characters"].values() if "voice_design" in c], engine)

    def test_the_table_says_a_voice_is_coming(self):
        cast, _ = self._run("breeze", self._two_who_want_high())
        rows, _ = chatterbox_ui.cast_rows(cast, "breeze")
        shown = {row[0]: row[5] for row in rows}
        self.assertEqual(shown["Lena"], "Bea")
        self.assertIn("a new voice is designed when the book starts", shown["Emera"])

    def test_a_designed_voice_survives_suggesting_again_but_a_pending_one_does_not(self):
        cast, path = self._run("breeze", self._two_who_want_high())
        cast["characters"]["lena"]["voice_design"] = {"status": "done", "description": "d", "file": "Bea.wav"}
        cleared = cast_store.clear_suggested_voices(cast)
        self.assertEqual(cleared, 1)
        self.assertEqual(cast["characters"]["lena"]["voice"], "Bea.wav")
        self.assertIsNone(cast["characters"]["emera"]["voice"])
        self.assertNotIn("voice_design", cast["characters"]["emera"])

    def test_a_designed_voice_deleted_from_the_library_is_suggested_again(self):
        chars = self._two_who_want_high()
        chars["lena"].update(voice="Gone (designed).wav",
                             voice_design={"status": "done", "description": "d", "file": "Gone (designed).wav"})
        cast, _ = self._run("breeze", chars)
        self.assertNotEqual(cast["characters"]["lena"]["voice"], "Gone (designed).wav")
        self.assertTrue(cast["characters"]["lena"]["voice"])

    def test_a_reanalysis_keeps_the_designed_voice(self):
        previous = _cast({"emera": _character("Emera", "female", "child", voice="Emera (designed).wav")})
        previous["characters"]["emera"]["voice_design"] = {"status": "done", "description": "d",
                                                           "file": "Emera (designed).wav"}
        fresh = {"emera": _character("Emera", "female", "child", voice=None)}
        self.assertEqual(cast_store.carry_voice_choices(previous, fresh, picked_only=True), 1)
        self.assertEqual((fresh["emera"]["voice"], fresh["emera"]["voice_design"]["status"]),
                         ("Emera (designed).wav", "done"))


class TestDesignPendingVoices(Workspace):

    def setUp(self):
        super().setUp()
        self.saved_path = self.path("saved.json")
        chars = {"anna": _character("Anna", lines=50, voice="Ada.wav"),
                 "emera": _character("Emera", "female", "child", lines=10, voice="Ada.wav"),
                 "bea": _character("Bea", lines=8, voice="Ada.wav")}
        for key in ("emera", "bea"):
            chars[key]["voice_design"] = {"status": "pending", "description": f"A {key}."}
        self.snapshot = self.path("snapshot.json")
        cast = _cast(chars)
        cast_store.save_cast(self.snapshot, cast)
        cast_store.save_cast(self.saved_path, cast)
        self.calls = []

    def _designer(self, fail=()):
        def design(name, description, gender, age):
            self.calls.append((name, description, gender, age))
            if name in fail:
                raise voice_design.DesignError("attempt 1: near-silence")
            file_name = f"{name} (designed).wav"
            _write_clip(os.path.join(self.voices, file_name))
            return file_name, _measurement()
        return design

    def test_both_casts_get_the_designed_voices(self):
        count = voice_design.design_pending_voices(self.snapshot, self.saved_path, self._designer())
        self.assertEqual(count, 2)
        self.assertEqual([c[0] for c in self.calls], ["Emera", "Bea"])  # most lines first
        self.assertEqual(self.calls[0], ("Emera", "A emera.", "female", "child"))
        for path in (self.snapshot, self.saved_path):
            characters = cast_store.load_cast(path)["characters"]
            self.assertEqual(characters["emera"]["voice"], "Emera (designed).wav")
            self.assertEqual(characters["emera"]["voice_design"],
                             {"status": "done", "description": "A emera.", "file": "Emera (designed).wav"})
            self.assertEqual(characters["anna"]["voice"], "Ada.wav")

    def test_a_failed_design_keeps_the_suggested_voice_and_the_book_goes_on(self):
        with self.assertLogs("audiobook_generator.core.voice_design", "WARNING"):
            count = voice_design.design_pending_voices(self.snapshot, self.saved_path, self._designer(fail=("Emera",)))
        self.assertEqual(count, 1)
        for path in (self.snapshot, self.saved_path):
            characters = cast_store.load_cast(path)["characters"]
            self.assertEqual(characters["emera"]["voice"], "Ada.wav")
            self.assertEqual(characters["emera"]["voice_design"]["status"], "failed")
            self.assertIn("near-silence", characters["emera"]["voice_design"]["error"])
            self.assertEqual(characters["bea"]["voice"], "Bea (designed).wav")

    def test_nothing_pending_does_nothing(self):
        voice_design.design_pending_voices(self.snapshot, self.saved_path, self._designer())
        self.calls.clear()
        self.assertEqual(voice_design.design_pending_voices(self.snapshot, self.saved_path, self._designer()), 0)
        self.assertEqual(self.calls, [])

    def test_a_voice_the_saved_cast_already_designed_is_adopted(self):
        self.add_voice("Emera (designed).wav")
        saved = cast_store.load_cast(self.saved_path)
        saved["characters"]["emera"]["voice"] = "Emera (designed).wav"
        saved["characters"]["emera"]["voice_design"] = {"status": "done", "description": "earlier",
                                                       "file": "Emera (designed).wav"}
        cast_store.save_cast(self.saved_path, saved)
        voice_design.design_pending_voices(self.snapshot, self.saved_path, self._designer())
        self.assertEqual([c[0] for c in self.calls], ["Bea"])
        snapshot = cast_store.load_cast(self.snapshot)["characters"]["emera"]
        self.assertEqual((snapshot["voice"], snapshot["voice_design"]["description"]),
                         ("Emera (designed).wav", "earlier"))

    def test_a_voice_the_owner_picked_since_is_left_alone_in_the_saved_cast(self):
        saved = cast_store.load_cast(self.saved_path)
        saved["characters"]["emera"].update(voice="Bea.wav", voice_picked=True)
        saved["characters"]["emera"].pop("voice_design")
        cast_store.save_cast(self.saved_path, saved)
        voice_design.design_pending_voices(self.snapshot, self.saved_path, self._designer())
        kept = cast_store.load_cast(self.saved_path)["characters"]["emera"]
        self.assertEqual((kept["voice"], "voice_design" in kept), ("Bea.wav", False))

    def test_edits_during_design_survive_in_the_saved_cast_only(self):
        design = self._designer()

        def edit_then_design(name, description, gender, age):
            if name in ("Emera", "Bea"):
                saved = cast_store.load_cast(self.saved_path)
                character = saved["characters"][name.lower()]
                if name == "Emera":
                    character.update(voice="Owner.wav", voice_picked=True, delivery="whisper")
                    character.pop("voice_design")
                else:
                    character["voice_design"] = {"status": "pending", "description": "Owner replacement."}
                cast_store.save_cast(self.saved_path, saved)
            return design(name, description, gender, age)

        count = voice_design.design_pending_voices(self.snapshot, self.saved_path, edit_then_design)

        self.assertEqual(count, 2)
        saved = cast_store.load_cast(self.saved_path)["characters"]
        self.assertEqual((saved["emera"]["voice"], saved["emera"]["voice_picked"],
                          saved["emera"]["delivery"]), ("Owner.wav", True, "whisper"))
        self.assertNotIn("voice_design", saved["emera"])
        self.assertEqual(saved["bea"]["voice"], "Ada.wav")
        self.assertEqual(saved["bea"]["voice_design"],
                         {"status": "pending", "description": "Owner replacement."})
        self.assertEqual(saved["anna"]["voice"], "Ada.wav")
        snapshot = cast_store.load_cast(self.snapshot)["characters"]
        self.assertEqual(snapshot["emera"]["voice"], "Emera (designed).wav")
        self.assertEqual(snapshot["emera"]["voice_design"]["description"], "A emera.")
        self.assertEqual(snapshot["bea"]["voice"], "Bea (designed).wav")

    def test_the_default_saved_cast_is_the_snapshot_s_key_in_the_casts_folder(self):
        with patch.object(cast_store, "CASTS_FOLDER", self.path("casts")), \
                patch.object(voice_design.cast_store, "cast_path", lambda key: self.path(f"casts/{key}.json")):
            cast_store.save_cast(self.path("casts/k.json"), cast_store.load_cast(self.snapshot))
            voice_design.design_pending_voices(self.snapshot, designer=self._designer())
        self.assertEqual(cast_store.load_cast(self.path("casts/k.json"))["characters"]["emera"]["voice"],
                         "Emera (designed).wav")


class TestBookStartHook(unittest.TestCase):

    def _generator(self, **config):
        base = dict(preview=False, model_name="breeze", voice_mode="cast", cast_file="cast.json")
        return AudiobookGenerator(SimpleNamespace(**{**base, **config}))

    def test_a_breeze_cast_book_designs_its_pending_voices(self):
        with patch("audiobook_generator.core.voice_design.design_pending_voices") as design:
            self._generator()._design_cast_voices()
        design.assert_called_once_with("cast.json")

    def test_other_books_and_previews_design_nothing(self):
        for config in ({"model_name": "chatterbox"}, {"model_name": "kokoro"}, {"voice_mode": "single"},
                       {"voice_mode": "dialogue"}, {"cast_file": None}, {"preview": True}):
            with patch("audiobook_generator.core.voice_design.design_pending_voices") as design:
                self._generator(**config)._design_cast_voices()
            design.assert_not_called()

    def test_a_failure_never_stops_the_book(self):
        with patch("audiobook_generator.core.voice_design.design_pending_voices", side_effect=RuntimeError("boom")), \
                self.assertLogs("audiobook_generator.core.audiobook_generator", "WARNING"):
            self._generator()._design_cast_voices()


class TestMeasureThroughBreeze(Workspace):

    def setUp(self):
        super().setUp()
        self.add_voice("Ada.wav")
        voice_transcripts.remember("Ada.wav", "Words in the clip.")
        self.requests = []

        def fake_batch(items, seed=None, **kwargs):
            self.requests.append(items)
            return [_tone(3000)]

        for p in (patch.object(chatterbox_ui.breeze_client, "synthesize_batch", side_effect=fake_batch),
                  patch.object(chatterbox_ui.voice_measure, "measure_audio", return_value=_measurement(210.0))):
            p.start()
            self.addCleanup(p.stop)

    def test_breeze_speaks_the_measuring_sentence_in_the_voice(self):
        with patch.object(chatterbox_ui, "_post_json") as chatterbox:
            measurement = chatterbox_ui.measure_voice("Ada.wav")
        chatterbox.assert_not_called()
        self.assertEqual(measurement["f0_median"], 210.0)
        self.assertEqual(self.requests, [[{"id": "measure", "text": voice_measure.MEASURE_TEXT, "voice": "Ada.wav",
                                           "ref_text": "Words in the clip.", "instruction": None,
                                           "cfg_scale": None}]])
        self.assertEqual(voice_measure.load_features()["Ada.wav"]["f0_median"], 210.0)

    def test_a_voice_with_no_transcript_cannot_be_measured(self):
        self.add_voice("Bea.wav")
        with self.assertRaisesRegex(ValueError, "Breeze needs the words spoken in Bea.wav"):
            chatterbox_ui.measure_voice("Bea.wav")

    def test_breeze_making_no_audio_is_an_error(self):
        with patch.object(chatterbox_ui.breeze_client, "synthesize_batch", return_value=["out of memory"]):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                chatterbox_ui.measure_voice("Ada.wav")

    def test_chatterbox_still_measures_when_breeze_is_not_configured(self):
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}), \
                patch.object(chatterbox_ui, "_post_json", return_value=b"wav") as chatterbox:
            chatterbox_ui.measure_voice("Ada.wav")
        self.assertEqual(chatterbox.call_args.args[0], "/tts")
        self.assertEqual(self.requests, [])

    def test_the_run_names_the_engine_it_could_not_reach(self):
        with patch.object(chatterbox_ui.breeze_client, "synthesize_batch", side_effect=OSError("refused")):
            self.assertIn("then Breeze couldn't be reached", chatterbox_ui.measure_voices())


class TestDesignHandlers(Workspace):

    def setUp(self):
        super().setUp()
        self.add_voice("Ada.wav", "female", 120.0)
        cast = _cast({"emera": _character("Emera", "female", "child", lines=10, voice="Ada.wav", pitch="high",
                                          note="playful")})
        cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        self.designs = []

        def design(name, description, gender, age):
            self.designs.append((name, description, gender, age))
            file_name = f"{name} (designed).wav"
            _write_clip(os.path.join(self.voices, file_name))
            return file_name, _measurement(354.0)

        p = patch.object(chatterbox_ui.voice_design, "design_voice", side_effect=design)
        p.start()
        self.addCleanup(p.stop)

    def _design(self, jobs=(), engine="breeze", description="A young girl.", key="emera"):
        return chatterbox_ui.design_character_voice("k", key, description, engine, list(jobs))

    def test_the_description_box_is_prefilled_from_the_profile_or_the_stored_design(self):
        shown = chatterbox_ui.design_description_for("k", "emera")["value"]
        self.assertEqual(shown, voice_design.describe(cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]["emera"]))
        self.assertIn("young girl", shown)
        cast = cast_store.load_cast(chatterbox_ui.cast_file_for("k"))
        cast["characters"]["emera"]["voice_design"] = {"status": "pending", "description": "Stored words."}
        cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        self.assertEqual(chatterbox_ui.design_description_for("k", "emera")["value"], "Stored words.")
        self.assertEqual(chatterbox_ui.design_description_for("k", "nobody")["value"], "")

    def test_designing_assigns_the_voice_as_the_owner_s_pick_and_plays_it(self):
        table, keys, message, voice_update, sample = self._design()
        self.assertEqual(self.designs, [("Emera", "A young girl.", "female", "child")])
        character = cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]["emera"]
        self.assertEqual((character["voice"], character["voice_picked"]), ("Emera (designed).wav", True))
        self.assertEqual(character["voice_design"],
                         {"status": "done", "description": "A young girl.", "file": "Emera (designed).wav"})
        self.assertEqual(voice_update["value"], "Emera (designed).wav")
        self.assertIn("Emera (designed).wav", [value for _, value in voice_update["choices"]])
        self.assertIn("354 Hz", message)
        self.assertTrue(os.path.isfile(sample))
        os.remove(sample)
        # Suggesting again keeps it: it is a pick.
        cast = cast_store.load_cast(chatterbox_ui.cast_file_for("k"))
        self.assertEqual(cast_store.clear_suggested_voices(cast), 0)

    def test_it_is_refused_with_a_reason(self):
        with self.assertRaisesRegex(gr.Error, "switch the Engine to Breeze"):
            self._design(engine="chatterbox")
        with self.assertRaisesRegex(gr.Error, "cast analysis is running"):
            self._design(jobs=[{"status": "running", "kind": "cast"}])
        with self.assertRaisesRegex(gr.Error, "Describe the voice"):
            self._design(description="  ")
        with self.assertRaisesRegex(gr.Error, "Click a character"):
            self._design(key=None)
        with self.assertRaisesRegex(gr.Error, "no longer in the cast"):
            self._design(key="nobody")
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}), self.assertRaisesRegex(gr.Error, "BREEZE_BASE_URL"):
            self._design()
        self.assertEqual(self.designs, [])

    def test_a_running_breeze_book_does_not_stop_a_single_design(self):
        self._design(jobs=[{"status": "running", "kind": "book", "settings": {"engine": "breeze"}}])
        self.assertEqual(len(self.designs), 1)

    def test_a_running_chatterbox_book_stops_a_single_design(self):
        with self.assertRaisesRegex(gr.Error, "Chatterbox book"):
            self._design(jobs=[{"status": "running", "kind": "book", "settings": {"engine": "chatterbox"}}])
        self.assertEqual(self.designs, [])

    def test_a_failed_design_is_a_clear_error_and_changes_nothing(self):
        with patch.object(chatterbox_ui.voice_design, "design_voice",
                          side_effect=voice_design.DesignError("attempt 1: near-silence")):
            with self.assertRaisesRegex(gr.Error, "Could not design a voice: attempt 1: near-silence"):
                self._design()
        character = cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]["emera"]
        self.assertEqual((character["voice"], "voice_design" in character), ("Ada.wav", False))

    def test_picking_another_voice_by_hand_drops_a_waiting_design(self):
        cast = cast_store.load_cast(chatterbox_ui.cast_file_for("k"))
        cast["characters"]["emera"]["voice_design"] = {"status": "pending", "description": "x"}
        cast_store.save_cast(chatterbox_ui.cast_file_for("k"), cast)
        chatterbox_ui.apply_cast_edit("k", "emera", "female", "Ada.wav", "breeze")
        character = cast_store.load_cast(chatterbox_ui.cast_file_for("k"))["characters"]["emera"]
        self.assertNotIn("voice_design", character)

    def test_starter_voices_are_designed_in_turn_skipping_existing_ones(self):
        starters = voice_design.STARTER_VOICES
        self.assertGreaterEqual(len(starters), 20)
        self.assertEqual({g for _, g, _, _ in starters}, {"female", "male"})
        self.assertEqual({a for _, _, a, _ in starters}, {"child", "adult", "elderly"})
        self.assertEqual(len({n for n, _, _, _ in starters}), len(starters))
        self.add_voice(voice_design.file_name_for(starters[0][0]))
        made = []

        def designer(name, description, gender, age):
            made.append(name)
            if name == starters[1][0]:
                raise voice_design.DesignError("attempt 1: near-silence")
            self.add_voice(voice_design.file_name_for(name))
            return voice_design.file_name_for(name), _measurement()

        with self.assertLogs(chatterbox_ui.logger.name, "WARNING"):
            progress = list(chatterbox_ui.design_starter_voices(lambda: [], designer))
        todo = len(starters) - 1
        self.assertEqual(made, [n for n, _, _, _ in starters[1:]])
        self.assertEqual(progress[0], f"Designing 1 of {todo}: **{starters[1][0]}** (about 40 s each)...")
        self.assertEqual(len(progress), todo + 1)
        self.assertIn(f"Designed {todo - 1} starter voices.", progress[-1])
        self.assertIn(f"Couldn't design {starters[1][0]} (attempt 1: near-silence)", progress[-1])
        # Run again: only the one that failed is left.
        made.clear()
        list(chatterbox_ui.design_starter_voices(lambda: [], designer))
        self.assertEqual(made, [starters[1][0]])

    def test_starter_voices_all_there_already(self):
        for name, *_ in voice_design.STARTER_VOICES:
            self.add_voice(voice_design.file_name_for(name))
        self.assertEqual(list(chatterbox_ui.design_starter_voices(lambda: [], self._never)),
                         [f"All {len(voice_design.STARTER_VOICES)} starter voices are already in the library."])

    def _never(self, *args):
        raise AssertionError("nothing should be designed")

    def test_starter_voices_wait_for_books_and_cast_analyses(self):
        for job in ({"status": "running", "kind": "book"}, {"status": "running", "kind": "cast"}):
            with self.assertRaisesRegex(gr.Error, "is running|is generating"):
                list(chatterbox_ui.design_starter_voices(lambda job=job: [job], self._never))
        queued = [{"status": "queued", "kind": "book"}, {"status": "done", "kind": "book"}]
        self.assertEqual(len(list(chatterbox_ui.design_starter_voices(lambda: queued, self.fake_ok))),
                         len(voice_design.STARTER_VOICES) + 1)

    def fake_ok(self, name, description, gender, age):
        self.add_voice(voice_design.file_name_for(name))
        return voice_design.file_name_for(name), _measurement()

    def test_starter_voices_stop_when_a_book_starts_meanwhile(self):
        states = iter([[], [], [{"status": "running", "kind": "book"}]])
        progress = list(chatterbox_ui.design_starter_voices(lambda: next(states), self.fake_ok))
        self.assertIn("Designed 1 of", progress[-1])
        self.assertIn("then stopped", progress[-1])
        self.assertEqual(len([f for f in os.listdir(self.voices) if "Starter" in f]), 1)

    def test_starter_voices_need_breeze(self):
        with patch.dict(os.environ, {"BREEZE_BASE_URL": ""}), self.assertRaisesRegex(gr.Error, "BREEZE_BASE_URL"):
            list(chatterbox_ui.design_starter_voices(lambda: [], self._never))


class TestLayout(Workspace):

    def _handlers(self, breeze: str) -> set:
        with patch.dict(os.environ, {"BREEZE_BASE_URL": breeze}):
            ui = chatterbox_ui.build_ui()
        return {getattr(fn.fn, "__name__", None) for fn in ui.fns.values()}

    def test_the_design_controls_are_wired_when_breeze_is_set_up(self):
        names = self._handlers("http://breeze:8005")
        self.assertTrue({"design_for_character", "design_starters", "design_description_for"} <= names)

    def test_without_breeze_there_is_no_starter_button_and_the_ui_still_builds(self):
        self.assertNotIn("design_starters", self._handlers(""))


if __name__ == "__main__":
    unittest.main()
