import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydub import AudioSegment

from audiobook_generator.core import speech_check
from audiobook_generator.core.speech_check import PASS_SCORE, carrier_bounds, match

CARRIER = "The room was quiet, and the window was open."
# Whisper small's word times for a real take of the lead-in and “Oh god!” (2026-09-30).
OH_GOD = [("The", 0, 300), ("room", 300, 480), ("was", 480, 740), ("quiet,", 740, 1100), ("and", 1300, 1400),
          ("the", 1400, 1520), ("window", 1520, 1800), ("was", 1800, 2100), ("open.", 2100, 2400),
          ("Oh,", 2840, 3080), ("God!", 3140, 3500)]


class TestMatch(unittest.TestCase):
    def test_garbled_takes_fail_and_clear_ones_pass(self):
        # Transcripts of takes the owner judged by ear, or marked while listening (WORKLOG §26).
        garbled = [("“Kiss me!”", "He's nearly."), ("“Oh,”", "They're going in the money"),
                   ("“About us.”", "Who cares?"), ("“I do,”", "Hello?"),
                   ("“Okay, okay!”", "Look here, okay, okay."), ("“Oh!”", "It's you!")]
        clear = [("“Kiss me!”", "Kiss me."), ("“Mmm, yummy,”", "Mmm yummy"), ("“Jeeesuss...”", "Jesus."),
                 ("Jon said.", "John said."), ("“Oh,”", "Oh!"), ("“It’s okay, Sam,”", "It's okay, Sam.")]
        for expected, heard in garbled:
            self.assertLess(match(expected, heard), PASS_SCORE, (expected, heard))
        for expected, heard in clear:
            self.assertGreaterEqual(match(expected, heard), PASS_SCORE, (expected, heard))

    def test_digits_whisper_writes_are_compared_as_words(self):
        # A clear "Three," heard as "3." scored 0 and was retried twice (WORKLOG §26.5).
        for expected, heard in (("“Three,”", "3."), ("“Twenty-one!”", "21!"), ("“First!”", "1st!"),
                                ("“Two thousand.”", "2,000."), ("“Four thirty,”", "4:30,"),
                                ("the twelfth time", "the 12th time")):
            self.assertEqual(match(expected, heard), 1.0, (expected, heard))
        self.assertLess(match("“Three,”", "3 times 3"), PASS_SCORE)

    def test_numbers_letterless_text_and_unheard_fillers_are_not_judged(self):
        self.assertIsNone(match("“Room 12!”", "Room twelve!"))
        self.assertIsNone(match("“...”", ""))
        self.assertIsNone(match("“Um...”", ""))  # Whisper often leaves hesitations out
        self.assertLess(match("“Um...”", "Oh, did you get gum?"), PASS_SCORE)
        self.assertEqual(match("“Mm-hmm.”", "Hmm"), 1.0)  # Whisper's spelling of a clear take
        self.assertLess(match("“Mm-hmm.”", "Do not tell!"), PASS_SCORE)
        self.assertEqual(match("“Kiss me!”", ""), 0)


class TestQuickPass(unittest.TestCase):
    def test_a_close_transcript_or_unjudgeable_text_settles_a_take(self):
        heard = lambda text: speech_check.Heard(text, [])
        self.assertTrue(speech_check.quick_pass("Kiss me now.", heard("Kiss me now.")))
        self.assertTrue(speech_check.quick_pass("Room 101.", heard("anything")))  # digits: match() can't judge
        # passes small's 0.70 but not the quick hearing's stricter mark: small hears it again
        self.assertGreaterEqual(match("Come here and kiss me now.", "Come here and kiss"), PASS_SCORE)  # 0.83
        self.assertFalse(speech_check.quick_pass("Come here and kiss me now.", heard("Come here and kiss")))
        self.assertFalse(speech_check.quick_pass("Kiss me now.", None))  # the quick hearing failed


class TestLeaked(unittest.TestCase):
    def test_a_cut_take_starting_with_a_lead_in_word_leaked(self):
        self.assertTrue(speech_check.leaked("“Kiss me!”", "Open. Kiss me!", CARRIER))
        self.assertTrue(speech_check.leaked("“Kiss me!”", "window was open, kiss me", CARRIER))
        self.assertGreaterEqual(match("“Kiss me!”", "Open. Kiss me!"), PASS_SCORE)  # why match isn't enough
        self.assertFalse(speech_check.leaked("“Kiss me!”", "Kiss me!", CARRIER))
        self.assertFalse(speech_check.leaked("“Was he your first?”", "Was he your first?", CARRIER))
        self.assertFalse(speech_check.leaked("“Um...”", "", CARRIER))


class TestClipped(unittest.TestCase):
    def test_a_cut_take_that_lost_its_first_word_is_clipped(self):
        self.assertTrue(speech_check.clipped("he grunts.", "Grunts"))  # a 30 ms rule cut after "he"
        self.assertTrue(speech_check.clipped("“Kiss me!”", "Me!"))
        for expected, heard in (("she said.", "She said."), ("“Jon, what?”", "John, what?"),
                                ("“Y-- yeah!”", "Why? Yeah!"), ("“Um... right.”", "Right."),
                                ("“Kiss me!”", ""), ("“Three,”", "3."), ("I stammer.", "Eye Stammer"),
                                ("“See! Asshole!”", "C. Asshole"), ("“Oh,”", "Ho!")):
            self.assertFalse(speech_check.clipped(expected, heard), (expected, heard))


class TestCarrierBounds(unittest.TestCase):
    def test_the_gap_after_the_lead_ins_last_word(self):
        self.assertEqual(carrier_bounds(OH_GOD, CARRIER), (2400, 2840))

    def test_a_misheard_word_is_tolerated_but_a_missing_lead_in_or_unit_is_not(self):
        misheard = [("quite",) + OH_GOD[3][1:] if i == 3 else w for i, w in enumerate(OH_GOD)]
        self.assertEqual(carrier_bounds(misheard, CARRIER), (2400, 2840))
        self.assertIsNone(carrier_bounds(OH_GOD[5:], CARRIER))
        self.assertEqual(carrier_bounds(OH_GOD[:9], CARRIER), (2400, None))  # nothing heard after it
        self.assertIsNone(carrier_bounds([], CARRIER))


class TestLoading(unittest.TestCase):
    def setUp(self):
        speech_check._loaded, speech_check._checker = False, None
        self.addCleanup(setattr, speech_check, "_loaded", False)
        self.addCleanup(setattr, speech_check, "_checker", None)

    def test_off_without_a_model_folder(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(speech_check.MODEL_ENV, None)
            self.assertIsNone(speech_check.get())

    def test_off_when_whisper_cannot_load(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {speech_check.MODEL_ENV: folder}), \
                patch.dict(sys.modules, {"faster_whisper": None}), \
                self.assertLogs(speech_check.logger, "WARNING"):
            self.assertIsNone(speech_check.get())

    def test_a_missing_model_is_downloaded_and_loaded_once(self):
        whisper = SimpleNamespace(WhisperModel=MagicMock())
        utils = SimpleNamespace(download_model=MagicMock())
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {speech_check.MODEL_ENV: folder}), \
                patch.dict(sys.modules, {"faster_whisper": whisper, "faster_whisper.utils": utils}):
            checker = speech_check.get()
            self.assertIs(speech_check.get(), checker)
            utils.download_model.assert_called_once_with("small", output_dir=folder)
            whisper.WhisperModel.assert_called_once()
            self.assertEqual(whisper.WhisperModel.call_args.args, (folder,))
            self.assertEqual(whisper.WhisperModel.call_args.kwargs["device"], "cpu")
            # Breeze's checks hear several takes at once; each worker gets its own threads.
            self.assertEqual(whisper.WhisperModel.call_args.kwargs["num_workers"], speech_check.WORKERS)
            self.assertLessEqual(whisper.WhisperModel.call_args.kwargs["cpu_threads"], 4)

    def test_the_quick_model_lives_beside_the_main_one_and_downloads_itself(self):
        whisper = SimpleNamespace(WhisperModel=MagicMock())
        utils = SimpleNamespace(download_model=MagicMock())
        self.addCleanup(setattr, speech_check, "_quick_loaded", False)
        self.addCleanup(setattr, speech_check, "_quick", None)
        speech_check._quick_loaded, speech_check._quick = False, None
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {speech_check.MODEL_ENV: os.path.join(folder, "faster-whisper-small")}), \
                patch.dict(sys.modules, {"faster_whisper": whisper, "faster_whisper.utils": utils}):
            os.environ.pop(speech_check.QUICK_MODEL_ENV, None)
            quick = speech_check.get_quick()
            self.assertIs(speech_check.get_quick(), quick)
            expected = os.path.join(folder, "faster-whisper-tiny.en")
            utils.download_model.assert_called_once_with("tiny.en", output_dir=expected)
            self.assertEqual(whisper.WhisperModel.call_args.args, (expected,))

    def test_the_quick_model_can_be_switched_off_and_needs_the_speech_check(self):
        for env in ({speech_check.MODEL_ENV: "/models/small", speech_check.QUICK_MODEL_ENV: "off"},
                    {speech_check.QUICK_MODEL_ENV: "/models/tiny"}):
            speech_check._quick_loaded, speech_check._quick = False, None
            with self.subTest(env=env), patch.dict(os.environ, env):
                if speech_check.MODEL_ENV not in env:
                    os.environ.pop(speech_check.MODEL_ENV, None)
                self.assertIsNone(speech_check.get_quick())
        speech_check._quick_loaded = False

    def test_transcribe_pads_the_take_and_reports_word_times_within_it(self):
        model = MagicMock()
        words = [SimpleNamespace(word=" Kiss", start=0.25, end=0.5), SimpleNamespace(word=" me!", start=0.55, end=0.9)]
        model.transcribe.return_value = (iter([SimpleNamespace(text=" Kiss me!", words=words)]), None)
        heard = speech_check.SpeechChecker(model).transcribe(AudioSegment.silent(1000, frame_rate=24000))
        self.assertEqual(heard.text, "Kiss me!")
        self.assertEqual(heard.words, [("Kiss", 50, 300), ("me!", 350, 700)])
        samples = model.transcribe.call_args.args[0]
        self.assertEqual(len(samples), 16000 * 1400 // 1000)
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "en")

    def test_word_times_are_only_asked_of_the_model_when_wanted(self):
        words = [SimpleNamespace(word=" Kiss", start=0.25, end=0.5)]
        model = MagicMock()
        # Like faster-whisper: no word list unless word_timestamps is on.
        model.transcribe.side_effect = lambda *a, **k: (
            iter([SimpleNamespace(text=" Kiss", words=words if k["word_timestamps"] else None)]), None)
        checker = speech_check.SpeechChecker(model)
        audio = AudioSegment.silent(1000, frame_rate=24000)
        heard = checker.transcribe(audio, words=False)
        self.assertIs(model.transcribe.call_args.kwargs["word_timestamps"], False)
        self.assertEqual((heard.text, heard.words), ("Kiss", []))
        self.assertTrue(checker.transcribe(audio).words)
        self.assertIs(model.transcribe.call_args.kwargs["word_timestamps"], True)

    def test_beam_search_unless_a_caller_asks_for_less(self):
        model = MagicMock()
        model.transcribe.side_effect = lambda *a, **k: (iter([SimpleNamespace(text=" Kiss", words=None)]), None)
        checker = speech_check.SpeechChecker(model)
        audio = AudioSegment.silent(1000, frame_rate=24000)
        checker.transcribe(audio)  # a voice clip's words, which Breeze clones from
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 5)
        checker.transcribe(audio, words=False, beam_size=speech_check.BATCH_BEAM)
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 1)


if __name__ == "__main__":
    unittest.main()
