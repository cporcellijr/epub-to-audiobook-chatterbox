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

    def test_numbers_letterless_text_and_unheard_fillers_are_not_judged(self):
        self.assertIsNone(match("“Room 12!”", "Room twelve!"))
        self.assertIsNone(match("“...”", ""))
        self.assertIsNone(match("“Um...”", ""))  # Whisper often leaves hesitations out
        self.assertLess(match("“Um...”", "Oh, did you get gum?"), PASS_SCORE)
        self.assertEqual(match("“Mm-hmm.”", "Hmm"), 1.0)  # Whisper's spelling of a clear take
        self.assertLess(match("“Mm-hmm.”", "Do not tell!"), PASS_SCORE)
        self.assertEqual(match("“Kiss me!”", ""), 0)


class TestLeaked(unittest.TestCase):
    def test_a_cut_take_starting_with_a_lead_in_word_leaked(self):
        self.assertTrue(speech_check.leaked("“Kiss me!”", "Open. Kiss me!", CARRIER))
        self.assertTrue(speech_check.leaked("“Kiss me!”", "window was open, kiss me", CARRIER))
        self.assertGreaterEqual(match("“Kiss me!”", "Open. Kiss me!"), PASS_SCORE)  # why match isn't enough
        self.assertFalse(speech_check.leaked("“Kiss me!”", "Kiss me!", CARRIER))
        self.assertFalse(speech_check.leaked("“Was he your first?”", "Was he your first?", CARRIER))
        self.assertFalse(speech_check.leaked("“Um...”", "", CARRIER))


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


if __name__ == "__main__":
    unittest.main()
