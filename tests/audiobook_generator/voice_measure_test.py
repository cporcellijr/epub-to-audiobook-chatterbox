"""Voice measurements: Praat on synthetic voices whose pitch and noise are known, the saved file,
which voices need measuring, and the within-gender percentiles cast suggestions match against."""
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydub import AudioSegment

from audiobook_generator.core import voice_measure

RATE = 24000


def _voice(f0_start: float, f0_end: float = None, noise: float = 0.0, seconds: float = 2.0) -> bytes:
    """A buzzy, voice-like tone (a fundamental plus harmonics) gliding from f0_start to f0_end, as WAV."""
    t = np.arange(int(RATE * seconds)) / RATE
    f0 = np.linspace(f0_start, f0_end or f0_start, len(t))
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    wave = sum(np.sin(k * phase) / k for k in range(1, 8))
    wave = wave + noise * np.random.default_rng(1).standard_normal(len(t))
    pcm = (0.3 * wave / np.max(np.abs(wave)) * 32767).astype(np.int16)
    out = io.BytesIO()
    AudioSegment(pcm.tobytes(), frame_rate=RATE, sample_width=2, channels=1).export(out, format="wav")
    return out.getvalue()


class TestMeasureAudio(unittest.TestCase):

    def test_pitch_range_and_huskiness_follow_the_voice(self):
        low, high = voice_measure.measure_audio(_voice(120)), voice_measure.measure_audio(_voice(220))
        self.assertAlmostEqual(low["f0_median"], 120, delta=3)
        self.assertAlmostEqual(high["f0_median"], 220, delta=4)
        self.assertLess(low["f0_range"], 1)  # a steady tone
        self.assertGreater(voice_measure.measure_audio(_voice(150, 300))["f0_range"], 6)  # an octave glide
        self.assertLess(voice_measure.measure_audio(_voice(150, noise=0.3))["hnr"],
                        voice_measure.measure_audio(_voice(150))["hnr"])  # noise sounds husky

    def test_silence_cannot_be_measured(self):
        out = io.BytesIO()
        AudioSegment.silent(duration=2000, frame_rate=RATE).export(out, format="wav")
        with self.assertRaises(ValueError):
            voice_measure.measure_audio(out.getvalue())


class TestSavedMeasurements(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "voice_features.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_load_and_forget(self):
        voice_measure.save_features("A.wav", {"f0_median": 150.0, "f0_range": 8.0, "hnr": 11.0}, "10:20", self.path)
        voice_measure.save_features("B.wav", {"f0_median": 200.0, "f0_range": 9.0, "hnr": 12.0}, "11:21", self.path)
        saved = voice_measure.load_features(self.path)
        self.assertEqual(set(saved), {"A.wav", "B.wav"})
        self.assertEqual(saved["A.wav"]["signature"], "10:20")
        voice_measure.forget_features("A.wav", self.path)
        voice_measure.forget_features("Never.wav", self.path)
        self.assertEqual(set(voice_measure.load_features(self.path)), {"B.wav"})

    def test_a_new_measuring_sentence_makes_old_measurements_stale(self):
        voice_measure.save_features("A.wav", {"f0_median": 150.0, "f0_range": 8.0, "hnr": 11.0}, "", self.path)
        with patch.object(voice_measure, "MEASURE_TEXT", "Another sentence."):
            self.assertEqual(voice_measure.load_features(self.path), {})
        self.assertEqual(voice_measure.load_features(os.path.join(self.tmp.name, "missing.json")), {})

    def test_new_and_replaced_voice_files_need_measuring(self):
        files = {}
        for name in ("Old.wav", "New.wav", "Replaced.wav"):
            files[name] = os.path.join(self.tmp.name, name)
            with open(files[name], "wb") as f:
                f.write(b"x" * 10)
        features = {"Old.wav": {"signature": voice_measure.file_signature(files["Old.wav"])},
                    "Replaced.wav": {"signature": "1:1"}}
        self.assertEqual(voice_measure.voices_to_measure(files, features), ["New.wav", "Replaced.wav"])


class TestTraits(unittest.TestCase):

    FEATURES = {
        "F1.wav": {"f0_median": 160, "f0_range": 6, "hnr": 10}, "F2.wav": {"f0_median": 200, "f0_range": 9, "hnr": 12},
        "F3.wav": {"f0_median": 240, "f0_range": 12, "hnr": 14}, "M1.wav": {"f0_median": 110, "f0_range": 9, "hnr": 9},
        "M2.wav": {"f0_median": 150, "f0_range": 9, "hnr": 9},
    }
    VOICES = [("F1.wav", "female"), ("F2.wav", "female"), ("F3.wav", "female"), ("M1.wav", "male"),
              ("M2.wav", "male"), ("Unmeasured.wav", "female")]

    def test_traits_rank_voices_within_their_gender(self):
        traits = voice_measure.voice_traits(self.VOICES, self.FEATURES)
        self.assertNotIn("Unmeasured.wav", traits)
        self.assertEqual([traits[v]["pitch"] for v in ("F1.wav", "F2.wav", "F3.wav")], [0.0, 0.5, 1.0])
        # 150 Hz is the higher of the two men, though lower than every woman.
        self.assertEqual((traits["M1.wav"]["pitch"], traits["M2.wav"]["pitch"]), (0.0, 1.0))
        self.assertEqual(traits["F1.wav"]["husky"], 1.0)  # the lowest harmonics-to-noise ratio
        self.assertEqual(traits["M1.wav"]["expressive"], 0.5)  # a tie shares the middle rank

    def test_a_neutral_voice_is_ranked_among_all_measured_voices(self):
        traits = voice_measure.voice_traits([("F1.wav", "female"), ("N.wav", "neutral")],
                                            {**self.FEATURES, "N.wav": {"f0_median": 300, "f0_range": 9, "hnr": 9}})
        self.assertEqual(traits["N.wav"]["pitch"], 1.0)

    def test_describe(self):
        self.assertEqual(voice_measure.describe({"pitch": 0.1, "husky": 0.9, "expressive": 0.1}, "female"),
                         "low for a woman, husky, even")
        self.assertEqual(voice_measure.describe({"pitch": 0.5, "husky": 0.5, "expressive": 0.9}, "male"),
                         "medium for a man, expressive")
        self.assertEqual(voice_measure.describe(None), "not measured")


if __name__ == "__main__":
    unittest.main()
