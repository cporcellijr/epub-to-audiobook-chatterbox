"""Per-voice tone matching: band balance, cut-only gains, the minimum speech before a voice is
trusted, and a filtered take landing near its clip's balance."""
import os
import tempfile
import unittest
import wave

import numpy as np

from audiobook_generator.core import tone_match

RATE = 24000


def _noise(seconds: float, seed: int = 0, level: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, level, int(RATE * seconds)).astype(np.float32)


def _darker(samples: np.ndarray) -> np.ndarray:
    """A gentle low-pass ([1, 2, 1] / 4): treble falls off steadily, -6 dB at 6 kHz and -35 dB by 11 kHz."""
    return np.convolve(samples, [0.25, 0.5, 0.25], mode="same").astype(np.float32)


def _write_wav(path: str, samples: np.ndarray) -> None:
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class TestBalanceAndCuts(unittest.TestCase):

    def test_bands_stop_below_nyquist(self):
        edges = tone_match.band_edges(RATE)
        self.assertEqual(edges[0], 1600)
        self.assertLess(edges[-1], RATE / 2)
        self.assertEqual(len(edges), 10)

    def test_brighter_audio_measures_higher_in_the_top_bands(self):
        white, _ = tone_match.speech_spectrum(_noise(4))
        dark, _ = tone_match.speech_spectrum(_darker(_noise(4)))
        self.assertGreater(tone_match.balance(white, RATE)[-1], tone_match.balance(dark, RATE)[-1] + 10)

    def test_cuts_never_boost_and_stop_at_the_cap(self):
        made = np.array([0.0, 0.0, 5.0, 30.0, 30.0, 30.0, -10.0, -10.0, -10.0])
        clip = np.zeros(9)
        cuts = tone_match.cuts_for(made, clip)
        self.assertTrue(np.all(cuts <= 0))
        self.assertAlmostEqual(cuts[0], 0.0)
        self.assertAlmostEqual(cuts[4], tone_match.MAX_CUT_DB)
        self.assertTrue(np.all(cuts[-2:] == 0))  # darker than the clip: left alone

    def test_apply_keeps_length_and_leaves_audio_alone_without_cuts(self):
        x = _noise(1.3)
        np.testing.assert_array_equal(tone_match.apply(x, RATE, np.zeros(9)), x)
        self.assertEqual(len(tone_match.apply(x, RATE, np.full(9, -6.0))), len(x))

    def test_describe(self):
        self.assertEqual(tone_match.describe(np.zeros(9), RATE), "none")
        cuts = np.array([0, 0, 0, -0.5, -3.0, -6.0, -9.0, -11.0, -11.0])
        self.assertEqual(tone_match.describe(cuts, RATE), "up to 11.0 dB from 4.5 kHz")


class TestToneMatcher(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _write_wav(os.path.join(self.tmp.name, "Dark.wav"), _darker(_noise(6, seed=1)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_voice_is_left_alone_until_enough_of_it_is_measured(self):
        matcher = tone_match.ToneMatcher(self.tmp.name)
        matcher.add("Dark.wav", _noise(3))
        self.assertIsNone(matcher.cuts("Dark.wav", RATE))
        matcher.add("Dark.wav", _noise(6, seed=2))
        self.assertIsNotNone(matcher.cuts("Dark.wav", RATE))

    def test_a_voice_without_a_readable_clip_is_left_alone(self):
        matcher = tone_match.ToneMatcher(self.tmp.name)
        matcher.add("Missing.wav", _noise(10))
        self.assertIsNone(matcher.cuts("Missing.wav", RATE))
        self.assertIsNone(tone_match.ToneMatcher(None).cuts("Dark.wav", RATE))

    def test_a_brighter_take_is_brought_close_to_its_clip(self):
        matcher = tone_match.ToneMatcher(self.tmp.name)
        made = _noise(10, seed=3)
        matcher.add("Dark.wav", made)
        cuts = matcher.cuts("Dark.wav", RATE)
        self.assertLess(cuts[-1], -8)                       # the top bands are cut hard
        filtered, _ = tone_match.speech_spectrum(tone_match.apply(made, RATE, cuts))
        clip, _ = tone_match.speech_spectrum(tone_match.read_clip(os.path.join(self.tmp.name, "Dark.wav"), RATE))
        before = tone_match.balance(tone_match.speech_spectrum(made)[0], RATE)
        after, target = tone_match.balance(filtered, RATE), tone_match.balance(clip, RATE)
        # Bands within the cap land near the clip; every band moves toward it.
        self.assertTrue(np.all(np.abs(after - target) <= np.abs(before - target) + 0.5))
        self.assertLess(abs(after[2] - target[2]), 3.0)


if __name__ == "__main__":
    unittest.main()
