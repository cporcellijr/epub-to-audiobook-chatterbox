"""Tone matching and the peak guard in the OpenAI provider: each voice's units are cut toward its
own clip (only voices brighter than their clip, only once enough of them is heard), the book can
turn matching off, Kokoro is never matched, and every unit of a lossy chapter peaks at most at
PEAK_GUARD_DBFS while a lossless chapter keeps its peaks."""
import io
import os
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core import delivery, tone_match
from audiobook_generator.tts_providers.openai_tts_provider import OpenAITTSProvider

RATE = 24000
# Four units of narration, about 4.5 s each, so one voice passes MIN_SPEECH_SECONDS within the chapter.
TEXT = " ".join(f"This is sentence number {n}, and it is easily long enough to be a unit of its own."
                for n in ("one", "two", "three", "four"))


def _wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _noise(seconds: float, seed: int, level: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, level, int(RATE * seconds)).astype(np.float32)


def _top_band(samples: np.ndarray) -> float:
    spectrum, _ = tone_match.speech_spectrum(samples)
    return float(tone_match.balance(spectrum, RATE)[-1])


class _NoiseResponses:
    """client.audio.speech.create stand-in: white noise (bright) for every request, optionally
    scaled to a given peak."""

    def __init__(self, peak=None):
        self.peak = peak
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        samples = _noise(4.5, seed=self.calls)
        if self.peak is not None:
            samples = samples * (self.peak / np.abs(samples).max())
        return SimpleNamespace(content=_wav(samples))


def _provider(**extra) -> OpenAITTSProvider:
    fields = dict(tts="openai", model_name="chatterbox", voice_name="Dark.wav", output_format="aac",
                  speed=1.0, instructions=None, language="en", sentence_pause_ms=100, paragraph_pause_ms=300,
                  paced_unit_mode="sentence", voice_mode="single", dialogue_voice=None, cast_file=None,
                  openai_base_url=None, adaptive_delivery=False, delivery_exaggeration=None,
                  delivery_cfg_weight=None, delivery_temperature=None)
    fields.update(extra)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
        return OpenAITTSProvider(GeneralConfig(SimpleNamespace(**fields)))


class TestToneMatchInTheProvider(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        dark = np.convolve(_noise(8, seed=99), [0.25, 0.5, 0.25], mode="same")
        with open(os.path.join(self.tmp.name, "Dark.wav"), "wb") as f:
            f.write(_wav(dark))
        with open(os.path.join(self.tmp.name, "Bright.wav"), "wb") as f:
            f.write(_wav(_noise(8, seed=98)))

    def tearDown(self):
        self.tmp.cleanup()

    def _units(self, provider, responses=None):
        """The unit audio handed to the chapter export, as float arrays (pauses dropped)."""
        responses = responses or _NoiseResponses()
        provider.client = MagicMock()
        provider.client.audio.speech.create.side_effect = lambda **kwargs: responses(**kwargs)
        captured = {}

        def export(pieces, audio_format, *args):
            captured["pieces"] = list(pieces)
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with patch.dict(os.environ, {"TTS_VOICES_DIR": self.tmp.name}), \
                patch.object(provider, "_combine_and_export", side_effect=export), \
                patch("audiobook_generator.tts_providers.openai_tts_provider._tiny_quote", return_value=False):
            provider.text_to_speech(TEXT, os.path.join(self.tmp.name, "out.aac"), tags)
        units = [np.frombuffer(p, dtype="<i2").astype(np.float32) / 32768 for p in captured["pieces"]]
        return [u for u in units if len(u) and np.abs(u).max() > 0]

    def test_a_voice_brighter_than_its_clip_is_cut(self):
        units = self._units(_provider())
        self.assertEqual(len(units), 4)
        for unit in units:
            self.assertLess(_top_band(unit), -8)  # white noise measures ~0 dB here before matching

    def test_a_voice_that_already_matches_its_clip_is_left_alone(self):
        units = self._units(_provider(voice_name="Bright.wav"))
        for unit in units:
            self.assertGreater(_top_band(unit), -2)

    def test_the_book_can_turn_matching_off(self):
        for unit in self._units(_provider(tone_match=False)):
            self.assertGreater(_top_band(unit), -2)

    def test_kokoro_is_never_matched(self):
        for unit in self._units(_provider(model_name="kokoro", voice_name="Dark.wav")):
            self.assertGreater(_top_band(unit), -2)

    def test_matching_carries_over_to_the_next_chapter(self):
        provider = _provider()
        self._units(provider)
        self.assertIsNotNone(provider._tone.cuts("Dark.wav", RATE))


class TestPeakGuardOnEveryUnit(unittest.TestCase):

    def _peaks(self, output_format):
        provider = _provider(output_format=output_format, tone_match=False)
        responses = _NoiseResponses(peak=0.99)
        provider.client = MagicMock()
        provider.client.audio.speech.create.side_effect = lambda **kwargs: responses(**kwargs)
        captured = {}
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(provider, "_combine_and_export",
                             side_effect=lambda pieces, *args: captured.setdefault("pieces", list(pieces))), \
                patch("audiobook_generator.tts_providers.openai_tts_provider._tiny_quote", return_value=False):
            provider.text_to_speech(TEXT, os.path.join(tmp, f"out.{output_format}"), tags)
        return [np.abs(np.frombuffer(p, dtype="<i2")).max() / 32768 for p in captured["pieces"] if len(p)]

    def test_a_lossy_chapter_has_headroom_on_every_unit(self):
        limit = 10 ** (delivery.PEAK_GUARD_DBFS / 20)
        self.assertTrue(all(peak <= limit + 1e-4 for peak in self._peaks("aac")))

    def test_a_lossless_chapter_keeps_its_peaks(self):
        self.assertGreater(max(self._peaks("wav")), 0.98)


if __name__ == "__main__":
    unittest.main()
