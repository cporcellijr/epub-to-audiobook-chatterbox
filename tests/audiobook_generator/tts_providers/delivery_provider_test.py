"""Adaptive delivery in the OpenAI provider: a per-book baseline sent via extra_body even with
adaptive delivery off, mood-based presets and the narrator voice for every segment in single voice
mode when adaptive delivery is on, gain plus the peak guard applied to the decoded audio, and
Kokoro's total exemption (no extra_body, no gain, regardless of what a config carries)."""
import io
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydub import AudioSegment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core import delivery
from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M
from audiobook_generator.tts_providers.openai_tts_provider import OpenAITTSProvider


def _wav_bytes(peak_amplitude=None, ms=300, frame_rate=24000) -> bytes:
    """A short WAV clip: silence by default, or a square wave with the given peak sample value."""
    if peak_amplitude is None:
        audio = AudioSegment.silent(duration=ms, frame_rate=frame_rate)
    else:
        sample_count = int(frame_rate * ms / 1000)
        samples = [peak_amplitude if i % 2 == 0 else -peak_amplitude for i in range(sample_count)]
        raw = struct.pack("<%dh" % sample_count, *samples)
        audio = AudioSegment(data=raw, sample_width=2, frame_rate=frame_rate, channels=1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()


class _ScriptedResponses:
    """A client.audio.speech.create stand-in: always answers with the same peak level, recording
    every call's kwargs (voice, extra_body, ...) for inspection."""

    def __init__(self, peak_amplitude=None):
        self.peak_amplitude = peak_amplitude
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=_wav_bytes(self.peak_amplitude))


def _provider(**extra) -> OpenAITTSProvider:
    fields = dict(tts="openai", model_name="chatterbox", voice_name="Narrator.wav", output_format="mp3",
                 speed=1.0, instructions=None, language="en", sentence_pause_ms=100, paragraph_pause_ms=300,
                 paced_unit_mode="sentence", voice_mode="single", dialogue_voice=None, cast_file=None,
                 openai_base_url=None, adaptive_delivery=False, delivery_exaggeration=None,
                 delivery_cfg_weight=None, delivery_temperature=None)
    fields.update(extra)
    config = GeneralConfig(SimpleNamespace(**fields))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
        provider = OpenAITTSProvider(config)
    return provider


def _speak(provider: OpenAITTSProvider, text: str, responses: _ScriptedResponses, output_file: str) -> None:
    provider.client = MagicMock()
    provider.client.audio.speech.create.side_effect = lambda **kwargs: responses(**kwargs)
    tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
    with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
        provider.text_to_speech(text, output_file, tags)


class TestBaselineWithAdaptiveOff(unittest.TestCase):

    def test_adaptive_off_and_no_baseline_sends_exactly_todays_request_kwargs(self):
        provider = _provider()
        responses = _ScriptedResponses()
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, '"Fine," she said.', responses, os.path.join(tmp, "out.mp3"))
        self.assertTrue(responses.calls)
        for call in responses.calls:
            self.assertNotIn("extra_body", call)

    def test_a_baseline_with_adaptive_off_sends_the_plain_baseline_on_every_request(self):
        provider = _provider(delivery_exaggeration=0.8, delivery_cfg_weight=0.45, delivery_temperature=0.5)
        responses = _ScriptedResponses()
        # A soft cue is present but ignored entirely: adaptive delivery itself is off.
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, '"Fine," she whispered.', responses, os.path.join(tmp, "out.mp3"))
        self.assertTrue(responses.calls)
        for call in responses.calls:
            self.assertEqual(call["extra_body"], {"exaggeration": 0.8, "cfg_weight": 0.45, "temperature": 0.5})

    def test_a_partial_baseline_is_filled_in_from_chatterbox_saved_defaults(self):
        provider = _provider(delivery_exaggeration=0.9)  # cfg/temperature left unset
        responses = _ScriptedResponses()
        with patch("audiobook_generator.core.delivery.saved_chatterbox_defaults",
                  return_value=delivery.Baseline(0.1, 0.2, 0.3)):
            with tempfile.TemporaryDirectory() as tmp:
                _speak(provider, '"Fine."', responses, os.path.join(tmp, "out.mp3"))
        self.assertEqual(responses.calls[0]["extra_body"], {"exaggeration": 0.9, "cfg_weight": 0.2, "temperature": 0.3})


class TestAdaptiveDeliveryOn(unittest.TestCase):

    def _baseline_kwargs(self):
        return dict(adaptive_delivery=True, delivery_exaggeration=0.73, delivery_cfg_weight=0.5,
                   delivery_temperature=0.61)

    def test_single_voice_mode_uses_the_narrator_voice_for_every_segment(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        text = f'She whispered, "Go now."{M}"Get out!" he shouted.'
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, text, responses, os.path.join(tmp, "out.mp3"))
        self.assertEqual({call["voice"] for call in responses.calls}, {"Narrator.wav"})

    def test_each_segments_mood_preset_is_sent_via_extra_body(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        text = f'She whispered, "Go now."{M}"Get out!" he shouted.'
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, text, responses, os.path.join(tmp, "out.mp3"))
        bodies = [call["extra_body"] for call in responses.calls]
        self.assertIn({"exaggeration": 0.35, "cfg_weight": 0.35, "temperature": 0.5}, bodies)   # soft
        self.assertIn({"exaggeration": 1.0, "cfg_weight": 0.4, "temperature": 0.7}, bodies)      # excited
        self.assertIn({"exaggeration": 0.73, "cfg_weight": 0.5, "temperature": 0.61}, bodies)    # normal narration

    def test_gain_and_peak_guard_are_applied_before_the_pauses_are_joined(self):
        provider = _provider(output_format="wav", **self._baseline_kwargs())
        responses = _ScriptedResponses(peak_amplitude=32767)  # 0 dBFS raw: excited's +1.5 dB would clip
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            _speak(provider, '"Get out!" he shouted.', responses, path)
            exported = AudioSegment.from_file(path)
        self.assertAlmostEqual(exported.max_dBFS, delivery.PEAK_GUARD_DBFS, delta=0.2)

    def test_soft_gain_reduces_volume_without_the_guard_engaging(self):
        provider = _provider(output_format="wav", **self._baseline_kwargs())
        responses = _ScriptedResponses(peak_amplitude=23197)  # about -3 dBFS raw
        # Every scripted unit is 300 ms, and the whispered quotation is the first unit: measure it
        # alone, since the narration unit after it ("she whispered.") stays at 0 dB.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            _speak(provider, '"I will do this," she whispered.', responses, path)
            exported = AudioSegment.from_file(path)
        self.assertAlmostEqual(exported[:300].max_dBFS, -9.0, delta=0.3)  # -3 dBFS raw, -6 dB gain
        self.assertAlmostEqual(exported.max_dBFS, -3.0, delta=0.3)  # the narration unit, untouched

    def test_mood_is_logged_like_voice_is_today(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        with self.assertLogs("audiobook_generator.tts_providers.openai_tts_provider", level="INFO") as log:
            with tempfile.TemporaryDirectory() as tmp:
                _speak(provider, '"Get out!" he shouted.', responses, os.path.join(tmp, "out.mp3"))
        self.assertTrue(any("mood=excited" in message for message in log.output))


class TestKokoroIgnoresDelivery(unittest.TestCase):

    def test_kokoro_engine_never_sends_extra_body_or_changes_gain(self):
        provider = _provider(model_name="kokoro", adaptive_delivery=True, delivery_exaggeration=0.9,
                             delivery_cfg_weight=0.3, delivery_temperature=0.9, output_format="wav")
        responses = _ScriptedResponses(peak_amplitude=32767)  # would be clipped by the guard if it ran
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            _speak(provider, '"Get out!" he shouted.', responses, path)
            exported = AudioSegment.from_file(path)
        self.assertTrue(responses.calls)
        for call in responses.calls:
            self.assertNotIn("extra_body", call)
        self.assertAlmostEqual(exported.max_dBFS, 0.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
