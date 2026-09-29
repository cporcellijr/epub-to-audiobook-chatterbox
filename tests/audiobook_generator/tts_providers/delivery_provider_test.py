"""Adaptive delivery in the OpenAI provider: a per-book baseline sent via extra_body even with
adaptive delivery off, mood-based presets and the narrator voice for every segment in single voice
mode when adaptive delivery is on, gain plus the peak guard applied to the decoded audio, and
Kokoro's total exemption (no extra_body, no gain, regardless of what a config carries)."""
import io
import hashlib
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydub import AudioSegment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core import delivery
from audiobook_generator.core.dialogue import DIALOGUE, PARAGRAPH_MARK as M, chapter_segments
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


def _settings(call: dict) -> dict:
    return {key: value for key, value in call.get("extra_body", {}).items() if key != "seed"}


class TestBaselineWithAdaptiveOff(unittest.TestCase):

    def test_adaptive_off_and_no_baseline_only_sends_a_seed_for_short_lines(self):
        provider = _provider()
        responses = _ScriptedResponses()
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, '"Fine," she said.', responses, os.path.join(tmp, "out.mp3"))
        self.assertTrue(responses.calls)
        for call in responses.calls:
            self.assertEqual(_settings(call), {})
            self.assertGreater(call["extra_body"]["seed"], 0)

    def test_a_baseline_with_adaptive_off_sends_the_plain_baseline_on_every_request(self):
        provider = _provider(delivery_exaggeration=0.8, delivery_cfg_weight=0.45, delivery_temperature=0.5)
        responses = _ScriptedResponses()
        # A soft cue is present but ignored entirely: adaptive delivery itself is off.
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, '"Fine," she whispered.', responses, os.path.join(tmp, "out.mp3"))
        self.assertTrue(responses.calls)
        for call in responses.calls:
            self.assertEqual(_settings(call), {"exaggeration": 0.8, "cfg_weight": 0.45, "temperature": 0.5})

    def test_a_partial_baseline_is_filled_in_from_chatterbox_saved_defaults(self):
        provider = _provider(delivery_exaggeration=0.9)  # cfg/temperature left unset
        responses = _ScriptedResponses()
        with patch("audiobook_generator.core.delivery.saved_chatterbox_defaults",
                  return_value=delivery.Baseline(0.1, 0.2, 0.3)):
            with tempfile.TemporaryDirectory() as tmp:
                _speak(provider, '"Fine."', responses, os.path.join(tmp, "out.mp3"))
        self.assertEqual(_settings(responses.calls[0]), {"exaggeration": 0.9, "cfg_weight": 0.2, "temperature": 0.3})

    def test_saved_defaults_are_read_once_per_chapter(self):
        provider = _provider(adaptive_delivery=True)  # every slider comes from Chatterbox's saved defaults
        responses = _ScriptedResponses()
        with patch("audiobook_generator.core.delivery.saved_chatterbox_defaults",
                   return_value=delivery.Baseline(0.5, 0.5, 0.5)) as saved:
            with tempfile.TemporaryDirectory() as tmp:
                _speak(provider, f'"Fine," she said.{M}"Get out!" he shouted.', responses,
                       os.path.join(tmp, "out.mp3"))
        self.assertGreater(len(responses.calls), 2)
        self.assertEqual(saved.call_count, 1)


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

    def test_very_short_mooded_quotes_use_the_book_baseline(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        text = f'She whispered, "Go now."{M}"Get out!" he shouted.'
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, text, responses, os.path.join(tmp, "out.mp3"))
        bodies = [_settings(call) for call in responses.calls]
        baseline = {"exaggeration": 0.73, "cfg_weight": 0.5, "temperature": 0.61}
        self.assertGreaterEqual(bodies.count(baseline), 2)  # both short mooded quotes
        self.assertIn(baseline, bodies)                     # normal narration

    def test_medium_mooded_quote_eases_parameters_toward_its_full_preset(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, 'She whispered, "I will remember this."', responses,
                   os.path.join(tmp, "out.mp3"))
        # The medium quote receives only part of the soft preset.
        quote_body = next(_settings(call) for call in responses.calls if _settings(call) != {
            "exaggeration": 0.73, "cfg_weight": 0.5, "temperature": 0.61})
        self.assertEqual(quote_body, {"exaggeration": 0.58, "cfg_weight": 0.44, "temperature": 0.57})

    def test_an_untagged_exclamation_uses_mild_emphasis(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        with tempfile.TemporaryDirectory() as tmp:
            _speak(provider, '"Get out!"', responses, os.path.join(tmp, "out.mp3"))
        self.assertEqual(_settings(responses.calls[0]),
                         {"exaggeration": 0.73, "cfg_weight": 0.5, "temperature": 0.61})

    def test_a_saved_excited_label_yields_to_the_new_punctuation_rule(self):
        text = '"Get out!"'
        provider = _provider(**self._baseline_kwargs())
        provider.cast = {"chapters": {hashlib.sha1(text.encode("utf-8")).hexdigest():
                                      {"moods": {"1": "excited"}}}}
        quote = next(piece for paragraph in chapter_segments(text) for piece in paragraph if piece.kind == DIALOGUE)
        self.assertEqual(provider._mood_of(text)(quote), delivery.MOOD_EMPHATIC)

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
            _speak(provider, '"No." she whispered.', responses, path)
            exported = AudioSegment.from_file(path)
        self.assertAlmostEqual(exported[:300].max_dBFS, -3.0, delta=0.3)  # short quote keeps baseline gain
        self.assertAlmostEqual(exported.max_dBFS, -3.0, delta=0.3)  # the narration unit, untouched

    def test_medium_excited_gain_is_applied_partially(self):
        provider = _provider(output_format="wav", **self._baseline_kwargs())
        responses = _ScriptedResponses(peak_amplitude=23197)  # about -3 dBFS raw
        quote = "An urgent reply."
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            _speak(provider, '"' + quote + '" he shouted.', responses, path)
            exported = AudioSegment.from_file(path)
        expected_gain = delivery.unit_preset(delivery.MOOD_EXCITED, delivery.APPROVED_BASELINE, quote)[3]
        self.assertGreater(expected_gain, 0.0)
        self.assertLess(expected_gain, delivery.preset(delivery.MOOD_EXCITED, delivery.APPROVED_BASELINE)[3])
        self.assertAlmostEqual(exported[:300].max_dBFS, -3.0 + expected_gain, delta=0.3)

    def test_mood_is_logged_like_voice_is_today(self):
        provider = _provider(**self._baseline_kwargs())
        responses = _ScriptedResponses()
        with self.assertLogs("audiobook_generator.tts_providers.openai_tts_provider", level="INFO") as log:
            with tempfile.TemporaryDirectory() as tmp:
                _speak(provider, '"Get out!" he shouted.', responses, os.path.join(tmp, "out.mp3"))
        self.assertTrue(any("mood=excited" in message for message in log.output))


class TestCharacterDelivery(unittest.TestCase):
    """Cast mode with adaptive delivery: an even or expressive character's lines are read around a
    shifted baseline; narration and everyone else keep the book's."""

    TEXT = '"Come here," said Ada. "Fine," said Tom. "Who knows," said a stranger.'

    def _cast_file(self, tmp: str, text: str = None) -> str:
        from audiobook_generator.core import cast as cast_store
        cast = cast_store.new_cast("k", "/x.epub", "T", "A", "chatterbox", "Narrator.wav", [1])
        cast["characters"] = {
            "ada": {"name": "Ada", "aliases": [], "gender": "female", "age": "adult", "lines": 5, "voice": "Ada.wav",
                    "profile": {"voice_targets": {"delivery": "expressive"}}},
            "tom": {"name": "Tom", "aliases": [], "gender": "male", "age": "adult", "lines": 4, "voice": "Tom.wav",
                    "delivery": "even"},
            # Not in this chapter; an even profile beside Ada's expressive one keeps the cast's centre at 0.
            "sam": {"name": "Sam", "aliases": [], "gender": "male", "age": "adult", "lines": 5, "voice": "Sam.wav",
                    "profile": {"voice_targets": {"delivery": "even"}}},
        }
        cast["chapters"][cast_store.text_hash(text or self.TEXT)] = {
            "number": 1, "lines": {"1": "ada", "2": "tom", "3": None}}
        path = os.path.join(tmp, "cast.json")
        cast_store.save_cast(path, cast)
        return path

    def _bodies(self, adaptive: bool, text: str = None) -> dict:
        text = text or self.TEXT
        with tempfile.TemporaryDirectory() as tmp:
            provider = _provider(voice_mode="cast", dialogue_voice="Dialogue.wav", cast_file=self._cast_file(tmp, text),
                                 adaptive_delivery=adaptive, delivery_exaggeration=0.73, delivery_cfg_weight=0.5,
                                 delivery_temperature=0.61)
            responses = _ScriptedResponses()
            # This test's fixed 300 ms fake response is deliberately too short for the longer
            # dialogue fixture; duration validation belongs to the paced speech tests.
            with patch("audiobook_generator.tts_providers.openai_tts_provider._implausible_quote_duration",
                       return_value=None):
                _speak(provider, text, responses, os.path.join(tmp, "out.mp3"))
        return {call["input"]: call["extra_body"]["exaggeration"] for call in responses.calls}

    def test_very_short_character_lines_use_the_book_baseline(self):
        bodies = self._bodies(adaptive=True)
        self.assertEqual(bodies['"Come here,"'], 0.73)
        self.assertEqual(bodies['"Fine,"'], 0.73)
        self.assertEqual(bodies['"Who knows,"'], 0.73)   # an unknown speaker: the book's own
        self.assertEqual(bodies["said Ada."], 0.73)      # narration: the book's own

    def test_long_character_lines_keep_their_cast_delivery(self):
        text = ('"Come here, I have something important to tell you now," said Ada. '
                '"I suppose that is all you have to say for now," said Tom.')
        bodies = self._bodies(adaptive=True, text=text)
        self.assertEqual(bodies['"Come here, I have something important to tell you now,"'], 0.85)
        self.assertEqual(bodies['"I suppose that is all you have to say for now,"'], 0.61)

    def test_without_adaptive_delivery_every_line_keeps_the_books_baseline(self):
        self.assertEqual(set(self._bodies(adaptive=False).values()), {0.73})

    def _voices(self, picked: bool) -> dict:
        """Voice per request in a first-person book where Ada tells the story."""
        from audiobook_generator.core import cast as cast_store
        with tempfile.TemporaryDirectory() as tmp:
            path = self._cast_file(tmp)
            cast = cast_store.load_cast(path)
            cast["book_tone"] = {"point_of_view": "first", "pov_key": "ada"}
            cast["characters"]["ada"]["voice_picked"] = picked
            cast_store.save_cast(path, cast)
            provider = _provider(voice_mode="cast", dialogue_voice="Dialogue.wav", cast_file=path,
                                 adaptive_delivery=True, delivery_exaggeration=0.73, delivery_cfg_weight=0.5,
                                 delivery_temperature=0.61)
            responses = _ScriptedResponses()
            _speak(provider, self.TEXT, responses, os.path.join(tmp, "out.mp3"))
        return {call["input"]: (call["voice"], call["extra_body"]["exaggeration"]) for call in responses.calls}

    def test_a_first_person_narrators_lines_are_read_in_the_narrators_voice(self):
        voices = self._voices(picked=False)
        self.assertEqual(voices['"Come here,"'], ("Narrator.wav", 0.73))  # the narrator's voice and delivery
        self.assertEqual(voices['"Fine,"'][0], "Tom.wav")
        # The owner gave Ada a voice of her own in the cast editor: it wins.
        self.assertEqual(self._voices(picked=True)['"Come here,"'][0], "Ada.wav")

    def test_each_chapter_has_its_own_first_person_narrator(self):
        # An anthology: this chapter is Tom's story, whatever the book's tone says, so Tom's own
        # voice narrates it (narration and his lines); a third-person chapter (narrator None) has
        # the book's narrator and everyone in their own voice.
        from audiobook_generator.core import cast as cast_store
        for narrator, expected in (("tom", ("Ada.wav", "Tom.wav", "Tom.wav")),
                                   (None, ("Ada.wav", "Tom.wav", "Narrator.wav"))):
            with tempfile.TemporaryDirectory() as tmp:
                path = self._cast_file(tmp)
                cast = cast_store.load_cast(path)
                cast["book_tone"] = {"point_of_view": "first", "pov_key": "ada"}
                cast["chapters"][cast_store.text_hash(self.TEXT)].update(point_of_view="first" if narrator else "third",
                                                                        narrator=narrator)
                cast_store.save_cast(path, cast)
                provider = _provider(voice_mode="cast", dialogue_voice="Dialogue.wav", cast_file=path,
                                     adaptive_delivery=True, delivery_exaggeration=0.73, delivery_cfg_weight=0.5,
                                     delivery_temperature=0.61)
                responses = _ScriptedResponses()
                _speak(provider, self.TEXT, responses, os.path.join(tmp, "out.mp3"))
            voices = {call["input"]: call["voice"] for call in responses.calls}
            self.assertEqual((voices['"Come here,"'], voices['"Fine,"'], voices["said Ada."]), expected)


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
