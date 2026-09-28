"""Multi-voice narration in the OpenAI provider: units never span a voice change, single voice
mode sends exactly today's requests, and per-unit voices follow the dialogue voice or the cast."""
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydub import AudioSegment

from audiobook_generator.core import cast as cast_store
from audiobook_generator.core.dialogue import DIALOGUE, Segment, chapter_segments
from audiobook_generator.tts_providers.openai_tts_provider import (
    MAX_REQUEST_CHARS, PARAGRAPH_MARK as M, OpenAITTSProvider, paced_units, paragraph_mode_units, voiced_units,
    voiced_paragraph_units,
)

# An invented passage: quotes, a tiny tag, a scene break, an oversized sentence (split with no pause)
# and short back-and-forth lines. Anything real would be private (see the build brief).
FIXTURE = (
    'The lamp had burned low by the time Ada Marsh came in from the yard. "You left the gate open again," '
    'she said. "The goats are in the beans." ' + M +
    '"They were in the beans before I got up," said her brother. He did not look up from the ledger. "Blame '
    'the goats." ' + M +
    'Ada set the lamp down. Outside, the wind was rising, and somewhere down the lane a dog began to bark and '
    'would not stop, a thin, complaining sound that carried over the fields, over the dark water of the '
    'millpond, over the roofs of the sleeping village and on into the hills, where nobody was awake to hear '
    'it except the shepherd, who heard everything and said nothing, as shepherds do, and who turned once in '
    'his blanket, pulled it higher against the cold, counted the stars he could see through the gap in the '
    'shutter, and went back to sleep without ever wondering what the dog had seen. ' + M +
    '* * *' + M +
    '"Well?" ' + M +
    '"Well what?" ' + M +
    '"Are you going to fetch them, or am I?" She was already pulling her boots back on.'
)

# The exact requests the provider sent for FIXTURE before multi-voice existed (recorded from the
# previous paced_units); single voice mode must keep sending precisely these.
TODAYS_UNITS = [
    (0, "The lamp had burned low by the time Ada Marsh came in from the yard.", False),
    (0, "\"You left the gate open again,\" she said. \"The goats are in the beans.\"", False),
    (1, "\"They were in the beans before I got up,\" said her brother.", False),
    (1, "He did not look up from the ledger. \"Blame the goats.\"", False),
    (2, "Ada set the lamp down. ", False),
    (2, "Outside, the wind was rising, and somewhere down the lane a dog began to bark and would not stop, a thin, "
        "complaining sound that carried over the fields, over the dark water of the millpond, over the roofs of the "
        "sleeping village and on into the hills, where nobody was awake to hear it except the shepherd, who heard "
        "everything and said nothing, as shepherds do, and who turned once in his blanket, pulled it higher against "
        "the cold,", True),
    (2, " counted the stars he could see through the gap in the shutter, and went back to sleep without ever "
        "wondering what the dog had seen.", True),
    (4, "\"Well?\"", False),
    (5, "\"Well what?\"", False),
    (6, "\"Are you going to fetch them, or am I?\" She was already pulling her boots back on.", False),
]

UNIT_MS = 500


def _wav_bytes(ms: int = UNIT_MS) -> bytes:
    buffer = io.BytesIO()
    AudioSegment.silent(duration=ms, frame_rate=24000).export(buffer, format="wav")
    return buffer.getvalue()


def _narrator_or_dialogue(piece: Segment) -> str:
    return "Dialogue.wav" if piece.kind == DIALOGUE else "Narrator.wav"


class TestVoicedUnits(unittest.TestCase):

    def test_single_voice_units_are_exactly_todays(self):
        self.assertEqual(paced_units(FIXTURE, "en"), TODAYS_UNITS)

    def test_units_never_span_a_voice_change(self):
        units = voiced_units(FIXTURE, "en", _narrator_or_dialogue)
        segments = [s for para in chapter_segments(FIXTURE) for s in para]
        for _, unit, _, voice in units:
            owners = {s.kind for s in segments if unit.strip() and unit.strip() in s.text}
            self.assertEqual(len(owners), 1, f"unit crosses segments: {unit!r}")
            self.assertEqual(voice, "Dialogue.wav" if owners == {DIALOGUE} else "Narrator.wav")
        # The narration between two quotes in one paragraph is its own unit even though it is short,
        # while short sentences inside one narration segment still join as before.
        self.assertIn((0, "she said.", False, "Narrator.wav"), units)
        self.assertIn((1, "said her brother. He did not look up from the ledger.", False, "Narrator.wav"), units)

    def test_paragraph_numbers_and_scene_breaks_match_single_voice_mode(self):
        units = voiced_units(FIXTURE, "en", _narrator_or_dialogue)
        self.assertEqual(sorted({p for p, _, _, _ in units}), [0, 1, 2, 4, 5, 6])  # "* * *" (3) is dropped

    def test_oversized_narration_is_still_split_with_no_pause(self):
        units = voiced_units(FIXTURE, "en", _narrator_or_dialogue)
        self.assertTrue(all(len(u) <= MAX_REQUEST_CHARS for _, u, _, _ in units))
        self.assertEqual([c for p, _, c, _ in units if p == 2], [False, True, True])

    def test_paragraph_mode_packs_within_segments(self):
        text = f'"One short line," said Ada. "Then another one." He waited a while.{M}Then nothing at all.'
        units = voiced_paragraph_units(text, "en", _narrator_or_dialogue)
        self.assertEqual([(u, n, v) for _, u, n, _, v in units], [
            ('"One short line,"', 1, "Dialogue.wav"), ("said Ada.", 1, "Narrator.wav"),
            ('"Then another one."', 1, "Dialogue.wav"), ("He waited a while.", 1, "Narrator.wav"),
            ("Then nothing at all.", 1, "Narrator.wav"),
        ])
        # Single voice paragraph mode keeps packing the whole paragraph as one request.
        single = paragraph_mode_units(text, "en")
        self.assertEqual(len(single), 2)
        self.assertEqual(single[0][1], text.split(M)[0])


class TestProviderVoices(unittest.TestCase):

    def _provider(self, **extra):
        fields = dict(tts="openai", model_name="chatterbox", voice_name="Narrator.wav", output_format="mp3",
                      speed=1.0, instructions=None, language="en", sentence_pause_ms=100, paragraph_pause_ms=300,
                      paced_unit_mode="sentence", voice_mode="single", dialogue_voice=None, cast_file=None,
                      openai_base_url=None)
        fields.update(extra)
        from audiobook_generator.config.general_config import GeneralConfig
        config = GeneralConfig(SimpleNamespace(**fields))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            provider = OpenAITTSProvider(config)
        provider.client = MagicMock()
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=_wav_bytes())
        return provider

    def _requests(self, provider, text):
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
                provider.text_to_speech(text, os.path.join(tmp, "out.mp3"), tags)
        return [(c.kwargs["voice"], c.kwargs["input"]) for c in provider.client.audio.speech.create.call_args_list]

    def test_single_voice_mode_sends_todays_requests_with_the_one_voice(self):
        requests = self._requests(self._provider(), FIXTURE)
        self.assertEqual(requests, [("Narrator.wav", unit) for _, unit, _ in TODAYS_UNITS])

    def test_a_job_without_a_voice_mode_is_single_voice(self):
        provider = self._provider(voice_mode=None, dialogue_voice="Dialogue.wav")
        self.assertEqual(provider.config.voice_mode, "single")
        self.assertTrue(all(v == "Narrator.wav" for v, _ in self._requests(provider, FIXTURE)))

    def test_dialogue_mode_sends_quoted_lines_with_the_dialogue_voice(self):
        requests = self._requests(self._provider(voice_mode="dialogue", dialogue_voice="Dialogue.wav"),
                                  f'She said, "Come in, the door is open." He did not.')
        self.assertEqual(requests, [("Narrator.wav", "She said,"), ("Dialogue.wav", '"Come in, the door is open."'),
                                    ("Narrator.wav", "He did not.")])

    def test_dialogue_mode_without_a_dialogue_voice_falls_back_to_the_narrator(self):
        requests = self._requests(self._provider(voice_mode="dialogue"), 'She said, "Come in." He did not.')
        self.assertTrue(all(v == "Narrator.wav" for v, _ in requests))

    def test_cast_mode_uses_each_speakers_voice_and_the_dialogue_voice_for_unknowns(self):
        text = f'"Come in," said Ada.{M}"Thanks," he said.{M}"Who is there?"'
        cast = cast_store.new_cast("k", "/x.epub", "T", "A", "chatterbox", "Narrator.wav", [1])
        cast["characters"] = {"ada marsh": {"name": "Ada Marsh", "voice": "Ada.wav", "lines": 1},
                              "guest": {"name": "Guest", "voice": None, "lines": 1}}
        cast["chapters"][cast_store.text_hash(text)] = {"number": 1, "lines": {"1": "ada marsh", "2": "guest", "3": None}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cast.json")
            cast_store.save_cast(path, cast)
            provider = self._provider(voice_mode="cast", dialogue_voice="Dialogue.wav", cast_file=path)
            requests = self._requests(provider, text)
        self.assertEqual(requests, [
            ("Ada.wav", '"Come in,"'), ("Narrator.wav", "said Ada."),
            ("Dialogue.wav", '"Thanks,"'), ("Narrator.wav", "he said."),  # character without a voice yet
            ("Dialogue.wav", '"Who is there?"'),                          # unknown speaker
        ])

    def test_cast_mode_with_an_unanalysed_chapter_speaks_all_quotes_with_the_dialogue_voice(self):
        cast = cast_store.new_cast("k", "/x.epub", "T", "A", "chatterbox", "Narrator.wav", [1])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cast.json")
            cast_store.save_cast(path, cast)
            provider = self._provider(voice_mode="cast", dialogue_voice="Dialogue.wav", cast_file=path)
            requests = self._requests(provider, '"Hello there," she said.')
        self.assertEqual(requests, [("Dialogue.wav", '"Hello there,"'), ("Narrator.wav", "she said.")])

    def test_cast_mode_needs_a_readable_cast_file(self):
        with self.assertRaises(ValueError):
            self._provider(voice_mode="cast", dialogue_voice="Dialogue.wav", cast_file="/nowhere/cast.json")

    def test_unknown_voice_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self._provider(voice_mode="chorus")

    def test_pauses_between_a_narration_and_a_dialogue_segment_are_sentence_pauses(self):
        provider = self._provider(voice_mode="dialogue", dialogue_voice="Dialogue.wav")
        text = f'She said, "Come in." He did.{M}Later.'
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp3")
            with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
                provider.text_to_speech(text, path, tags)
            duration = len(AudioSegment.from_file(path))
        # 4 units: 3 in the first paragraph (2 sentence pauses of 100 ms) + 1 paragraph pause of 300 ms
        self.assertAlmostEqual(duration, 4 * UNIT_MS + 2 * 100 + 300, delta=80)
