"""The Breeze path of the OpenAI provider: a chapter's units go to the server in batches of 32 (64 when
short) with their voice and its transcript (longest text first, directed units apart from plain ones), the checks
of one batch run in a worker thread while the next batch generates, units that fail are sent again
together with a new seed (the best take is kept), server errors count as failed attempts, and tone
matching and the clip map work as for Chatterbox. The server is a fake; nothing here talks to a real
one."""
import json
import os
import re
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core import speech_check, tone_match
from audiobook_generator.core.dialogue import PARAGRAPH_MARK
from audiobook_generator.tts_providers import openai_tts_provider
from audiobook_generator.tts_providers.openai_tts_provider import (BREEZE_BATCH_SIZE, BREEZE_SHORT_BATCH_CHARS,
                                                                    BREEZE_SHORT_BATCH_SIZE, OpenAITTSProvider,
                                                                    _breeze_batches)

RATE = 24000
PROVIDER = "audiobook_generator.tts_providers.openai_tts_provider"
TAGS = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)


def _text(count: int) -> str:
    return " ".join(f"Sentence {chr(65 + n // 26)}{chr(65 + n % 26)} is long enough to be a unit of its own."
                    for n in range(count))


def _unit_text(index: int) -> str:
    return f"Sentence {chr(65 + index // 26)}{chr(65 + index % 26)} is long enough to be a unit of its own."


def _tone(ms: int) -> AudioSegment:
    return Sine(440).to_audio_segment(duration=ms, volume=-12).set_frame_rate(RATE).set_channels(1).set_sample_width(2)


def _chunk(item: dict) -> int:
    return int(re.search(r"chunk_(\d+)_of", item["id"]).group(1))


class FakeBreeze:
    """breeze_client.synthesize_batch stand-in: a tone of 1000 + chunk number ms for every item,
    unless `behave(item, round)` (round = how often this chunk was asked for, from 1) says otherwise:
    an AudioSegment, or a string for a server error."""

    def __init__(self, behave=None):
        self.calls = []
        self.asked = {}
        self.behave = behave

    def __call__(self, items, seed=None):
        self.calls.append((list(items), seed))
        results = []
        for item in items:
            number = _chunk(item)
            self.asked[number] = self.asked.get(number, 0) + 1
            made = self.behave(item, self.asked[number]) if self.behave else None
            results.append(made if made is not None else _tone(1000 + number))
        return results


def _provider(**extra) -> OpenAITTSProvider:
    fields = dict(tts="openai", model_name="breeze", voice_name="Dark.wav", output_format="wav",
                  speed=1.0, instructions=None, language="en", sentence_pause_ms=100, paragraph_pause_ms=300,
                  paced_unit_mode="sentence", voice_mode="single", dialogue_voice=None, cast_file=None,
                  openai_base_url=None, adaptive_delivery=False, delivery_exaggeration=None,
                  delivery_cfg_weight=None, delivery_temperature=None, tone_match=False)
    fields.update(extra)
    return OpenAITTSProvider(GeneralConfig(SimpleNamespace(**fields)))


def _first_attempt(server) -> list:
    """Every item sent with the first call's seed: the first attempt, however many requests it took."""
    seed = server.calls[0][1]
    return [item for items, call_seed in server.calls if call_seed == seed for item in items]


class _Run:
    """One chapter through the provider with a fake server, a fake transcript and a fake checker."""

    def __init__(self, test, text, server=None, checker=None, transcript=lambda voice: f"words of {voice}",
                 quick=None, **extra):
        self.server = server or FakeBreeze()
        self.provider = _provider(**extra)
        self.dir = tempfile.TemporaryDirectory()
        test.addCleanup(self.dir.cleanup)
        self.output = os.path.join(self.dir.name, "out.wav")
        self.pieces = None
        original = self.provider._combine_and_export

        def capture(pieces, *args):
            self.pieces = list(pieces)
            original(pieces, *args)
        with patch(f"{PROVIDER}.breeze_client.synthesize_batch", self.server), \
                patch(f"{PROVIDER}.voice_transcripts.transcript", side_effect=transcript) as self.transcripts, \
                patch(f"{PROVIDER}.speech_check.get", return_value=checker), \
                patch(f"{PROVIDER}.speech_check.get_quick", return_value=quick), \
                patch.object(self.provider, "_combine_and_export", side_effect=capture):
            self.error = None
            try:
                self.provider.text_to_speech(text, self.output, TAGS)
            except Exception as error:
                self.error = error

    def clips(self):
        with open(f"{self.output}.clips.json", encoding="utf-8") as f:
            return json.load(f)["clips"]


class TestBreezeBatches(unittest.TestCase):

    def test_short_units_go_out_64_at_a_time_in_order(self):
        self.assertEqual((BREEZE_BATCH_SIZE, BREEZE_SHORT_BATCH_SIZE), (32, 64))
        self.assertLessEqual(len(_unit_text(0)), BREEZE_SHORT_BATCH_CHARS)
        run = _Run(self, _text(70))
        self.assertIsNone(run.error)
        self.assertEqual([len(items) for items, _ in run.server.calls], [64, 6])
        sent = [_chunk(item) for items, _ in run.server.calls for item in items]
        self.assertEqual(sent, list(range(1, 71)))
        # The chapter is assembled in unit order: take n lasts 1000 + n ms.
        takes = [len(AudioSegment(data=p, frame_rate=RATE, channels=1, sample_width=2)) for p in run.pieces
                 if len(p) > 20000]
        self.assertEqual(takes, [1000 + n for n in range(1, 71)])

    def test_every_item_carries_its_text_voice_and_transcript(self):
        run = _Run(self, _text(3), voice_mode="dialogue", dialogue_voice="Teen.mp3",
                   transcript=lambda voice: {"Dark.wav": "dark words", "Teen.mp3": "teen words"}[voice])
        self.assertIsNone(run.error)
        items = run.server.calls[0][0]
        self.assertEqual([item["text"] for item in items], [_unit_text(n) for n in range(3)])
        # Narration is the narrator's voice; no quotes here, so every unit is Dark.wav.
        self.assertEqual({(item["voice"], item["ref_text"]) for item in items}, {("Dark.wav", "dark words")})
        self.assertTrue(all(item["instruction"] is None and item["cfg_scale"] is None for item in items))

    def test_a_dialogue_unit_is_sent_with_the_dialogue_voice_and_its_own_transcript(self):
        text = f"{_unit_text(0)} “Where are you going to be tomorrow evening?” she asked him quietly."
        run = _Run(self, text, voice_mode="dialogue", dialogue_voice="Teen.mp3",
                   transcript=lambda voice: {"Dark.wav": "dark words", "Teen.mp3": "teen words"}[voice])
        self.assertIsNone(run.error)
        voices = {(item["voice"], item["ref_text"]) for item in run.server.calls[0][0]}
        self.assertIn(("Teen.mp3", "teen words"), voices)
        self.assertIn(("Dark.wav", "dark words"), voices)

    def test_a_voice_without_a_transcript_fails_the_chapter_before_any_request(self):
        run = _Run(self, _text(3), transcript=lambda voice: None)
        self.assertIsInstance(run.error, ValueError)
        self.assertIn("Dark.wav", str(run.error))
        self.assertEqual(run.server.calls, [])

    def test_paragraph_mode_and_the_openai_client_are_not_used(self):
        provider = _provider(paced_unit_mode="paragraph")
        self.assertIsNone(provider.client)
        run = _Run(self, _text(2), paced_unit_mode="paragraph")
        self.assertIsNone(run.error)
        self.assertEqual(len(run.server.calls[0][0]), 2)  # sentence units, not one paragraph request

    def test_adaptive_delivery_off_sends_no_instructions_whatever_the_moods(self):
        run = _Run(self, MOODY, adaptive_delivery=False)
        self.assertIsNone(run.error)
        self.assertTrue(all(item["instruction"] is None for items, _ in run.server.calls for item in items))
        self.assertEqual({clip["mood"] for clip in run.clips()}, {"normal"})
        self.assertTrue(all("instruction" not in clip for clip in run.clips()))


MOODY = (f"{_unit_text(0)}{PARAGRAPH_MARK}"
         f"\u201cWhere are you going to be tomorrow evening?\u201d she whispered.{PARAGRAPH_MARK}"
         f"\u201cGet out of this house right now, all of you!\u201d he shouted.{PARAGRAPH_MARK}"
         f"\u201cI cannot believe that you would do this to me!\u201d she said.{PARAGRAPH_MARK}"
         f"\u201cThe weather was fine all week,\u201d said Tom.")


def _sentence(n: int, extra: int) -> str:
    return f"Sentence {chr(65 + n // 26)}{chr(65 + n % 26)} is long enough to be a unit of its own{' really' * extra}."


def _takes(run) -> list:
    return [len(AudioSegment(data=p, frame_rate=RATE, channels=1, sample_width=2)) for p in run.pieces
            if len(p) > 20000]


class TestBreezeBatchOrder(unittest.TestCase):

    def test_a_chapter_of_mixed_lengths_goes_out_longest_first_and_is_assembled_in_unit_order(self):
        extras = [(n * 7) % 11 for n in range(40)]
        with patch.object(openai_tts_provider, "BREEZE_SHORT_BATCH_SIZE", 32):  # the order, not the size
            run = _Run(self, " ".join(_sentence(n, extras[n]) for n in range(40)))
        self.assertIsNone(run.error)
        self.assertEqual([len(items) for items, _ in run.server.calls], [32, 8])
        lengths = [len(item["text"]) for items, _ in run.server.calls for item in items]
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        # The partial last request holds the shortest units.
        self.assertLessEqual(max(len(i["text"]) for i in run.server.calls[1][0]),
                             min(len(i["text"]) for i in run.server.calls[0][0]))
        self.assertNotEqual([_chunk(i) for items, _ in run.server.calls for i in items], list(range(1, 41)))
        self.assertEqual(_takes(run), [1000 + n for n in range(1, 41)])

    def test_directed_units_never_share_a_request_with_plain_ones(self):
        run = _Run(self, MOODY, adaptive_delivery=True)
        self.assertIsNone(run.error)
        self.assertGreater(len(run.server.calls), 1)
        for items, _ in run.server.calls:
            self.assertEqual(len({bool(item["instruction"]) for item in items}), 1)
        self.assertTrue(any(item["instruction"] for items, _ in run.server.calls for item in items))
        self.assertEqual(_takes(run), [1000 + n for n in range(1, len(_takes(run)) + 1)])

    def test_the_breeze_checker_is_asked_for_no_word_times_and_a_greedy_transcript(self):
        asked = []

        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                asked.append((words, beam_size))
                return speech_check.Heard("words of Dark.wav", [])
        run = _Run(self, _text(3), checker=Checker())
        self.assertIsNone(run.error)
        self.assertTrue(asked)
        self.assertEqual(set(asked), {(False, speech_check.BATCH_BEAM)})

    def test_the_log_marks_a_directed_batch(self):
        with self.assertLogs(PROVIDER, level="INFO") as logs:
            run = _Run(self, MOODY, adaptive_delivery=True)
        self.assertIsNone(run.error)
        batch_lines = [line for line in logs.output if ", batch " in line and " of " in line and "units" in line]
        self.assertTrue(any("(directed)" in line for line in batch_lines))
        self.assertTrue(any("(directed)" not in line for line in batch_lines))


class TestBreezeBatchesFunction(unittest.TestCase):

    @staticmethod
    def _items(*specs):
        return [{"text": "x" * length, "instruction": "Whisper." if directed else None} for length, directed in specs]

    def test_plain_units_come_first_longest_text_first_and_directed_ones_after(self):
        items = self._items((5, False), (9, True), (7, False), (3, True), (8, False))
        self.assertEqual(_breeze_batches(list(range(5)), items), [[4, 2, 0], [1, 3]])

    def test_equal_lengths_keep_book_order(self):
        items = self._items(*[(10, False)] * 5)
        self.assertEqual(_breeze_batches([3, 0, 4, 1, 2], items), [[3, 0, 4, 1, 2]])
        self.assertEqual(_breeze_batches(list(range(5)), items), [list(range(5))])

    def test_each_group_is_cut_into_requests_of_32_or_64_by_its_longest_text(self):
        # lengths 100 down to 31: a request starting above 80 characters takes 32 units, below 64
        items = self._items(*[(100 - n, False) for n in range(70)], *[(50, True)] * 33)
        batches = _breeze_batches(list(range(103)), items)
        self.assertEqual([len(b) for b in batches], [32, 38, 33])
        self.assertEqual(batches[0], list(range(32)))
        self.assertEqual(batches[1], list(range(32, 70)))
        self.assertEqual(batches[2], list(range(70, 103)))

    def test_long_units_stay_32_at_a_time(self):
        items = self._items(*[(BREEZE_SHORT_BATCH_CHARS + 1, False)] * 70)
        self.assertEqual([len(b) for b in _breeze_batches(list(range(70)), items)], [32, 32, 6])
        items = self._items(*[(BREEZE_SHORT_BATCH_CHARS, False)] * 70)
        self.assertEqual([len(b) for b in _breeze_batches(list(range(70)), items)], [64, 6])

    def test_only_the_pending_units_are_batched(self):
        items = self._items((5, False), (9, False), (7, True), (6, False), (8, True))
        self.assertEqual(_breeze_batches([3, 0, 4], items), [[3, 0], [4]])
        self.assertEqual(_breeze_batches([], items), [])


class TestBreezeAdaptiveDelivery(unittest.TestCase):
    """Moods become voice directions on the batch items; the voice is still the clip's."""

    def test_only_soft_units_carry_an_instruction(self):
        run = _Run(self, MOODY, adaptive_delivery=True)
        self.assertIsNone(run.error)
        items = _first_attempt(run.server)
        by_mood = {clip["chunk"]: clip["mood"] for clip in run.clips()}
        self.assertEqual({"normal", "soft", "excited", "emphatic"}, set(by_mood.values()))
        for item in items:
            mood = by_mood[_chunk(item)]
            self.assertEqual(item["instruction"] is None, mood != "soft", (mood, item["text"]))
        self.assertTrue(all(item["voice"] == "Dark.wav" and item["ref_text"] == "words of Dark.wav"
                            and item["cfg_scale"] is None for item in items))

    def test_the_tags_verb_picks_the_wording(self):
        run = _Run(self, MOODY, adaptive_delivery=True)
        by_text = {item["text"]: item["instruction"] for item in _first_attempt(run.server)}
        pick = lambda words: [i for t, i in by_text.items() if words in t]
        self.assertEqual(pick("tomorrow evening"), ["Whisper this softly."])
        self.assertEqual(pick("right now"), [None])  # loud lines stay plain (the owner's ear, 2026-10-01)
        self.assertEqual(pick("believe"), [None])
        self.assertEqual(pick("weather"), [None])

    def test_the_clip_map_records_mood_and_instruction(self):
        run = _Run(self, MOODY, adaptive_delivery=True)
        clips = run.clips()
        sent = {_chunk(item): item["instruction"] for item in _first_attempt(run.server)}
        self.assertEqual(len([c for c in clips if c["mood"] != "normal"]), 3)
        for clip in clips:
            if sent[clip["chunk"]] is None:
                self.assertNotIn("instruction", clip)
            else:
                self.assertEqual(clip["instruction"], sent[clip["chunk"]])
        self.assertEqual(next(c for c in clips if c["mood"] == "soft")["instruction"], "Whisper this softly.")

    def test_no_gain_or_peak_change_is_applied_to_a_directed_take(self):
        with patch(f"{PROVIDER}.delivery.guarded_gain") as gain:
            run = _Run(self, MOODY, adaptive_delivery=True)
        self.assertIsNone(run.error)
        gain.assert_not_called()
        takes = [len(AudioSegment(data=p, frame_rate=RATE, channels=1, sample_width=2)) for p in run.pieces
                 if len(p) > 20000]
        self.assertEqual(takes, [1000 + n for n in range(1, len(takes) + 1)])

    def test_adaptive_is_active_for_breeze_only_when_the_book_asks(self):
        self.assertTrue(_provider(adaptive_delivery=True)._adaptive_active())
        self.assertFalse(_provider(adaptive_delivery=False)._adaptive_active())

    def test_the_mood_is_logged_with_a_rejected_take(self):
        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                return speech_check.Heard("completely different words", [])
        with self.assertLogs(PROVIDER, level="INFO") as logs:
            run = _Run(self, MOODY, FakeBreeze(), checker=Checker(), adaptive_delivery=True)
        self.assertIsNone(run.error)
        text = "\n".join(logs.output)
        self.assertIn("mood=soft", text)
        self.assertIn("mood=excited", text)

    def test_a_directed_unit_is_retried_with_its_instruction(self):
        def behave(item, round):
            return AudioSegment.silent(3500, frame_rate=RATE) if item["instruction"] and round == 1 else None
        run = _Run(self, MOODY, FakeBreeze(behave), adaptive_delivery=True)
        self.assertIsNone(run.error)
        first_seed = run.server.calls[0][1]
        resent = [item for items, seed in run.server.calls if seed != first_seed for item in items]
        self.assertIn("Whisper this softly.", [item["instruction"] for item in resent])
        self.assertTrue(all(c["attempts"] >= 2 for c in run.clips() if c["mood"] == "soft"))
        self.assertTrue(all(c["attempts"] == 1 for c in run.clips() if c["mood"] == "normal"))


class TestBreezeRetries(unittest.TestCase):

    def test_a_near_silent_take_is_sent_again_with_a_new_seed(self):
        def behave(item, round):
            return AudioSegment.silent(3500, frame_rate=RATE) if _chunk(item) == 2 and round == 1 else None
        run = _Run(self, _text(3), FakeBreeze(behave))
        self.assertIsNone(run.error)
        (first, first_seed), (second, second_seed) = run.server.calls
        self.assertEqual([_chunk(i) for i in first], [1, 2, 3])
        self.assertEqual([_chunk(i) for i in second], [2])
        self.assertNotEqual(first_seed, second_seed)
        clips = run.clips()
        self.assertEqual([(c["attempts"], c["seed"]) for c in clips],
                         [(1, first_seed), (2, second_seed), (1, first_seed)])
        self.assertTrue(all("flagged" not in c for c in clips))

    def test_failed_units_of_a_big_chapter_are_resent_together(self):
        def behave(item, round):
            return AudioSegment.silent(3500, frame_rate=RATE) if _chunk(item) in (1, 40) and round == 1 else None
        with patch.object(openai_tts_provider, "BREEZE_SHORT_BATCH_SIZE", 32):  # two requests, then the retry
            run = _Run(self, _text(40), FakeBreeze(behave))
        self.assertIsNone(run.error)
        self.assertEqual([[_chunk(i) for i in items] for items, _ in run.server.calls],
                         [list(range(1, 33)), list(range(33, 41)), [1, 40]])

    def test_the_best_take_is_kept_when_none_passes_and_the_chapter_goes_on(self):
        expected = _unit_text(1)
        heard_by_length = {1002: "completely different words", 1102: " ".join(expected.split()[:4]),
                           1202: "nothing alike"}

        def behave(item, round):
            return _tone(1002 + (round - 1) * 100) if _chunk(item) == 2 else None

        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                if len(audio) in heard_by_length:
                    return speech_check.Heard(heard_by_length[len(audio)], [])
                return speech_check.Heard(_unit_text(len(audio) - 1001), [])
        run = _Run(self, _text(3), FakeBreeze(behave), checker=Checker())
        self.assertIsNone(run.error)
        self.assertEqual(len(run.server.calls), 3)  # the first try and _BAD_CLIP_RETRIES more
        seeds = [seed for _, seed in run.server.calls]
        self.assertEqual(len(set(seeds)), 3)
        clip = run.clips()[1]
        self.assertEqual((clip["attempts"], clip["seed"], clip["flagged"]), (3, seeds[1], "speech mismatch"))
        self.assertLess(clip["match"], speech_check.PASS_SCORE)
        # Attempt 2's take (the closest match, 1102 ms) is what the chapter holds.
        lengths = [len(AudioSegment(data=p, frame_rate=RATE, channels=1, sample_width=2)) for p in run.pieces
                   if len(p) > 20000]
        self.assertEqual(lengths, [1001, 1102, 1003])

    def test_a_server_error_counts_as_a_failed_attempt(self):
        run = _Run(self, _text(3), FakeBreeze(lambda item, round: "CUDA out of memory"
                                              if _chunk(item) == 3 and round == 1 else None))
        self.assertIsNone(run.error)
        self.assertEqual([[_chunk(i) for i in items] for items, _ in run.server.calls], [[1, 2, 3], [3]])
        self.assertEqual([c["attempts"] for c in run.clips()], [1, 1, 2])

    def test_a_unit_the_server_never_makes_audio_for_fails_the_chapter(self):
        run = _Run(self, _text(2), FakeBreeze(lambda item, round: "no good" if _chunk(item) == 2 else None))
        self.assertIsInstance(run.error, RuntimeError)
        self.assertIn("no good", str(run.error))
        self.assertEqual(len(run.server.calls), 3)

    def test_an_always_near_silent_take_is_kept_rather_than_failing_the_chapter(self):
        run = _Run(self, _text(2), FakeBreeze(lambda item, round:
                                              AudioSegment.silent(3500, frame_rate=RATE) if _chunk(item) == 2 else None))
        self.assertIsNone(run.error)
        self.assertEqual(run.clips()[1]["flagged"], "near-silent audio")


class TestBreezeChecksInAThread(unittest.TestCase):

    def test_batch_one_is_checked_while_batch_two_generates(self):
        second_batch_started = threading.Event()
        checked_in = []
        gate_opened = []

        class Server(FakeBreeze):
            def __call__(self, items, seed=None):
                if len(self.calls) == 1:
                    second_batch_started.set()
                return super().__call__(items, seed)

        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                checked_in.append(threading.current_thread().name)
                if not gate_opened:  # the very first take waits for batch two's request
                    gate_opened.append(second_batch_started.wait(timeout=10))
                return speech_check.Heard(_unit_text(len(audio) - 1001), [])
        with patch.object(openai_tts_provider, "BREEZE_SHORT_BATCH_SIZE", 32):  # two requests
            run = _Run(self, _text(40), Server(), checker=Checker())
        self.assertIsNone(run.error)
        self.assertEqual(gate_opened, [True])
        self.assertEqual(len(checked_in), 40)
        self.assertTrue(all(name.startswith("breeze-check") for name in checked_in))
        self.assertEqual(len(run.server.calls), 2)  # every take matched: nothing retried

    def test_generation_never_runs_more_than_one_batch_ahead_of_the_checks(self):
        checked = []
        ahead = []

        class Server(FakeBreeze):
            def __call__(self, items, seed=None):
                ahead.append(len(checked))
                return super().__call__(items, seed)

        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                checked.append(1)
                return speech_check.Heard(_unit_text(len(audio) - 1001), [])
        with patch.object(openai_tts_provider, "BREEZE_SHORT_BATCH_SIZE", 32):  # three requests
            run = _Run(self, _text(96), Server(), checker=Checker())
        self.assertIsNone(run.error)
        self.assertEqual(len(ahead), 3)
        self.assertGreaterEqual(ahead[2], 32)  # batch 1 is fully checked before batch 3 is requested

    def test_a_batchs_takes_are_heard_several_at_a_time_and_each_against_its_own_unit(self):
        lock = threading.Lock()
        active, peak = [0], [0]

        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.05)
                with lock:
                    active[0] -= 1
                return speech_check.Heard(_unit_text(len(audio) - 1001), [])
        run = _Run(self, _text(12), checker=Checker())
        self.assertIsNone(run.error)
        self.assertEqual(peak[0], speech_check.WORKERS)
        self.assertEqual(len(run.server.calls), 1)  # a transcript paired with another unit would fail
        self.assertTrue(all(clip["match"] == 1.0 for clip in run.clips()))

    def test_a_failed_speech_check_never_fails_the_chapter(self):
        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                raise RuntimeError("whisper broke")
        run = _Run(self, _text(2), checker=Checker())
        self.assertIsNone(run.error)
        self.assertEqual(len(run.server.calls), 1)


class TestBreezeToneMatch(unittest.TestCase):

    def _top_band(self, samples):
        spectrum, _ = tone_match.speech_spectrum(samples)
        return float(tone_match.balance(spectrum, RATE)[-1])

    def _bright_run(self, **extra):
        voices = tempfile.TemporaryDirectory()
        self.addCleanup(voices.cleanup)
        rng = np.random.default_rng(1)

        def wav(samples):
            return AudioSegment(data=(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes(),
                                frame_rate=RATE, channels=1, sample_width=2)
        dark = np.convolve(rng.normal(0, 0.1, RATE * 8), [0.25, 0.5, 0.25], mode="same")
        wav(dark).export(os.path.join(voices.name, "Dark.wav"), format="wav")

        def behave(item, round):
            return wav(np.random.default_rng(_chunk(item)).normal(0, 0.1, int(RATE * 4.5)))
        with patch.dict(os.environ, {"TTS_VOICES_DIR": voices.name}):
            return _Run(self, _text(4), FakeBreeze(behave), **extra)

    def test_a_brighter_than_its_clip_voice_is_cut(self):
        run = self._bright_run(tone_match=True)
        self.assertIsNone(run.error)
        units = [np.frombuffer(p, dtype="<i2").astype(np.float32) / 32768 for p in run.pieces if len(p) > 20000]
        self.assertEqual(len(units), 4)
        for unit in units:
            self.assertLess(self._top_band(unit), -8)  # white noise measures ~0 dB before matching

    def test_the_book_can_turn_matching_off(self):
        run = self._bright_run(tone_match=False)
        for piece in (p for p in run.pieces if len(p) > 20000):
            self.assertGreater(self._top_band(np.frombuffer(piece, dtype="<i2").astype(np.float32) / 32768), -2)


class TestBreezeClipMap(unittest.TestCase):

    def test_the_clip_map_has_the_chatterbox_keys(self):
        run = _Run(self, _text(3))
        self.assertIsNone(run.error)
        clips = run.clips()
        self.assertEqual(len(clips), 3)
        for number, clip in enumerate(clips, 1):
            self.assertEqual(clip["chunk"], number)
            self.assertEqual(clip["voice"], "Dark.wav")
            self.assertEqual(clip["mood"], "normal")
            self.assertEqual(clip["attempts"], 1)
            self.assertEqual(clip["seed"], run.server.calls[0][1])
            self.assertEqual(clip["text_length"], len(_unit_text(number - 1)))
            self.assertEqual(clip["settings"], {})
        self.assertLess(clips[0]["end_ms"], clips[1]["start_ms"])

    def test_the_speech_checks_match_is_recorded(self):
        class Checker:
            def transcribe(self, audio, words=True, beam_size=5):
                return speech_check.Heard(_unit_text(len(audio) - 1001), [])
        run = _Run(self, _text(2), checker=Checker())
        self.assertEqual([c["match"] for c in run.clips()], [1.0, 1.0])


class Hearing:
    """A fake Whisper: hears each take's own unit, except `garbles` (take numbers) or raises for `fails`."""

    def __init__(self, garbles=(), fails=()):
        self.garbles, self.fails, self.heard = set(garbles), set(fails), []

    def transcribe(self, audio, words=True, beam_size=5):
        number = len(audio) - 1000
        self.heard.append(number)
        if number in self.fails:
            raise RuntimeError("whisper broke")
        return speech_check.Heard("completely different words" if number in self.garbles
                                  else _unit_text(number - 1), [])


class TestBreezeQuickHearing(unittest.TestCase):

    def test_takes_the_quick_hearing_passes_never_reach_whisper_small(self):
        quick, small = Hearing(), Hearing()
        run = _Run(self, _text(5), checker=small, quick=quick)
        self.assertIsNone(run.error)
        self.assertEqual(sorted(quick.heard), [1, 2, 3, 4, 5])
        self.assertEqual(small.heard, [])
        self.assertEqual([c["match"] for c in run.clips()], [1.0] * 5)

    def test_only_doubted_takes_are_heard_again_and_small_decides_them(self):
        quick, small = Hearing(garbles={2}, fails={4}), Hearing()
        with self.assertLogs(PROVIDER, level="INFO") as logs:
            run = _Run(self, _text(5), checker=small, quick=quick)
        self.assertIsNone(run.error)
        self.assertEqual(sorted(small.heard), [2, 4])
        self.assertEqual([c["attempts"] for c in run.clips()], [1] * 5)  # small heard them right
        self.assertTrue(any("checked 5 takes" in line and "(2 heard again by Whisper small)" in line
                            for line in logs.output))

    def test_a_take_both_hearings_fail_is_rejected_and_sent_again(self):
        quick, small = Hearing(garbles={3}), Hearing(garbles={3})
        run = _Run(self, _text(4), checker=small, quick=quick)
        self.assertIsNone(run.error)
        self.assertEqual([_chunk(item) for item in run.server.calls[1][0]], [3])  # only unit 3 again
        self.assertGreater(run.clips()[2]["attempts"], 1)


class TestOtherEnginesUnchanged(unittest.TestCase):

    def test_kokoro_and_chatterbox_still_make_one_request_per_unit_and_never_touch_breeze(self):
        for model in ("kokoro", "chatterbox"):
            with self.subTest(model=model), patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                provider = _provider(model_name=model, output_format="mp3")
                requests = []

                def create(**kwargs):
                    requests.append(kwargs)
                    return SimpleNamespace(content=_tone(1500).export(format="wav").read())
                provider.client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
                with tempfile.TemporaryDirectory() as folder, \
                        patch(f"{PROVIDER}.breeze_client.synthesize_batch") as batch, \
                        patch(f"{PROVIDER}.speech_check.get", return_value=None):
                    provider.text_to_speech(_text(3), os.path.join(folder, "out.mp3"), TAGS)
                self.assertEqual(len(requests), 3)
                self.assertEqual({r["model"] for r in requests}, {model})
                batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
