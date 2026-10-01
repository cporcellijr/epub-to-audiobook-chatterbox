"""Adaptive delivery: presets around a book's baseline, the peak guard, the rule-based mood cues
and reading Chatterbox's saved generation defaults."""
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

import yaml
from pydub import AudioSegment

from audiobook_generator.core import delivery
from audiobook_generator.core.delivery import (
    APPROVED_BASELINE, MOOD_EMPHATIC, MOOD_EXCITED, MOOD_NORMAL, MOOD_SOFT, Baseline, mood_of, peak_guard, preset,
    saved_chatterbox_defaults, segment_moods, segment_moods_and_cues,
)
from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M, chapter_segments


def _tone(peak_amplitude: int, frame_rate: int = 24000, duration_ms: int = 200) -> AudioSegment:
    """A square wave whose peak absolute sample is exactly peak_amplitude, for a known max_dBFS."""
    sample_count = int(frame_rate * duration_ms / 1000)
    samples = [peak_amplitude if i % 2 == 0 else -peak_amplitude for i in range(sample_count)]
    raw = struct.pack("<%dh" % sample_count, *samples)
    return AudioSegment(data=raw, sample_width=2, frame_rate=frame_rate, channels=1)


class TestPreset(unittest.TestCase):

    def test_approved_baseline_reproduces_the_approved_numbers_exactly(self):
        self.assertEqual(preset(MOOD_SOFT, APPROVED_BASELINE), (0.35, 0.35, 0.5, -6.0))
        self.assertEqual(preset(MOOD_NORMAL, APPROVED_BASELINE), (0.73, 0.5, 0.61, 0.0))
        self.assertEqual(preset(MOOD_EMPHATIC, APPROVED_BASELINE), (0.85, 0.46, 0.64, 0.5))
        self.assertEqual(preset(MOOD_EXCITED, APPROVED_BASELINE), (0.91, 0.45, 0.7, 1.5))

    def test_relative_presets_at_another_baseline_including_the_floors(self):
        baseline = Baseline(0.5, 0.6, 0.4)
        # soft exaggeration (0.5*0.48=0.24) and soft temperature (0.4-0.11=0.29) both hit their floor.
        self.assertEqual(preset(MOOD_SOFT, baseline), (0.25, 0.45, 0.3, -6.0))
        self.assertEqual(preset(MOOD_NORMAL, baseline), (0.5, 0.6, 0.4, 0.0))
        self.assertEqual(preset(MOOD_EXCITED, baseline), (0.68, 0.55, 0.49, 1.5))

    def test_excited_exaggeration_and_temperature_ceilings(self):
        baseline = Baseline(0.9, 0.5, 0.85)
        exaggeration, _, temperature, _ = preset(MOOD_EXCITED, baseline)
        self.assertEqual(exaggeration, 1.0)   # 0.9 + 0.18 = 1.08, capped at 1.0
        self.assertEqual(temperature, 0.9)    # 0.85 + 0.09 = 0.94, capped at 0.9

    def test_unrecognised_mood_behaves_like_normal(self):
        self.assertEqual(preset("shouting-into-a-pillow", APPROVED_BASELINE), preset(MOOD_NORMAL, APPROVED_BASELINE))

    def test_very_short_units_use_baseline_and_longer_units_ease_into_full_mood(self):
        short = delivery.unit_preset(MOOD_EXCITED, APPROVED_BASELINE, '"Stop!"')
        normal = delivery.unit_preset(MOOD_NORMAL, APPROVED_BASELINE, '"Stop!"')
        medium = delivery.unit_preset(MOOD_EXCITED, APPROVED_BASELINE, "An urgent reply.")
        full = delivery.unit_preset(MOOD_EXCITED, APPROVED_BASELINE, "A much longer expressive line." * 2)

        self.assertEqual(short, preset(MOOD_NORMAL, APPROVED_BASELINE))
        self.assertEqual(normal, preset(MOOD_NORMAL, APPROVED_BASELINE))
        self.assertGreater(medium[0], APPROVED_BASELINE.exaggeration)
        self.assertLess(medium[0], preset(MOOD_EXCITED, APPROVED_BASELINE)[0])
        self.assertGreater(medium[1], APPROVED_BASELINE.cfg_weight - 0.1)
        self.assertLess(medium[1], APPROVED_BASELINE.cfg_weight)
        self.assertGreater(medium[2], APPROVED_BASELINE.temperature)
        self.assertLess(medium[2], preset(MOOD_EXCITED, APPROVED_BASELINE)[2])
        self.assertGreater(medium[3], 0.0)
        self.assertLess(medium[3], preset(MOOD_EXCITED, APPROVED_BASELINE)[3])
        self.assertEqual(full, preset(MOOD_EXCITED, APPROVED_BASELINE))


class TestPeakGuard(unittest.TestCase):

    def test_a_clip_over_the_guard_is_turned_down_to_exactly_minus_one_dbfs(self):
        loud = _tone(32767)  # 0 dBFS
        guarded = peak_guard(loud)
        self.assertAlmostEqual(guarded.max_dBFS, delivery.PEAK_GUARD_DBFS, delta=0.1)

    def test_a_clip_already_under_the_guard_is_unchanged(self):
        quiet = _tone(16423)  # about -6 dBFS
        self.assertIs(peak_guard(quiet), quiet)

    def test_silence_is_unchanged(self):
        silence = AudioSegment.silent(duration=200)
        self.assertIs(peak_guard(silence), silence)


def _sine(peak_dbfs: float, frame_rate: int = 24000, duration_ms: int = 300) -> AudioSegment:
    """A 220 Hz sine peaking at peak_dbfs: unlike a square wave, clipping visibly changes its shape."""
    import math
    amplitude = 32767 * 10 ** (peak_dbfs / 20)
    count = int(frame_rate * duration_ms / 1000)
    samples = [int(round(amplitude * math.sin(2 * math.pi * 220 * i / frame_rate))) for i in range(count)]
    return AudioSegment(data=struct.pack("<%dh" % count, *samples), sample_width=2, frame_rate=frame_rate, channels=1)


class TestGuardedGain(unittest.TestCase):

    def test_an_excited_gain_on_a_near_full_scale_clip_never_clips(self):
        # Chatterbox's clips peak near -0.4 dBFS: +1.5 dB first and the guard after flattened the peaks.
        clip = _sine(-0.4)
        out = delivery.guarded_gain(clip, 1.5)
        self.assertAlmostEqual(out.max_dBFS, delivery.PEAK_GUARD_DBFS, delta=0.1)
        before, after = clip.get_array_of_samples(), out.get_array_of_samples()
        factor = max(map(abs, after)) / max(map(abs, before))
        self.assertLessEqual(max(abs(b * factor - a) for a, b in zip(after, before)), 2)  # just scaled, same shape

    def test_the_full_gain_applies_when_there_is_headroom_and_silence_is_left_alone(self):
        self.assertAlmostEqual(delivery.guarded_gain(_sine(-10.0), 1.5).max_dBFS, -8.5, delta=0.1)
        self.assertAlmostEqual(delivery.guarded_gain(_sine(-3.0), -6.0).max_dBFS, -9.0, delta=0.1)
        silence = AudioSegment.silent(duration=200)
        self.assertIs(delivery.guarded_gain(silence, 1.5), silence)


class TestMoodOf(unittest.TestCase):

    def test_no_cues_is_normal(self):
        self.assertEqual(mood_of("", '"Fine."', "she said."), MOOD_NORMAL)

    def test_soft_verb_after(self):
        self.assertEqual(mood_of("", '"Go now."', "she whispered."), MOOD_SOFT)

    def test_soft_verb_before_only_counts_as_a_lead_in_ending_in_comma_or_colon(self):
        self.assertEqual(mood_of("She whispered,", '"Go now."', ""), MOOD_SOFT)
        self.assertEqual(mood_of("She said, and then whispered:", '"Go now."', ""), MOOD_SOFT)
        # No trailing ',' or ':': not a lead-in, so the cue is not counted.
        self.assertEqual(mood_of("She whispered softly", '"Go now."', ""), MOOD_NORMAL)

    def test_soft_adverb_phrases(self):
        for after in ("she said softly.", "she said quietly.", "she said gently.",
                      "she said under his breath.", "she said under her breath.", "she said under their breath.",
                      "she said in a whisper.", "she said in a low voice."):
            self.assertEqual(mood_of("", '"Go now."', after), MOOD_SOFT, after)

    def test_excited_verbs(self):
        for after in ("she shouted.", "she yelled.", "she screamed.", "she shrieked.", "she roared.",
                      "she bellowed.", "she cried.", "she cried out.", "she exclaimed."):
            self.assertEqual(mood_of("", '"Get out."', after), MOOD_EXCITED, after)

    def test_excited_adverbs(self):
        for after in ("he said loudly.", "he said angrily.", "he said furiously."):
            self.assertEqual(mood_of("", '"Get out."', after), MOOD_EXCITED, after)

    def test_exclamation_mark_alone_is_mildly_emphatic(self):
        self.assertEqual(mood_of("", '"Get out!"', ""), MOOD_EMPHATIC)
        self.assertEqual(mood_of("", "“Get out!”", ""), MOOD_EMPHATIC)  # curly quotes
        self.assertEqual(mood_of("", '"Get out!"', "she shouted."), MOOD_EXCITED)

    def test_soft_wins_over_an_exclamation_mark(self):
        self.assertEqual(mood_of("", '"Get out!"', "she whispered."), MOOD_SOFT)

    def test_soft_wins_over_an_excited_cue(self):
        self.assertEqual(mood_of("", '"Get out."', "she whispered, though he had shouted a moment before."),
                         MOOD_SOFT)

    def test_case_insensitive(self):
        self.assertEqual(mood_of("", '"Go now."', "She WHISPERED."), MOOD_SOFT)
        self.assertEqual(mood_of("", '"Go now."', "She SHOUTED."), MOOD_EXCITED)


class TestCuesComeFromTheTag(unittest.TestCase):

    def test_manner_words_inside_the_spoken_line_are_not_a_cue(self):
        self.assertEqual(mood_of("", '"I whispered it to him yesterday."', "she said."), MOOD_NORMAL)
        self.assertEqual(mood_of("", '"He shouted at me."', "she said."), MOOD_NORMAL)

    def test_a_closing_exclamation_in_the_line_still_counts(self):
        self.assertEqual(mood_of("", '"He shouted at me!"', "she said."), MOOD_EMPHATIC)


class TestSegmentMoods(unittest.TestCase):

    def test_ordinary_lines_use_mood_of(self):
        text = f'"Fine," she said.{M}"Get out!" he shouted.{M}"Shh," she whispered.'
        moods = segment_moods(chapter_segments(text))
        self.assertEqual(moods, {1: MOOD_NORMAL, 2: MOOD_EXCITED, 3: MOOD_SOFT})

    def test_narration_never_appears(self):
        moods = segment_moods(chapter_segments(f"Nothing happens here.{M}Nor here."))
        self.assertEqual(moods, {})

    def test_a_continued_line_inherits_the_speechs_mood(self):
        text = f'She whispered, "First part.{M}"Second part," she said.'
        self.assertEqual(segment_moods(chapter_segments(text)), {1: MOOD_SOFT, 2: MOOD_SOFT})

    def test_an_exclamation_alone_never_calms_or_raises_a_continued_speech(self):
        shout = f'He shouted, "Run to the gate.{M}"Don\'t stop!'
        self.assertEqual(segment_moods(chapter_segments(shout)), {1: MOOD_EXCITED, 2: MOOD_EXCITED})
        whisper = f'She whispered, "Stay low.{M}"Not a sound!'
        self.assertEqual(segment_moods(chapter_segments(whisper)), {1: MOOD_SOFT, 2: MOOD_SOFT})
        plain = f'"Run to the gate.{M}"Don\'t stop!'
        self.assertEqual(segment_moods(chapter_segments(plain)), {1: MOOD_NORMAL, 2: MOOD_EMPHATIC})

    def test_a_clear_new_cue_in_a_continued_paragraph_overrides_the_inherited_mood(self):
        text = f'She whispered, "First part.{M}"Second part," he shouted.'
        self.assertEqual(segment_moods(chapter_segments(text)), {1: MOOD_SOFT, 2: MOOD_EXCITED})


class TestParagraphMood(unittest.TestCase):
    """A speech tag's cue sets the manner for the rest of its paragraph."""

    def moods(self, text):
        return delivery.segment_moods(chapter_segments(text))

    def test_a_whisper_carries_to_the_untagged_quotation_after_it(self):
        text = '\u201cTom, are you awake?\u201d she whispered. \u201cDon\u2019t wake Mother.\u201d'
        self.assertEqual(self.moods(text), {1: "soft", 2: "soft"})

    def test_a_shout_carries_to_the_untagged_quotation_before_it(self):
        text = '\u201cNo.\u201d He stood up. \u201cGet out,\u201d he shouted.'
        self.assertEqual(self.moods(text), {1: "excited", 2: "excited"})

    def test_a_quotation_with_its_own_tag_keeps_its_own_mood(self):
        text = '\u201cHush,\u201d she whispered. \u201cWhy?\u201d he asked.'
        self.assertEqual(self.moods(text), {1: "soft", 2: "normal"})

    def test_a_shout_outranks_an_untagged_exclamation_in_its_paragraph(self):
        text = '“Go!” He stood up. “Get out,” he shouted.'
        self.assertEqual(self.moods(text), {1: "excited", 2: "excited"})

    def test_a_whisper_does_not_soften_an_untagged_exclamation(self):
        text = '“Go!” He stood up. “Get out,” she whispered.'
        self.assertEqual(self.moods(text), {1: "emphatic", 2: "soft"})

    def test_conflicting_cues_lend_nothing(self):
        text = '\u201cHush,\u201d she whispered. \u201cNo,\u201d he shouted. \u201cWell.\u201d'
        self.assertEqual(self.moods(text), {1: "soft", 2: "excited", 3: "normal"})

    def test_a_cue_does_not_cross_into_the_next_paragraph(self):
        text = f'\u201cHush,\u201d she whispered.{M}\u201cWhat?\u201d'
        self.assertEqual(self.moods(text), {1: "soft", 2: "normal"})


class TestSegmentCues(unittest.TestCase):
    """The speech tag's cue word travels with a line's mood, for Breeze's per-verb directions."""

    def _cues(self, text):
        return segment_moods_and_cues(chapter_segments(text))[1]

    def test_the_tags_verb_is_found_after_or_before_the_line(self):
        self.assertEqual(self._cues('"Shh," she Hissed.'), {1: "hissed"})
        self.assertEqual(self._cues('He roared, "Get out!"'), {1: "roared"})
        self.assertEqual(self._cues('"Go," she said in a whisper.'), {1: "in a whisper"})

    def test_no_cue_word_for_normal_or_punctuation_only_lines(self):
        self.assertEqual(self._cues(f'"Fine," she said.{M}"Get out!" she said.'), {})

    def test_moods_are_the_same_as_segment_moods(self):
        text = f'"Tom?" she whispered. "Quiet."{M}"Go!" he shouted.{M}"Now!" she said.{M}He hissed, "Stay,{M}"down!"'
        self.assertEqual(segment_moods_and_cues(chapter_segments(text))[0], segment_moods(chapter_segments(text)))

    def test_a_borrowed_mood_borrows_the_cue_word(self):
        self.assertEqual(self._cues('"Tom?" she muttered. "Do not wake Mother."'), {1: "muttered", 2: "muttered"})

    def test_a_continued_line_inherits_the_cue_word_or_takes_its_own(self):
        self.assertEqual(self._cues(f'He hissed, "Stay down, I mean it,{M}"and stay quiet."'),
                         {1: "hissed", 2: "hissed"})
        self.assertEqual(self._cues(f'He hissed, "Stay down, I mean it,{M}"and run!" he shouted.'),
                         {1: "hissed", 2: "shouted"})


class TestBreezeInstruction(unittest.TestCase):

    def test_normal_and_unknown_moods_are_not_directed(self):
        self.assertIsNone(delivery.breeze_instruction(MOOD_NORMAL, "Hello there, friend."))
        self.assertIsNone(delivery.breeze_instruction(MOOD_NORMAL, "Hello there, friend.", "whispered"))
        self.assertIsNone(delivery.breeze_instruction("sleepy", "Hello there, friend."))

    def test_only_soft_speech_is_directed(self):
        """The owner's ear (2026-10-01): loud directions distorted the voice; quiet ones helped."""
        self.assertEqual(delivery.breeze_instruction(MOOD_SOFT, "Hello."),
                         "Say this softly and quietly, close to a whisper.")
        for mood in (MOOD_EXCITED, MOOD_EMPHATIC):
            for cue in (None, "screamed", "roared", "shouted", "yelled", "whispered"):
                self.assertIsNone(delivery.breeze_instruction(mood, "Hello!", cue), (mood, cue))

    def test_a_known_soft_verb_gets_its_own_direction_and_muttering_stays_plain(self):
        soft = {"whispered": "Whisper this softly.", "HISSED": "Hiss this through clenched teeth, quietly.",
                "in a whisper": "Whisper this softly.", "murmured": "Murmur this softly and quietly.",
                "under his breath": "Say this under your breath, very quietly."}
        for cue, expected in soft.items():
            self.assertEqual(delivery.breeze_instruction(MOOD_SOFT, "Hello.", cue), expected, cue)
        for cue in ("muttered", "mumbled"):
            self.assertIsNone(delivery.breeze_instruction(MOOD_SOFT, "Hello.", cue), cue)

    def test_an_unlisted_cue_falls_back_to_the_soft_default(self):
        self.assertEqual(delivery.breeze_instruction(MOOD_SOFT, "Hello.", "softly"),
                         delivery.breeze_instruction(MOOD_SOFT, "Hello."))
        self.assertEqual(delivery.breeze_instruction(MOOD_SOFT, "Hello.", "shouted"),
                         delivery.breeze_instruction(MOOD_SOFT, "Hello."))

    def test_every_direction_is_a_short_sentence(self):
        cues = [None, *(stem for rows in delivery._BREEZE_CUE_DIRECTIONS.values() for stem, _ in rows)]
        for cue in cues:
            text = delivery.breeze_instruction(MOOD_SOFT, "Hello.", cue)
            self.assertTrue(text is None or (text.endswith(".") and len(text) < 80), text)

    def test_a_cued_mood_is_still_its_mood_string(self):
        mood = delivery.CuedMood(MOOD_SOFT, "hissed")
        self.assertEqual((mood, mood.cue, {mood: 1}[MOOD_SOFT]), (MOOD_SOFT, "hissed", 1))
        self.assertEqual(delivery.breeze_instruction(mood, "Hello.", mood.cue),
                         "Hiss this through clenched teeth, quietly.")


class TestSavedChatterboxDefaults(unittest.TestCase):

    def test_no_config_path_returns_the_approved_baseline(self):
        with patch.dict(os.environ, {"CHATTERBOX_CONFIG": ""}):
            self.assertEqual(saved_chatterbox_defaults(), APPROVED_BASELINE)

    def test_reads_generation_defaults_from_the_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"generation_defaults": {"exaggeration": 0.6, "cfg_weight": 0.45, "temperature": 0.7}}, f)
            with patch.dict(os.environ, {"CHATTERBOX_CONFIG": path}):
                self.assertEqual(saved_chatterbox_defaults(), Baseline(0.6, 0.45, 0.7))

    def test_a_field_missing_from_the_file_falls_back_to_the_approved_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"generation_defaults": {"exaggeration": 0.9}}, f)
            with patch.dict(os.environ, {"CHATTERBOX_CONFIG": path}):
                self.assertEqual(saved_chatterbox_defaults(),
                                 Baseline(0.9, APPROVED_BASELINE.cfg_weight, APPROVED_BASELINE.temperature))

    def test_missing_file_retries_once_then_falls_back(self):
        sleeps = []
        with patch.dict(os.environ, {"CHATTERBOX_CONFIG": "/nowhere/config.yaml"}):
            result = saved_chatterbox_defaults(sleep=sleeps.append)
        self.assertEqual(result, APPROVED_BASELINE)
        self.assertEqual(len(sleeps), 1)


if __name__ == "__main__":
    unittest.main()
