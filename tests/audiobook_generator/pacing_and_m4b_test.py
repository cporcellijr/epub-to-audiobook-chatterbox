import io
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydub import AudioSegment
from pydub.generators import Sine
from mutagen.flac import FLAC
from mutagen.oggopus import OggOpus
from mutagen.wave import WAVE

from audiobook_generator.book_parsers.epub_book_parser import EpubBookParser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audiobook_generator import AudiobookGenerator, CHAPTER_WORK_FOLDER
from audiobook_generator.core.m4b import BadChapterFileError, build_m4b, safe_book_file_name
from audiobook_generator.core import speech_check
from audiobook_generator.tts_providers.openai_tts_provider import (
    PARAGRAPH_MARK,
    OpenAITTSProvider,
    paced_units,
    paragraph_mode_units,
    _chatterbox_input,
    _SHORT_QUOTE_CONTEXT,
    _tiny_quote,
    _quote_context_cut,
    _carrier_cut,
    _silence_runs,
    _stretch_sentence_gaps,
)

UNIT_MS = 1000


def _wav_bytes(ms: int = UNIT_MS) -> bytes:
    buffer = io.BytesIO()
    Sine(300).to_audio_segment(duration=ms).set_frame_rate(24000).set_channels(1).export(buffer, format="wav")
    return buffer.getvalue()


def _tone(ms: int) -> AudioSegment:
    return Sine(300).to_audio_segment(duration=ms).set_frame_rate(24000).set_channels(1)


def _silence(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms, frame_rate=24000).set_channels(1)


def _wav_from_segment(segment: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def _mp3(path: str, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=300:duration={seconds}", "-ar", "24000", "-ac", "1", path], check=True)


def _aac(path: str, seconds: float) -> None:
    """A standalone ADTS AAC chapter file, the format M4B chapters use since F-06."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=300:duration={seconds}", "-ar", "24000", "-ac", "1",
                    "-c:a", "aac", "-f", "adts", path], check=True)


def _cover(path: str) -> None:
    """A tiny cover image; ffmpeg picks the encoder from the file extension."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=blue:s=64x64", "-frames:v", "1", path], check=True)


def _ffprobe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-show_streams", "-show_format",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


class TestParagraphDetection(unittest.TestCase):

    def test_paragraphs_found_from_html_blocks_even_with_single_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
                z.writestr("META-INF/container.xml",
                           '<?xml version="1.0"?><container version="1.0" '
                           'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                           '<rootfile full-path="c.opf" media-type="application/oebps-package+xml"/>'
                           '</rootfiles></container>')
                z.writestr("c.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
                                    'version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                                    '<dc:title>T</dc:title></metadata><manifest>'
                                    '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
                                    '</manifest><spine><itemref idref="c1"/></spine></package>')
                z.writestr("c1.xhtml", '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>\n'
                                       '<p>First paragraph.</p>\n<p>Second paragraph.</p>\n<p>Third.</p>\n'
                                       '</body></html>')
            config = GeneralConfig(SimpleNamespace(input_file=path, title_mode="first_few", newline_mode="double",
                                                   remove_endnotes=False, remove_reference_numbers=False,
                                                   search_and_replace_file=None))
            marked = EpubBookParser(config).get_chapters(f" {PARAGRAPH_MARK}")[0][1]
            plain = EpubBookParser(config).get_chapters(" ")[0][1]
        self.assertEqual([p.strip() for p in marked.split(PARAGRAPH_MARK) if p.strip()],
                         ["First paragraph.", "Second paragraph.", "Third."])
        self.assertEqual(plain, "First paragraph. Second paragraph. Third.")


class TestPacedUnits(unittest.TestCase):

    def test_sentences_become_units_and_short_ones_join_the_next(self):
        text = (f"The rain had stopped by the time they reached the bridge. Yes. She pulled her coat "
                f"tighter and looked back at the village.{PARAGRAPH_MARK}No. The road home was quiet and dark.")
        self.assertEqual(paced_units(text, "en"), [
            (0, "The rain had stopped by the time they reached the bridge.", False),
            (0, "Yes. She pulled her coat tighter and looked back at the village.", False),
            (1, "No. The road home was quiet and dark.", False),
        ])

    def test_trailing_short_sentence_joins_previous_unit(self):
        units = paced_units("She pulled her coat tighter against the cold wind. Then she ran.", "en")
        self.assertEqual(units, [(0, "She pulled her coat tighter against the cold wind. Then she ran.", False)])

    def test_long_comma_only_sentence_is_split_so_no_unit_reaches_the_token_cap(self):
        # sentencex will not split this: no sentence-ending punctuation until the very end.
        clause = "the quiet valley held its breath under a pale and heavy sky"
        text = ", ".join([clause] * 14) + "."
        self.assertGreater(len(text), 800)  # matches the WORKLOG's observed truncation range

        units = paced_units(text, "en")

        self.assertGreater(len(units), 1, "an oversized unit must be split into several requests")
        for _, unit, _ in units:
            self.assertLessEqual(len(unit), 450)
        # Reassembling the pieces (space-joined, as split_long_sentence cuts on spaces/commas)
        # must not lose any non-whitespace content.
        rejoined = "".join(unit for _, unit, _ in units)
        self.assertEqual("".join(rejoined.split()), "".join(text.split()))
        # All pieces after the first are mid-sentence continuations: 0 ms gap, never a pause.
        self.assertFalse(units[0][2])
        self.assertTrue(all(continues for _, _, continues in units[1:]))

    def test_scene_break_and_symbol_only_paragraphs_are_dropped(self):
        text = (f"Yes.{PARAGRAPH_MARK}* * *{PARAGRAPH_MARK}…{PARAGRAPH_MARK}"
                f"—{PARAGRAPH_MARK}No, she said.")
        units = paced_units(text, "en")
        spoken = [unit for _, unit, _ in units]
        self.assertEqual(spoken, ["Yes.", "No, she said."])


class TestParagraphModeUnits(unittest.TestCase):
    """F-05: paragraph_mode_units packs a whole paragraph into one request when it fits."""

    def test_whole_paragraph_becomes_one_unit_with_a_sentence_count(self):
        text = (f"One sentence here. Two sentences now. Three now.{PARAGRAPH_MARK}"
                f"A second paragraph line.")
        self.assertEqual(paragraph_mode_units(text, "en"), [
            (0, "One sentence here. Two sentences now. Three now.", 3, False),
            (1, "A second paragraph line.", 1, False),
        ])

    def test_long_paragraph_is_split_at_sentence_boundaries_not_mid_sentence(self):
        sentence = "This is one sentence of a certain moderate length that repeats."
        text = " ".join([sentence] * 8)
        self.assertGreater(len(text), 450)

        units = paragraph_mode_units(text, "en")

        self.assertGreater(len(units), 1)
        for _, unit, _, _ in units:
            self.assertLessEqual(len(unit), 450)
            self.assertTrue(unit.endswith("repeats."), "must cut between sentences, not inside one")

    def test_oversized_single_sentence_is_split_like_sentence_mode(self):
        # Same F-07 shape as paced_units: sentencex won't split this, so it must still be
        # broken up so no single request nears the server's token cap.
        clause = "the quiet valley held its breath under a pale and heavy sky"
        text = ", ".join([clause] * 14) + "."
        self.assertGreater(len(text), 800)

        units = paragraph_mode_units(text, "en")

        self.assertGreater(len(units), 1)
        for _, unit, sentence_count, _ in units:
            self.assertLessEqual(len(unit), 450)
            self.assertEqual(sentence_count, 1)  # no internal gap to detect in a mid-sentence cut
        self.assertFalse(units[0][3])
        self.assertTrue(all(continues for _, _, _, continues in units[1:]))

    def test_scene_break_paragraphs_are_dropped(self):
        text = f"Real line here.{PARAGRAPH_MARK}* * *{PARAGRAPH_MARK}Another real line."
        units = paragraph_mode_units(text, "en")
        spoken = [unit for _, unit, _, _ in units]
        self.assertEqual(spoken, ["Real line here.", "Another real line."])


class TestGapDetection(unittest.TestCase):
    """F-05: find the model's own inter-sentence gaps in one multi-sentence response and
    stretch them to the configured sentence pause."""

    def _three_sentence_clip(self) -> AudioSegment:
        # 3 "sentences" of tone separated by two uneven, short "natural" gaps (300 ms, 80 ms).
        return _tone(500) + _silence(300) + _tone(500) + _silence(80) + _tone(500)

    def test_silence_runs_finds_both_internal_gaps(self):
        runs = _silence_runs(self._three_sentence_clip())
        self.assertEqual(len(runs), 2)
        (s1, e1), (s2, e2) = runs
        self.assertAlmostEqual(e1 - s1, 300, delta=20)
        self.assertAlmostEqual(e2 - s2, 80, delta=20)

    def test_leading_and_trailing_silence_is_not_a_gap(self):
        clip = _silence(200) + self._three_sentence_clip() + _silence(200)
        runs = _silence_runs(clip)
        self.assertEqual(len(runs), 2, "the added leading/trailing silence must not be picked")

    def test_stretch_sets_the_chosen_gaps_to_exactly_the_target_length(self):
        stretched = _stretch_sentence_gaps(self._three_sentence_clip(), gap_count=2, target_ms=200)
        self.assertAlmostEqual(len(stretched), 500 * 3 + 200 * 2, delta=30)

    def test_stretch_keeps_the_quiet_end_of_the_word_before_the_gap(self):
        # A word ending that trails off (150 ms at about -30 dBFS: under the gap threshold, but
        # audible) then 200 ms of true silence. Only the silence is resized; the tail stays whole.
        tail = _tone(150).apply_gain(-30 - _tone(150).dBFS)
        clip = _tone(500) + tail + _silence(200) + _tone(500)
        stretched = _stretch_sentence_gaps(clip, gap_count=1, target_ms=350)
        self.assertAlmostEqual(len(stretched), 500 + 150 + 350 + 500, delta=20)
        self.assertAlmostEqual(stretched[510:640].dBFS, tail[10:140].dBFS, delta=1.0)

    def test_sound_between_two_silent_patches_is_kept(self):
        # A word's tail, a brief silence, a consonant release, then the real 200 ms gap: only the
        # longest unbroken silence is resized, so the release survives.
        quiet = lambda ms: _tone(ms).apply_gain(-30 - _tone(ms).dBFS)
        clip = _tone(500) + quiet(100) + _silence(30) + quiet(40) + _silence(200) + _tone(500)
        stretched = _stretch_sentence_gaps(clip, gap_count=1, target_ms=350)
        self.assertAlmostEqual(len(stretched), 500 + 100 + 30 + 40 + 350 + 500, delta=20)
        self.assertGreater(stretched[635:665].dBFS, -40)  # the release is still there

    def test_a_gap_with_no_true_silence_gets_the_pause_inserted_and_loses_nothing(self):
        murmur = _tone(200).apply_gain(-30 - _tone(200).dBFS)  # quiet but never silent
        clip = _tone(500) + murmur + _tone(500)
        stretched = _stretch_sentence_gaps(clip, gap_count=1, target_ms=300)
        self.assertAlmostEqual(len(stretched), 500 + 200 + 300 + 500, delta=20)

    def test_stretch_picks_the_longest_gaps_first_and_leaves_the_rest(self):
        # Only 1 gap requested (as for a 2-sentence unit): the longer 300 ms gap must be the one
        # replaced; the shorter 80 ms gap is left alone.
        stretched = _stretch_sentence_gaps(self._three_sentence_clip(), gap_count=1, target_ms=200)
        self.assertAlmostEqual(len(stretched), 500 * 3 + 200 + 80, delta=30)

    def test_zero_gap_count_or_target_is_a_no_op(self):
        clip = self._three_sentence_clip()
        self.assertEqual(len(_stretch_sentence_gaps(clip, 0, 200)), len(clip))
        self.assertEqual(len(_stretch_sentence_gaps(clip, 2, 0)), len(clip))


class TestPacedSpeech(unittest.TestCase):

    def _provider(self, sentence_ms, paragraph_ms, speed=1.0):
        config = GeneralConfig(SimpleNamespace(
            tts="openai", model_name="chatterbox", voice_name="Elena.wav", output_format="mp3", speed=speed,
            instructions=None, language="en", sentence_pause_ms=sentence_ms, paragraph_pause_ms=paragraph_ms))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            provider = OpenAITTSProvider(config)
        provider.client = MagicMock()
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=_wav_bytes())
        return provider

    def _speak(self, provider, text):
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp3")
            with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
                provider.text_to_speech(text, path, tags)
            return len(AudioSegment.from_file(path))

    TEXT = (f"The rain had stopped by the time they reached the bridge. She pulled her coat tighter "
            f"and looked back at the village.{PARAGRAPH_MARK}The road home was quiet, dark and very long.")

    def test_pauses_inserted_between_sentences_and_paragraphs(self):
        provider = self._provider(400, 1000)
        duration = self._speak(provider, self.TEXT)
        self.assertAlmostEqual(duration, 3 * UNIT_MS + 400 + 1000, delta=80)
        requests = [call.kwargs for call in provider.client.audio.speech.create.call_args_list]
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(r["response_format"] == "wav" and PARAGRAPH_MARK not in r["input"] for r in requests))

    def test_pauses_shrink_with_speed(self):
        # F-27: speed is applied ONCE client-side over the whole finished chapter (speech and
        # unscaled pauses together), not per unit server-side, so the entire unscaled timeline
        # (3 units + full 400 ms + full 1000 ms pauses) is what ends up divided by speed.
        duration = self._speak(self._provider(400, 1000, speed=2.0), self.TEXT)
        self.assertAlmostEqual(duration, (3 * UNIT_MS + 400 + 1000) / 2.0, delta=150)

    def test_speed_is_requested_once_client_side_not_per_unit(self):
        # F-27: every unit is requested at speed 1.0 regardless of the configured speed; the
        # server would otherwise run its own per-unit ffmpeg atempo ~330 times per chapter.
        provider = self._provider(400, 1000, speed=2.0)
        self._speak(provider, self.TEXT)
        requests = [call.kwargs for call in provider.client.audio.speech.create.call_args_list]
        self.assertTrue(requests, "expected at least one request")
        self.assertTrue(all(r["speed"] == 1.0 for r in requests))

    def test_retries_a_speech_clip_with_a_long_silent_gap(self):
        provider = self._provider(400, 1000)
        bad = _wav_from_segment(_tone(200) + _silence(4000) + _tone(200))
        provider.client.audio.speech.create.side_effect = [SimpleNamespace(content=bad),
                                                           SimpleNamespace(content=_wav_bytes())]
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow", return_value=123):
            self._speak(provider, "The lantern burned steadily beside the window.")
        self.assertEqual(provider.client.audio.speech.create.call_count, 2)
        first, second = [call.kwargs for call in provider.client.audio.speech.create.call_args_list]
        self.assertNotIn("seed", first.get("extra_body", {}))
        self.assertEqual(second["extra_body"]["seed"], 124)

    def test_repeated_silent_clips_fail_instead_of_entering_the_book(self):
        provider = self._provider(400, 1000)
        bad = _wav_from_segment(_tone(200) + _silence(4000) + _tone(200))
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=bad)
        with self.assertRaisesRegex(RuntimeError, "repeated near-silent audio"):
            self._speak(provider, "The lantern burned steadily beside the window.")
        self.assertEqual(provider.client.audio.speech.create.call_count, 3)

    def test_short_quote_context_preserves_voice_text_map_and_quiet_word_onset(self):
        word = _tone(20).apply_gain(-42) + _tone(480)
        contextual_audio = _tone(1800) + _silence(200) + word
        expected = contextual_audio[1900:]
        for mode in ("sentence", "paragraph"):
            with self.subTest(mode=mode):
                provider = self._provider(100, 300)
                provider.config.paced_unit_mode = mode
                provider.config.voice_mode = "dialogue"
                provider.config.dialogue_voice = "Gianna.wav"
                provider.config.output_format = "wav"
                provider.client.audio.speech.create.side_effect = lambda **kw: SimpleNamespace(
                    content=_wav_from_segment(contextual_audio if kw['input'].startswith(
                        _SHORT_QUOTE_CONTEXT) else _tone(400)))
                tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "out.wav")
                    with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                               return_value=123):
                        provider.text_to_speech('“Oh!” she said.', path, tags)
                    output = AudioSegment.from_file(path)
                    with open(path + ".clips.json", encoding="utf8") as f:
                        clips = json.load(f)["clips"]
                self.assertEqual(output[:len(expected)].raw_data, expected.raw_data)
                self.assertEqual(len(output), len(expected) + 100 + 400)
                request = provider.client.audio.speech.create.call_args_list[0].kwargs
                self.assertEqual(request['input'], _SHORT_QUOTE_CONTEXT + ' “Oh!”')
                self.assertEqual(request['voice'], 'Gianna.wav')
                self.assertEqual(request['extra_body'], {'seed': 124})
                self.assertEqual(clips[0]['text_sha1'], hashlib.sha1('“Oh!”'.encode()).hexdigest())
                self.assertEqual((clips[0]['text_length'], clips[0]['context_trim_ms']), (5, 1900))
                self.assertEqual((clips[0]['end_ms'], clips[0]['attempts']), (600, 1))
                self.assertNotIn('context_trim_ms', clips[1])

    def test_uncertain_context_boundaries_retry_then_speak_without_the_lead_in(self):
        provider = self._provider(100, 300)
        bad = _tone(2400)
        ambiguous = _tone(1800) + _silence(100) + _tone(200) + _silence(100) + _tone(400)
        self.assertIsNone(_quote_context_cut(ambiguous))
        self.assertIsNone(_quote_context_cut(_tone(1800) + _silence(40) + _tone(400)))
        good = _tone(1800) + _silence(200) + _tone(500)
        provider.client.audio.speech.create.side_effect = [SimpleNamespace(content=_wav_from_segment(a))
                                                          for a in (bad, ambiguous, good)]
        kwargs = dict(model='chatterbox', voice='Gianna.wav', input='“Oh!”', response_format='wav',
                      extra_body={'seed': 10, 'exaggeration': .78, 'cfg_weight': .45, 'temperature': .6})
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                   side_effect=[20, 30]):
            audio, params, attempts, flagged = provider._speak_take(kwargs, '“Oh!”', 'test')
        self.assertEqual(audio.raw_data, good[1900:].raw_data)
        self.assertEqual((params['seed'], attempts, flagged), (31, 3, None))
        self.assertEqual(kwargs['input'], '“Oh!”')
        # No attempt separates: one more request without the lead-in, kept and flagged, so one
        # short unit can't fail the chapter and the lead-in never reaches the book.
        provider.client.audio.speech.create.side_effect = None
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=_wav_from_segment(bad))
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                   side_effect=[40, 50, 60]):
            audio, params, attempts, flagged = provider._speak_take(dict(kwargs), '“Oh!”', 'test')
        last = provider.client.audio.speech.create.call_args.kwargs
        self.assertEqual((last['input'], last['extra_body']['seed']), ('“Oh!”', 61))
        self.assertEqual((audio.raw_data, attempts, flagged), (bad.raw_data, 4, 'unseparated context'))
        self.assertNotIn('_context_trim_ms', params)
        silent = _tone(200) + _silence(4000) + _tone(200)
        provider.client.audio.speech.create.side_effect = [SimpleNamespace(content=_wav_from_segment(a))
                                                          for a in (bad, bad, bad, silent, silent, silent)]
        with patch.object(provider, '_combine_and_export') as export:
            with self.assertRaisesRegex(RuntimeError, 'repeated unseparated short-quote context'):
                self._speak(provider, '“Oh!”')
            export.assert_not_called()

    def test_context_scope_excludes_prose_phrases_ellipsis_and_other_engines_or_languages(self):
        for text in ('“Oh!”', '“Nothing!”', '“Oh,”', '"Don\'t!"'):
            self.assertTrue(_tiny_quote(text), text)
        for text in ('Oh!', '“Of course!”', '“Well…”', '“Wait—”', '“Oh!“', '„Ja“', '“...”',
                     '“Extraordinarilylongquotation!”'):
            self.assertFalse(_tiny_quote(text), text)
        for model, language in (('kokoro', 'en'), ('chatterbox', 'es')):
            provider = self._provider(100, 300)
            provider.config.model_name, provider.config.language = model, language
            audio, params, _, _ = provider._speak_take(
                dict(input='“Oh!”', model=model, voice='Elena.wav', response_format='wav'), '“Oh!”', 'test')
            self.assertEqual(provider.client.audio.speech.create.call_args.kwargs['input'], '“Oh!”')
            self.assertNotIn('_context_trim_ms', params)
            self.assertEqual(len(audio), UNIT_MS)

    def _checked_provider(self, *cut_transcripts):
        """A provider whose speech check hears the lead-in end at 1790 ms and the unit start at 2010 ms
        in a whole take, and the given transcripts, in turn, in each cut take."""
        provider = self._provider(100, 300)
        provider.config.output_format = "wav"
        cut_heard = list(cut_transcripts)
        checker = MagicMock()

        def transcribe(audio):
            if len(audio) >= 2000:
                return speech_check.Heard("", self.HEARD_TWO_WORDS)
            item = cut_heard.pop(0)
            if isinstance(item, Exception):
                raise item
            return speech_check.Heard(item, [])
        checker.transcribe.side_effect = transcribe
        patcher = patch("audiobook_generator.tts_providers.openai_tts_provider.speech_check.get",
                        return_value=checker)
        patcher.start()
        self.addCleanup(patcher.stop)
        return provider, checker

    # The lead-in, then a two-word quote with its own pause: the only silence in the take's latter
    # half is the quote's, so only Whisper's word times find the right gap.
    TWO_WORDS = _tone(1800) + _silence(200) + _tone(300) + _silence(150) + _tone(400)
    HEARD_TWO_WORDS = ([(w, i * 190, i * 190 + 170) for i, w in enumerate(_SHORT_QUOTE_CONTEXT.split()[:-1])]
                       + [("open.", 1520, 1790), ("Oh,", 2010, 2290), ("God!", 2460, 2850)])

    def test_speech_check_cuts_a_short_phrase_after_the_lead_ins_last_word(self):
        self.assertIsNone(_quote_context_cut(self.TWO_WORDS))
        self.assertEqual(_carrier_cut(self.TWO_WORDS, self.HEARD_TWO_WORDS), 1900)
        self.assertIsNone(_carrier_cut(self.TWO_WORDS, self.HEARD_TWO_WORDS[-3:]))  # lead-in not heard
        provider, checker = self._checked_provider("Oh, God!")
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(self.TWO_WORDS))
        kwargs = dict(model='chatterbox', voice='Gianna.wav', input='“Oh god!”', response_format='wav',
                      extra_body={'seed': 10})
        audio, params, attempts, flagged = provider._speak_take(kwargs, '“Oh god!”', 'test')
        self.assertEqual(provider.client.audio.speech.create.call_args.kwargs['input'],
                         _SHORT_QUOTE_CONTEXT + ' “Oh god!”')
        self.assertEqual(audio.raw_data, self.TWO_WORDS[1900:].raw_data)
        self.assertEqual((params['_context_trim_ms'], params['_match'], attempts, flagged), (1900, 1.0, 1, None))
        self.assertEqual(checker.transcribe.call_count, 2)  # the whole take, then the cut

    def test_a_pause_before_the_lead_ins_last_word_is_never_the_cut(self):
        # "...was" pause "open." gap "Um": Whisper ends "open." early (1450 ms; it runs 100-170 ms
        # early), and nothing is heard after it. Cutting in the first pause would keep "open.".
        audio = _tone(1300) + _silence(100) + _tone(400) + _silence(200) + _tone(500)
        words = self.HEARD_TWO_WORDS[:-3] + [("open.", 1200, 1450)]
        self.assertEqual(_carrier_cut(audio, words), 1900)
        self.assertIsNone(_carrier_cut(audio, words[:-1] + [("open.", 1600, 1850)]))  # late: retry

    def test_a_short_pause_is_enough_where_whisper_places_the_gap(self):
        # Some voices pause only 30-70 ms after the lead-in; the silence-only rule wants 80.
        words = self.HEARD_TWO_WORDS[:-2] + [("Oh,", 1850, 2100)]
        self.assertEqual(_carrier_cut(_tone(1800) + _silence(40) + _tone(500), words), 1820)
        self.assertIsNone(_carrier_cut(_tone(1800) + _silence(20) + _tone(500), words))
        self.assertIsNone(_quote_context_cut(_tone(1800) + _silence(40) + _tone(500)))
        # No pause after the lead-in, only one after the unit's first word ("he" grunts): never cut there.
        audio = _tone(1800) + _silence(20) + _tone(150) + _silence(40) + _tone(400)
        self.assertIsNone(_carrier_cut(audio, self.HEARD_TWO_WORDS[:-2] + [("he", 1820, 1960), ("grunts.", 2010, 2400)]))

    def test_lead_in_speech_left_after_the_cut_or_a_lost_first_word_is_retried(self):
        for heard in ("Open. Kiss me!", "Me!"):
            with self.subTest(heard=heard):
                self._retried_after(heard)

    def _retried_after(self, first_heard):
        provider, _ = self._checked_provider(first_heard, "Kiss me!")
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(self.TWO_WORDS))
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow", return_value=20):
            audio, params, attempts, flagged = provider._speak_take(
                dict(model='chatterbox', voice='Gianna.wav', response_format='wav', input='“Kiss me!”',
                     extra_body={'seed': 10}), '“Kiss me!”', 'test')
        self.assertEqual((params['seed'], params['_match'], attempts, flagged), (21, 1.0, 2, None))

    def test_takes_without_the_lead_in_get_the_length_checks_and_count_every_attempt(self):
        provider = self._provider(100, 300)
        unseparable = [_tone(2400)] * 3
        kwargs = dict(model='chatterbox', voice='Gianna.wav', input='“Oh!”', response_format='wav',
                      extra_body={'seed': 10})
        for plain, kept, attempts, flagged in (([7000, 500], 500, 5, 'unseparated context'),
                                               ([7000, 6500, 8000], 6500, 6, 'overlong short dialogue')):
            with self.subTest(plain=plain):
                provider.client.audio.speech.create.side_effect = [
                    SimpleNamespace(content=_wav_from_segment(a)) for a in unseparable + [_tone(ms) for ms in plain]]
                audio, _, made, flag = provider._speak_take(dict(kwargs), '“Oh!”', 'test')
                self.assertEqual((len(audio), made, flag), (kept, attempts, flagged))

    def test_a_filler_whisper_leaves_out_is_cut_at_the_first_silence_after_the_lead_in(self):
        provider, checker = self._checked_provider("")
        checker.transcribe.side_effect = [speech_check.Heard("", self.HEARD_TWO_WORDS[:-2]),
                                          speech_check.Heard("", [])]
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(self.TWO_WORDS))
        audio, params, attempts, flagged = provider._speak_take(
            dict(model='chatterbox', voice='Gianna.wav', response_format='wav', input='“Um...”',
                 extra_body={'seed': 10}), '“Um...”', 'test')
        self.assertEqual((params['_context_trim_ms'], attempts, flagged), (1900, 1, None))
        self.assertNotIn('_match', params)  # nothing heard is no verdict for a filler

    def test_an_unseparated_unit_is_retried_without_the_lead_in_until_it_passes(self):
        provider, checker = self._checked_provider()
        plain = ["Oh, did you get gum?", "Kiss me!"]
        checker.transcribe.side_effect = lambda audio: speech_check.Heard(
            "" if len(audio) >= 2000 else plain.pop(0), [])  # the lead-in is never heard: no cut
        provider.client.audio.speech.create.side_effect = [
            SimpleNamespace(content=_wav_from_segment(take)) for take in [self.TWO_WORDS] * 3 + [_tone(500)] * 2]
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                   side_effect=[20, 30, 40, 50]):
            audio, params, attempts, flagged = provider._speak_take(
                dict(model='chatterbox', voice='Gianna.wav', response_format='wav', input='“Kiss me!”',
                     extra_body={'seed': 10}), '“Kiss me!”', 'test')
        inputs = [call.kwargs['input'] for call in provider.client.audio.speech.create.call_args_list]
        self.assertEqual(inputs, [_SHORT_QUOTE_CONTEXT + ' “Kiss me!”'] * 3 + ['“Kiss me!”'] * 2)
        self.assertEqual((params['seed'], params['_match'], attempts, flagged), (51, 1.0, 5, None))
        self.assertEqual(len(audio), 500)

    def test_speech_check_retries_a_take_that_does_not_say_its_text(self):
        provider, _ = self._checked_provider("He's nearly.", "Kiss me.")
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(self.TWO_WORDS))
        kwargs = dict(model='chatterbox', voice='Gianna.wav', input='“Kiss me!”', response_format='wav',
                      extra_body={'seed': 10})
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow", return_value=20):
            audio, params, attempts, flagged = provider._speak_take(kwargs, '“Kiss me!”', 'test')
        self.assertEqual((params['seed'], params['_match'], attempts, flagged), (21, 1.0, 2, None))

    def test_the_best_matching_take_is_kept_and_flagged_in_the_clip_map(self):
        heard = ("He's nearly.", "Kiss it, Lee.", "Is it, Lee?")
        provider, _ = self._checked_provider(*heard)
        takes = [self.TWO_WORDS + _silence(10 * i) for i in range(3)]  # tell the kept take apart
        provider.client.audio.speech.create.side_effect = [
            SimpleNamespace(content=_wav_from_segment(take)) for take in takes]
        scores = [speech_check.match('“Kiss me!”', text) for text in heard]
        self.assertLess(max(scores), speech_check.PASS_SCORE)
        best = scores.index(max(scores))
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.wav")
            with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                       side_effect=[10, 20, 30]):
                provider.text_to_speech('“Kiss me!”', path, tags)
            output = AudioSegment.from_file(path)
            with open(path + ".clips.json", encoding="utf8") as f:
                clip = json.load(f)["clips"][0]
        self.assertEqual(output.raw_data, takes[best][1900:].raw_data)
        self.assertEqual((clip['seed'], clip['attempts'], clip['flagged']),
                         ((11, 21, 31)[best], 3, 'speech mismatch'))
        self.assertEqual((clip['match'], clip['context_trim_ms']), (round(max(scores), 2), 1900))

    def test_speech_check_skips_long_units_and_a_failed_check_keeps_the_take(self):
        provider, checker = self._checked_provider(RuntimeError("model gone"))
        prose = 'The lantern burned steadily beside the window.'
        provider._speak_take(dict(model='chatterbox', voice='Elena.wav', response_format='wav', input=prose),
                             prose, 'test')
        self.assertEqual(provider.client.audio.speech.create.call_args.kwargs['input'], prose)
        checker.transcribe.assert_not_called()
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(self.TWO_WORDS))
        audio, params, attempts, flagged = provider._speak_take(
            dict(model='chatterbox', voice='Elena.wav', response_format='wav', input='she said.',
                 extra_body={'seed': 10}), 'she said.', 'test')
        self.assertEqual((len(audio), attempts, flagged), (len(self.TWO_WORDS) - 1900, 1, None))
        # After the lead-in a lowercase tag starts its own sentence, as Chatterbox would capitalise it alone.
        self.assertEqual(provider.client.audio.speech.create.call_args.kwargs['input'],
                         _SHORT_QUOTE_CONTEXT + ' She said.')
        self.assertNotIn('_match', params)

    def test_keeps_the_closest_take_when_every_retry_has_an_implausible_length(self):
        # A quick "Well…" can be real speech: three short takes must not fail the chapter (and
        # with it the M4B). The one nearest a plausible length is kept, with its own seed.
        provider = self._provider(400, 1000)
        provider.client.audio.speech.create.side_effect = [
            SimpleNamespace(content=_wav_from_segment(_tone(ms))) for ms in (700, 880, 800)]
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp3")
            with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"), patch(
                    "audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                    side_effect=[10, 20, 30]):
                provider.text_to_speech("“Well…”", path, tags)
            duration = len(AudioSegment.from_file(path))
            with open(path + ".clips.json", encoding="utf-8") as file:
                clip = json.load(file)["clips"][0]
        self.assertEqual(provider.client.audio.speech.create.call_count, 3)
        self.assertAlmostEqual(duration, 880, delta=80)
        self.assertEqual((clip["seed"], clip["attempts"], clip["flagged"]), (21, 3, "too-short ellipsis audio"))

    def test_a_take_with_a_long_silent_gap_is_never_the_one_kept(self):
        provider = self._provider(400, 1000)
        silent = _wav_from_segment(_tone(200) + _silence(4000) + _tone(200))
        provider.client.audio.speech.create.side_effect = [
            SimpleNamespace(content=content) for content in (silent, _wav_from_segment(_tone(700)), silent)]
        self.assertAlmostEqual(self._speak(provider, "“Well…”"), 700, delta=80)

    def test_repairs_an_opening_quote_used_to_close_any_line(self):
        self.assertEqual(_chatterbox_input("“Wait—“"), "“Wait—”")
        self.assertEqual(_chatterbox_input("“Who’s there?“"), "“Who’s there?”")
        self.assertEqual(_chatterbox_input("„Ja“"), "„Ja“")  # German quotes close with “
        self.assertEqual(_chatterbox_input("“Fine.”"), "“Fine.”")

    def test_retries_a_truncated_short_line_ending_in_ellipsis(self):
        provider = self._provider(400, 1000)
        provider.client.audio.speech.create.side_effect = [
            SimpleNamespace(content=_wav_from_segment(_tone(780))),
            SimpleNamespace(content=_wav_from_segment(_tone(1220))),
        ]
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow", return_value=123):
            self._speak(provider, "“She’s…”")
        self.assertEqual(provider.client.audio.speech.create.call_count, 2)
        first, second = [call.kwargs for call in provider.client.audio.speech.create.call_args_list]
        self.assertEqual(first["extra_body"]["seed"], 124)
        self.assertEqual(second["extra_body"]["seed"], 125)

    def test_retries_dialogue_that_loops_or_cuts_off(self):
        cases = [
            ("“Stop now!”", 8000, 1060),
            ("“I know, isn’t it?”", 7130, 1220),
            ("“Also, I know everything. I know about you and your affair.”", 1200, 3660),
        ]
        for line, bad_ms, good_ms in cases:
            with self.subTest(line=line):
                provider = self._provider(400, 1000)
                provider.client.audio.speech.create.side_effect = [
                    SimpleNamespace(content=_wav_from_segment(_tone(bad_ms))),
                    SimpleNamespace(content=_wav_from_segment(_tone(good_ms))),
                ]
                with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow",
                           return_value=123):
                    self._speak(provider, line)
                self.assertEqual(provider.client.audio.speech.create.call_count, 2)
                requests = [call.kwargs for call in provider.client.audio.speech.create.call_args_list]
                if len(line.strip()) <= 25:
                    self.assertEqual(requests[0]["extra_body"]["seed"], 124)
                    self.assertEqual(requests[1]["extra_body"]["seed"], 125)
                else:
                    self.assertNotIn("seed", requests[0].get("extra_body", {}))
                    self.assertEqual(requests[1]["extra_body"]["seed"], 124)

    def test_retries_an_overlong_short_narration_tag(self):
        provider = self._provider(400, 1000)
        provider.client.audio.speech.create.side_effect = [
            SimpleNamespace(content=_wav_from_segment(_tone(7000))),
            SimpleNamespace(content=_wav_from_segment(_tone(1200))),
        ]
        with patch("audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow", return_value=123):
            self._speak(provider, "she whispered.")
        self.assertEqual(provider.client.audio.speech.create.call_count, 2)
        requests = [call.kwargs for call in provider.client.audio.speech.create.call_args_list]
        self.assertEqual(requests[0]["extra_body"]["seed"], 124)
        self.assertEqual(requests[1]["extra_body"]["seed"], 125)

    def test_clip_map_records_short_seed_settings_and_chapter_times(self):
        provider = self._provider(400, 1000)
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp3")
            with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"), patch(
                "audiobook_generator.tts_providers.openai_tts_provider.secrets.randbelow", return_value=123
            ):
                provider.text_to_speech(f'The door opened.{PARAGRAPH_MARK}She spoke.', path, tags)
            with open(path + ".clips.json", encoding="utf-8") as file:
                clip_map = json.load(file)
        self.assertEqual(clip_map["version"], 1)
        self.assertEqual(len(clip_map["clips"]), 2)
        self.assertEqual([(entry["start_ms"], entry["end_ms"]) for entry in clip_map["clips"]],
                         [(0, 1000), (2000, 3000)])
        self.assertEqual([entry["seed"] for entry in clip_map["clips"]], [124, 124])
        self.assertEqual(clip_map["clips"][0]["voice"], "Elena.wav")
        self.assertEqual(clip_map["clips"][0]["settings"], {})
        self.assertEqual(clip_map["clips"][0]["text_sha1"], hashlib.sha1(
            b"The door opened.").hexdigest())
        self.assertNotIn("text", clip_map["clips"][0])

    def test_repairs_a_mismatched_quote_on_interrupted_dialogue(self):
        provider = self._provider(400, 1000)
        self._speak(provider, "“What…what are you-“")
        request = provider.client.audio.speech.create.call_args.kwargs
        self.assertEqual(request["input"], "What... what are you—")

    def test_without_pauses_marks_are_never_spoken(self):
        provider = self._provider(None, None)
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3", response=MagicMock())
        with patch("audiobook_generator.tts_providers.openai_tts_provider.merge_audio_segments"), \
                patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
            provider.text_to_speech(self.TEXT, "/tmp/unused.mp3", SimpleNamespace(idx=1, title="Ch"))
        spoken = " ".join(c.kwargs["input"] for c in provider.client.audio.speech.create.call_args_list)
        self.assertNotIn(PARAGRAPH_MARK, spoken)
        self.assertIn("village. The road", spoken)


class TestParagraphModeSpeech(unittest.TestCase):
    """F-05: opt-in paced_unit_mode="paragraph" cuts the request count; default is unchanged."""

    # Each sentence is long enough on its own (>= MIN_UNIT_CHARS) that sentence mode keeps them
    # as 3 separate requests, so the comparison against paragraph mode's 1 request is clean.
    THREE_SENTENCES = ("The lantern swung gently on its rusted iron hook by the door. "
                        "A cold draft slipped in beneath the warped floorboards. "
                        "Somewhere upstairs a single floorboard creaked twice.")

    def _provider(self, sentence_ms, paragraph_ms, mode=None):
        config = GeneralConfig(SimpleNamespace(
            tts="openai", model_name="chatterbox", voice_name="Elena.wav", output_format="mp3", speed=1.0,
            instructions=None, language="en", sentence_pause_ms=sentence_ms, paragraph_pause_ms=paragraph_ms,
            paced_unit_mode=mode))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            provider = OpenAITTSProvider(config)
        provider.client = MagicMock()
        return provider

    def _speak(self, provider, text):
        tags = SimpleNamespace(title="Ch", author="A", book_title="B", idx=1, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp3")
            with patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
                provider.text_to_speech(text, path, tags)
            return len(AudioSegment.from_file(path))

    def test_default_paced_unit_mode_is_sentence(self):
        provider = self._provider(400, 1000, mode=None)
        self.assertEqual(provider.config.paced_unit_mode, "sentence")

    def test_paragraph_mode_sends_one_request_for_a_whole_paragraph(self):
        provider = self._provider(400, 1000, mode="paragraph")
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(_tone(300) + _silence(150) + _tone(300) + _silence(150) + _tone(300)))
        self._speak(provider, self.THREE_SENTENCES)
        self.assertEqual(provider.client.audio.speech.create.call_count, 1)

    def test_paragraph_mode_uses_fewer_requests_than_sentence_mode_for_the_same_text(self):
        sentence_provider = self._provider(400, 1000, mode="sentence")
        sentence_provider.client.audio.speech.create.return_value = SimpleNamespace(content=_wav_bytes())
        paragraph_provider = self._provider(400, 1000, mode="paragraph")
        paragraph_provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(_tone(300) + _silence(150) + _tone(300) + _silence(150) + _tone(300)))

        self._speak(sentence_provider, self.THREE_SENTENCES)
        self._speak(paragraph_provider, self.THREE_SENTENCES)

        self.assertEqual(sentence_provider.client.audio.speech.create.call_count, 3)
        self.assertEqual(paragraph_provider.client.audio.speech.create.call_count, 1)

    def test_paragraph_mode_stretches_internal_gaps_to_the_configured_sentence_pause(self):
        provider = self._provider(400, 1000, mode="paragraph")
        provider.client.audio.speech.create.return_value = SimpleNamespace(
            content=_wav_from_segment(_tone(300) + _silence(150) + _tone(300) + _silence(150) + _tone(300)))
        duration = self._speak(provider, self.THREE_SENTENCES)
        # 3 tones (300 ms) + the 2 natural 150 ms gaps stretched to the configured 400 ms.
        self.assertAlmostEqual(duration, 3 * 300 + 2 * 400, delta=120)


class TestLooseFileTagging(unittest.TestCase):
    """F-26: wav/flac/opus loose chapter files get native tags instead of a raw ID3 write,
    which is inert on wav/flac and non-conformant on opus (see openai_tts_provider._tag_loose_file)."""

    def _provider(self, output_format):
        config = GeneralConfig(SimpleNamespace(
            tts="openai", model_name="chatterbox", voice_name="Elena.wav", output_format=output_format,
            speed=1.0, instructions=None, language="en", sentence_pause_ms=400, paragraph_pause_ms=1000))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            provider = OpenAITTSProvider(config)
        provider.client = MagicMock()
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=_wav_bytes(300))
        return provider

    def test_wav_flac_opus_chapters_are_tagged_in_their_native_format(self):
        tags = SimpleNamespace(title="Chapter One", author="Author Name", book_title="Test Book",
                                idx=3, cover=None)
        readers = {"wav": WAVE, "flac": FLAC, "opus": OggOpus}
        for output_format, reader in readers.items():
            with self.subTest(output_format=output_format):
                provider = self._provider(output_format)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, f"out.{output_format}")
                    provider.text_to_speech("A short paced sentence for tagging.", path, tags)
                    audio = reader(path)  # raises if the fix broke the container header
                    if output_format == "wav":
                        self.assertEqual(audio.tags["TIT2"].text[0], "Chapter One")
                        self.assertEqual(audio.tags["TALB"].text[0], "Test Book")
                    else:
                        self.assertEqual(audio["title"][0], "Chapter One")
                        self.assertEqual(audio["album"][0], "Test Book")

    def test_aac_chapter_is_tagged_via_id3_and_does_not_fail(self):
        # Note (F-26): ADTS has no native tag container; mutagen's own docs say to use ID3
        # directly, which is what set_audio_tags already does. Must never raise for aac.
        provider = self._provider("aac")
        tags = SimpleNamespace(title="Chapter One", author="Author Name", book_title="Test Book",
                                idx=3, cover=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.aac")
            provider.text_to_speech("A short paced sentence for tagging.", path, tags)
            self.assertTrue(os.path.exists(path))


class TestBuildM4b(unittest.TestCase):

    def test_chapters_cover_and_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = os.path.join(tmp, "1.mp3"), os.path.join(tmp, "2.mp3")
            _mp3(one, 2)
            _mp3(two, 3)
            cover = os.path.join(tmp, "cover.jpg")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", "color=c=red:s=64x64", "-frames:v", "1", cover], check=True)
            output = os.path.join(tmp, "Book = One; #2.m4b")
            build_m4b([("Chapter One", one), ("Chapter Two", two)], output, "Book = One; #2", "Jane Doe", cover)
            info = _ffprobe(output)
            leftovers = [n for n in os.listdir(tmp) if n.endswith(".part")]
        self.assertEqual([c["tags"]["title"] for c in info["chapters"]], ["Chapter One", "Chapter Two"])
        self.assertAlmostEqual(float(info["chapters"][1]["start_time"]), 2.0, delta=0.1)
        self.assertEqual(info["format"]["tags"]["title"], "Book = One; #2")
        self.assertEqual(info["format"]["tags"]["artist"], "Jane Doe")
        self.assertIn("aac", [s["codec_name"] for s in info["streams"]])
        self.assertTrue(any(s.get("disposition", {}).get("attached_pic") for s in info["streams"]))
        self.assertEqual(leftovers, [])

    def test_safe_book_file_name(self):
        self.assertEqual(safe_book_file_name('Car Load: A Ride / On "A" Lap? '), "Car Load A Ride On A Lap")
        self.assertEqual(safe_book_file_name(""), "audiobook")

    def test_safe_book_file_name_truncation_does_not_reintroduce_a_trailing_space(self):
        # F-34: safe_book_file_name used to strip trailing space/dot, THEN truncate to
        # 150 chars, so the cut itself could reintroduce one at the new end of the string.
        title = "A" * 149 + " " + "B" * 10
        result = safe_book_file_name(title)
        self.assertEqual(result, "A" * 149)
        self.assertFalse(result.endswith(" "))

    def test_safe_book_file_name_avoids_windows_reserved_device_names(self):
        # F-34: none of the three sanitizers guarded against CON, NUL, COM1, ..., which
        # cannot be created as a real file or folder on Windows even with an extension.
        self.assertEqual(safe_book_file_name("con"), "con_")
        self.assertEqual(safe_book_file_name("NUL"), "NUL_")
        self.assertEqual(safe_book_file_name("COM1"), "COM1_")
        self.assertEqual(safe_book_file_name("Conquest"), "Conquest")

    def test_gif_and_webp_covers_build_successfully(self):
        # F-04: ffmpeg's MP4 muxer only accepts JPEG/PNG as an attached picture with
        # "-c:v copy"; GIF and WebP covers used to fail the whole M4B build.
        with tempfile.TemporaryDirectory() as tmp:
            chapter = os.path.join(tmp, "1.mp3")
            _mp3(chapter, 1)
            for ext in ("gif", "webp"):
                cover = os.path.join(tmp, f"cover.{ext}")
                _cover(cover)
                output = os.path.join(tmp, f"Book.{ext}.m4b")
                build_m4b([("One", chapter)], output, "Book", "Author", cover)
                info = _ffprobe(output)
                self.assertTrue(
                    any(s.get("disposition", {}).get("attached_pic") for s in info["streams"]),
                    f"no attached picture for a {ext} cover",
                )

    def test_aac_chapters_are_stream_copied_not_reencoded(self):
        # F-06: chapters generated as ADTS AAC must be remuxed with "-c:a copy
        # -bsf:a aac_adtstoasc" instead of a second lossy re-encode to AAC.
        with tempfile.TemporaryDirectory() as tmp:
            one, two = os.path.join(tmp, "1.aac"), os.path.join(tmp, "2.aac")
            _aac(one, 2)
            _aac(two, 3)
            output = os.path.join(tmp, "Book.m4b")
            calls = []
            real_run = subprocess.run

            def _spy(cmd, *args, **kwargs):
                calls.append(cmd)
                return real_run(cmd, *args, **kwargs)

            with patch("audiobook_generator.core.m4b.subprocess.run", side_effect=_spy):
                build_m4b([("One", one), ("Two", two)], output, "Book", "Author")

            build_command = calls[-1]
            self.assertIn("copy", build_command)
            self.assertIn("aac_adtstoasc", build_command)
            self.assertNotIn("64k", build_command)
            info = _ffprobe(output)
            self.assertEqual([c["tags"]["title"] for c in info["chapters"]], ["One", "Two"])
            self.assertAlmostEqual(float(info["chapters"][1]["start_time"]), 2.0, delta=0.1)
            self.assertEqual(info["streams"][0]["codec_name"], "aac")

    def test_aac_chapter_joins_follow_the_real_audio_not_a_bitrate_estimate(self):
        # ADTS AAC has no duration header, so ffprobe estimates its length from the bitrate
        # of the first frames. A chapter that opens loud and ends quiet was declared far
        # shorter than it is: its chapter marker came early, and the stream copy overlapped
        # the next chapter, which ffmpeg "fixes" by squashing the overlapping packets to zero
        # length ("Non-monotonic DTS").
        with tempfile.TemporaryDirectory() as tmp:
            one, two = os.path.join(tmp, "1.aac"), os.path.join(tmp, "2.aac")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", "aevalsrc=exprs='0.5*sin(2*PI*300*t)*lt(t,1)':s=24000:d=8",
                            "-ac", "1", "-c:a", "aac", "-f", "adts", one], check=True)
            _aac(two, 3)
            output = os.path.join(tmp, "Book.m4b")
            chapter_seconds = build_m4b([("One", one), ("Two", two)], output, "Book", "Author")
            info = _ffprobe(output)
            packets = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                      "packet=duration_time", "-of", "csv=p=0", output],
                                     capture_output=True, text=True, check=True).stdout.split()
        durations = [float(value.strip(",")) for value in packets]
        self.assertAlmostEqual(float(info["chapters"][1]["start_time"]), 8.0, delta=0.15)
        # The durations it returns are the ones the chapter markers were laid out with.
        self.assertEqual(len(chapter_seconds), 2)
        self.assertAlmostEqual(float(info["chapters"][1]["start_time"]), chapter_seconds[0], delta=0.001)
        self.assertEqual([d for d in durations if d < 0.01], [])
        self.assertAlmostEqual(sum(durations), 11.0, delta=0.2)

    def test_bad_chapter_file_names_the_chapter_that_is_unreadable(self):
        # F-33: an empty/corrupt chapter file used to fail the M4B with a bare
        # ffprobe/ffmpeg error that never said which chapter was at fault.
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "1.mp3")
            _mp3(good, 1)
            bad = os.path.join(tmp, "2.mp3")
            open(bad, "wb").close()  # empty file: ffprobe cannot read a duration from it
            output = os.path.join(tmp, "Book.m4b")
            with self.assertRaises(BadChapterFileError) as ctx:
                build_m4b([("One", good), ("Two", bad)], output, "Book", "Author")
        self.assertEqual(ctx.exception.path, bad)
        self.assertEqual(ctx.exception.title, "Two")
        self.assertIn("Two", str(ctx.exception))

    def test_missing_ffmpeg_raises_a_clear_error(self):
        # F-33: previously the first ffprobe/ffmpeg call raised a raw FileNotFoundError
        # deep inside the build instead of one clear, immediate message.
        with tempfile.TemporaryDirectory() as tmp:
            chapter = os.path.join(tmp, "1.mp3")
            _mp3(chapter, 1)
            output = os.path.join(tmp, "Book.m4b")
            with patch("audiobook_generator.core.m4b.shutil.which", return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    build_m4b([("One", chapter)], output, "Book", "Author")
        self.assertIn("ffmpeg", str(ctx.exception))
        self.assertIn("ffprobe", str(ctx.exception))


class _InlinePool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, tasks):
        return map(func, tasks)


class TestGeneratorM4b(unittest.TestCase):

    def _run(self, fail_title=None, cover=False, corrupt_title=None):
        tmp = tempfile.mkdtemp()
        output = os.path.join(tmp, "Book")
        config = SimpleNamespace(
            output_folder=output, preview=False, output_text=False, log="INFO", log_file=None, no_prompt=True,
            worker_count=1, chapter_start=1, chapter_end=-1, chapter_selection=None, skip_existing=False,
            output_m4b=True)
        cover_obj = None
        if cover:
            cover_source = os.path.join(tmp, "source_cover.jpg")
            _cover(cover_source)  # a real, ffmpeg-decodable jpeg (not fake byte content)
            with open(cover_source, "rb") as f:
                cover_obj = SimpleNamespace(mime="image/jpeg", data=f.read())
        parser = SimpleNamespace(get_book_title=lambda: "My Book", get_book_author=lambda: "Author",
                                 get_book_cover=lambda: cover_obj,
                                 get_chapters=lambda _: [("One", "a" * 50), ("Two", "b" * 50)])

        # Snapshot of the visible output folder while chapter Two (the last one) is
        # generating, i.e. after chapter One and the cover already exist somewhere (F-23).
        seen_while_generating = []

        class FakeProvider:
            def get_break_string(self):
                return " "

            def estimate_cost(self, _):
                return 0.0

            def get_output_file_extension(self):
                return "mp3"

            def text_to_speech(self, text, path, tags):
                if tags.title == "Two":
                    seen_while_generating.extend(os.listdir(output))
                if tags.title == fail_title:
                    raise RuntimeError("tts failed")
                if tags.title == corrupt_title:
                    open(path, "wb").close()  # empty/corrupt chapter file (F-33)
                    return
                _mp3(path + ".mp3", 1)
                os.replace(path + ".mp3", path)

        with patch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=parser), \
                patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=FakeProvider()), \
                patch("audiobook_generator.core.audiobook_generator.multiprocessing.Pool", _InlinePool):
            result = AudiobookGenerator(config).run()
        return output, seen_while_generating, result

    def test_finished_book_is_one_m4b_and_work_folder_removed(self):
        output, _, result = self._run()
        self.assertTrue(result)
        self.assertEqual(sorted(os.listdir(output)), ["My Book.m4b"])
        self.assertEqual([c["tags"]["title"] for c in _ffprobe(os.path.join(output, "My Book.m4b"))["chapters"]],
                         ["One", "Two"])

    def test_failed_chapter_keeps_parts_for_resume_and_builds_no_m4b(self):
        output, _, result = self._run(fail_title="Two")
        self.assertFalse(result)
        self.assertEqual(os.listdir(output), [CHAPTER_WORK_FOLDER])
        # Also the F-11 manifest (original chapter number + text hash) for the one chapter
        # that did finish.
        self.assertEqual(sorted(os.listdir(os.path.join(output, CHAPTER_WORK_FOLDER))),
                         [".manifest.json", "0001_One.mp3"])

    def test_cover_is_hidden_while_generating_and_kept_beside_the_finished_m4b(self):
        # F-23: the cover used to be written straight into the visible output folder
        # before any chapter existed, so a library scanner could see a cover-only "book"
        # for the whole time the book was generating.
        output, seen_while_generating, result = self._run(cover=True)
        self.assertTrue(result)
        self.assertEqual(seen_while_generating, [CHAPTER_WORK_FOLDER])
        self.assertEqual(sorted(os.listdir(output)), ["My Book.m4b", "cover.jpg"])

    def test_corrupt_chapter_file_is_deleted_so_a_retry_can_regenerate_it(self):
        # F-33: build_m4b names the unreadable chapter; the generator deletes it instead
        # of leaving the book permanently stuck on the same failure.
        output, _, result = self._run(corrupt_title="Two")
        self.assertFalse(result)
        chapters_dir = os.path.join(output, CHAPTER_WORK_FOLDER)
        self.assertEqual(os.listdir(output), [CHAPTER_WORK_FOLDER])
        self.assertEqual(sorted(os.listdir(chapters_dir)), [".manifest.json", "0001_One.mp3"])


if __name__ == "__main__":
    unittest.main()
