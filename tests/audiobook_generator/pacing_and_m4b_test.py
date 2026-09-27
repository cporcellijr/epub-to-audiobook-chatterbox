import io
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

from audiobook_generator.book_parsers.epub_book_parser import EpubBookParser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audiobook_generator import AudiobookGenerator, CHAPTER_WORK_FOLDER
from audiobook_generator.core.m4b import build_m4b, safe_book_file_name
from audiobook_generator.tts_providers.openai_tts_provider import (
    PARAGRAPH_MARK,
    OpenAITTSProvider,
    paced_units,
)

UNIT_MS = 1000


def _wav_bytes(ms: int = UNIT_MS) -> bytes:
    buffer = io.BytesIO()
    Sine(300).to_audio_segment(duration=ms).set_frame_rate(24000).set_channels(1).export(buffer, format="wav")
    return buffer.getvalue()


def _mp3(path: str, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=300:duration={seconds}", "-ar", "24000", "-ac", "1", path], check=True)


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
            (0, "The rain had stopped by the time they reached the bridge."),
            (0, "Yes. She pulled her coat tighter and looked back at the village."),
            (1, "No. The road home was quiet and dark."),
        ])

    def test_trailing_short_sentence_joins_previous_unit(self):
        units = paced_units("She pulled her coat tighter against the cold wind. Then she ran.", "en")
        self.assertEqual(units, [(0, "She pulled her coat tighter against the cold wind. Then she ran.")])


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
        duration = self._speak(self._provider(400, 1000, speed=2.0), self.TEXT)
        self.assertAlmostEqual(duration, 3 * UNIT_MS + 200 + 500, delta=80)

    def test_without_pauses_marks_are_never_spoken(self):
        provider = self._provider(None, None)
        provider.client.audio.speech.create.return_value = SimpleNamespace(content=b"ID3", response=MagicMock())
        with patch("audiobook_generator.tts_providers.openai_tts_provider.merge_audio_segments"), \
                patch("audiobook_generator.tts_providers.openai_tts_provider.set_audio_tags"):
            provider.text_to_speech(self.TEXT, "/tmp/unused.mp3", SimpleNamespace(idx=1, title="Ch"))
        spoken = " ".join(c.kwargs["input"] for c in provider.client.audio.speech.create.call_args_list)
        self.assertNotIn(PARAGRAPH_MARK, spoken)
        self.assertIn("village. The road", spoken)


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

    def _run(self, fail_title=None):
        tmp = tempfile.mkdtemp()
        output = os.path.join(tmp, "Book")
        config = SimpleNamespace(
            output_folder=output, preview=False, output_text=False, log="INFO", log_file=None, no_prompt=True,
            worker_count=1, chapter_start=1, chapter_end=-1, chapter_selection=None, skip_existing=False,
            output_m4b=True)
        parser = SimpleNamespace(get_book_title=lambda: "My Book", get_book_author=lambda: "Author",
                                 get_book_cover=lambda: None,
                                 get_chapters=lambda _: [("One", "a" * 50), ("Two", "b" * 50)])

        class FakeProvider:
            def get_break_string(self):
                return " "

            def estimate_cost(self, _):
                return 0.0

            def get_output_file_extension(self):
                return "mp3"

            def text_to_speech(self, text, path, tags):
                if tags.title == fail_title:
                    raise RuntimeError("tts failed")
                _mp3(path + ".mp3", 1)
                os.replace(path + ".mp3", path)

        with patch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=parser), \
                patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=FakeProvider()), \
                patch("audiobook_generator.core.audiobook_generator.multiprocessing.Pool", _InlinePool):
            AudiobookGenerator(config).run()
        return output

    def test_finished_book_is_one_m4b_and_work_folder_removed(self):
        output = self._run()
        self.assertEqual(sorted(os.listdir(output)), ["My Book.m4b"])
        self.assertEqual([c["tags"]["title"] for c in _ffprobe(os.path.join(output, "My Book.m4b"))["chapters"]],
                         ["One", "Two"])

    def test_failed_chapter_keeps_parts_for_resume_and_builds_no_m4b(self):
        output = self._run(fail_title="Two")
        self.assertEqual(os.listdir(output), [CHAPTER_WORK_FOLDER])
        self.assertEqual(os.listdir(os.path.join(output, CHAPTER_WORK_FOLDER)), ["0001_One.mp3"])


if __name__ == "__main__":
    unittest.main()
