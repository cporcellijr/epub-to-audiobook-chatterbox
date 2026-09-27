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
from audiobook_generator.core.m4b import BadChapterFileError, build_m4b, safe_book_file_name
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
