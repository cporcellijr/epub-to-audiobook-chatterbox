import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.core.audiobook_generator import AudiobookGenerator


def _provider(write: bool = True, fail: bool = False, clip_map=None) -> MagicMock:
    provider = MagicMock()
    provider.get_output_file_extension.return_value = "mp3"

    def text_to_speech(text, output_file, audio_tags):
        if write:
            with open(output_file, "wb") as f:
                f.write(b"audio")
            if clip_map is not None:
                with open(output_file + ".clips.json", "w", encoding="utf-8") as f:
                    json.dump(clip_map, f)
        if fail:
            raise RuntimeError("tts failed")

    provider.text_to_speech.side_effect = text_to_speech
    return provider


class TestProcessChapterPartialWrite(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        config = MagicMock(output_folder=self.tmp.name, output_text=False, preview=False, skip_existing=False)
        self.generator = AudiobookGenerator(config)

    def tearDown(self):
        self.tmp.cleanup()

    def _files(self):
        return sorted(os.listdir(self.tmp.name))

    def test_completed_chapter_is_renamed_into_place(self):
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=_provider()):
            self.assertTrue(self.generator.process_chapter(1, "Chapter One", "text"))
        # Also writes the F-11 manifest (original chapter number + text hash) next to it.
        self.assertEqual(self._files(), [".manifest.json", "0001_Chapter_One.mp3"])

    def test_provider_writes_to_hidden_partial_path(self):
        provider = _provider()
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=provider):
            self.generator.process_chapter(1, "Chapter One", "text")
        written_to = provider.text_to_speech.call_args[0][1]
        self.assertTrue(os.path.basename(written_to).startswith("."))
        self.assertTrue(written_to.endswith(".part"))

    def test_failed_chapter_leaves_no_file_for_skip_existing_to_keep(self):
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider",
                   return_value=_provider(write=True, fail=True)):
            self.assertFalse(self.generator.process_chapter(1, "Chapter One", "text"))
        self.assertEqual(self._files(), [])

    def test_completed_chapter_moves_provider_clip_map_next_to_audio(self):
        clip_map = {"version": 1, "duration_ms": 800, "clips": [{"start_ms": 0, "end_ms": 800,
                                                                      "text_sha1": "abc"}]}
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider",
                   return_value=_provider(clip_map=clip_map)):
            self.assertTrue(self.generator.process_chapter(1, "Chapter One", "text"))
        audio = os.path.join(self.tmp.name, "0001_Chapter_One.mp3")
        with open(audio + ".clips.json", encoding="utf-8") as f:
            self.assertEqual(json.load(f), clip_map)
        self.assertFalse(any(name.endswith(".part.clips.json") for name in self._files()))

    def test_failed_chapter_removes_partial_clip_map(self):
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider",
                   return_value=_provider(fail=True, clip_map={"version": 1, "duration_ms": 1, "clips": []})):
            self.assertFalse(self.generator.process_chapter(1, "Chapter One", "text"))
        self.assertEqual(self._files(), [])


class TestMergeClipMaps(unittest.TestCase):

    def test_scales_chapter_times_and_offsets_across_missing_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "one.aac")
            second = os.path.join(tmp, "two.aac")
            third = os.path.join(tmp, "three.aac")
            with open(first + ".clips.json", "w", encoding="utf-8") as f:
                json.dump({"version": 1, "duration_ms": 1000, "clips": [
                    {"start_ms": 100, "end_ms": 500, "text_sha1": "one", "voice": "A"}]}, f)
            with open(third + ".clips.json", "w", encoding="utf-8") as f:
                json.dump({"version": 1, "duration_ms": 2000, "clips": [
                    {"start_ms": 0, "end_ms": 2000, "text_sha1": "three", "voice": "B"}]}, f)
            output = os.path.join(tmp, "book.m4b")
            durations = {first: 2.0, second: 3.0, third: 1.0}
            with patch("audiobook_generator.core.audiobook_generator._duration_seconds",
                       side_effect=lambda path: durations[path]):
                AudiobookGenerator._merge_clip_maps([("one", first), ("two", second), ("three", third)], output)
            with open(output + ".clips.json", encoding="utf-8") as f:
                result = json.load(f)
        self.assertEqual(result["duration_ms"], 6000)
        self.assertEqual([(c["book_start_ms"], c["book_end_ms"]) for c in result["clips"]],
                         [(200, 1000), (5000, 6000)])
        self.assertEqual([c["text_sha1"] for c in result["clips"]], ["one", "three"])

    def test_chapter_offsets_follow_cumulative_m4b_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            chapters = [(str(number), os.path.join(tmp, f"{number}.aac")) for number in range(3)]
            for _, audio in chapters:
                with open(audio + ".clips.json", "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "duration_ms": 1000,
                               "clips": [{"start_ms": 0, "end_ms": 1000}]}, f)
            output = os.path.join(tmp, "book.m4b")
            with patch("audiobook_generator.core.audiobook_generator._duration_seconds",
                       return_value=1.0006):
                AudiobookGenerator._merge_clip_maps(chapters, output)
            with open(output + ".clips.json", encoding="utf-8") as f:
                result = json.load(f)
        self.assertEqual([clip["book_start_ms"] for clip in result["clips"]], [0, 1001, 2001])
        self.assertEqual(result["duration_ms"], 3002)


if __name__ == "__main__":
    unittest.main()
