import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.core.audiobook_generator import AudiobookGenerator


def _provider(write: bool = True, fail: bool = False) -> MagicMock:
    provider = MagicMock()
    provider.get_output_file_extension.return_value = "mp3"

    def text_to_speech(text, output_file, audio_tags):
        if write:
            with open(output_file, "wb") as f:
                f.write(b"audio")
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
        self.assertEqual(self._files(), ["0001_Chapter_One.mp3"])

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


if __name__ == "__main__":
    unittest.main()
