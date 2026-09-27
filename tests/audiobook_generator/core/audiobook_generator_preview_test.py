import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audiobook_generator import AudiobookGenerator


def _config(output_folder: str, output_text: bool) -> GeneralConfig:
    return GeneralConfig(SimpleNamespace(
        input_file="examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub",
        output_folder=output_folder, preview=True, output_text=output_text, log="INFO",
        no_prompt=True, worker_count=1, skip_existing=False, use_pydub_merge=False,
        title_mode="auto", newline_mode="double", chapter_start=1, chapter_end=2,
        remove_endnotes=False, remove_reference_numbers=False, search_and_replace_file=None,
        tts="openai", language="en-US", voice_name="alloy", output_format="mp3",
        model_name="chatterbox", instructions=None, speed=1.0,
    ))


class TestPreviewWritesNothing(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = os.path.join(self.tmp.name, "Robinson Crusoe")
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_preview_leaves_no_folder_or_cover_in_the_library(self):
        AudiobookGenerator(_config(self.output, output_text=False)).run()
        self.assertFalse(os.path.exists(self.output))

    def test_preview_with_chapter_text_writes_text_but_no_cover(self):
        AudiobookGenerator(_config(self.output, output_text=True)).run()
        files = os.listdir(self.output)
        self.assertTrue(files)
        self.assertTrue(all(name.endswith(".txt") for name in files), files)


if __name__ == "__main__":
    unittest.main()
