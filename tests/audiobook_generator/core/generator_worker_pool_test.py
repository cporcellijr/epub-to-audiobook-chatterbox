import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from audiobook_generator.core.audiobook_generator import AudiobookGenerator


class FakeProvider:
    def get_break_string(self):
        return " "

    def estimate_cost(self, _):
        return 0.0

    def get_output_file_extension(self):
        return "mp3"

    def text_to_speech(self, text, path, tags):
        with open(path, "wb") as f:
            f.write(b"audio")


class TestSingleWorkerRunsInProcess(unittest.TestCase):
    """F-01: "Stop current book" only terminates the job process. When chapters ran
    through a multiprocessing.Pool(1), that Pool worker (the process actually talking to
    the TTS server) was reparented to PID 1 on terminate() and kept generating for up to
    the rest of the chapter, even though the queue already reported the job STOPPED.
    worker_count is always 1 in the UI, so chapters must run directly in this process
    instead: terminating the job process then has nothing left running.
    """

    def _run(self, worker_count: int):
        tmp = tempfile.mkdtemp()
        config = SimpleNamespace(
            output_folder=tmp, preview=False, output_text=False, log="INFO", log_file=None,
            no_prompt=True, worker_count=worker_count, chapter_start=1, chapter_end=-1,
            chapter_selection=None, skip_existing=False, output_m4b=False)
        parser = SimpleNamespace(get_book_title=lambda: "Book", get_book_author=lambda: "Author",
                                 get_book_cover=lambda: None,
                                 get_chapters=lambda _: [("One", "a" * 50)])
        with patch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=parser), \
                patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=FakeProvider()), \
                patch("audiobook_generator.core.audiobook_generator.multiprocessing.Pool") as pool_cls:
            result = AudiobookGenerator(config).run()
        return tmp, result, pool_cls

    def test_no_pool_is_created_for_a_single_worker(self):
        tmp, result, pool_cls = self._run(worker_count=1)
        # The crux of the fix: nothing is spawned that a "Stop current book" signal to
        # this process alone could fail to reach.
        pool_cls.assert_not_called()
        self.assertTrue(result)
        # Also the F-11 manifest, written alongside the chapter in this non-M4B config.
        self.assertEqual(sorted(os.listdir(tmp)), [".manifest.json", "0001_One.mp3"])

    def test_pool_is_still_used_for_more_than_one_worker(self):
        # The CLI's --worker_count > 1 path is unchanged.
        _, _, pool_cls = self._run(worker_count=2)
        pool_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
