import unittest
from types import SimpleNamespace
from unittest.mock import patch

from audiobook_generator.core.audiobook_generator import AudiobookGenerator
from audiobook_generator.core.chapter_selection import preselect_chapters, supplement_score

STORY = "It was a dark and stormy night. " * 120  # ~3,800 characters


class TestPreselectChapters(unittest.TestCase):

    def test_front_matter_from_a_real_collection_is_skipped(self):
        # "Story Collection 2": 17-char title page, 201-char collection blurb, then the stories.
        chapters = [
            ("Story_Collection_2", "Story Collection 2"),
            ("Five_Book_Story_Collection", "Five Book Story Collection By A. N. Author " * 4),
            ("The_First_Story", STORY),
            ("The_Second_Story", STORY),
        ]
        self.assertEqual(preselect_chapters(chapters), [False, False, True, True])

    def test_typical_publisher_front_and_back_matter(self):
        chapters = [
            ("Copyright 2019", "Copyright © 2019 Jon Athan. All Rights Reserved. ISBN 123"),
            ("Contents", "Contents Chapter One Chapter Two Chapter Three"),
            ("For my family", "For my family, who put up with me."),
            ("Chapter One", STORY),
            ("Chapter Two", STORY),
            ("Acknowledgments", "I first want to thank my editor. " * 40),
            ("About the Author", "Jon Athan lives in California. Visit www.example.com"),
            ("Join the mailing list", "Join the mailing list! Sign up for news."),
        ]
        self.assertEqual(preselect_chapters(chapters), [False, False, False, True, True, False, False, False])

    def test_mid_book_part_divider_is_kept(self):
        chapters = [("Chapter 1", STORY), ("PART TWO", "PART TWO"), ("Chapter 2", STORY)]
        self.assertEqual(preselect_chapters(chapters), [True, True, True])

    def test_short_final_chapter_is_kept(self):
        epilogue = "The white walls were back. The humming lights flickered once more. " * 13  # ~900 chars
        chapters = [("Chapter 1", STORY), ("Epilogue-ish", epilogue), ("ALSO BY SK PRYNTZ", "Also by SK Pryntz")]
        self.assertEqual(preselect_chapters(chapters), [True, True, False])

    def test_whole_novel_in_one_section_is_never_dropped(self):
        novel = "ENDYMION Copyright (c) 1995 by Dan Simmons. " + STORY * 10
        chapters = [("ENDYMION Copyright c 1995", novel), ("About the Author", "Dan Simmons lives in Colorado.")]
        self.assertEqual(preselect_chapters(chapters), [True, False])

    def test_long_chapter_mentioning_keywords_is_story(self):
        text = STORY + " She raised her index finger. The publisher's permission was copyright. " + STORY
        self.assertLess(supplement_score("A Quiet Morning", text), 1.9)

    def test_book_word_only_counts_as_content_at_the_start_of_a_title(self):
        self.assertLess(supplement_score("Book Two", "Book Two"), 0)
        self.assertGreater(supplement_score("Five Book Collection", "Five Book Collection"), 0)

    def test_single_chapter_and_never_all_unticked(self):
        self.assertEqual(preselect_chapters([("Copyright", "Copyright © 2020")]), [True])
        flags = preselect_chapters([("Copyright", "Copyright © 2020 all rights reserved"),
                                    ("Contents", "Contents " * 30)])
        self.assertEqual(flags.count(True), 1)


class _InlinePool:
    """multiprocessing.Pool stand-in that runs tasks in-process."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, tasks):
        return map(func, tasks)


class TestGeneratorChapterSelection(unittest.TestCase):

    def _run(self, selection):
        config = SimpleNamespace(
            output_folder="/tmp/unused", preview=True, output_text=False, log="INFO", log_file=None,
            no_prompt=True, worker_count=1, chapter_start=1, chapter_end=-1, chapter_selection=selection,
        )
        parser = SimpleNamespace(
            get_book_title=lambda: "Book", get_book_author=lambda: "Author", get_book_cover=lambda: None,
            get_chapters=lambda _: [("Title Page", "Title"), ("Blurb", "Blurb"), ("One", "a" * 50),
                                    ("Two", "b" * 50), ("Three", "c" * 50)],
        )
        tts = SimpleNamespace(get_break_string=lambda: " ", estimate_cost=lambda _: 0.0)
        calls = []
        generator = AudiobookGenerator(config)
        with patch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=parser), \
                patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=tts), \
                patch("audiobook_generator.core.audiobook_generator.multiprocessing.Pool", _InlinePool), \
                patch.object(AudiobookGenerator, "process_chapter",
                             lambda self, idx, title, text: calls.append((idx, title)) or True):
            generator.run()
        return calls

    def test_selected_chapters_are_renumbered_from_one(self):
        self.assertEqual(self._run([3, 5]), [(1, "One"), (2, "Three")])

    def test_no_selection_keeps_original_numbering(self):
        self.assertEqual([idx for idx, _ in self._run(None)], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
