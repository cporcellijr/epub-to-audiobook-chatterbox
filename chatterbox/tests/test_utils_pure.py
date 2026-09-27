"""Pure-Python tests for the parts of chatterbox/utils.py that need neither a GPU
nor a running server: filename sanitization, path-traversal guarding, and the
sentence chunker. No GPU or running server needed.

Run inside the image (has the pip-installed chatterbox/torch/etc.):
    docker run --rm --entrypoint python3 -v <repo>/chatterbox:/app -w /app \
        chatterbox-tts-server:local -m unittest discover -s tests
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


class TestSanitizeFilename(unittest.TestCase):
    def test_ordinary_name_keeps_letters_and_replaces_spaces(self) -> None:
        self.assertEqual(utils.sanitize_filename("My Voice.wav"), "My_Voice.wav")

    def test_directory_components_are_stripped(self) -> None:
        result = utils.sanitize_filename("../../etc/passwd")
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)

    def test_disallowed_characters_are_removed(self) -> None:
        result = utils.sanitize_filename("weird<>:name??.wav")
        for bad_char in "<>:?":
            self.assertNotIn(bad_char, result)
        self.assertTrue(result.endswith(".wav"))

    def test_empty_input_returns_a_generated_name(self) -> None:
        result = utils.sanitize_filename("")
        self.assertTrue(result.startswith("unnamed_file_"))

    def test_dots_only_name_returns_a_generated_name(self) -> None:
        # "..." contains only characters the sanitizer would otherwise keep, but
        # lstrip("._") reduces it to nothing, so it must fall back to a generated name.
        result = utils.sanitize_filename("...")
        self.assertTrue(result.startswith("sanitized_file_"))

    def test_long_name_is_truncated_but_keeps_the_extension(self) -> None:
        long_name = ("a" * 150) + ".wav"
        result = utils.sanitize_filename(long_name)
        self.assertLessEqual(len(result), 110)
        self.assertTrue(result.endswith(".wav"))


class TestSafeResolveWithin(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="chatterbox_safe_resolve_test_"))
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_plain_filename_resolves_inside_base_dir(self) -> None:
        resolved = utils.safe_resolve_within(self.tmp_dir, "song.wav")
        self.assertEqual(resolved, (self.tmp_dir / "song.wav").resolve())

    def test_multi_segment_traversal_is_neutralized_to_its_basename(self) -> None:
        # Path(...).name strips directory components first, so this cannot escape
        # base_dir; it resolves to base_dir/passwd rather than raising.
        resolved = utils.safe_resolve_within(self.tmp_dir, "../../etc/passwd")
        self.assertEqual(resolved, (self.tmp_dir / "passwd").resolve())

    def test_bare_dotdot_escapes_and_is_rejected(self) -> None:
        # Unlike a multi-segment path, Path("..").name == "..", so this resolves to
        # base_dir's parent and must be caught and rejected.
        with self.assertRaises(ValueError):
            utils.safe_resolve_within(self.tmp_dir, "..")

    def test_empty_filename_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            utils.safe_resolve_within(self.tmp_dir, "")


class TestChunkTextBySentences(unittest.TestCase):
    def test_empty_or_whitespace_only_returns_no_chunks(self) -> None:
        self.assertEqual(utils.chunk_text_by_sentences("", 500), [])
        self.assertEqual(utils.chunk_text_by_sentences("   \n  ", 500), [])

    def test_short_text_is_a_single_chunk(self) -> None:
        chunks = utils.chunk_text_by_sentences("Hello there. How are you?", 500)
        self.assertEqual(len(chunks), 1)

    def test_long_text_is_split_into_more_than_one_chunk(self) -> None:
        sentence = "This is one ordinary sentence of moderate length. "
        long_text = sentence * 40  # ~2000 characters, well past a 500-char target
        chunks = utils.chunk_text_by_sentences(long_text, 500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_single_overlong_sentence_is_kept_whole_not_split(self) -> None:
        # F-07: the chunker only splits at sentence boundaries; a sentence with none
        # becomes one over-target chunk rather than being cut mid-sentence here.
        one_long_sentence = "word " * 200 + "end."  # no internal sentence boundary
        chunks = utils.chunk_text_by_sentences(one_long_sentence, 100)
        self.assertEqual(len(chunks), 1)
        self.assertGreater(len(chunks[0]), 100)


if __name__ == "__main__":
    unittest.main()
