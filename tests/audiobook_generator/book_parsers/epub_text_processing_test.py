"""Regression tests for epub_book_parser text-cleaning and title fixes: F-13, F-14, F-15,
F-16, F-32 (see docs/chatterbox-edition/REVIEW_FINDINGS.md)."""
import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock

from audiobook_generator.book_parsers.epub_book_parser import EpubBookParser
from audiobook_generator.config.general_config import GeneralConfig


def _write_epub(path: str, heading_html: str, body_html: str) -> None:
    """Minimal single-chapter EPUB2 book for text-cleaning unit tests."""
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
        z.writestr("c1.xhtml",
                   '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>'
                   f'{heading_html}{body_html}</body></html>')


def _config(input_file: str, **overrides) -> GeneralConfig:
    args = dict(
        input_file=input_file, output_folder="output", preview=False, output_text=False,
        title_mode="auto", log="INFO", newline_mode="double", chapter_start=1, chapter_end=-1,
        remove_endnotes=False, remove_reference_numbers=False, search_and_replace_file=None,
        tts="openai", language="en-US", voice_name="alloy", output_format="mp3", model_name="",
    )
    args.update(overrides)
    return GeneralConfig(MagicMock(**args))


class TestChapterTitlesStayReadable(unittest.TestCase):
    """F-13: get_chapters must return a readable title, not the file-name-sanitized one."""

    def _title(self, heading: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            _write_epub(path, f"<h1>{heading}</h1>", "<p>Body text.</p>")
            title, _ = EpubBookParser(_config(path)).get_chapters(" @BRK#")[0]
            return title

    def test_colon_and_apostrophe_are_kept(self):
        self.assertEqual(self._title("Chapter One: Don’t Look Back"), "Chapter One: Don’t Look Back")

    def test_hyphenated_words_are_not_fused(self):
        self.assertEqual(self._title("Twenty-One Nights"), "Twenty-One Nights")

    def test_em_dash_is_kept(self):
        self.assertEqual(self._title("Part I—The Fall"), "Part I—The Fall")


class TestTableCellSeparation(unittest.TestCase):
    """F-14: adjacent table cells without whitespace must not be read as one word."""

    def test_packed_cells_get_a_separating_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            body = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Ann</td><td>30</td></tr></table>"
            _write_epub(path, "<h1>Table</h1>", body)
            _, text = EpubBookParser(_config(path, newline_mode="none")).get_chapters(" ")[0]
        self.assertIn("Name Age", text)
        self.assertNotIn("NameAge", text)
        self.assertIn("Ann 30", text)

    def test_inline_em_inside_a_word_is_not_split(self):
        """Guard against the rejected fix (a global get_text separator would split this)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            _write_epub(path, "<h1>T</h1>", "<p>over<em>whelmed</em></p>")
            _, text = EpubBookParser(_config(path, newline_mode="none")).get_chapters(" ")[0]
        self.assertIn("overwhelmed", text)


class TestSearchAndReplaceParsing(unittest.TestCase):
    """F-15: robust parsing of the search-and-replace rules file."""

    def _rules(self, file_text: str) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            epub_path = os.path.join(tmp, "b.epub")
            _write_epub(epub_path, "<h1>T</h1>", "<p>Body.</p>")
            rules_path = os.path.join(tmp, "rules.txt")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(file_text)
            parser = EpubBookParser(_config(epub_path, search_and_replace_file=rules_path))
            return parser.get_search_and_replaces()

    def test_last_line_without_trailing_newline_keeps_its_last_character(self):
        rules = self._rules("foo==bar\nbaz==qux")  # no trailing newline
        self.assertEqual(rules[-1], {"search": "baz", "replace": "qux"})

    def test_double_equals_in_the_replacement_is_kept_intact(self):
        rules = self._rules("a==b==c\n")
        self.assertEqual(rules, [{"search": "a", "replace": "b==c"}])

    def test_invalid_pattern_names_its_line_number(self):
        with self.assertRaises(ValueError) as ctx:
            self._rules("good==fine\nbad(==oops\n")
        self.assertIn("line 2", str(ctx.exception))

    def test_comments_and_blank_lines_are_ignored(self):
        rules = self._rules("# a comment\n\nfoo==bar\n")
        self.assertEqual(rules, [{"search": "foo", "replace": "bar"}])

    def test_empty_replacement_deletes_the_match(self):
        rules = self._rules("unwanted-header==\n")
        self.assertEqual(rules, [{"search": "unwanted-header", "replace": ""}])


class TestRemoveEndnotesRegex(unittest.TestCase):
    """F-16: strip short inline endnote markers without corrupting ordinary text."""

    def _clean(self, body_text: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            _write_epub(path, "<h1>T</h1>", f"<p>{body_text}</p>")
            _, text = EpubBookParser(_config(path, remove_endnotes=True, newline_mode="none")).get_chapters(" ")[0]
            return text

    def test_acronym_number_compounds_survive(self):
        self.assertIn("COVID19", self._clean("The COVID19 outbreak changed plans."))
        self.assertIn("B2B", self._clean("This is a B2B marketing term."))

    def test_quoted_decade_survives(self):
        self.assertIn('"1990s"', self._clean('He said "1990s" was his favorite decade.'))

    def test_real_endnote_marker_is_still_removed(self):
        cleaned = self._clean("This remarkable claim.12 The next chapter begins.")
        self.assertIn("claim. The next chapter", cleaned)
        self.assertNotIn("12", cleaned)

    def test_marker_glued_before_the_period_is_still_removed(self):
        cleaned = self._clean("This remarkable claim12. The next chapter begins.")
        self.assertIn("claim. The next chapter", cleaned)
        self.assertNotIn("12", cleaned)


class TestNumericHeadingFallback(unittest.TestCase):
    """F-32: a whitespace-padded numeric heading must still trigger the text-preview fallback."""

    def test_padded_numeric_heading_uses_text_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            _write_epub(path, "<h1>\n 12 \n</h1>", "<p>The story begins on a quiet morning in the village.</p>")
            title, _ = EpubBookParser(_config(path, title_mode="auto")).get_chapters(" ")[0]
        self.assertNotEqual(title, "12")
        self.assertIn("The story begins", title)


if __name__ == "__main__":
    unittest.main()


class TestFallbackTitleHasNoMarkerFragment(unittest.TestCase):

    def test_title_cut_at_60_chars_never_ends_in_part_of_the_paragraph_marker(self):
        # First paragraph is 58 characters, so a 60-character cut of the marked text would
        # land inside the " @BRK#" paragraph marker and leave "@" in the title.
        first = "An untitled opening paragraph that runs for fifty-eight ch"
        self.assertEqual(len(first), 58)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.epub")
            _write_epub(path, "", f"<p>{first}</p><p>Second paragraph.</p>")
            title, _ = EpubBookParser(_config(path, title_mode="first_few")).get_chapters(" @BRK#")[0]
        self.assertNotIn("@", title)
        self.assertTrue(title.startswith(first))
