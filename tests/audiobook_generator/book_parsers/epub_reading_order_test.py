import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock

from audiobook_generator.book_parsers.epub_book_parser import EpubBookParser
from audiobook_generator.config.general_config import GeneralConfig


def _write_epub(path: str) -> None:
    """EPUB whose manifest order differs from its spine, with a nav doc and a non-linear item."""
    chapters = {
        "c1": "Chapter One text.",
        "c2": "Chapter Two text.",
        "c3": "Chapter Three text.",
        "notes": "Endnote text.",
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        # Manifest deliberately lists chapters 3, 1, 2; spine says 1, 2, 3 then a non-linear note.
        manifest = "".join(
            f'<item id="{k}" href="{k}.xhtml" media-type="application/xhtml+xml"/>'
            for k in ("c3", "c1", "c2", "notes")
        )
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
                   'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:identifier id="id">t</dc:identifier><dc:title>Order Test</dc:title>'
                   '<dc:creator>Tester</dc:creator><dc:language>en</dc:language></metadata><manifest>'
                   '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
                   f'{manifest}</manifest><spine>'
                   '<itemref idref="nav"/><itemref idref="c1"/><itemref idref="c2"/><itemref idref="c3"/>'
                   '<itemref idref="notes" linear="no"/></spine></package>')
        z.writestr("OEBPS/nav.xhtml",
                   '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" '
                   'xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol>'
                   '<li><a href="c1.xhtml">Table of Contents Entry</a></li></ol></nav></body></html>')
        for key, body in chapters.items():
            z.writestr(f"OEBPS/{key}.xhtml",
                       '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head>'
                       f'<title>{key}</title></head><body><p>{body}</p></body></html>')


def _config(input_file: str) -> GeneralConfig:
    return GeneralConfig(MagicMock(
        input_file=input_file, output_folder="output", preview=False, output_text=False,
        title_mode="auto", log="INFO", newline_mode="double", chapter_start=1, chapter_end=-1,
        remove_endnotes=False, remove_reference_numbers=False, search_and_replace_file=None,
        tts="openai", language="en-US", voice_name="alloy", output_format="mp3", model_name="",
    ))


def _write_guide_epub(path: str, toc_body: str) -> None:
    """EPUB2 book whose <guide> points a generically titled page at the table of contents."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
                   'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:identifier id="id">t</dc:identifier><dc:title>Guide Test</dc:title>'
                   '<dc:creator>Tester</dc:creator><dc:language>en</dc:language></metadata><manifest>'
                   '<item id="toc" href="toc.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
                   '</manifest><spine><itemref idref="toc"/><itemref idref="c1"/></spine>'
                   '<guide><reference type="toc" title="Contents" href="toc.xhtml"/></guide>'
                   '</package>')
        z.writestr("OEBPS/toc.xhtml",
                   '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head>'
                   f'<title>Contents</title></head><body><p>{toc_body}</p></body></html>')
        z.writestr("OEBPS/c1.xhtml",
                   '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head>'
                   '<title>c1</title></head><body><p>Chapter one real text.</p></body></html>')


class TestEpubGuideToc(unittest.TestCase):
    """F-17: an EPUB2 <guide type="toc"> page must not be narrated as a chapter, unless it's
    long enough that it reads as a publisher's mistagged real content page instead."""

    def _reading_order_names(self, toc_body: str) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "guide.epub")
            _write_guide_epub(path, toc_body)
            parser = EpubBookParser(_config(path))
            return [item.get_name() for item in parser._reading_order_documents()]

    def test_short_guide_toc_page_is_excluded(self):
        names = self._reading_order_names("Chapter One .. 1\nChapter Two .. 15")
        self.assertNotIn("toc.xhtml", names)
        self.assertIn("c1.xhtml", names)

    def test_long_guide_toc_page_is_kept_as_content(self):
        long_body = "This is really a chapter, not a contents page. " * 120  # > 5,000 chars
        self.assertGreater(len(long_body), 5000)
        names = self._reading_order_names(long_body)
        self.assertIn("toc.xhtml", names)


class TestEpubReadingOrder(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "order.epub")
        _write_epub(self.path)
        self.parser = EpubBookParser(_config(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_chapters_follow_spine_not_manifest(self):
        texts = [text for _, text in self.parser.get_chapters(" ")]
        self.assertEqual(texts, ["Chapter One text.", "Chapter Two text.", "Chapter Three text."])

    def test_nav_document_is_not_read_as_a_chapter(self):
        texts = " ".join(text for _, text in self.parser.get_chapters(" "))
        self.assertNotIn("Table of Contents Entry", texts)

    def test_non_linear_items_are_skipped(self):
        texts = " ".join(text for _, text in self.parser.get_chapters(" "))
        self.assertNotIn("Endnote text.", texts)


if __name__ == "__main__":
    unittest.main()
