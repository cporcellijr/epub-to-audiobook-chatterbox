import os
import tempfile
import unittest
import zipfile

from audiobook_generator.ui.web_ui import safe_folder_name, suggest_output_dir, timestamped_output_dir


def _epub_with_title(path: str, title: str) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
                   'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:identifier id="id">t</dc:identifier><dc:title>{title}</dc:title>'
                   '<dc:language>en</dc:language></metadata><manifest>'
                   '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
                   '<spine><itemref idref="c1"/></spine></package>')
        z.writestr("c1.xhtml", '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                               '<body><p>x</p></body></html>')


class TestOutputDir(unittest.TestCase):

    def test_safe_folder_name_keeps_spaces_and_strips_illegal_characters(self):
        self.assertEqual(safe_folder_name('Book: One / Two? "Three"'), "Book One Two Three")
        self.assertEqual(safe_folder_name("  The Book That Wouldn’t Burn.  "), "The Book That Wouldn’t Burn")

    def test_upload_suggests_folder_named_after_epub_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "whatever.epub")
            _epub_with_title(path, "The Atrocity Archives")
            update = suggest_output_dir(path)
        self.assertEqual(update["value"], os.path.join("audiobook_output", "The Atrocity Archives"))

    def test_cleared_upload_leaves_folder_unchanged(self):
        self.assertNotIn("value", suggest_output_dir(None))

    def test_default_folder_is_computed_per_call(self):
        self.assertTrue(timestamped_output_dir().startswith("audiobook_output"))


if __name__ == "__main__":
    unittest.main()
