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

    def test_safe_folder_name_truncation_does_not_reintroduce_a_trailing_space(self):
        # F-34: shared with safe_book_file_name; truncating must happen before stripping
        # a trailing space/dot, or the cut can reintroduce one.
        title = "A" * 149 + " " + "B" * 10
        result = safe_folder_name(title)
        self.assertEqual(result, "A" * 149)
        self.assertFalse(result.endswith(" "))

    def test_safe_folder_name_avoids_windows_reserved_device_names(self):
        # F-34: CON, NUL, COM1, ... cannot be created as a real folder on Windows.
        self.assertEqual(safe_folder_name("con"), "con_")
        self.assertEqual(safe_folder_name("NUL"), "NUL_")
        self.assertEqual(safe_folder_name("Conquest"), "Conquest")

    def test_safe_folder_name_empty_input_has_no_fallback(self):
        # Unlike safe_book_file_name, callers rely on "" being falsy to fall back to the
        # EPUB filename (see suggest_output_dir).
        self.assertEqual(safe_folder_name(""), "")

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
