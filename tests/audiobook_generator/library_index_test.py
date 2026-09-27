import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from audiobook_generator.ui import library_index


def _epub(path: str, title: str, author: str) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:title id="t">{title}</dc:title><dc:creator opf:role="aut">{author}</dc:creator>'
                   '</metadata></package>')


class TestLibraryIndex(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.library = os.path.join(self.tmp.name, "library")
        os.makedirs(os.path.join(self.library, "nested"))
        self.detour = os.path.join(self.library, "01. Detour (1988).epub")
        self.archives = os.path.join(self.library, "nested", "archives.epub")
        _epub(self.detour, "Detour", "Jane Doe")
        _epub(self.archives, "The Atrocity Archives &amp; More", "Charles Stross")
        with open(os.path.join(self.library, "broken.epub"), "wb") as f:
            f.write(b"not a zip")
        with open(os.path.join(self.library, "notes.txt"), "w") as f:
            f.write("ignored")
        self.env = patch.dict(os.environ, {
            "EBOOK_LIBRARY_DIR": self.library,
            "EBOOK_INDEX_FILE": os.path.join(self.tmp.name, "index.json"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_lists_epubs_recursively_with_metadata_labels(self):
        choices = library_index.book_choices(library_index.refresh_index())
        self.assertEqual(choices, [
            ("broken", os.path.join(self.library, "broken.epub")),
            ("Detour — Jane Doe", self.detour),
            ("The Atrocity Archives & More — Charles Stross", self.archives),
        ])

    def test_unchanged_files_are_not_reread(self):
        library_index.refresh_index()
        with patch.object(library_index, "read_epub_metadata", wraps=library_index.read_epub_metadata) as reader:
            library_index.refresh_index()
        reader.assert_not_called()

    def test_changed_file_is_reread_and_deleted_file_dropped(self):
        library_index.refresh_index()
        _epub(self.detour, "Detour (Revised Edition)", "Jane Doe")
        os.utime(self.detour, (1, 1))
        os.remove(self.archives)
        index = library_index.refresh_index()
        self.assertEqual(index[self.detour]["title"], "Detour (Revised Edition)")
        self.assertNotIn(self.archives, index)

    def test_cache_persists_between_loads(self):
        library_index.refresh_index()
        self.assertEqual(library_index.load_index()[self.detour]["author"], "Jane Doe")

    def test_book_title_falls_back_to_file_name(self):
        self.assertEqual(library_index.book_title("/x/Some Book.epub", {}), "Some Book")

    def test_no_library_configured(self):
        with patch.dict(os.environ, {"EBOOK_LIBRARY_DIR": ""}):
            self.assertEqual(library_index.refresh_index(), {})


if __name__ == "__main__":
    unittest.main()
