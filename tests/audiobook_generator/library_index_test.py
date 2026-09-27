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
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
                   'xmlns:opf="http://www.idpf.org/2007/opf">'
                   f'<dc:title id="t">{title}</dc:title><dc:creator opf:role="aut">{author}</dc:creator>'
                   '</metadata></package>')


def _opf_epub(path: str, opf_body: str) -> None:
    """EPUB with a hand-built <metadata> body, for OPF edge cases `_epub` doesn't cover."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
                   f'xmlns:opf="http://www.idpf.org/2007/opf">{opf_body}</metadata></package>')


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


class TestReadEpubMetadataOpfEdgeCases(unittest.TestCase):
    """F-20: a sort title, a CDATA title or an editor-before-author must not win the label."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "book.epub")

    def _metadata(self, opf_body: str):
        _opf_epub(self.path, opf_body)
        return library_index.read_epub_metadata(self.path)

    def test_sort_title_listed_before_the_main_title_is_not_picked(self):
        title, _ = self._metadata(
            '<dc:title id="t1">Archives, The</dc:title>'
            '<meta refines="#t1" property="file-as">Archives, The</meta>'
            '<dc:title id="t2">The Archives</dc:title>'
        )
        self.assertEqual(title, "The Archives")

    def test_title_type_main_wins_even_if_listed_second(self):
        title, _ = self._metadata(
            '<dc:title id="t1">Sort Form</dc:title>'
            '<dc:title id="t2">Display Title</dc:title>'
            '<meta refines="#t2" property="title-type">main</meta>'
        )
        self.assertEqual(title, "Display Title")

    def test_cdata_title_is_not_shown_with_its_wrapper(self):
        title, _ = self._metadata('<dc:title><![CDATA[My Book Title]]></dc:title>')
        self.assertEqual(title, "My Book Title")

    def test_editor_before_author_is_not_picked(self):
        _, author = self._metadata(
            '<dc:creator id="c1" opf:role="edt">Editor Name</dc:creator>'
            '<dc:creator id="c2" opf:role="aut">Author Name</dc:creator>'
        )
        self.assertEqual(author, "Author Name")

    def test_epub3_role_meta_is_also_honored(self):
        _, author = self._metadata(
            '<dc:creator id="c1">Editor Name</dc:creator>'
            '<meta refines="#c1" property="role">edt</meta>'
            '<dc:creator id="c2">Author Name</dc:creator>'
            '<meta refines="#c2" property="role">aut</meta>'
        )
        self.assertEqual(author, "Author Name")

    def test_single_title_and_author_still_work(self):
        title, author = self._metadata('<dc:title>Plain Book</dc:title><dc:creator>Plain Author</dc:creator>')
        self.assertEqual((title, author), ("Plain Book", "Plain Author"))

    def test_malformed_opf_yields_empty_strings_not_a_crash(self):
        title, author = self._metadata('<dc:title>Unterminated')
        self.assertEqual((title, author), ("", ""))


if __name__ == "__main__":
    unittest.main()
