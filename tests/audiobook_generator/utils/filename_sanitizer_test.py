import unittest

from audiobook_generator.utils.filename_sanitizer import make_safe_filename


class TestMakeSafeFilenameReservedNames(unittest.TestCase):
    """F-34: none of the three sanitizers guarded against Windows-reserved device names
    (CON, NUL, COM1, ...), which cannot be created as a real file even with an extension.
    """

    def test_bare_reserved_name_gets_a_safe_suffix(self):
        # idx=None is the one case where the sanitized title becomes the whole file stem,
        # so a chapter literally titled "CON" would otherwise produce an unwritable name.
        name = make_safe_filename(title="CON", idx=None, output_dir=".", ext=".mp3", collision_check=False)
        self.assertEqual(name, "CON_.mp3")

    def test_numbered_reserved_name_still_gets_the_suffix(self):
        # The numeric prefix already makes "0001_CON.mp3" harmless, but the guard applies
        # uniformly rather than special-casing the (common) prefixed path.
        name = make_safe_filename(title="CON", idx=1, output_dir=".", ext=".mp3", collision_check=False)
        self.assertEqual(name, "0001_CON_.mp3")

    def test_ordinary_titles_containing_a_reserved_word_are_untouched(self):
        name = make_safe_filename(title="Conquest", idx=1, output_dir=".", ext=".mp3", collision_check=False)
        self.assertEqual(name, "0001_Conquest.mp3")


class TestMakeSafeFilenameForbiddenChars(unittest.TestCase):
    """F-34: the forbidden-character set is now shared with safe_book_file_name and
    safe_folder_name, so control characters beyond \\n\\r\\t are also replaced."""

    def test_control_characters_are_replaced(self):
        name = make_safe_filename(title="Chapter\x07One", idx=1, output_dir=".", ext=".mp3",
                                  collision_check=False)
        self.assertEqual(name, "0001_Chapter_One.mp3")


if __name__ == "__main__":
    unittest.main()
