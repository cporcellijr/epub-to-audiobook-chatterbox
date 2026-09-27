import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from audiobook_generator.core.audiobook_generator import AudiobookGenerator


def _provider():
    provider = MagicMock()
    provider.get_output_file_extension.return_value = "mp3"

    def text_to_speech(text, output_file, audio_tags):
        with open(output_file, "wb") as f:
            f.write(text.encode())

    provider.text_to_speech.side_effect = text_to_speech
    return provider


class TestSkipExistingManifest(unittest.TestCase):
    """F-11: skip_existing used to trust any file whose OUTPUT position and (sanitized)
    title matched, with nothing recording which original chapter actually produced it. If
    a book was re-added with a changed chapter selection and a repeated title (a common
    running head, or every "tag_text" chapter titled the same), the new chapter at that
    output position silently kept the old chapter's audio under the new number.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = MagicMock(output_folder=self.tmp.name, output_text=False, preview=False,
                                 skip_existing=True, output_m4b=False)
        self.generator = AudiobookGenerator(self.config)
        self.output_file = os.path.join(self.tmp.name, "0001_Intro.mp3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_changed_selection_regenerates_instead_of_reusing_a_same_named_file(self):
        # First run: original chapter 1 (text "AAA") becomes output chapter 1, "Intro".
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=_provider()):
            self.generator.original_numbers = {1: 1}
            self.assertTrue(self.generator.process_chapter(1, "Intro", "AAA"))
        with open(self.output_file, "rb") as f:
            self.assertEqual(f.read(), b"AAA")

        # Second run: the selection changed so a DIFFERENT original chapter (3) now lands
        # at output position 1, with the same repeated title "Intro" and different text.
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=_provider()):
            self.generator.original_numbers = {1: 3}
            self.assertTrue(self.generator.process_chapter(1, "Intro", "ZZZ"))
        with open(self.output_file, "rb") as f:
            self.assertEqual(f.read(), b"ZZZ")  # regenerated, not the stale chapter-1 audio

    def test_unchanged_selection_still_skips(self):
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=_provider()):
            self.generator.original_numbers = {1: 1}
            self.assertTrue(self.generator.process_chapter(1, "Intro", "AAA"))

        provider = _provider()
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=provider):
            self.generator.original_numbers = {1: 1}
            self.assertTrue(self.generator.process_chapter(1, "Intro", "AAA"))
        provider.text_to_speech.assert_not_called()

    def test_same_original_chapter_with_edited_text_regenerates(self):
        # Same original chapter number, but the text itself changed (e.g. a search-and-
        # replace rule was added): the manifest's text hash catches this too.
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=_provider()):
            self.generator.original_numbers = {1: 1}
            self.assertTrue(self.generator.process_chapter(1, "Intro", "AAA"))

        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=_provider()):
            self.generator.original_numbers = {1: 1}
            self.assertTrue(self.generator.process_chapter(1, "Intro", "EDITED"))
        with open(self.output_file, "rb") as f:
            self.assertEqual(f.read(), b"EDITED")

    def test_missing_manifest_trusts_the_existing_file(self):
        # A book started before this manifest existed (or resumed mid-flight): no
        # manifest entry at all. Forcing a full regeneration here would be far more
        # costly than the rare mislabeling this manifest guards against, so the existing
        # file is trusted rather than treated as a mismatch.
        os.makedirs(self.generator.chapter_folder(), exist_ok=True)
        with open(self.output_file, "wb") as f:
            f.write(b"OLD")

        provider = _provider()
        with patch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=provider):
            self.generator.original_numbers = {1: 3}  # selection "changed", but no manifest exists
            self.assertTrue(self.generator.process_chapter(1, "Intro", "ZZZ"))
        provider.text_to_speech.assert_not_called()
        with open(self.output_file, "rb") as f:
            self.assertEqual(f.read(), b"OLD")


if __name__ == "__main__":
    unittest.main()
