"""Speech tags that name the speaker, found without an LLM (core.speech_tags)."""
import unittest

from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M, chapter_segments
from audiobook_generator.core.speech_tags import first_person_tagged, has_speech_tag, tagged_speakers


def tags(text: str) -> dict:
    return tagged_speakers(chapter_segments(text))


class TestNamedTags(unittest.TestCase):

    def test_verb_then_name_after_the_quotation(self):
        self.assertEqual(tags('"You left the gate open," said Ada.'), {1: "Ada"})

    def test_name_after_quote_following_a_mistyped_curly_closer(self):
        text = f'“I saved somebody yesterday. “{M}“That is right!” said Ada.'
        self.assertEqual(tags(text), {2: "Ada"})

    def test_name_then_verb_after_the_quotation(self):
        self.assertEqual(tags('"I did not," Tom said quietly.'), {1: "Tom"})

    def test_titles_and_multi_word_names(self):
        self.assertEqual(tags('"Out, both of you," said Mrs. Marsh.'), {1: "Mrs. Marsh"})
        self.assertEqual(tags('"Steady," Captain Ferris said.'), {1: "Captain Ferris"})
        self.assertEqual(tags('"The ferry is late," said Old Hobb.'), {1: "Old Hobb"})

    def test_tag_before_the_quotation(self):
        self.assertEqual(tags('Mrs. Marsh said quietly, "Out, both of you."'), {1: "Mrs. Marsh"})
        self.assertEqual(tags('She put the cup down. Then Tom said, "Fine."'), {1: "Tom"})

    def test_family_words_count_as_names(self):
        self.assertEqual(tags('"Supper," said Mother.'), {1: "Mother"})

    def test_a_named_tag_is_lent_to_the_other_untagged_quotation_in_its_paragraph(self):
        text = '"Bring him up to the house," said Mrs. Marsh. "Ada, run ahead and build the fire up."'
        self.assertEqual(tags(text), {1: "Mrs. Marsh", 2: "Mrs. Marsh"})


class TestDirection(unittest.TestCase):
    """A tag belongs to one quotation; a wrong anchor is final, so doubt goes to the model (WORKLOG §27)."""

    def test_a_colon_introduction_belongs_to_the_next_quotation(self):
        # A real book's case, sanitised: the question was locked to the person who answers it.
        text = 'Then I heard Jon\'s voice: "When can we meet?" Ann answered without a pause: "Monday."'
        self.assertEqual(tags(text), {2: "Ann"})

    def test_two_speakers_introduced_in_one_paragraph(self):
        self.assertEqual(tags('Tom said: "Ready?" Ada answered at once: "Always."'), {1: "Tom", 2: "Ada"})

    def test_interrupted_speech_keeps_its_speaker(self):
        self.assertEqual(tags('"We should," Alice said, "leave."'), {1: "Alice", 2: "Alice"})

    def test_a_speaker_continues_through_their_own_action(self):
        text = '"Bring him up," said Mrs. Marsh. She turned to the door. "Ada, run ahead."'
        self.assertEqual(tags(text), {1: "Mrs. Marsh", 2: "Mrs. Marsh"})

    def test_an_earlier_quotation_never_borrows(self):
        self.assertEqual(tags('"Where were you?" "Out," said Tom.'), {2: "Tom"})

    def test_narration_naming_or_introducing_someone_else_ends_the_run(self):
        self.assertEqual(tags('"Hello," said Tom. Ada looked up. "Well?"'), {1: "Tom"})
        self.assertEqual(tags('"Hello," said Tom. She looked up: "Well?"'), {1: "Tom"})

    def test_contradictory_tags_are_left_to_the_model(self):
        self.assertEqual(tags('Tom said, "Fine," said Ada.'), {})


class TestTenses(unittest.TestCase):

    def test_present_tense_and_voiced_tags(self):
        self.assertEqual(tags('"Stay close," Mara whispers.'), {1: "Mara"})
        self.assertEqual(tags('"Oh, Daddy," Ann panted.'), {1: "Ann"})
        for tag in ("he murmurs.", "she sobs.", "he snarls.", "she panted.", "I stammer.", "they whisper."):
            self.assertTrue(has_speech_tag("", tag), tag)

    def test_first_person_tags_in_either_tense_and_direction(self):
        self.assertEqual(first_person_tagged(chapter_segments('"Why?" I ask.')), [1])
        self.assertEqual(first_person_tagged(chapter_segments('"Who?" I answered without thinking: "You."')), [2])


class TestNotNames(unittest.TestCase):

    def test_pronoun_and_description_tags_are_left_to_the_model(self):
        self.assertEqual(tags('"Fine," she said.'), {})
        self.assertEqual(tags('"It is a formality," said the doctor.'), {})

    def test_possessives_and_bare_titles_are_not_names(self):
        self.assertEqual(tags('"Come here," said Tom\'s mother.'), {})
        self.assertEqual(tags('"Come here," said Old.'), {})

    def test_an_untagged_line_is_not_guessed(self):
        self.assertEqual(tags(f'"Then who did?"{M}"The wind, probably."'), {})

    def test_a_pronoun_tagged_quotation_does_not_borrow_the_paragraph_speaker(self):
        text = '"Hello," said Tom. "Hello yourself," she answered.'
        self.assertEqual(tags(text), {1: "Tom"})

    def test_two_named_speakers_in_one_paragraph_lend_nothing(self):
        text = '"Hello," said Tom. "Hello," said Ada. "Well then."'
        self.assertEqual(tags(text), {1: "Tom", 2: "Ada"})

    def test_a_continued_quotation_is_left_to_the_previous_line(self):
        self.assertEqual(tags(f'“First part.{M}“Second part,” said Ada.'), {})


if __name__ == "__main__":
    unittest.main()
