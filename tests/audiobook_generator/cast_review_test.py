"""Which attributed lines get a second look (core.cast_review), with sanitised cases from a real book."""
import unittest

from audiobook_generator.core import cast_review
from audiobook_generator.core.cast_review import CONTINUES, CONTRADICTED, GENDER, NARRATOR, UNKNOWN
from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M, chapter_segments
from audiobook_generator.core.speech_tags import tagged_speakers

PEOPLE = {"sam": {"name": "Sam", "gender": "male"}, "ann": {"name": "Ann", "gender": "female"},
          "walt": {"name": "Walt", "gender": "male"}}


def flags(text, result, people=PEOPLE):
    paragraphs = chapter_segments(text)
    return cast_review.flag_lines(paragraphs, result, tagged_speakers(paragraphs), people)


class TestFlags(unittest.TestCase):

    def test_a_line_with_no_speaker(self):
        self.assertEqual(flags('"Hello?"', {1: None}), {1: [UNKNOWN]})

    def test_she_said_given_to_a_man(self):
        # Sanitised from the book: the first half of one sentence went to the narrator.
        text = 'I poured her a drink. "So, Dad," she said after a sip, "Sam knows about us."'
        found = flags(text, {1: "sam", 2: "ann"})
        self.assertIn(GENDER, found[1])
        self.assertEqual((sorted(found), found[2]), ([1, 2], [CONTINUES]))  # one sentence, two speakers: both asked

    def test_an_i_tagged_line_not_given_to_the_chapters_clear_narrator(self):
        text = f'"Yes," I said.{M}"No," I said.{M}"Maybe," I said.'
        self.assertEqual(flags(text, {1: "sam", 2: "sam", 3: "ann"}), {3: [NARRATOR]})
        self.assertEqual(flags(text, {1: "sam", 2: "ann", 3: "walt"}), {})  # no clear narrator

    def test_a_speaker_change_where_nothing_else_names_anyone(self):
        # Sanitised from the book: "... looked at me." made the model hand the line to the narrator.
        text = '"And the best part was the end." Ann looked at me. "Isn\'t that sweet?"'
        self.assertEqual(flags(text, {1: "ann", 2: "sam"}), {1: [CONTINUES], 2: [CONTINUES]})
        text = '"And the best part was the end." Walt looked at me. "Isn\'t that sweet?"'
        self.assertEqual(flags(text, {1: "ann", 2: "sam"}), {})  # someone else is named: fine

    def test_a_sentence_split_by_a_tag_is_one_speakers(self):
        text = '"She\'s a good girl," he said, "but sometimes she talks too much."'
        # "he said," tags both halves, so giving the second to a woman is a gender clash as well
        self.assertEqual(flags(text, {1: "walt", 2: "ann"}), {1: [CONTINUES], 2: [GENDER, CONTINUES]})

    def test_contradictory_tags(self):
        self.assertEqual(flags('Tom said, "Fine," said Ada.', {1: "ann"}), {1: [CONTRADICTED]})

    def test_a_consistent_chapter_and_anchored_lines_are_left_alone(self):
        text = f'"Hello," said Ann. "How are you?"{M}"Well enough," Walt said.{M}"Good."'
        self.assertEqual(flags(text, {1: "ann", 2: "ann", 3: "walt", 4: "ann"}), {})
        self.assertEqual(flags('"Hello," said Ann.', {1: None}), {})  # tag-named: never asked


class TestNarrator(unittest.TestCase):

    def test_the_narrator_needs_two_i_tagged_votes_and_a_majority(self):
        paragraphs = chapter_segments(f'"Yes," I said.{M}"No," I said.{M}"Maybe," I said.')
        self.assertEqual(cast_review.narrator_by_tags(paragraphs, {1: "sam", 2: "sam", 3: "ann"}), "sam")
        self.assertIsNone(cast_review.narrator_by_tags(paragraphs, {1: "sam", 2: None, 3: None}))


class TestGroupsAndRendering(unittest.TestCase):

    def test_nearby_lines_share_a_request_and_far_ones_do_not(self):
        paragraphs = chapter_segments(M.join(f'"Line {n}."' for n in range(1, 21)))
        found = cast_review.groups(paragraphs, [2, 4, 15])
        self.assertEqual([ids for _, _, ids in found], [[2, 4], [15]])
        self.assertEqual(found[0][:2], (0, 4 + cast_review.CONTEXT_AFTER))
        self.assertEqual(found[1][:2], (14 - cast_review.CONTEXT_BEFORE, 14 + cast_review.CONTEXT_AFTER + 1))

    def test_asked_certain_and_guessed_lines_are_marked_differently(self):
        paragraphs = chapter_segments('"One," said Ann. "Two." "Three."')
        shown = cast_review.render(paragraphs, 0, 1, [3], {1: "Ann"}, {2: "Ann"})
        self.assertEqual(shown, '[Ann] "One," said Ann. [Ann?] "Two." [#3] "Three."')


if __name__ == "__main__":
    unittest.main()
