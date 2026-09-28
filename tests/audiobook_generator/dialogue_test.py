import unittest

from audiobook_generator.core.dialogue import (
    DIALOGUE, NARRATION, PARAGRAPH_MARK as M, Segment, chapter_segments, detect_quote_style, dialogue_lines,
    paragraphs_of,
)


def _flat(text):
    """[(kind initial, line id, text, continues)] over the whole chapter."""
    return [(s.kind[0], s.line_id, s.text, s.continues) for para in chapter_segments(text) for s in para]


class TestDoubleQuotes(unittest.TestCase):

    def test_straight_quotes_split_narration_and_speech_in_order(self):
        text = 'He looked up. "Where have you been?" she asked, and he did not answer. "It\'s late."'
        self.assertEqual(_flat(text), [
            ("n", 0, "He looked up.", False),
            ("d", 1, '"Where have you been?"', False),
            ("n", 0, "she asked, and he did not answer.", False),
            ("d", 2, '"It\'s late."', False),
        ])

    def test_curly_quotes_and_ids_run_on_across_paragraphs(self):
        text = f"“Come in,” she said.{M}He did.{M}“Sit.”"
        self.assertEqual([(k, i) for k, i, _, _ in _flat(text)], [("d", 1), ("n", 0), ("n", 0), ("d", 2)])

    def test_apostrophes_inside_double_quoted_speech_do_not_split_it(self):
        text = '"Don\'t touch O\'Brien\'s things," he said.'
        self.assertEqual([t for _, _, t, _ in _flat(text)], ['"Don\'t touch O\'Brien\'s things,"', "he said."])

    def test_single_quote_inside_double_quoted_speech_stays_inside(self):
        text = '"She called it \'the pit\' and laughed," he said.'
        self.assertEqual(_flat(text)[0][2], '"She called it \'the pit\' and laughed,"')

    def test_segments_partition_the_paragraph(self):
        text = 'Before. "One," said A. Between. "Two." After.'
        self.assertEqual(" ".join(t for _, _, t, _ in _flat(text)), text)


class TestMultiParagraphQuotes(unittest.TestCase):
    def test_mistyped_curly_closer_does_not_continue_into_the_next_speaker(self):
        text = f'“I saved somebody yesterday. “{M}“That is right!” said Ada.'
        self.assertEqual(_flat(text), [
            ("d", 1, "“I saved somebody yesterday. “", False),
            ("d", 2, "“That is right!”", False),
            ("n", 0, "said Ada.", False),
        ])

    def test_continuation_paragraph_is_marked_and_still_gets_its_own_id(self):
        text = (f"“First part of a long speech that goes on.{M}"
                f"“Second part,” he said, “and the end.”{M}"
                f"Nothing more was said.")
        self.assertEqual(_flat(text), [
            ("d", 1, "“First part of a long speech that goes on.", False),
            ("d", 2, "“Second part,”", True),
            ("n", 0, "he said,", False),
            ("d", 3, "“and the end.”", False),
            ("n", 0, "Nothing more was said.", False),
        ])

    def test_straight_quote_continuation_does_not_flip_open_and_close(self):
        text = f'"Part one of the speech.{M}"Part two," he said.'
        self.assertEqual([(k, t) for k, _, t, _ in _flat(text)],
                         [("d", '"Part one of the speech.'), ("d", '"Part two,"'), ("n", "he said.")])

    def test_unclosed_quote_followed_by_plain_narration_does_not_swallow_it(self):
        text = f'"Missing close mark here{M}Then the narration goes on quite normally.'
        self.assertEqual([(k, c) for k, _, _, c in _flat(text)], [("d", False), ("n", False)])


class TestSingleQuotes(unittest.TestCase):

    def test_british_single_quote_dialogue_with_apostrophes_and_a_possessive(self):
        text = "'Come in,' she said. 'It's cold out there, and James' hat is soaked.'"
        self.assertEqual([t for _, _, t, _ in _flat(text)], [
            "'Come in,'", "she said.", "'It's cold out there, and James' hat is soaked.'",
        ])

    def test_curly_single_quotes(self):
        text = "‘Well?’ he asked. ‘I don’t know,’ she said."
        self.assertEqual([k for k, _, _, _ in _flat(text)], ["d", "n", "d", "n"])

    def test_apostrophes_alone_are_not_dialogue(self):
        text = "It wasn't the dog's fault, and it wasn't o'clock yet."
        self.assertIsNone(detect_quote_style(paragraphs_of(text)))
        self.assertEqual(_flat(text), [("n", 0, text, False)])

    def test_the_majority_style_wins_for_the_chapter(self):
        double_heavy = f'"One." "Two." "Three." Then he said \'Tis nothing.'
        self.assertEqual(detect_quote_style(paragraphs_of(double_heavy)), "double")
        single_heavy = f"'One.' 'Two.' 'Three.' She wrote \"fin\" on the page."
        self.assertEqual(detect_quote_style(paragraphs_of(single_heavy)), "single")
        self.assertEqual([k for k, _, _, _ in _flat(single_heavy)], ["d", "d", "d", "n"])


class TestDashDialogue(unittest.TestCase):

    def test_em_dash_paragraphs_alternate_speech_and_narration(self):
        text = f"— Where were you? — he asked. — Nowhere.{M}She shrugged."
        self.assertEqual(_flat(text), [
            ("d", 1, "Where were you?", False), ("n", 0, "he asked.", False), ("d", 2, "Nowhere.", False),
            ("n", 0, "She shrugged.", False),
        ])

    def test_dashes_are_ignored_in_a_chapter_that_uses_quote_marks(self):
        text = f'"Hello," he said.{M}— and then silence.'
        self.assertEqual([k for k, _, _, _ in _flat(text)], ["d", "n", "n"])


class TestHelpers(unittest.TestCase):

    def test_dialogue_lines_lists_only_speech_in_order(self):
        text = f'A. "One." B.{M}"Two," said C. "Three."'
        self.assertEqual([(s.line_id, s.text) for s in dialogue_lines(text)],
                         [(1, '"One."'), (2, '"Two,"'), (3, '"Three."')])

    def test_paragraph_list_matches_paced_units_numbering(self):
        text = f"One.{M}{M}  Two  words. {M}* * *{M}Three."
        self.assertEqual(paragraphs_of(text), ["One.", "Two words.", "* * *", "Three."])
        self.assertEqual(len(chapter_segments(text)), 4)
        self.assertEqual(chapter_segments(text)[2], [Segment(NARRATION, 0, "* * *")])

    def test_kind_constants(self):
        self.assertEqual((NARRATION, DIALOGUE), ("narration", "dialogue"))
