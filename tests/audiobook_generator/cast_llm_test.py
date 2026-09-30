"""Speaker attribution: windows, strict reply parsing, alias merging and the retry-then-unknown rule,
all with a scripted stand-in for the chat endpoint."""
import json
import logging
import unittest
from unittest.mock import patch

from audiobook_generator.core import cast_llm as cast_llm_module
from audiobook_generator.core.cast_llm import (
    PROMPTS, WINDOW_LINES, AttributionError, Roster, attribute_chapter, build_windows, parse_reply,
)
from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M, chapter_segments

CHAPTER = (
    f'Ada Marsh put the lamp down. "You left the gate open," she said.{M}'
    f'"I did not," said Tom. He went on writing.{M}'
    f'"Then who did?"{M}'
    f'"The wind, probably." He shrugged.{M}'
    f'Their mother came in from the yard. "Both of you, out. The goats are in the beans."{M}'
    f'"Yes, Mrs. Marsh," said Tom, who liked to annoy her.'
)


class ScriptedChat:
    """Returns the scripted replies in order; records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages):
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        return json.dumps(reply) if isinstance(reply, dict) else reply


def _reply(speakers, characters=(), moods=None):
    reply = {"speakers": {str(k): v for k, v in speakers.items()}, "characters": list(characters)}
    if moods is not None:
        reply["moods"] = {str(k): v for k, v in moods.items()}
    return reply


class TestWindows(unittest.TestCase):

    def test_one_small_chapter_is_one_window_with_numbered_lines(self):
        windows = build_windows(chapter_segments(CHAPTER))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].ids, [1, 2, 3, 4, 5, 6])
        self.assertIn('[#1] "You left the gate open,"', windows[0].passage)
        self.assertIn("Ada Marsh put the lamp down.", windows[0].passage)

    def test_windows_are_cut_by_line_count_and_carry_context(self):
        text = M.join(f'"Line {n}," said Person {n % 3}.' for n in range(1, 2 * WINDOW_LINES + 6))
        windows = build_windows(chapter_segments(text), context_paragraphs=2)
        self.assertEqual([len(w.ids) for w in windows], [WINDOW_LINES, WINDOW_LINES, 5])
        self.assertEqual(windows[1].ids[0], WINDOW_LINES + 1)
        # The two paragraphs before the window are repeated as unmarked context.
        self.assertIn(f'"Line {WINDOW_LINES - 1},"', windows[1].passage)
        self.assertNotIn(f"[#{WINDOW_LINES - 1}]", windows[1].passage)
        self.assertIn(f"[#{WINDOW_LINES + 1}]", windows[1].passage)

    def test_windows_are_cut_by_character_budget_too(self):
        text = M.join(f'"{"word " * 200}" said someone.' for _ in range(6))
        windows = build_windows(chapter_segments(text), max_chars=2500, context_paragraphs=0)
        self.assertEqual([len(w.ids) for w in windows], [2, 2, 2])

    def test_narration_only_paragraphs_make_no_window(self):
        self.assertEqual(build_windows(chapter_segments(f"Nothing.{M}Nothing again.")), [])

    def test_continued_lines_are_not_asked(self):
        text = f'“First part.{M}“Second part,” he said.'
        windows = build_windows(chapter_segments(text))
        self.assertEqual((windows[0].ids, windows[0].continued), ([1], [2]))
        self.assertNotIn("[#2]", windows[0].passage)


class TestParseReply(unittest.TestCase):

    def test_good_json(self):
        speakers, characters, moods = parse_reply(json.dumps(_reply({1: "Ada Marsh", 2: "unknown"}, [
            {"name": "Ada Marsh", "gender": "female", "age": "adult", "aliases": ["Ada"]}])), [1, 2])
        self.assertEqual(speakers, {1: "Ada Marsh", 2: None})
        self.assertEqual(characters, [{"name": "Ada Marsh", "gender": "female", "age": "adult", "aliases": ["Ada"]}])
        self.assertEqual(moods, {1: "normal", 2: "normal"})  # no "moods" in the reply: every id defaults normal

    def test_fenced_json_and_hash_ids_are_tolerated(self):
        reply = "Here you go:\n```json\n{\"speakers\": {\"#1\": \"Tom\", \"[#2]\": \"Ada\"}}\n```"
        self.assertEqual(parse_reply(reply, [1, 2])[0], {1: "Tom", 2: "Ada"})

    def test_broken_json_is_an_error(self):
        with self.assertRaises(AttributionError):
            parse_reply('{"speakers": {"1": "Tom",', [1])
        with self.assertRaises(AttributionError):
            parse_reply("I think Tom says the first line.", [1])

    def test_missing_id_is_an_error(self):
        with self.assertRaises(AttributionError):
            parse_reply(json.dumps(_reply({1: "Tom"})), [1, 2])

    def test_invented_id_is_an_error(self):
        with self.assertRaises(AttributionError):
            parse_reply(json.dumps(_reply({1: "Tom", 9: "Ada"})), [1])

    def test_non_string_speaker_is_an_error(self):
        with self.assertRaises(AttributionError):
            parse_reply(json.dumps({"speakers": {"1": ["Tom"]}}), [1])

    def test_bad_gender_or_age_become_unknown_and_nameless_characters_are_dropped(self):
        _, characters, _ = parse_reply(json.dumps(_reply({1: "Tom"}, [
            {"name": "Tom", "gender": "boy", "age": "teen"}, {"gender": "male"}, {"name": "  "}])), [1])
        self.assertEqual(characters, [{"name": "Tom", "gender": "unknown", "age": "unknown", "aliases": []}])

    def test_unknown_listed_as_a_character_is_dropped(self):
        # Seen live: the model listed "Unknown" among the characters, and it got a voice.
        _, characters, _ = parse_reply(json.dumps(_reply({1: "Tom"}, [
            {"name": "Unknown", "gender": "unknown", "age": "unknown"}, {"name": "Tom", "gender": "male"}])), [1])
        self.assertEqual([c["name"] for c in characters], ["Tom"])

    def test_moods_are_parsed_and_invalid_or_missing_ones_become_normal(self):
        reply = json.dumps({"speakers": {"1": "Tom", "2": "Ada", "3": "Bea"},
                            "moods": {"1": "soft", "2": "not-a-mood", "9": "excited"}})
        _, _, moods = parse_reply(reply, [1, 2, 3])
        # id 1: valid; id 2: invalid value -> normal; id 3: no entry at all -> normal; id 9 (not asked) ignored.
        self.assertEqual(moods, {1: "soft", 2: "normal", 3: "normal"})

    def test_a_non_dict_moods_value_is_tolerated_and_ignored(self):
        reply = json.dumps({"speakers": {"1": "Tom"}, "moods": "soft"})
        _, _, moods = parse_reply(reply, [1])
        self.assertEqual(moods, {1: "normal"})


class TestRoster(unittest.TestCase):

    def test_first_name_and_full_name_merge_into_one_character(self):
        roster = Roster()
        key = roster.add("Tom", "male", "adult")
        self.assertEqual(roster.add("Thomas Baker"), key)       # the full name pulls it together
        self.assertEqual(roster.characters[key]["name"], "Thomas Baker")
        self.assertEqual(roster.add("tom"), key)
        self.assertEqual(roster.add("Thomas"), key)
        self.assertEqual(roster.add("Tom Baker"), key)
        self.assertCountEqual(roster.characters[key]["aliases"], ["Tom", "Thomas", "Tom Baker"])
        self.assertEqual(len(roster.characters), 1)

    def test_a_bare_surname_is_never_merged_by_code_only_by_a_declared_alias(self):
        roster = Roster()
        key = roster.add("Ada Marsh", "female")
        self.assertNotEqual(roster.add("Mrs. Marsh"), key)      # the mother, not the daughter
        self.assertEqual(len(roster.characters), 2)
        other = Roster()
        key = other.add("Thomas Baker", "male", aliases=["Mr. Baker"])
        self.assertEqual(other.add("Mr. Baker"), key)           # the model said they are one person
        self.assertEqual(len(other.characters), 1)

    def test_short_forms_and_prefixes_are_the_same_person(self):
        roster = Roster()
        key = roster.add("Bill")
        self.assertEqual(roster.add("Will"), key)          # both short for William
        self.assertEqual(roster.add("William Hale"), key)
        self.assertEqual(roster.add("Ben"), roster.add("Benjamin"))
        self.assertEqual(len(roster.characters), 2)

    def test_a_first_name_shared_by_two_people_is_not_merged(self):
        roster = Roster()
        roster.add("Anne Baker")
        roster.add("Anne Hale")
        self.assertIsNone(roster.resolve("Anne"))
        self.assertEqual(len(roster.characters), 2)

    def test_people_of_different_known_genders_stay_apart(self):
        roster = Roster()
        chris = roster.add("Chris", "female")
        christopher = roster.add("Christopher", "male")
        self.assertNotEqual(chris, christopher)
        self.assertEqual(roster.add("Christopher Hale"), christopher)  # the identically spelt first name wins
        self.assertEqual(len(roster.characters), 2)

    def test_gender_and_age_are_filled_once_known_and_not_overwritten(self):
        roster = Roster()
        key = roster.add("Ada")
        roster.add("Ada", "female", "adult")
        roster.add("Ada", "male", "child")
        self.assertEqual((roster.characters[key]["gender"], roster.characters[key]["age"]), ("female", "adult"))

    def test_explicit_aliases_from_the_model_are_recorded(self):
        roster = Roster()
        key = roster.add("Margaret Hale", "female", aliases=["Mother", "Mrs. Hale"])
        self.assertEqual(roster.add("Mother"), key)
        self.assertEqual(roster.add("Mrs Hale"), key)

    def test_words_anyone_may_be_called_are_never_aliases(self):
        # Seen live: Chris collected "he", "honey" and "child"; Nina "his mom".
        roster = Roster()
        chris = roster.add("Chris", "male", aliases=["honey", "he", "child", "Chris", "Chrissy"])
        self.assertEqual(roster.characters[chris]["aliases"], ["Chrissy"])
        nina = roster.add("Nina", "female", aliases=["his mom", "Mom", "the woman"])
        self.assertEqual(roster.add("Mom"), nina)  # a family word names her in this chapter
        self.assertEqual(roster.characters[nina]["aliases"], [])  # ...but is never saved with her
        oliver = roster.add("Oliver", "male", aliases=["I"])  # a first-person book's narrator
        self.assertEqual(roster.add("I"), oliver)
        self.assertNotEqual(roster.add("Honey"), chris)  # someone may really be called Honey
        woman = roster.add("The Woman", "female")  # a character's own name is kept even when generic
        self.assertEqual(roster.add("the woman"), woman)

    def test_a_family_word_names_one_person_only_within_a_chapter(self):
        # Seen live: a collection's six stories each had a "Mom", and all of them became one character.
        roster = Roster()
        terri = roster.add("Terri", "female", aliases=["Mom", "Mrs. McCallister"])
        self.assertEqual(roster.add("Mom"), terri)
        roster.new_chapter()
        self.assertNotEqual(roster.add("Mom"), terri)  # the next story's mother is someone else
        self.assertEqual(roster.add("Mrs McCallister"), terri)  # a real name still merges across chapters
        # Seen live too: "Himself" as an alias, and "little brother" / "Son-in-law" kept across stories.
        nate = roster.add("Nate", "male", aliases=["Himself", "Son-in-law", "little brother"])
        self.assertEqual((roster.characters[nate]["aliases"], roster.add("Son-in-law")), ([], nate))
        roster.new_chapter()
        self.assertNotEqual(roster.add("little brother"), nate)
        ruth = roster.add("Ruth", "female", aliases=["Ma"])
        self.assertEqual(roster.add("Ma"), ruth)
        # A saved cast's family-word aliases are not restored as names either.
        restored = Roster({"terri": {"name": "Terri", "aliases": ["Mom", "Terr"], "gender": "female", "lines": 3}})
        self.assertEqual((restored.resolve("Terr"), restored.resolve("Mom")), ("terri", None))

    def test_a_pronoun_as_the_speaker_is_an_unknown_speaker(self):
        reply = json.dumps({"speakers": {"1": "he", "2": "She", "3": "I"},
                            "characters": [{"name": "he", "gender": "male"}]})
        speakers, characters, _ = parse_reply(reply, [1, 2, 3])
        self.assertEqual(speakers, {1: None, 2: None, 3: "I"})
        self.assertEqual(characters, [])

    def test_a_saved_cast_restores_the_roster(self):
        roster = Roster({"thomas baker": {"name": "Thomas Baker", "aliases": ["Tom"], "gender": "male",
                                          "age": "adult", "lines": 3}})
        self.assertEqual(roster.add("Tom"), "thomas baker")


class TestAttributeChapter(unittest.TestCase):

    def setUp(self):
        self.paragraphs = chapter_segments(CHAPTER)
        self.log = logging.getLogger("test-cast")

    def test_good_replies_attribute_every_line_and_count_them(self):
        chat = ScriptedChat(_reply({1: "Ada Marsh", 2: "Tom", 3: "Ada", 4: "Tom", 5: "Mrs. Marsh", 6: "Tom"}, [
            {"name": "Ada Marsh", "gender": "female", "age": "adult"},
            {"name": "Tom", "gender": "male", "age": "child"},
            {"name": "Mrs. Marsh", "gender": "female", "age": "adult", "aliases": ["their mother"]}]))
        roster, stats = Roster(), {}
        lines, moods = attribute_chapter(self.paragraphs, roster, chat, stats, self.log)
        self.assertEqual(lines, {1: "ada marsh", 2: "tom", 3: "ada marsh", 4: "tom", 5: "marsh", 6: "tom"})
        self.assertEqual({k: c["lines"] for k, c in roster.characters.items()}, {"ada marsh": 2, "tom": 3, "marsh": 1})
        self.assertEqual((stats["windows"], stats["invalid_json"], stats["lines"], stats["unknown_lines"]), (1, 0, 6, 0))
        self.assertIn(PROMPTS["roster_empty"], chat.prompts[0][1]["content"])

    def test_a_broken_reply_is_retried_once_and_the_retry_can_succeed(self):
        chat = ScriptedChat("not json at all", _reply({n: "Ada" for n in range(1, 7)}))
        stats = {}
        lines, moods = attribute_chapter(self.paragraphs, Roster(), chat, stats, self.log)
        self.assertEqual(set(lines.values()), {"ada", "tom"})  # lines 2 and 6 are tagged "said Tom"
        self.assertEqual((stats["invalid_json"], stats.get("invalid_after_retry", 0), len(chat.prompts)), (1, 0, 2))

    def test_two_bad_replies_leave_the_window_unknown(self):
        # Missing ids, then an invented id; the review of the four unknown lines is unusable too.
        chat = ScriptedChat(_reply({1: "Ada"}), _reply({n: "Ada" for n in range(1, 8)}), "no JSON here")
        stats = {}
        lines, moods = attribute_chapter(self.paragraphs, Roster(), chat, stats, self.log)
        # The tagged lines ("said Tom") never depended on the model.
        self.assertEqual(lines, {1: None, 2: "tom", 3: None, 4: None, 5: None, 6: "tom"})
        self.assertEqual((stats["invalid_json"], stats["invalid_after_retry"], stats["unknown_lines"]), (1, 1, 4))
        self.assertEqual((stats["review_requests"], stats["review_lines"], stats["review_unusable"]), (1, 4, 1))

    def test_unknown_speakers_stay_unknown_and_known_names_reach_the_next_prompt(self):
        chat = ScriptedChat(_reply({1: "Ada Marsh", 2: "unknown", 3: "Ada Marsh", 4: "", 5: "narrator", 6: "Tom"}),
                            _reply({4: "unknown", 5: "unknown"}))  # the review can't tell either
        roster = Roster()
        lines, moods = attribute_chapter(self.paragraphs, roster, chat, {}, self.log)
        # Line 2 is tagged "said Tom", so the model's "unknown" for it is not even asked for.
        self.assertEqual(lines, {1: "ada marsh", 2: "tom", 3: "ada marsh", 4: None, 5: None, 6: "tom"})
        next_chat = ScriptedChat(_reply({1: "Tom"}))
        attribute_chapter(chapter_segments('"Again," he said.'), roster, next_chat, {}, self.log)
        self.assertIn("Ada Marsh, Tom", next_chat.prompts[0][1]["content"])

    def test_continued_lines_inherit_the_previous_speaker(self):
        text = f'“First part of the speech.{M}“Second part,” said Ada.{M}“New line,” said Tom.'
        chat = ScriptedChat(_reply({1: "Ada", 3: "Tom"}))
        roster = Roster()
        lines, moods = attribute_chapter(chapter_segments(text), roster, chat, {}, self.log)
        self.assertEqual(lines, {1: "ada", 2: "ada", 3: "tom"})
        self.assertEqual(roster.characters["ada"]["lines"], 2)


class TestReview(unittest.TestCase):
    """The review pass (core.cast_review + review_lines), on the three kinds of error a real book's
    first chapter showed, sanitised (WORKLOG §28)."""

    # A first-person chapter: the narrator's own "I said" lines, one left unknown; a "she said" line
    # given to the narrator; a line after "Ann looked at me." given to the narrator.
    CHAPTER = (f'"Morning," I said.{M}"Coffee?" I said.{M}'
               f'"Damn!" I said. "You mean he was too young?"{M}'
               f'"Yes, at first." Ann smiled.{M}'
               f'I poured her a drink. "So, Dad," she said after a sip, "Sam knows about us."{M}'
               f'"And the best part was the end." Ann looked at me. "Isn\'t that sweet?"')
    FIRST = {1: "Sam", 2: "Sam", 3: "Narrator", 4: "Narrator", 5: "unknown", 6: "Sam", 7: "Ann",
             8: "Ann", 9: "Sam"}
    PEOPLE = [{"name": "Sam", "gender": "male"}, {"name": "Ann", "gender": "female"}]

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def test_flagged_lines_are_asked_again_and_corrected_before_counting(self):
        chat = ScriptedChat(_reply(self.FIRST, self.PEOPLE),
                            _reply({3: "Sam", 4: "Sam", 5: "Ann", 6: "Ann", 7: "Ann", 8: "Ann", 9: "Ann"}))
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(self.CHAPTER), roster, chat, stats, self.log)
        self.assertEqual(lines, {1: "sam", 2: "sam", 3: "sam", 4: "sam", 5: "ann", 6: "ann", 7: "ann",
                                 8: "ann", 9: "ann"})
        self.assertEqual((roster.characters["sam"]["lines"], roster.characters["ann"]["lines"]), (4, 5))
        self.assertEqual((stats["review_requests"], stats["review_changed"], stats["unknown_lines"]), (1, 5, 0))
        prompt = chat.prompts[1][1]["content"]
        self.assertIn('The narrator, who says "I" in the narration, is Sam', prompt)
        self.assertIn('[#6] "So, Dad,"', prompt)
        self.assertIn('[Sam?] "Morning,"', prompt)  # an earlier guess, shown as one
        self.assertIn("Ids to answer: 3, 4, 5, 6, 7, 8, 9", prompt)

    def test_an_unknown_review_answer_keeps_the_first_answer(self):
        chat = ScriptedChat(_reply(self.FIRST, self.PEOPLE),
                            _reply({3: "unknown", 4: "unknown", 5: "unknown", 6: "unknown", 7: "unknown",
                                    8: "unknown", 9: "unknown"}))
        stats = {}
        lines, _ = attribute_chapter(chapter_segments(self.CHAPTER), Roster(), chat, stats, self.log)
        self.assertEqual((lines[6], lines[9], lines[3]), ("sam", "sam", None))
        self.assertEqual(stats["review_changed"], 0)

    def test_an_unnamed_narrator_is_described_not_named_i(self):
        text = f'"Morning," I said.{M}"Coffee?" I said.{M}"Now?" I said.'
        chat = ScriptedChat(_reply({1: "I", 2: "I", 3: "unknown"}), _reply({3: "I"}))
        lines, _ = attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        self.assertIn('is never named: answer "I"', chat.prompts[1][1]["content"])
        self.assertEqual(lines[3], lines[1])

    def test_a_consistent_chapter_asks_nothing_more(self):
        chat = ScriptedChat(_reply({1: "Ada"}))
        stats = {}
        attribute_chapter(chapter_segments('"Hello," Ada smiled.'), Roster(), chat, stats, self.log)
        self.assertEqual((len(chat.prompts), stats.get("review_requests", 0)), (1, 0))


class TestTaggedLines(unittest.TestCase):
    """Lines a speech tag names are not asked; the model sees them, and earlier decisions, as [Name]."""

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def test_tagged_lines_are_shown_as_known_and_not_asked(self):
        windows = build_windows(chapter_segments(CHAPTER), known={2: "Tom", 6: "Tom"})
        self.assertEqual((windows[0].ids, windows[0].anchored), ([1, 3, 4, 5], [2, 6]))
        self.assertIn('[Tom] "I did not,"', windows[0].passage)
        self.assertNotIn("[#2]", windows[0].passage)

    def test_a_question_before_an_introduced_answer_is_asked_not_locked(self):
        # Sanitised from a real book: "X answered ...:" had locked the question to X as well.
        from audiobook_generator.core.speech_tags import tagged_speakers
        paragraphs = chapter_segments('Then I heard Jon\'s voice: "When can we meet?" '
                                      'Ann answered without a pause: "Monday."')
        windows = build_windows(paragraphs, known=tagged_speakers(paragraphs))
        self.assertEqual((windows[0].ids, windows[0].anchored), ([1], [2]))
        self.assertIn('[Ann] "Monday."', windows[0].passage)

    def test_the_model_is_not_asked_about_tagged_lines_and_their_answers_are_ignored(self):
        chat = ScriptedChat(_reply({1: "Ada Marsh", 2: "Somebody Else", 3: "Ada Marsh", 4: "Tom", 5: "Mrs. Marsh",
                                    6: "Somebody Else"}))
        lines, moods = attribute_chapter(chapter_segments(CHAPTER), Roster(), chat, {}, self.log)
        prompt = chat.prompts[0][1]["content"]
        self.assertIn("Ids to answer: 1, 3, 4, 5", prompt)
        self.assertEqual((lines[2], lines[6]), ("tom", "tom"))

    def test_a_tag_name_joins_the_character_the_model_gave_that_alias(self):
        text = f'"Supper is ready," said Mother.{M}"Coming," said Ada.{M}"Wash your hands first."'
        chat = ScriptedChat(_reply({3: "Mrs. Marsh"}, [
            {"name": "Mrs. Marsh", "gender": "female", "age": "adult", "aliases": ["Mother"]}]))
        roster = Roster()
        lines, moods = attribute_chapter(chapter_segments(text), roster, chat, {}, self.log)
        self.assertEqual(lines[1], lines[3])
        self.assertEqual(roster.characters[lines[1]]["name"], "Mrs. Marsh")

    def test_later_windows_see_earlier_decisions_as_context(self):
        exchange = [f'"Line {n}."' for n in range(1, 31)]
        text = M.join(exchange)
        first = _reply({n: ("Ada" if n % 2 else "Tom") for n in range(1, 21)})
        second = _reply({n: ("Ada" if n % 2 else "Tom") for n in range(21, 31)})
        chat = ScriptedChat(first, second)
        attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        second_prompt = chat.prompts[1][1]["content"]
        self.assertIn('[Tom] "Line 20."', second_prompt)
        self.assertIn('[Ada] "Line 19."', second_prompt)
        self.assertIn('[#21] "Line 21."', second_prompt)

    def test_a_chapter_where_every_line_is_tagged_needs_no_request(self):
        text = f'"Hello," said Ada.{M}"Hello yourself," Tom said.'
        chat = ScriptedChat()
        stats = {}
        lines, moods = attribute_chapter(chapter_segments(text), Roster(), chat, stats, self.log)
        self.assertEqual((lines, chat.prompts, stats["tagged_lines"]), ({1: "ada", 2: "tom"}, [], 2))


class TestAttributeChapterMoods(unittest.TestCase):
    """Moods combine core.delivery's rule cues with the LLM's own guess: a rule cue always wins."""

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def test_moods_are_not_requested_in_the_prompt_by_default(self):
        # ASK_LLM_FOR_MOODS is off by default: measured 2026-09-28 (WORKLOG #14) to cost speaker
        # accuracy (221/270 -> 198/270 on the labelled fixture), well past the 219/270 floor.
        self.assertFalse(cast_llm_module.ASK_LLM_FOR_MOODS)
        chat = ScriptedChat(_reply({1: "Ada Marsh", 2: "Tom", 3: "Ada", 4: "Tom", 5: "Mrs. Marsh", 6: "Tom"}))
        attribute_chapter(chapter_segments(CHAPTER), Roster(), chat, {}, self.log)
        self.assertNotIn('"moods"', chat.prompts[0][1]["content"])

    def test_moods_are_requested_when_the_module_constant_is_switched_on(self):
        chat = ScriptedChat(_reply({1: "Ada Marsh", 2: "Tom", 3: "Ada", 4: "Tom", 5: "Mrs. Marsh", 6: "Tom"}))
        with patch.object(cast_llm_module, "ASK_LLM_FOR_MOODS", True):
            attribute_chapter(chapter_segments(CHAPTER), Roster(), chat, {}, self.log)
        self.assertIn('"moods"', chat.prompts[0][1]["content"])

    @patch.object(cast_llm_module, "ASK_LLM_FOR_MOODS", True)
    def test_llm_moods_flow_through_when_rules_are_silent(self):
        chat = ScriptedChat(_reply({1: "Ada Marsh", 2: "Tom", 3: "Ada", 4: "Tom", 5: "Mrs. Marsh", 6: "Tom"},
                                   moods={1: "excited", 3: "soft", 4: "not-a-mood"}))
        _, moods = attribute_chapter(chapter_segments(CHAPTER), Roster(), chat, {}, self.log)
        # CHAPTER has no rule cues anywhere, so every asked line's mood is the LLM's guess (an
        # invalid value defaults to normal); lines 2 and 6 are tagged ("said Tom") and never asked.
        self.assertEqual(moods, {1: "excited", 2: "normal", 3: "soft", 4: "normal", 5: "normal", 6: "normal"})

    @patch.object(cast_llm_module, "ASK_LLM_FOR_MOODS", True)
    def test_a_rule_cue_overrides_the_llms_mood(self):
        text = f'"Go now," she whispered.{M}"Fine."'
        chat = ScriptedChat(_reply({1: "Ada", 2: "Ada"}, moods={1: "excited", 2: "excited"}))
        _, moods = attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        # Line 1 has a rule cue (whispered): soft wins over the LLM's "excited". Line 2 has none,
        # so the LLM's guess is used.
        self.assertEqual(moods, {1: "soft", 2: "excited"})

    def test_lines_never_asked_get_rule_moods_only(self):
        text = f'"Go now," Ada whispered.{M}"Get out!" Tom shouted.'  # both tagged: neither is asked
        chat = ScriptedChat()
        _, moods = attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        self.assertEqual(chat.prompts, [])
        self.assertEqual(moods, {1: "soft", 2: "excited"})

    def test_continued_lines_inherit_the_previous_lines_final_combined_mood(self):
        text = f'She whispered, "First part.{M}"Second part," he said.'
        chat = ScriptedChat(_reply({1: "Ada"}, moods={1: "excited"}))  # the rule cue overrides this for line 1
        _, moods = attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        self.assertEqual(moods, {1: "soft", 2: "soft"})


class TestMoodsStayRulesOnly(unittest.TestCase):

    def test_moods_a_reply_volunteers_are_ignored_while_llm_moods_are_off(self):
        import audiobook_generator.core.cast_llm as cast_llm_module
        self.assertFalse(cast_llm_module.ASK_LLM_FOR_MOODS)
        reply = {"speakers": {"1": "Ada"}, "moods": {"1": "excited"}, "characters": []}
        _, moods = attribute_chapter(chapter_segments('"Then who did?"'), Roster(), ScriptedChat(reply), {},
                                     logging.getLogger("test-cast"))
        self.assertEqual(moods, {1: "normal"})

