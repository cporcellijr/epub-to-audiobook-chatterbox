"""Speaker attribution: windows, strict reply parsing, alias merging and the retry-then-unknown rule,
all with a scripted stand-in for the chat endpoint."""
import json
import logging
import unittest
from unittest.mock import patch

from audiobook_generator.core import cast_llm as cast_llm_module, cast_review
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

    def test_repeated_conflicting_genders_reuse_the_compatible_character(self):
        for name in ("Alex", "Alex Baker", "Dr. Alex"):
            with self.subTest(name=name):
                roster = Roster()
                first = roster.add(name, "male")
                second = roster.add(name, "female")
                self.assertNotEqual(first, second)
                for _ in range(3):
                    self.assertEqual(roster.add(name, "female"), second)
                    self.assertEqual(roster.add(name, "male"), first)
                self.assertEqual(len(roster.characters), 2)
                saved = Roster(roster.characters)
                self.assertEqual(saved.add(name, "female"), second)
                self.assertEqual(saved.add(name, "male"), first)

    def test_explicit_aliases_from_the_model_are_recorded(self):
        roster = Roster()
        key = roster.add("Margaret Hale", "female", aliases=["Mother", "Mrs. Hale"])
        self.assertEqual(roster.add("Mother"), key)
        self.assertEqual(roster.add("Mrs Hale"), key)

    def test_possessive_relationship_name_stays_distinct_from_its_owner(self):
        for label in ("Jimmy's companion", "Jimmy’s companion"):
            for companion_first in (True, False):
                with self.subTest(label=label, companion_first=companion_first):
                    roster = Roster()
                    if companion_first:
                        companion = roster.add(label, "male", aliases=["Jimmy"])
                        jimmy = roster.add("Jimmy", "male")
                    else:
                        jimmy = roster.add("Jimmy", "male")
                        companion = roster.add(label, "male", aliases=["Jimmy"])
                    self.assertNotEqual(companion, jimmy)
                    self.assertEqual(roster.add("Jimmy"), jimmy)
                    self.assertEqual(roster.add(label), companion)

            companion_key = cast_llm_module.normalize_name(label)
            restored = Roster({companion_key: {
                "name": label, "aliases": ["Jimmy"], "gender": "male", "lines": 1
            }})
            self.assertIsNone(restored.resolve(label))  # relationship labels are chapter-local
            companion = restored.add(label, "male")
            jimmy = restored.add("Jimmy", "male")
            self.assertNotEqual(companion, companion_key)  # a fresh chapter-local character
            self.assertNotEqual(jimmy, companion)
            self.assertNotEqual(jimmy, companion_key)
            self.assertEqual(restored.aliases["jimmy"], jimmy)

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

    def test_narrator_references_are_scoped_to_the_chapter(self):
        roster = Roster()
        polly = roster.add("Polly", "female", aliases=["I"])
        roster.set_chapter_narrator(polly, ["Polly Mary"])
        self.assertEqual([roster.add(name) for name in ("I", "narrator", "the narrator", "Polly Mary")],
                         [polly] * 4)
        roster.new_chapter()
        jane = roster.add("Jane", "female")
        roster.set_chapter_narrator(jane)
        self.assertEqual(roster.add("I"), jane)
        self.assertEqual(roster.add("the narrator"), jane)
        self.assertNotEqual(jane, polly)

    def test_unnamed_narrators_do_not_merge_across_chapters(self):
        roster = Roster()
        first = roster.add("I")
        self.assertEqual(roster.add("the narrator"), first)
        roster.new_chapter()
        second = roster.add("narrator")
        self.assertNotEqual(first, second)
        self.assertEqual(roster.names_for_prompt(), ["The Narrator"])

    def test_current_reply_can_bind_the_scoped_narrator_to_a_named_character(self):
        roster = Roster()
        anonymous = roster.add("I")
        polly = roster.add("Polly", "female", aliases=["I"])
        self.assertEqual(roster.chapter_narrator, polly)
        self.assertEqual(roster.canonical_key(anonymous), polly)
        self.assertFalse(roster.is_scoped_narrator(polly))

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

    def test_narrator_labels_are_valid_scoped_references(self):
        reply = json.dumps({"speakers": {"1": "narrator", "2": "the narrator", "3": "I"}})
        speakers, _, _ = parse_reply(reply, [1, 2, 3])
        self.assertEqual(speakers, {1: "narrator", 2: "the narrator", 3: "I"})
        roster = Roster()
        self.assertEqual({roster.add(name) for name in speakers.values()}, {"narrator"})

    def test_a_saved_cast_restores_the_roster(self):
        roster = Roster({"thomas baker": {"name": "Thomas Baker", "aliases": ["Tom"], "gender": "male",
                                          "age": "adult", "lines": 3}})
        self.assertEqual(roster.add("Tom"), "thomas baker")


class TestConservativeMerging(unittest.TestCase):
    """A wrong merge gives a whole character the wrong voice (WORKLOG §29)."""

    def test_a_gendered_title_keeps_two_people_with_one_surname_apart(self):
        for genders in (("male", "female"), ("unknown", "unknown")):
            with self.subTest(genders=genders):
                roster = Roster()
                mr, mrs = roster.add("Mr. Smith", genders[0]), roster.add("Mrs. Smith", genders[1])
                self.assertEqual((mr, mrs), ("smith", "mrs smith"))
                self.assertEqual((roster.add("Mrs. Smith"), roster.add("Mr. Smith")), (mrs, mr))
                self.assertEqual((roster.characters[mr]["gender"], roster.characters[mrs]["gender"]),
                                 ("male", "female"))

    def test_the_other_order_keys_the_second_with_its_title(self):
        roster = Roster()
        self.assertEqual([roster.add(n) for n in ("Mrs. Smith", "Mr. Smith", "Mr. Smith")],
                         ["smith", "mr smith", "mr smith"])

    def test_a_known_gender_is_checked_on_an_exact_name_too(self):
        roster = Roster()
        roster.add("Sam", "male")
        self.assertNotEqual(roster.add("Sam", "female"), "sam")
        self.assertEqual(roster.add("Sam", "male"), "sam")

    def test_a_name_one_letter_longer_is_another_name(self):
        for short, long in (("Ann", "Anna"), ("Paul", "Paula"), ("Carl", "Carla"), ("Dan", "Dana")):
            roster = Roster()
            self.assertNotEqual(roster.add(short), roster.add(long), (short, long))
        for short, long in (("Ben", "Benjamin"), ("Chris", "Christopher"), ("Ann", "Annie"), ("Tom", "Thomas")):
            roster = Roster()
            self.assertEqual(roster.add(short), roster.add(long), (short, long))

    def test_a_family_word_alias_still_resolves(self):
        roster = Roster()
        key = roster.add("Mrs. Marsh", "female", aliases=["Mother"])
        self.assertEqual(roster.add("Mother"), key)


class TestOnePersonPerName(unittest.TestCase):
    """Names that aren't one person, and one person under two names (WORKLOG §30)."""

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def test_a_pair_answered_as_one_speaker_is_its_first_person(self):
        reply = _reply({1: "Jonathon and Jess"}, [{"name": "Jonathon and Jess", "gender": "male",
                                                   "aliases": ["Jonathon", "Jon & Jess"]}])
        speakers, characters, _ = parse_reply(json.dumps(reply), [1])
        self.assertEqual((speakers[1], characters[0]["name"], characters[0]["aliases"]),
                         ("Jonathon", "Jonathon", ["Jonathon"]))

    TEXT = (f'"Good morning," said Dr. Hale. "Please call me Lena."{M}"Thank you, Lena."{M}'
            '"You are welcome," Lena said.')

    def test_a_speaker_introducing_themselves_absorbs_that_name(self):
        chat = ScriptedChat(_reply({3: "Nora"}))
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(self.TEXT), roster, chat, stats, self.log)
        self.assertEqual(lines[4], lines[1])
        self.assertNotIn("lena", roster.characters)
        self.assertIn("Lena", roster.characters[lines[1]]["aliases"])
        self.assertEqual((stats["merged_introductions"], roster.characters[lines[1]]["lines"]), (1, 3))

    # Seen live: the model gave the introduction to the new name itself, splitting one doctor in two.
    GIVEN_TO_NEW_NAME = (f'"Good morning," said Dr. Hale.{M}"Morning," said Nora.{M}'
                         f'"I am glad. And please call me Lena."{M}"Thank you, Lena," said Nora.{M}'
                         '"You are welcome," Lena said.')

    def test_an_introduction_given_to_the_new_name_asks_who_it_belongs_to(self):
        chat = ScriptedChat(_reply({3: "Lena"}), {"same_as": "Dr. Hale"})
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(self.GIVEN_TO_NEW_NAME), roster, chat, stats, self.log)
        self.assertEqual((lines[3], lines[5]), (lines[1], lines[1]))
        self.assertNotIn("lena", roster.characters)
        self.assertEqual((stats["identity_questions"], stats["merged_introductions"]), (1, 1))
        self.assertIn("one of these characters who spoke just before: Dr. Hale, Nora", chat.prompts[1][1]["content"])

    def test_a_newcomer_introducing_themselves_stays_new(self):
        chat = ScriptedChat(_reply({3: "Lena"}), {"same_as": "new"})
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(self.GIVEN_TO_NEW_NAME), roster, chat, stats, self.log)
        self.assertEqual(lines[3], lines[5])
        self.assertNotEqual(lines[3], lines[1])
        self.assertNotIn("merged_introductions", stats)

    def test_someone_met_in_an_earlier_chapter_is_not_merged(self):
        roster = Roster()
        roster.add("Lena", "female")
        roster.count_line("lena")
        lines, _ = attribute_chapter(chapter_segments(self.TEXT), roster, ScriptedChat(_reply({3: "Nora"})), {},
                                     self.log)
        self.assertEqual(roster.characters[lines[4]]["name"], "Lena")

    def test_a_denied_name_is_no_introduction_and_a_new_one_becomes_an_alias(self):
        text = f'"Don\'t call me Rex," said Anna.{M}"Fine," Rex said.{M}"Call me Annie," said Anna.'
        roster = Roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(), {}, self.log)
        self.assertNotEqual(lines[1], lines[2])
        self.assertEqual(roster.resolve("Annie"), lines[1])

    def test_offered_names_are_introductions_too(self):
        # Seen in a real book: "You can call me <name>." (§33)
        for speech in ("You can call me Lena.", "Just call me Lena.", "Please, call me Lena.",
                       "Please just call me Lena."):
            with self.subTest(speech=speech):
                text = self.TEXT.replace("Please call me Lena.", speech)
                roster, stats = Roster(), {}
                lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(_reply({3: "Nora"})),
                                             stats, self.log)
                self.assertEqual(lines[4], lines[1])
                self.assertNotIn("lena", roster.characters)
                self.assertEqual(stats["merged_introductions"], 1)

    def test_denied_or_reported_introductions_do_not_merge_people(self):
        for speech in ("Don't ever call me Beth.", "Don't you dare call me Beth.",
                       "You can't call me Beth.", "Don't just call me Beth.",
                       "Yesterday John told me, 'Call me Beth.'",
                       "John said, ‘Hello. My name is Beth.’", "'Call me Beth,' John said.",
                       "He said my name is Beth."):
            with self.subTest(speech=speech):
                text = f'Beth said, "Hello."{M}Ann said, "{speech}"'
                roster, stats, chat = Roster(), {}, ScriptedChat()
                lines, _ = attribute_chapter(chapter_segments(text), roster, chat, stats, self.log)
                self.assertEqual(lines, {1: "beth", 2: "ann"})
                self.assertEqual(set(roster.characters), {"beth", "ann"})
                self.assertNotIn("merged_introductions", stats)
                self.assertEqual(chat.prompts, [])


class TestLocalReferences(unittest.TestCase):
    """Descriptions ("the young woman") name someone only within their chapter."""

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def test_a_declared_description_alias_is_the_named_character_in_its_chapter_only(self):
        text = f'"Wait for me," said the young woman.{M}"Hurry," the young woman called.'
        reply = _reply({1: "young woman", 2: "young woman"},
                       [{"name": "Mara", "gender": "female", "aliases": ["young woman"]}])
        roster = Roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(reply), {}, self.log)
        self.assertEqual(lines, {1: "mara", 2: "mara"})
        self.assertEqual(set(roster.characters), {"mara"})
        later, _ = attribute_chapter(chapter_segments('"Please," said the young woman.'), roster,
                                     ScriptedChat(_reply({1: "young woman"})), {}, self.log)
        self.assertNotEqual(later[1], "mara")
        self.assertEqual(roster.characters["mara"]["lines"], 2)

    def test_the_same_description_in_two_chapters_is_two_characters(self):
        roster = Roster()
        first = roster.add("one of the guys", "male")
        self.assertEqual(roster.add("One of the guys"), first)
        roster.new_chapter()
        self.assertNotEqual(roster.add("one of the guys", "male"), first)

    def test_a_description_never_absorbs_a_proper_name(self):
        roster = Roster()
        man = roster.add("the man", "male")
        ray = roster.add("Man Ray", "male")  # same first word as the description, still not it
        self.assertNotEqual(ray, man)
        self.assertEqual(roster._candidates("man"), [ray])
        self.assertEqual(roster.add("the man"), man)

    # Seen live (WORKLOG §64): the model declared "the girl" with the alias "Gemma"; she stayed
    # "the girl", and the label went on to take another girl's and the narrator's lines.
    def test_a_description_declared_with_a_new_name_becomes_that_person(self):
        text = f'"Hi," said Polly.{M}"Hello yourself," the girl answered.{M}"Again?"'
        reply = _reply({2: "the girl", 3: "the girl"},
                       [{"name": "the girl", "gender": "female", "aliases": ["Gemma"]}])
        roster = Roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(reply), {}, self.log)
        gemma = lines[2]
        self.assertEqual((lines[3], roster.characters[gemma]["name"]), (gemma, "Gemma"))
        self.assertNotIn("reference_scope", roster.characters[gemma])
        roster.new_chapter()
        self.assertEqual(roster.resolve("Gemma"), gemma)
        self.assertIn("Gemma", roster.names_for_prompt())

    def test_a_phrase_declared_as_a_name_does_not_rename_a_description(self):
        # Seen live (WORKLOG §65): "Miss me" became a character, then a chapter's narrator.
        roster = Roster()
        girl = roster.add("Girl", "female", aliases=["Miss me", "Don't be late Jimmy"])
        self.assertEqual((roster.characters[girl]["name"], roster.characters[girl].get("reference_scope")),
                         ("Girl", "chapter"))

    def test_a_description_declared_as_the_i_is_left_to_the_narrator_choice(self):
        roster = Roster()
        girl = roster.add("the girl", "female", aliases=["I", "Gemma"])
        self.assertEqual((roster.characters[girl]["name"], roster.characters[girl].get("reference_scope")),
                         ("the girl", "chapter"))

    def test_a_description_declared_with_someone_elses_name_stays_apart(self):
        roster = Roster()
        gemma = roster.add("Gemma", "female")
        roster.new_chapter()
        girl = roster.add("the girl", "female", aliases=["Gemma"])
        self.assertNotEqual(girl, gemma)
        self.assertEqual((roster.characters[girl]["name"], roster.characters[girl].get("reference_scope")),
                         ("the girl", "chapter"))
        self.assertEqual(roster.characters[gemma]["name"], "Gemma")


class TestInventedNames(unittest.TestCase):
    """Names the model makes up rather than reads (WORKLOG §67)."""

    TEXT = M.join(['"Status?" asked Dr. Joanna Glass.', '"Holding," Captain Katrina de la Cruz said.'])

    def _roster(self, text=TEXT):
        roster = Roster()
        roster.read(chapter_segments(text))
        joanna = roster.add("Dr. Joanna Glass", "female")
        katrina = roster.add("Katrina de la Cruz", "female")
        return roster, joanna, katrina

    def test_a_spliced_name_the_book_never_writes_is_the_first_names_owner(self):
        # Seen live: the doctor's lines under "Joanna de la Cruz", a second voice for one person.
        roster, joanna, katrina = self._roster()
        self.assertEqual(roster.add("Joanna de la Cruz", "female"), joanna)
        self.assertEqual(roster.speaker_key("Joanna de la Cruz"), joanna)
        self.assertEqual((set(roster.characters), roster.characters[joanna]["name"]),
                         ({joanna, katrina}, "Dr. Joanna Glass"))

    def test_a_spliced_name_the_book_writes_is_a_new_person(self):
        roster, joanna, katrina = self._roster(self.TEXT + f'{M}"Joanna de la Cruz, my sister," Katrina said.')
        self.assertNotIn(roster.add("Joanna de la Cruz", "female"), (joanna, katrina))

    def test_an_invented_full_name_does_not_rename_a_first_name(self):
        roster = Roster()
        roster.read(chapter_segments('"Here," Joanna said.' + M + '"Good," said Katrina de la Cruz.'))
        joanna = roster.add("Joanna", "female")
        roster.add("Katrina de la Cruz", "female")
        self.assertEqual(roster.add("Joanna de la Cruz", "female"), joanna)
        self.assertEqual(roster.characters[joanna]["name"], "Joanna")

    def test_without_the_books_text_names_are_taken_as_given(self):
        roster = Roster()
        joanna = roster.add("Dr. Joanna Glass", "female")
        roster.add("Katrina de la Cruz", "female")
        self.assertNotEqual(roster.add("Joanna de la Cruz", "female"), joanna)

    def test_a_guidance_example_alias_the_book_never_writes_is_dropped(self):
        roster = Roster()
        roster.read(chapter_segments('"Soup," said Maria.'))
        maria = roster.add("Maria Arena", "female", aliases=["Squirrelly"])
        self.assertEqual(roster.characters[maria]["aliases"], [])
        roster.read(chapter_segments('"Squirrelly, come here," Maria\'s mother called.'))
        roster.add("Maria Arena", "female", aliases=["Squirrelly"])
        self.assertEqual(roster.characters[maria]["aliases"], ["Squirrelly"])

    def test_the_guarded_example_names_are_the_guidances_own(self):
        guidance = " ".join(PROMPTS.values()).lower()
        for name in cast_llm_module.PROMPT_EXAMPLE_NAMES:
            self.assertIn(name, guidance)


class TestLaterIntroductions(unittest.TestCase):
    """A name given in answer to "what's your name?" or "I'm X", after the speaker spoke as a description."""

    def test_a_stutter_repeats_the_names_start_and_a_hyphenated_name_stays_whole(self):
        bare = cast_llm_module._bare_name
        self.assertEqual([bare(t) for t in ("D-D-Dex,", "T-Tom.", "Jo-Ann", "Dex!", "G-G-Dex", "dex")],
                         ["Dex", "Tom", "Jo-Ann", "Dex", None, None])

    SCENE = (f'"Look, man, I didn\'t agree to this," the man stutters to Rob.{M}'
             f'The corridor smelled of old rain and wet wool.{M}'
             f'"You first, tell me your name," I say to the mystery man.{M}'
             '"D-D-Dex," he stutters.')
    PEOPLE = [{"name": "Rob", "gender": "male"}, {"name": "Rob's companion", "gender": "male"}]

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def _earlier_chapter(self, roster):
        roster.add("Rob", "male")
        roster.add("one of the guys", "male")
        roster.new_chapter()

    def test_a_description_learns_its_name_from_the_answer_to_a_name_question(self):
        roster, stats = Roster(), {}
        self._earlier_chapter(roster)
        guy = next(k for k, c in roster.characters.items() if c["name"] == "one of the guys")
        chat = ScriptedChat(_reply({1: "Rob's companion", 3: "Rob's companion"}, self.PEOPLE))
        lines, _ = attribute_chapter(chapter_segments(self.SCENE), roster, chat, stats, self.log)
        key = lines[1]
        self.assertEqual(lines[3], key)
        self.assertEqual(roster.characters[key]["name"], "Dex")
        self.assertNotIn("reference_scope", roster.characters[key])
        self.assertEqual(roster.aliases["dex"], key)
        self.assertNotEqual(key, guy)
        self.assertEqual(roster.characters[guy]["name"], "one of the guys")
        roster.new_chapter()
        self.assertEqual(roster.resolve("Dex"), key)
        self.assertIsNone(roster.resolve("Rob's companion"))

    def test_a_reply_given_to_the_name_asks_whether_it_is_the_earlier_description(self):
        roster, stats = Roster(), {}
        self._earlier_chapter(roster)
        chat = ScriptedChat(_reply({1: "Rob's companion", 3: "Dex"}, self.PEOPLE + [{"name": "Dex", "gender": "male"}]),
                            {"same_as": "Rob's companion"})
        lines, _ = attribute_chapter(chapter_segments(self.SCENE), roster, chat, stats, self.log)
        self.assertEqual(lines[1], lines[3])
        self.assertEqual(roster.characters[lines[1]]["name"], "Dex")
        self.assertEqual((stats["identity_questions"], stats["merged_introductions"]), (1, 1))
        self.assertNotIn("dex", [k for k in roster.characters if k != lines[1]])

    def test_a_bare_name_that_answers_nothing_is_not_an_introduction(self):
        text = (f'"Look, man, I didn\'t agree to this," the man stutters to Rob.{M}'
                f'"Hey, over here," I say to the mystery man.{M}'
                '"Dex!" he shouts.')
        roster, stats = Roster(), {}
        chat = ScriptedChat(_reply({1: "Rob's companion", 3: "Rob's companion"}, self.PEOPLE))
        lines, _ = attribute_chapter(chapter_segments(text), roster, chat, stats, self.log)
        self.assertEqual(roster.characters[lines[1]]["name"], "Rob's companion")
        self.assertNotIn("dex", roster.aliases)

    def test_i_am_with_a_name_introduces_the_speaker(self):
        text = f'"I\'m Dex," the man says.{M}"Nice to meet you," said Rob.'
        roster = Roster()
        chat = ScriptedChat(_reply({1: "the man", 2: "Rob"}, [{"name": "Rob", "gender": "male"}]))
        lines, _ = attribute_chapter(chapter_segments(text), roster, chat, {}, self.log)
        self.assertEqual(roster.characters[lines[1]]["name"], "Dex")

    def test_an_unnamed_i_who_gives_a_known_name_becomes_that_character(self):
        # Seen replaying a real book: the narrator's "Call me ..." line was tagged "I say", so it went
        # to the chapter's unnamed "I", and the named narrator was folded into "The Narrator".
        text = f'"Hello," Wren said.{M}"Call me Wren," I say.{M}"Fine," said Rob.'
        roster = Roster()
        chat = ScriptedChat(_reply({1: "Wren", 3: "Rob"}, [{"name": "Wren", "gender": "female"},
                                                           {"name": "Rob", "gender": "male"}]))
        lines, _ = attribute_chapter(chapter_segments(text), roster, chat, {}, self.log)
        self.assertEqual(lines[2], lines[1])
        self.assertEqual(roster.characters[lines[2]]["name"], "Wren")
        self.assertNotIn("The Narrator", [c["name"] for c in roster.characters.values()])
        self.assertEqual(roster.add("I"), lines[1])  # this chapter's "I" is Wren now

    def test_an_unnamed_i_who_gives_a_new_name_takes_it(self):
        text = f'"Call me Wren," I say.{M}"Fine," said Rob.'
        roster = Roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(), {}, self.log)
        self.assertEqual(roster.characters[lines[1]]["name"], "Wren")
        self.assertFalse(roster.is_scoped_narrator(lines[1]))


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
        self.assertEqual(lines, {1: "ada marsh", 2: "tom", 3: "ada marsh", 4: None, 5: "narrator", 6: "tom"})
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
                            _reply({5: "Ann", 6: "Ann", 7: "Ann", 8: "Ann", 9: "Ann"}))
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(self.CHAPTER), roster, chat, stats, self.log,
                                     narrator="Sam")
        self.assertEqual(lines, {1: "sam", 2: "sam", 3: "sam", 4: "sam", 5: "ann", 6: "ann", 7: "ann",
                                 8: "ann", 9: "ann"})
        self.assertEqual((roster.characters["sam"]["lines"], roster.characters["ann"]["lines"]), (4, 5))
        self.assertEqual((stats["review_requests"], stats["review_changed"], stats["unknown_lines"]), (1, 3, 0))
        prompt = chat.prompts[1][1]["content"]
        self.assertIn('In this chapter, the narrator is Sam', prompt)
        self.assertIn('[#6] "So, Dad,"', prompt)
        self.assertIn('[Sam] "Morning,"', prompt)  # explicit first-person tag, not a guess
        self.assertIn("Ids to answer: 5, 6, 7, 8, 9", prompt)

    def test_an_unknown_review_answer_keeps_the_first_answer(self):
        chat = ScriptedChat(_reply(self.FIRST, self.PEOPLE),
                            _reply({3: "unknown", 4: "unknown", 5: "unknown", 6: "unknown", 7: "unknown",
                                    8: "unknown", 9: "unknown"}))
        stats = {}
        lines, _ = attribute_chapter(chapter_segments(self.CHAPTER), Roster(), chat, stats, self.log,
                                     narrator="Sam")
        self.assertEqual((lines[6], lines[9], lines[3]), ("sam", "sam", "sam"))
        self.assertEqual(stats["review_changed"], 0)

    def test_an_unnamed_narrator_is_a_chapter_scoped_identity(self):
        text = f'"Morning," I said.{M}"Coffee?" I said.{M}"Now?" I said.'
        chat = ScriptedChat()
        lines, _ = attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        self.assertEqual(lines[3], lines[1])
        self.assertEqual(lines[2], lines[1])
        self.assertEqual(chat.prompts, [])

    def test_current_reply_alias_i_guides_the_next_attribution_window(self):
        text = f'"Intro," I said.{M}' + M.join(f'"Line {i}?"' for i in range(2, 26))
        first = _reply({i: "Polly" for i in range(2, 22)},
                       [{"name": "Polly", "gender": "female", "aliases": ["I"]}])
        second = _reply({i: "Polly" for i in range(22, 26)})
        chat = ScriptedChat(first, second)
        lines, _ = attribute_chapter(chapter_segments(text), Roster(), chat, {}, self.log)
        self.assertEqual(set(lines.values()), {"polly"})
        self.assertIn("the narrator is Polly", chat.prompts[1][1]["content"])

    def test_a_consistent_chapter_asks_nothing_more(self):
        chat = ScriptedChat(_reply({1: "Ada"}))
        stats = {}
        attribute_chapter(chapter_segments('"Hello," Ada smiled.'), Roster(), chat, stats, self.log)
        self.assertEqual((len(chat.prompts), stats.get("review_requests", 0)), (1, 0))

    def test_a_review_reply_naming_the_i_moves_the_narrators_lines_before_the_next_group(self):
        # Seen live (WORKLOG §60): the first review group's reply named the unnamed "I", which merged
        # the chapter's narrator key away while the "I said" lines still held it; the next group
        # looked the old key up and the whole analysis failed.
        text = M.join(['"Morning," I said.', '"Hello there."', "The rain went on.", "Nothing moved.",
                       "The clock ticked.", "Somebody coughed.", '"Who is it?"', '"Me," I said.'])
        chat = ScriptedChat(_reply({2: "unknown", 3: "unknown"}),
                            _reply({2: "William"}, [{"name": "William", "gender": "male", "aliases": ["I"]}]),
                            _reply({3: "Ann"}, [{"name": "Ann", "gender": "female"}]))
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(text), roster, chat, stats, self.log)
        self.assertEqual(lines, {1: "william", 2: "william", 3: "ann", 4: "william"})
        self.assertEqual(stats["review_requests"], 2)
        self.assertIn('[William] "Me,"', chat.prompts[2][1]["content"])
        self.assertEqual(roster.characters["william"]["lines"], 3)


class TestReviewIsAccountable(unittest.TestCase):
    """A review answer the line's own text rules out never replaces the first answer."""

    TEXT = f'"No! Do not!" he cries out.{M}"Quiet," said Ada.'
    PEOPLE = [{"name": "Ada", "gender": "female"}, {"name": "Ben", "gender": "male"}]

    def setUp(self):
        self.log = logging.getLogger("test-cast")

    def _run(self, first, second=None):
        chat = ScriptedChat(*[_reply(r, self.PEOPLE) for r in (first, second) if r])
        roster, stats = Roster(), {}
        paragraphs = chapter_segments(self.TEXT)
        lines, _ = attribute_chapter(paragraphs, roster, chat, stats, self.log)
        return paragraphs, roster, stats, lines, chat

    def test_a_review_answer_against_the_pronoun_tag_is_rejected(self):
        paragraphs, roster, stats, lines, _ = self._run({1: "Ada"}, {1: "Ada"})
        self.assertEqual((lines[1], stats["review_rejected"], stats["review_changed"]), ("ada", 1, 0))
        flags = cast_review.flag_lines(paragraphs, lines, {}, roster.characters)
        self.assertIn("pronoun gender", flags[1])

    def test_a_review_answer_matching_the_pronoun_tag_is_accepted(self):
        _, roster, stats, lines, _ = self._run({1: "Ada"}, {1: "Ben"})
        self.assertEqual((lines[1], stats.get("review_rejected", 0), stats["review_changed"]), ("ben", 0, 1))

    def test_a_correct_first_answer_is_never_asked_again(self):
        _, _, stats, lines, chat = self._run({1: "Ben"})
        self.assertEqual((lines[1], len(chat.prompts), stats.get("review_requests", 0)), ("ben", 1, 0))

    def test_i_said_is_not_given_to_a_review_answer_other_than_the_narrator(self):
        text = f'"Morning," I said.{M}"Coffee?" I said.{M}"Sure," said Ada.{M}"Fine," I said.'
        first = {3: "Ada"}
        chat = ScriptedChat(_reply(first, self.PEOPLE))
        roster, stats = Roster(), {}
        lines, _ = attribute_chapter(chapter_segments(text), roster, chat, stats, self.log, narrator="Ben")
        self.assertEqual(set(lines[i] for i in (1, 2, 4)), {"ben"})
        paragraphs = chapter_segments(text)
        self.assertEqual(cast_review.hard_violation(paragraphs, 2, "ada", roster.characters, "ben"),
                         "I-tag not the narrator")
        self.assertIsNone(cast_review.hard_violation(paragraphs, 2, "ben", roster.characters, "ben"))


class TestNamesFromTheModel(unittest.TestCase):
    """Qualifiers, nicknames and bare surnames in the names a model or a tag gives."""

    def test_a_parenthetical_qualifier_is_not_part_of_the_name(self):
        reply = json.dumps(_reply({1: "Kenjiro Mori (the clone)", 2: "Wren [younger]", 3: "(unknown)",
                                   4: "Kenjiro Mori, Ninth of the line"}, [
            {"name": "Kenjiro Mori (the clone)", "gender": "male", "age": "adult", "aliases": ["Ken (the clone)"]}]))
        speakers, characters, _ = parse_reply(reply, [1, 2, 3, 4])
        self.assertEqual(speakers, {1: "Kenjiro Mori", 2: "Wren", 3: None, 4: "Kenjiro Mori"})
        self.assertEqual((characters[0]["name"], characters[0]["aliases"]), ("Kenjiro Mori", ["Ken"]))
        roster = Roster()
        for character in characters:
            roster.add(character["name"], character["gender"], character["age"], character["aliases"])
        self.assertEqual(roster.add(speakers[1]), roster.add("Kenjiro Mori"))
        self.assertEqual(len(roster.characters), 1)

    def test_a_nickname_that_ends_the_first_name_is_the_same_person_in_either_order(self):
        roster = Roster()
        key = roster.add("Kenjiro Mori", "male")
        self.assertEqual(roster.add("Jiro"), key)
        later = Roster()
        key = later.add("Jiro")
        self.assertEqual(later.add("Kenjiro Mori"), key)
        self.assertEqual(later.add("Drew"), later.add("Andrew"))

    def test_an_ending_is_no_nickname_when_short_ambiguous_or_of_another_gender(self):
        short = Roster()
        short.add("Joann Pike")
        self.assertNotEqual(short.add("Ann"), "joann pike")
        self.assertEqual(len(short.characters), 2)
        twice = Roster()
        twice.add("Kenjiro Mori")
        twice.add("Shinjiro Ito")
        twice.add("Jiro")
        self.assertEqual(len(twice.characters), 3)
        gendered = Roster()
        gendered.add("Kenjiro Mori", "male")
        gendered.add("Jiro", "female")
        self.assertEqual(len(gendered.characters), 2)

    def test_a_bare_surname_joins_the_one_character_with_that_last_name(self):
        roster = Roster()
        key = roster.add("Detective Nora Vale", "female")
        self.assertEqual(roster.add("Vale"), key)
        self.assertEqual(roster.add("Vale", "male"), "vale")  # a man of that name is someone else
        self.assertEqual(len(roster.characters), 2)

    def test_a_bare_surname_stays_apart_when_shared_titled_or_a_first_name(self):
        shared = Roster()
        shared.add("Nora Vale")
        shared.add("Peter Vale")
        self.assertIsNone(shared.resolve("Vale"))
        titled = Roster()
        key = titled.add("Nora Vale", "female")
        self.assertNotEqual(titled.add("Mrs. Vale"), key)
        first = Roster()
        first.add("Nora Vale")
        first.add("Vale Wen")
        self.assertEqual(first.resolve("Vale"), "vale wen")  # as a first name it is Vale Wen's, not Nora's


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

    def test_a_family_word_tag_is_asked_and_the_answer_joins_the_character_given_that_alias(self):
        text = f'"Supper is ready," said Mother.{M}"Coming," said Ada.{M}"Wash your hands first."'
        chat = ScriptedChat(_reply({1: "Mother", 3: "Mrs. Marsh"}, [
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



class TestQuotedTermsInAChapter(unittest.TestCase):

    def test_quoted_terms_are_not_asked_shown_unmarked_and_have_no_speaker(self):
        text = (f'She pointed at the “spare two” crates.{M}'
                f'“Who is there?” asked Ada.{M}'
                f'“Nobody.”{M}'
                f'So much for the “one small glass” of wine. “Pour it,” said Tom.{M}'
                f'“Fine.”')
        chat = ScriptedChat(_reply({3: "Ada", 6: "Tom"}))
        stats = {}
        lines, moods = attribute_chapter(chapter_segments(text), Roster(), chat, stats, logging.getLogger("t"))
        prompt = chat.prompts[0][1]["content"]
        self.assertIn("Ids to answer: 3, 6", prompt)
        self.assertIn("the “spare two” crates", prompt)
        self.assertNotIn("[#1]", prompt)
        self.assertNotIn("[#4]", prompt)
        self.assertEqual((lines[1], lines[4]), (None, None))
        self.assertEqual((lines[5], lines[6]), ("tom", "tom"))
        self.assertEqual((stats["quoted_terms"], stats["unknown_lines"]), (2, 0))


class TestSourceReviewedIdentities(unittest.TestCase):
    """Regressions from the October 6 source review, with short neutral passages."""

    def roster(self):
        return Roster({
            "hee haw": {"name": "Hee Haw", "gender": "male", "age": "adult", "aliases": ["Colin"], "lines": 10},
            "hee haw 2": {"name": "Hee Haw", "gender": "female", "age": "child", "aliases": ["Alice"], "lines": 1},
        })

    def test_shared_names_have_distinct_tokens_and_ambiguous_tags_are_asked(self):
        text = (f'"Enough," she said. "Leave."{M}'
                f'"No," he said.{M}"Now," Hee Haw barked.')
        chat = ScriptedChat(_reply({1: "@hee haw 2", 2: "@hee haw 2", 3: "@hee haw", 4: "@hee haw 2"}))
        roster = self.roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, chat, {})
        self.assertEqual(lines, {1: "hee haw 2", 2: "hee haw 2", 3: "hee haw", 4: "hee haw 2"})
        self.assertIn("Alice = Hee Haw (female; aliases: Alice)", chat.prompts[0][1]["content"])
        self.assertIn("[#4]", chat.prompts[0][1]["content"])
        self.assertEqual(len(roster.characters), 2)

    def test_bare_shared_names_use_a_pronoun_without_inventing_a_character(self):
        text = f'"Enough," she said.{M}"No," he said.'
        roster = self.roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(_reply({1: "Hee Haw", 2: "Hee Haw"})), {})
        self.assertEqual(lines, {1: "hee haw 2", 2: "hee haw"})
        self.assertEqual(len(roster.characters), 2)
        self.assertIsNone(roster.speaker_key("Hee Haw"))

    def test_unknown_identity_tokens_do_not_become_characters(self):
        roster = self.roster()
        with patch.object(cast_llm_module, "REVIEW_FLAGGED_LINES", False):
            lines, _ = attribute_chapter(chapter_segments('"Hello."'), roster,
                                         ScriptedChat(_reply({1: "@invented"})), {})
        self.assertEqual(lines, {1: None})
        self.assertEqual(len(roster.characters), 2)

    def test_bad_identity_metadata_does_not_abort_a_cast_or_replace_known_gender(self):
        roster = self.roster()
        reply = _reply({1: "@hee haw 2"}, [{"name": "@hee haw", "gender": "female"},
                                         {"name": "@invented", "gender": "male"}])
        lines, _ = attribute_chapter(chapter_segments('"Enough," she said.'), roster, ScriptedChat(reply), {})
        self.assertEqual(lines, {1: "hee haw 2"})
        self.assertEqual(roster.characters["hee haw"]["gender"], "male")
        self.assertEqual(len(roster.characters), 2)

    def test_ambiguous_shared_name_metadata_does_not_invent_a_third_person(self):
        roster = self.roster()
        reply = _reply({1: "Hee Haw"}, [{"name": "Hee Haw", "gender": "unknown"}])
        with patch.object(cast_llm_module, "REVIEW_FLAGGED_LINES", False):
            lines, _ = attribute_chapter(chapter_segments('"Enough."'), roster, ScriptedChat(reply), {})
        self.assertEqual(lines, {1: None})
        self.assertEqual(len(roster.characters), 2)

    def test_identity_tokens_preserve_titles_in_keys_and_can_fill_unknown_metadata(self):
        roster = Roster()
        man = roster.add("Mr. Smith", "male")
        woman = roster.add("Mrs. Smith", "female")
        self.assertEqual(roster.resolve(f"@{woman}"), woman)
        self.assertNotEqual(roster.resolve(f"@{woman}"), man)
        unknown = roster.add("Alex")
        self.assertEqual(roster.add(f"@{unknown}", "female", "adult"), unknown)
        self.assertEqual(roster.characters[unknown]["gender"], "female")
        self.assertIn(f"@{woman} = Mrs. Smith (female; aliases: none)", roster.names_for_prompt())

    def test_a_reply_declaring_the_shared_names_identity_keeps_its_unqualified_answers(self):
        roster = self.roster()
        text = f'"Enough."{M}"No," he said.'
        reply = _reply({1: "Hee Haw", 2: "Colin"}, [{"name": "Hee Haw", "gender": "female"}])
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(reply), {})
        self.assertEqual(lines, {1: "hee haw 2", 2: "hee haw"})

    def test_a_real_name_with_a_quoted_nickname_promotes_a_description(self):
        text = '“No sir, my real name’s Shirley but everybody calls me ‘Squirrelly’,” she said.'
        reply = _reply({1: "the girl"}, [{"name": "the girl", "gender": "female", "aliases": ["Shirley", "Squirrelly"]}])
        roster = Roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(reply), {})
        key = lines[1]
        self.assertEqual(roster.characters[key]["name"], "Shirley")
        self.assertNotIn("reference_scope", roster.characters[key])
        roster.new_chapter()
        self.assertEqual(roster.resolve("Squirrelly"), key)

    def test_someone_elses_quoted_introduction_does_not_name_the_outer_speaker(self):
        text = '“He said ‘My name is Bob,’ and then he left,” she said.'
        roster = Roster()
        lines, _ = attribute_chapter(chapter_segments(text), roster, ScriptedChat(_reply({1: "the girl"})), {})
        self.assertEqual(roster.characters[lines[1]]["name"], "the girl")
        self.assertIsNone(roster.resolve("Bob"))

    def test_a_shared_two_word_self_name_survives_into_the_next_chapter(self):
        roster = Roster()
        man = roster.add("Hee Haw", "male", aliases=["Colin"])
        girl = roster.add("Alice", "female")
        lines, _ = attribute_chapter(chapter_segments('"I am Hee Haw," she said.'), roster,
                                     ScriptedChat(_reply({1: "Alice"})), {})
        self.assertEqual(lines[1], girl)
        self.assertIn("Hee Haw", roster.characters[girl]["aliases"])
        roster.new_chapter()
        self.assertEqual(roster.resolve("Hee Haw", "female"), girl)
        self.assertEqual(roster.resolve("Hee Haw", "male"), man)
        self.assertIsNone(roster.resolve("Hee Haw"))
        chat = ScriptedChat(_reply({1: "Alice"}))
        lines, _ = attribute_chapter(chapter_segments('"Enough," Hee Haw barked.'), roster, chat, {})
        self.assertEqual(lines[1], girl)
        self.assertIn("[#1]", chat.prompts[0][1]["content"])

    def test_generic_descriptions_still_create_distinct_people_in_new_chapters(self):
        roster = Roster()
        keys = []
        for _ in range(3):
            lines, _ = attribute_chapter(chapter_segments('"Hello," she said.'), roster,
                                         ScriptedChat(_reply({1: "the girl"}, [{"name": "the girl", "gender": "female"}])), {})
            keys.append(lines[1])
        self.assertEqual(len(set(keys)), 3)

    def test_an_explicit_new_speaker_overrides_a_broken_quote(self):
        text = (f'“Miles, what the he-“ Leon doubled over.{M}'
                '“Are you listening?”Miles screamed at him.')
        paragraphs = chapter_segments(text)
        self.assertTrue(paragraphs[1][0].continues)
        lines, _ = attribute_chapter(paragraphs, Roster(), ScriptedChat(_reply({1: "Leon"})), {})
        self.assertEqual(lines, {1: "leon", 2: "miles"})
