"""Character profiles: which passages a character's profile is written from, strict reply parsing,
and the pass over a cast with a scripted stand-in for the chat endpoint."""
import json
import unittest

from audiobook_generator.core.cast_profiles import (
    ChapterText, Passage, ProfileError, _spread, apply_chapter_narrators, chapter_narrators, chapter_point_of_view,
    character_passages, describe_book, drop_unsupported_accents, first_lines, name_forms, narration_passages,
    parse_profile, parse_tone, profile_cast, profile_candidates, render_excerpts, select_passages,
)
from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M, chapter_segments
from audiobook_generator.core.speech_tags import first_person_tagged

CHAPTER = (
    f'Ada Marsh put the lamp down. "You left the gate open," she said.{M}'
    f'"I did not," said Tom. He went on writing.{M}'
    f'"Then who did?"{M}'
    f'Tom shrugged and looked at the window.{M}'
    f'The goats were in the beans again.{M}'
    f'"Both of you, out," said Mrs. Marsh.'
)
LINES = {1: "ada marsh", 2: "tom", 3: "ada marsh", 4: "marsh"}


def _characters():
    return {
        "ada marsh": {"name": "Ada Marsh", "aliases": ["Ada"], "gender": "female", "age": "adult", "lines": 5},
        "tom": {"name": "Tom", "aliases": [], "gender": "unknown", "age": "unknown", "lines": 3},
        "marsh": {"name": "Mrs. Marsh", "aliases": [], "gender": "female", "age": "adult", "lines": 1},
    }


def _chapters():
    return [ChapterText(1, chapter_segments(CHAPTER), dict(LINES))]


def _profile(**overrides):
    reply = {"role": "supporting", "gender": "unknown", "age": "unknown", "description": "A farm girl.",
             "relationships": "", "voice": "bright young woman"}
    reply.update(overrides)
    return json.dumps(reply)


class ScriptedChat:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages):
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class TestPassages(unittest.TestCase):

    def test_name_forms_add_the_first_name_but_never_a_bare_surname(self):
        self.assertEqual(name_forms({"name": "Mrs. Ada Marsh", "aliases": []}), ["Mrs. Ada Marsh", "Ada"])
        self.assertEqual(name_forms({"name": "Mrs. Marsh", "aliases": ["Mother"]}), ["Mrs. Marsh", "Mother"])
        # A description's first word would match nearly every paragraph.
        self.assertEqual(name_forms({"name": "the man", "aliases": []}), ["the man"])
        self.assertEqual(name_forms({"name": "The Doctor", "aliases": []}), ["The Doctor"])

    def test_passages_are_where_the_character_speaks_or_is_named(self):
        characters = _characters()
        tom = character_passages(_chapters(), "tom", characters)
        self.assertEqual([p.paragraph for p in tom], [1, 3])
        self.assertEqual(tom[0].text, '[Tom] "I did not," said Tom. He went on writing.')
        ada = character_passages(_chapters(), "ada marsh", characters)
        # "Mrs. Marsh" in the last paragraph is someone else: a shared surname is not a mention.
        self.assertEqual([p.paragraph for p in ada], [0, 2])
        self.assertIn('[Ada Marsh] "You left the gate open,"', ada[0].text)

    def test_spread_order_covers_the_range_evenly(self):
        self.assertEqual(_spread(0), [])
        self.assertEqual(_spread(1), [0])
        order = _spread(10)
        self.assertEqual(sorted(order), list(range(10)))
        self.assertEqual(order[:3], [0, 8, 4])

    def test_selection_keeps_the_first_passages_then_spreads_within_the_budget(self):
        passages = [Passage(0, i, f"p{i:02d} " + "x" * 16) for i in range(40)]  # 20 characters each
        chosen = select_passages(passages, max_chars=10 * 22, lead=3)
        indexes = [p.paragraph for p in chosen]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(indexes[:3], [0, 1, 2])
        self.assertEqual(len(indexes), 10)
        self.assertGreater(max(indexes), 30)  # the spread reaches the end of the book, not just its start

    def test_a_single_passage_over_the_budget_is_cut_to_it(self):
        chosen = select_passages([Passage(0, 0, "word " * 100)], max_chars=50)
        self.assertLessEqual(len(chosen[0].text), 50)
        self.assertTrue(chosen[0].text.endswith("…"))

    def test_excerpts_mark_chapters_and_gaps(self):
        chapters = [ChapterText(3, [], {}), ChapterText(4, [], {})]
        text = render_excerpts([Passage(0, 1, "a"), Passage(0, 2, "b"), Passage(0, 7, "c"), Passage(1, 0, "d")], chapters)
        self.assertEqual(text, "(Chapter 3)\n\na\n\nb\n\n[...]\n\nc\n\n(Chapter 4)\n\nd")

    def test_first_lines_are_quoted_from_the_book(self):
        lines = first_lines(_chapters())
        self.assertEqual(lines["tom"], {"chapter": 1, "text": '"I did not,"'})
        self.assertEqual(lines["marsh"]["text"], '"Both of you, out,"')

    def test_candidates_are_the_most_spoken_with_enough_lines(self):
        self.assertEqual(profile_candidates(_characters(), min_lines=3, limit=15), ["ada marsh", "tom"])
        self.assertEqual(profile_candidates(_characters(), min_lines=1, limit=1), ["ada marsh"])


class TestParseProfile(unittest.TestCase):

    def test_a_good_reply(self):
        profile = parse_profile(_profile(role="Main character", gender="female", age="child"))
        self.assertEqual(profile, {"role": "protagonist", "gender": "female", "age": "child",
                                   "description": "A farm girl.", "relationships": "", "voice": "bright young woman",
                                   "voice_targets": {"pitch": None, "quality": None, "delivery": None}})

    def test_voice_targets_accept_the_words_a_model_uses(self):
        cases = [({"pitch": "high", "quality": "husky", "delivery": "expressive"}, ("high", "husky", "expressive")),
                 ({"pitch": "Deep", "quality": "breathy", "delivery": "calm"}, ("low", "husky", "even")),
                 ({"pitch": "medium-high", "quality": "either", "delivery": "unknown"}, ("medium", None, None)),
                 ({"pitch": 3, "quality": "crisp"}, (None, "clear", None))]
        for reply, expected in cases:
            targets = parse_profile(_profile(**reply))["voice_targets"]
            self.assertEqual((targets["pitch"], targets["quality"], targets["delivery"]), expected, reply)

    def test_fences_and_odd_values_are_tolerated(self):
        reply = "```json\n" + _profile(role="the villain", gender="woman", age="old", relationships="unknown") + "\n```"
        profile = parse_profile(reply)
        self.assertEqual((profile["role"], profile["gender"], profile["age"], profile["relationships"]),
                         ("antagonist", "unknown", "unknown", ""))

    def test_long_fields_are_clipped(self):
        profile = parse_profile(_profile(voice="very " * 100))
        self.assertLessEqual(len(profile["voice"]), 160)

    def test_filler_about_characters_the_excerpts_never_connect_is_dropped(self):
        profile = parse_profile(_profile(relationships="Bea: rough with her; Cal, Dee: not mentioned in the excerpts"))
        self.assertEqual(profile["relationships"], "Bea: rough with her")
        profile = parse_profile(_profile(relationships="Dee: receptionist; Bea: unknown; Dr. Lowe: Unknown."))
        self.assertEqual(profile["relationships"], "Dee: receptionist")

    def test_an_accent_needs_the_excerpts_to_name_it(self):
        note = "calm and introspective, with a slight southern drawl"
        self.assertEqual(drop_unsupported_accents(note, "Tom sat down and sighed."), "calm and introspective")
        self.assertEqual(drop_unsupported_accents(note, "He spoke in a lazy Southern drawl."), note)
        self.assertEqual(drop_unsupported_accents("gruff Scottish sailor", "The sailor spat."), "")

    def test_no_description_or_no_json_is_unusable(self):
        for reply in ("I can't help with that.", _profile(description=""), _profile(description="unknown"), "[1, 2]"):
            with self.assertRaises(ProfileError, msg=reply):
                parse_profile(reply)


class TestProfileCast(unittest.TestCase):

    def _cast(self):
        return {"characters": _characters(), "stats": {}}

    def test_profiles_for_the_main_characters_and_a_first_line_for_everyone(self):
        cast, saves = self._cast(), []
        chat = ScriptedChat(_profile(role="protagonist", gender="male", description="A sharp farm girl."),
                            _profile(gender="male", age="child", description="Her younger brother."))
        profile_cast(cast, _chapters(), chat, save=lambda: saves.append(1))
        ada, tom, marsh = (cast["characters"][k] for k in ("ada marsh", "tom", "marsh"))
        self.assertEqual(ada["profile"]["description"], "A sharp farm girl.")
        self.assertEqual(ada["profile"]["role"], "protagonist")
        self.assertEqual(ada["gender"], "female")  # a known gender is never overwritten by the profile
        self.assertEqual((tom["gender"], tom["age"]), ("male", "child"))  # an unknown one is filled in
        self.assertEqual(tom["profile"]["first_line"], {"chapter": 1, "text": '"I did not,"'})
        self.assertEqual(marsh["profile"], {"first_line": {"chapter": 1, "text": '"Both of you, out,"'}})
        self.assertEqual((cast["profiles_done"], cast["profiles_total"], cast["profile_error"]), (2, 2, ""))
        self.assertEqual(cast["stats"]["profiles"], 2)
        self.assertEqual(len(saves), 3)  # once when the stage starts, then after each character
        prompt = chat.prompts[0][1]["content"]
        self.assertIn("Character: Ada Marsh (also called Ada)", prompt)
        self.assertIn("Other characters in the book: Tom, Mrs. Marsh", prompt)
        self.assertIn('[Ada Marsh] "You left the gate open,"', prompt)
        self.assertNotIn("goats", prompt)  # a paragraph that neither names her nor has her speak

    def test_a_small_part_is_minor_whatever_the_model_says_unless_an_antagonist(self):
        for said, shown in (("supporting", "minor"), ("protagonist", "minor"), ("antagonist", "antagonist")):
            cast = self._cast()
            cast["characters"]["ada marsh"]["lines"] = 50  # Tom's 3 lines are under a tenth of hers
            profile_cast(cast, _chapters(), ScriptedChat(_profile(role="supporting"), _profile(role=said)))
            self.assertEqual(cast["characters"]["ada marsh"]["profile"]["role"], "supporting")
            self.assertEqual(cast["characters"]["tom"]["profile"]["role"], shown, said)

    def test_an_unusable_reply_is_asked_again_then_skipped(self):
        cast = self._cast()
        chat = ScriptedChat("not json", _profile(), "still not json", "{}")
        profile_cast(cast, _chapters(), chat)
        self.assertEqual(cast["characters"]["ada marsh"]["profile"]["description"], "A farm girl.")
        self.assertEqual(set(cast["characters"]["tom"]["profile"]), {"first_line"})
        self.assertEqual((cast["stats"]["profiles"], cast["stats"]["profiles_unusable"]), (1, 1))
        self.assertEqual(cast["profiles_done"], 2)

    def test_an_llm_error_ends_the_pass_without_raising(self):
        cast = self._cast()
        chat = ScriptedChat(_profile(), ConnectionError("LLM down"))
        profile_cast(cast, _chapters(), chat)
        self.assertEqual(cast["profile_error"], "LLM down")
        self.assertEqual(cast["profiles_done"], 1)
        self.assertIn("description", cast["characters"]["ada marsh"]["profile"])
        self.assertNotIn("description", cast["characters"]["tom"]["profile"])


class TestBookTone(unittest.TestCase):

    TONE = {"point_of_view": "first person", "pov_character": "Tom", "tone": "wry and warm", "pace": "fast",
            "intensity": "subdued", "narrator_gender": "Male", "narrator_pitch": "medium", "narrator_quality": "either",
            "narrator_delivery": "calm"}

    def test_a_reply_is_read_leniently(self):
        tone = parse_tone(json.dumps(self.TONE))
        self.assertEqual(tone, {"point_of_view": "first", "pov_character": "Tom", "tone": "wry and warm",
                                "pace": "brisk", "intensity": "restrained",
                                "narrator": {"gender": "male", "pitch": "medium", "quality": None, "delivery": "even"}})
        third = parse_tone(json.dumps({**self.TONE, "point_of_view": "third", "pov_character": "Tom"}))
        self.assertEqual(third["pov_character"], "")  # only a first-person book has a narrating character
        with self.assertRaises(ProfileError):
            parse_tone(json.dumps({**self.TONE, "tone": ""}))

    def test_only_paragraphs_without_dialogue_are_the_narrators(self):
        self.assertEqual([p.paragraph for p in narration_passages(_chapters())], [3, 4])

    def test_the_book_is_described_once_and_a_narrating_character_is_found(self):
        cast = {"characters": _characters()}
        chat = ScriptedChat("not json", json.dumps(self.TONE))
        describe_book(cast, _chapters(), chat)
        self.assertEqual(cast["book_tone"]["pov_key"], "tom")
        prompt = chat.prompts[0][1]["content"]
        self.assertIn("Ada Marsh (female)", prompt)
        self.assertIn("The goats were in the beans again.", prompt)
        self.assertNotIn("You left the gate open", prompt)  # dialogue is not the narrator's

    def test_a_failure_leaves_no_book_tone(self):
        cast = {"characters": _characters()}
        describe_book(cast, _chapters(), ScriptedChat(ConnectionError("LLM down")))
        self.assertNotIn("book_tone", cast)
        describe_book(cast, _chapters(), ScriptedChat("no", "still no"))
        self.assertNotIn("book_tone", cast)


# A man tells the first story and is only ever named when spoken to; the narration names only her.
_FIRST_PERSON = (f"I walked her home along the river, and my hands would not stay still. " * 20 + M
                 + f'"You worry too much, Oliver," Bettie said.{M}'
                 + f'"I know," I told her.{M}'
                 + f'"Stay a while," I said.{M}'
                 + f"Bettie laughed at me, and I did not mind at all. " * 10)
_THIRD_PERSON = (f"The sea rose against the harbour wall while Tom mended the nets. " * 25 + M
                 + f'"Storm coming," said Tom.{M}'
                 + f'"Then we sail at dawn," he said.')
_ANTHOLOGY_CHARACTERS = {
    "bettie": {"name": "Bettie", "aliases": [], "gender": "female", "age": "adult", "lines": 1},
    "oliver": {"name": "Oliver", "aliases": [], "gender": "male", "age": "adult", "lines": 2},
    "tom": {"name": "Tom", "aliases": [], "gender": "male", "age": "adult", "lines": 2},
}


def _story(number, text, lines):
    return ChapterText(number, chapter_segments(text), lines)


class TestChapterNarrators(unittest.TestCase):
    """Whether each chapter is told in the first person, and by whom (anthologies change both)."""

    FIRST = {1: "bettie", 2: "oliver", 3: "oliver"}
    THIRD = {1: "tom", 2: "tom"}

    def _cast(self, *numbers):
        return {"characters": {k: dict(v) for k, v in _ANTHOLOGY_CHARACTERS.items()},
                "chapters": {f"h{n}": {"number": n, "lines": {}} for n in numbers}}

    def test_point_of_view_comes_from_the_narration_not_the_dialogue(self):
        self.assertEqual(chapter_point_of_view(_story(1, _FIRST_PERSON, self.FIRST)), "first")
        self.assertEqual(chapter_point_of_view(_story(1, _THIRD_PERSON, self.THIRD)), "third")
        self.assertIsNone(chapter_point_of_view(_chapters()[0]))  # too little narration to tell

    def test_the_i_said_lines_name_the_narrator_over_the_tones_guess(self):
        # Seen live: the tone saw only narration, where she is named and he never is, and answered her.
        cast = self._cast(1)
        chapters = [_story(1, _FIRST_PERSON, self.FIRST)]
        wrong = dict(TestBookTone.TONE, pov_character="Bettie")
        describe_book(cast, chapters, ScriptedChat(json.dumps(wrong)))
        self.assertEqual((cast["book_tone"]["pov_key"], cast["book_tone"]["pov_character"]), ("oliver", "Oliver"))
        self.assertEqual(cast["chapters"]["h1"], {"number": 1, "lines": {}, "point_of_view": "first",
                                                  "narrator": "oliver"})

    def test_an_anthology_keeps_each_storys_point_of_view_and_narrator(self):
        cast = self._cast(1, 2, 3)
        chapters = [_story(1, _THIRD_PERSON, self.THIRD), _story(2, _FIRST_PERSON, self.FIRST),
                    _story(3, _FIRST_PERSON, self.FIRST)]
        found = chapter_narrators(cast, chapters, None)
        self.assertEqual(found, {1: {"point_of_view": "third", "narrator": None},
                                 2: {"point_of_view": "first", "narrator": "oliver"},
                                 3: {"point_of_view": "first", "narrator": "oliver"}})
        apply_chapter_narrators(cast, chapters, found)
        self.assertEqual(cast["book_tone"]["point_of_view"], "first")  # most of the narration is Oliver's
        self.assertEqual(cast["chapters"]["h1"]["narrator"], None)

    def test_side_by_side_first_person_stories_keep_their_own_narrators(self):
        # Seen live: a collection's first two stories are both in the first person, told by different people.
        untagged = _FIRST_PERSON.replace(" I told her", "").replace(" I said", "")
        chapters = [_story(1, _FIRST_PERSON, self.FIRST), _story(2, _FIRST_PERSON, {1: "oliver", 2: "bettie", 3: "bettie"}),
                    _story(3, untagged, {1: "tom", 2: "tom", 3: "tom"})]
        found = chapter_narrators(self._cast(1, 2, 3), chapters, None)
        # A chapter with too few "I said" lines of its own follows the one before it.
        self.assertEqual([found[n]["narrator"] for n in (1, 2, 3)], ["oliver", "bettie", "bettie"])

    def test_a_story_without_i_said_lines_asks_the_llm_about_that_story_alone(self):
        untagged = _FIRST_PERSON.replace(" I told her", "").replace(" I said", "")
        chapters = [_story(1, _FIRST_PERSON, self.FIRST), _story(2, _THIRD_PERSON, self.THIRD),
                    _story(3, untagged, {1: "bettie", 2: "tom", 3: "tom"})]
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Tom")))
        found = chapter_narrators(self._cast(1, 2, 3), chapters, None, chat)
        self.assertEqual([found[n]["narrator"] for n in (1, 2, 3)], ["oliver", None, "tom"])
        self.assertIn("(Chapter 3)", chat.prompts[0][1]["content"])
        self.assertNotIn("(Chapter 1)", chat.prompts[0][1]["content"])
        # Without an LLM (re-deriving a saved cast) that story's narrator stays unknown.
        self.assertIsNone(chapter_narrators(self._cast(1, 2, 3), chapters, None)[3]["narrator"])

    def test_first_person_tags_name_the_is_lines(self):
        paragraphs = chapter_segments(f'"I know," I told her.{M}"Go," she said.{M}"Stay," said I.{M}I said, "Fine."')
        self.assertEqual(len(first_person_tagged(paragraphs)), 3)

    def test_mostly_third_person_drops_the_tones_narrator(self):
        cast = self._cast(1, 2)
        cast["book_tone"] = {"point_of_view": "first", "pov_key": "tom", "pov_character": "Tom"}
        chapters = [_story(1, _THIRD_PERSON * 3, self.THIRD), _story(2, _FIRST_PERSON, self.FIRST)]
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, cast["book_tone"]))
        self.assertEqual(cast["book_tone"]["point_of_view"], "third")
        self.assertNotIn("pov_key", cast["book_tone"])
        self.assertEqual(cast["chapters"]["h2"]["narrator"], "oliver")  # still read as Oliver's in its chapter


if __name__ == "__main__":
    unittest.main()
