"""Character profiles: which passages a character's profile is written from, strict reply parsing,
and the pass over a cast with a scripted stand-in for the chat endpoint."""
import json
import unittest

from audiobook_generator.core.cast_profiles import (
    ChapterText, Passage, ProfileError, _spread, character_passages, drop_unsupported_accents, first_lines,
    name_forms, parse_profile, profile_cast, profile_candidates, render_excerpts, select_passages,
)
from audiobook_generator.core.dialogue import PARAGRAPH_MARK as M, chapter_segments

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


if __name__ == "__main__":
    unittest.main()
