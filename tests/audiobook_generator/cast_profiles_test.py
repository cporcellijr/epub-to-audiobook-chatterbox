"""Character profiles: which passages a character's profile is written from, strict reply parsing,
and the pass over a cast with a scripted stand-in for the chat endpoint."""
import json
import logging
import unittest

from audiobook_generator.core.cast_profiles import (
    ChapterText, Passage, ProfileError, _ask_tone, _spread, addressed_tellers, addresses, apply_chapter_narrators, chapter_narrators, chapter_point_of_view,
    character_passages, could_say_i, describe_book, drop_unsupported_accents, first_lines, name_forms, narration_passages,
    parse_profile, parse_tone, profile_cast, profile_candidates, render_excerpts, select_passages,
    turn_taking,
)
from audiobook_generator.core import cast as cast_store
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

    def test_pov_samples_skip_only_explicitly_framed_inset_documents(self):
        framing = "I enter the study. " + ("She watches the fire in silence. " * 24)
        opening = "I begin to browse through the notebook and stop at an entry."
        inset = "I loved him, I wrote to him, I hoped he would answer. " * 20
        closing = "Jesus Christ. I slam the notebook with a loud clap."
        chapter = ChapterText(1, chapter_segments(M.join((framing, opening, inset, closing))), {})
        self.assertEqual([p.paragraph for p in narration_passages([chapter])], [0, 1, 3])
        self.assertEqual(chapter_point_of_view(chapter), "third")

    def test_unframed_diary_and_later_anthology_chapter_remain_eligible(self):
        diary = ChapterText(1, chapter_segments("I wrote to my sister before dawn."), {})
        story = ChapterText(2, chapter_segments("I woke in a different town with no memory."), {})
        self.assertEqual([p.chapter for p in narration_passages([diary, story])], [0, 1])

    def test_explicit_backmatter_heading_excludes_the_remaining_document(self):
        text = M.join(("She crossed the yard and closed the door.", "About the Author",
                       "I grew up near the sea and wrote my first book at twenty."))
        chapter = ChapterText(1, chapter_segments(text), {})
        self.assertEqual([p.paragraph for p in narration_passages([chapter])], [0])

    def test_unclosed_document_frame_excludes_nothing_and_does_not_hide_the_chapter_pov(self):
        text = M.join(("She walks through the quiet house. " * 30,
                       "I open the diary and begin to read.", "I loved him and hoped he would return. " * 30))
        chapter = ChapterText(1, chapter_segments(text), {})
        self.assertEqual(chapter_point_of_view(chapter), "first")  # the unframed diary counts as evidence
        self.assertEqual([p.paragraph for p in narration_passages([chapter])], [0, 1, 2])

    def test_a_stray_open_with_a_close_far_later_excludes_nothing(self):
        filler = ["She watched the rain and counted the hours until morning. " * 3] * 29
        text = M.join(("I open the letter and set it down again.", *filler,
                       "Mara closes the letter and puts it away."))
        chapter = ChapterText(1, chapter_segments(text), {})
        self.assertEqual(len(narration_passages([chapter])), 31)

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
_UNADDRESSED = _FIRST_PERSON.replace(", Oliver,", ",")  # the same, but nobody calls him by name
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

    def _disputed_chapter(self):
        cast = self._cast(1)
        cast["characters"].update({
            "polly": {"name": "Polly", "aliases": [], "gender": "female", "age": "adult", "lines": 10},
            "dermatologist": {"name": "Self-proclaimed dermatologist", "aliases": [], "gender": "female",
                              "age": "adult", "lines": 0},
        })
        chapter = _story(1, _FIRST_PERSON, {})
        dialogue_ids = [s.line_id for p in chapter.paragraphs for s in p if s.kind == "dialogue"]
        tagged = first_person_tagged(chapter.paragraphs)
        chapter.lines.update({line_id: "bettie" for line_id in dialogue_ids})
        chapter.lines.update({line_id: "dermatologist" for line_id in tagged})
        cast["characters"]["bettie"]["lines"] = len(dialogue_ids) - len(tagged)
        cast["characters"]["dermatologist"]["lines"] = len(tagged)
        cast["chapters"]["h1"]["lines"] = {str(line_id): speaker for line_id, speaker in chapter.lines.items()}
        return cast, chapter, tagged

    def _book_tone(self, name="Polly"):
        return {**TestBookTone.TONE, "point_of_view": "first", "pov_key": name.lower(), "pov_character": name}

    def test_point_of_view_comes_from_the_narration_not_the_dialogue(self):
        self.assertEqual(chapter_point_of_view(_story(1, _FIRST_PERSON, self.FIRST)), "first")
        self.assertEqual(chapter_point_of_view(_story(1, _THIRD_PERSON, self.THIRD)), "third")
        self.assertIsNone(chapter_point_of_view(_chapters()[0]))  # too little narration to tell

    def test_the_i_said_lines_name_the_narrator_over_the_tones_guess(self):
        # Seen live: the tone saw only narration, where she is named and he never is, and answered her.
        cast = self._cast(1)
        chapters = [_story(1, _FIRST_PERSON, self.FIRST)]
        wrong = dict(TestBookTone.TONE, pov_character="Bettie")
        describe_book(cast, chapters, ScriptedChat(json.dumps(wrong),
                                                   json.dumps(dict(TestBookTone.TONE, pov_character="Oliver"))))
        self.assertEqual((cast["book_tone"]["pov_key"], cast["book_tone"]["pov_character"]), ("oliver", "Oliver"))
        self.assertEqual(cast["chapters"]["h1"], {"number": 1, "lines": {}, "point_of_view": "first",
                                                "narrator": "oliver", "unknown": 0})

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
                    _story(3, untagged, {1: "tom", 2: "bettie", 3: "tom"})]
        found = chapter_narrators(self._cast(1, 2, 3), chapters, None)
        # A chapter with too few "I said" lines of its own follows the nearest one whose teller speaks in it.
        self.assertEqual([found[n]["narrator"] for n in (1, 2, 3)], ["oliver", "bettie", "bettie"])

    def test_narration_confirms_an_anthologys_alternate_narrator(self):
        chapters = [_story(1, _FIRST_PERSON, self.FIRST),
                    _story(2, _FIRST_PERSON, {1: "oliver", 2: "bettie", 3: "bettie"})]
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Bettie")))
        found = chapter_narrators(self._cast(1, 2), chapters, self._book_tone("Oliver"), chat)
        self.assertEqual([found[n]["narrator"] for n in (1, 2)], ["oliver", "bettie"])
        self.assertEqual(len(chat.prompts), 1)
        self.assertIn("(Chapter 2)", chat.prompts[0][1]["content"])

    def _anonymous_story(self, number, key, text=_UNADDRESSED):
        return _story(number, text, {1: "bettie", 2: key, 3: key})

    def _anonymous_cast(self, keys, numbers):
        cast = self._cast(*numbers)
        for key in keys:
            cast["characters"][key] = {"name": "The Narrator", "aliases": [], "gender": "unknown",
                                       "age": "unknown", "lines": 2}
        for n, key in zip(numbers, keys):
            cast["chapters"][f"h{n}"].update(narrator_reference=key, lines={"1": "bettie", "2": key, "3": key})
        return cast

    def test_an_unnamed_narrator_stays_one_identity_across_a_first_person_run(self):
        keys = ["narrator", "narrator 2", "narrator 3"]
        cast = self._anonymous_cast(keys, (1, 2, 3))
        chapters = [self._anonymous_story(n, key) for n, key in zip((1, 2, 3), keys)]
        found = chapter_narrators(cast, chapters, None)
        apply_chapter_narrators(cast, chapters, found)
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)], ["narrator"] * 3)
        self.assertEqual([found[n]["narrator"] for n in (1, 2, 3)], ["narrator"] * 3)
        self.assertEqual([dict(c.lines) for c in chapters], [{1: "bettie", 2: "narrator", 3: "narrator"}] * 3)
        self.assertEqual(cast["chapters"]["h3"]["lines"], {"1": "bettie", "2": "narrator", "3": "narrator"})
        self.assertNotIn("narrator 2", cast["characters"])
        self.assertNotIn("narrator 3", cast["characters"])
        self.assertEqual(cast["characters"]["narrator"]["lines"], 6)
        self.assertEqual(cast["book_tone"]["pov_key"], "narrator")
        self.assertEqual(cast_store.narrating_character(cast), "narrator")
        self.assertNotIn("issues", cast)

    def test_an_owner_voiced_unnamed_narrator_is_kept_but_its_lines_still_fold(self):
        keys = ["narrator", "narrator 2"]
        cast = self._anonymous_cast(keys, (1, 2))
        cast["characters"]["narrator 2"].update(voice="Mine.wav", voice_picked=True)
        chapters = [self._anonymous_story(n, key) for n, key in zip((1, 2), keys)]
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual(cast["chapters"]["h2"]["narrator"], "narrator")
        self.assertIn("narrator 2", cast["characters"])
        self.assertEqual(cast["characters"]["narrator 2"]["lines"], 0)

    def test_a_story_that_only_ever_says_i_keeps_one_unnamed_narrator_whatever_the_guess(self):
        # Seen live: the model never named a story's "I", and the book-wide guess (a woman he talks to)
        # was made its narrator; her voice read his story. The guess no longer names an unnamed "I".
        untagged = _UNADDRESSED.replace(" I told her", "").replace(" I said", "")
        cast = self._anonymous_cast(["narrator", "narrator 2"], (1, 2))
        cast["chapters"]["h3"] = {"number": 3, "lines": {"1": "bettie", "2": "oliver", "3": "oliver"}}
        chapters = [self._anonymous_story(n, key) for n, key in zip((1, 2), ("narrator", "narrator 2"))]
        chapters.append(_story(3, untagged, {1: "bettie", 2: "oliver", 3: "oliver"}))
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Oliver")))
        found = chapter_narrators(cast, chapters, self._book_tone("Oliver"), chat)
        apply_chapter_narrators(cast, chapters, found)
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)], ["narrator"] * 3)
        self.assertEqual([dict(c.lines) for c in chapters[:2]], [{1: "bettie", 2: "narrator", 3: "narrator"}] * 2)
        self.assertNotIn("narrator 2", cast["characters"])
        self.assertEqual(cast["book_tone"]["pov_key"], "narrator")
        self.assertEqual(chat.prompts, [])

    def test_the_person_the_others_call_by_name_tells_a_story_that_only_says_i(self):
        keys = ["narrator", "narrator 2", "narrator 3"]
        cast = self._anonymous_cast(keys, (1, 2, 3))
        chapters = [self._anonymous_story(n, key, _FIRST_PERSON) for n, key in zip((1, 2, 3), keys)]
        chat = ScriptedChat()
        found = chapter_narrators(cast, chapters, self._book_tone("Bettie"), chat)
        apply_chapter_narrators(cast, chapters, found)
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)], ["oliver"] * 3)
        self.assertEqual([dict(c.lines) for c in chapters], [{1: "bettie", 2: "oliver", 3: "oliver"}] * 3)
        self.assertFalse(set(keys) & set(cast["characters"]))
        self.assertEqual(cast["book_tone"]["pov_key"], "oliver")
        self.assertEqual(chat.prompts, [])

    def test_an_untagged_story_after_one_its_teller_named_is_not_lent_its_unnamed_i(self):
        keys = ["narrator", "narrator 2", "narrator 3"]
        cast = self._anonymous_cast(keys, (1, 2, 3))
        for key in ("elgin", "mabel", "dora"):
            cast["characters"][key] = {"name": key.title(), "aliases": [], "gender": "unknown", "lines": 1}
        text = (f"I walked into the lab at dawn and the machines hummed. " * 20 + M
                + f'"Sit down," Elgin said.{M}"Coffee?" Mabel asked.{M}"Fine," said Dora.')
        chapters = [self._anonymous_story(n, key, _FIRST_PERSON) for n, key in zip((1, 2, 3), keys)]
        chapters.append(_story(4, text, {1: "elgin", 2: "mabel", 3: "dora"}))
        cast["chapters"]["h4"] = {"number": 4, "lines": {"1": "elgin", "2": "mabel", "3": "dora"}}
        found = chapter_narrators(cast, chapters, None)
        self.assertEqual([found[n]["narrator"] for n in (1, 2, 3, 4)], ["oliver", "oliver", "oliver", None])

    def test_a_narrator_only_described_takes_the_name_others_call_him_beside_another_tellers_chapter(self):
        # Seen live: a novel's "I" was labelled "Man" in most chapters and read in another voice than his
        # own name's; its "Nora's PoV" chapters, whose narration names him, had blocked the teller.
        cast = self._cast(1, 2, 3, 4)
        cast["characters"]["man"] = {"name": "Man", "aliases": [], "gender": "male", "age": "adult", "lines": 6,
                                     "reference_scope": "chapter"}
        chapters = [_story(n, _FIRST_PERSON, {1: "bettie", 2: "man", 3: "man"}) for n in (1, 2, 3)]
        hers = f"Oliver waited for me by the gate, as he always did.{M}" * 6 + _FIRST_PERSON
        chapters.append(_story(4, hers, {1: "oliver", 2: "bettie", 3: "bettie"}))
        for chapter in chapters:
            cast["chapters"][f"h{chapter.number}"]["lines"] = {str(k): v for k, v in chapter.lines.items()}
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3, 4)],
                         ["oliver", "oliver", "oliver", "bettie"])
        self.assertEqual([dict(c.lines) for c in chapters[:3]], [{1: "bettie", 2: "oliver", 3: "oliver"}] * 3)
        self.assertNotIn("man", cast["characters"])

    def test_a_teller_needs_three_addresses_twice_anyone_elses_and_a_narration_that_never_names_them(self):
        def story(*extra):
            text = _FIRST_PERSON + "".join(M + line for line in extra)
            return [_story(n, text, {}) for n in (1, 2, 3)]

        characters = {k: dict(v) for k, v in _ANTHOLOGY_CHARACTERS.items()}
        self.assertEqual(addressed_tellers(story(), characters), {1: "oliver", 2: "oliver", 3: "oliver"})
        self.assertEqual(addressed_tellers(story()[:2], characters), {})  # twice is not enough
        self.assertEqual(addressed_tellers(story('"Hush, Bettie."'), characters), {})  # as often as her
        named = story(*[f"Oliver walked on alone, as Oliver always did." for _ in range(5)])
        self.assertEqual(addressed_tellers(named, characters), {})  # the narration calls him by name

    def test_a_story_break_ignores_whom_the_model_gave_the_lines_to(self):
        # Seen live: lines of the next story went to the last story's narrator, joining them up.
        cast, chapters = self._collection()
        third = _story(3, _FIRST_PERSON, dict(self.FIRST))
        other = chapters[2]
        chapters = chapters[:2] + [third, ChapterText(4, other.paragraphs, other.lines)]
        for key in ("elgin", "mabel", "dora"):
            cast["characters"][key]["name"] = key.title()
        self.assertEqual(addressed_tellers(chapters, cast["characters"]), {1: "oliver", 2: "oliver", 3: "oliver"})

    def _she_is_named(self, lines, tagged_to):
        """A first-person chapter whose narration names Bettie in six paragraphs (and never Oliver),
        with its "I said" lines given to `tagged_to`."""
        text = f"Bettie waited for me by the gate, as she always did.{M}" * 6 + _FIRST_PERSON
        chapter = _story(1, text, dict(lines))
        chapter.lines.update({line_id: tagged_to for line_id in first_person_tagged(chapter.paragraphs)})
        return chapter

    def test_nobody_the_narration_names_becomes_its_narrator(self):
        # Seen live: the "I said" lines went to the stepsister the narration talks about; and the
        # book-wide guess (a woman he talks to, named in 25 paragraphs) overruled his own vote.
        cast = self._cast(1)
        chapter = self._she_is_named({}, "bettie")
        self.assertEqual(chapter_narrators(cast, [chapter], self._book_tone("Oliver"))[1]["narrator"], "oliver")
        chapter = self._she_is_named({}, "oliver")
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Bettie")))
        self.assertEqual(chapter_narrators(cast, [chapter], self._book_tone("Bettie"), chat)[1]["narrator"], "oliver")
        self.assertEqual(chat.prompts, [])  # a guess the narration rules out is not worth confirming
        untagged = self._she_is_named({}, "bettie")
        untagged.lines.clear()
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Bettie")))
        self.assertIsNone(chapter_narrators(cast, [untagged], self._book_tone("Bettie"), chat)[1]["narrator"])

    def test_an_unnamed_i_is_not_folded_into_a_teller_its_narration_names(self):
        # Review finding: chapters alternate between two first-person tellers; the model named one
        # and left the other's "I" unnamed, and the fold handed the second teller's chapter to the first.
        cast = self._anonymous_cast(["narrator 2"], (2,))
        cast["chapters"]["h1"] = {"number": 1, "lines": {"1": "bettie", "2": "oliver", "3": "oliver"}}
        hers = f"Oliver waited for me by the gate, as he always did.{M}" * 6 + _FIRST_PERSON
        chapters = [_story(1, _FIRST_PERSON, {1: "bettie", 2: "oliver", 3: "oliver"}),
                    _story(2, hers, {1: "bettie", 2: "narrator 2", 3: "narrator 2"})]
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2)], ["oliver", "narrator 2"])
        self.assertEqual(dict(chapters[1].lines), {1: "bettie", 2: "narrator 2", 3: "narrator 2"})
        self.assertEqual(cast["chapters"]["h2"]["lines"], {"1": "bettie", "2": "narrator 2", "3": "narrator 2"})

    def test_message_labels_and_framed_documents_do_not_name_the_narrator(self):
        oliver = _ANTHOLOGY_CHARACTERS["oliver"]
        texts = f"Oliver: on my way{M}" * 6 + f"I opened the letter.{M}" + f"Dear Oliver, come home.{M}" * 3 \
            + f"I closed the letter.{M}" + f"I walked her home along the river.{M}" * 20
        chapter = _story(1, texts, {})
        self.assertTrue(could_say_i(chapter, "oliver", {"oliver": oliver}))
        chapter = _story(1, f"Oliver walked her home.{M}" * 5, {})
        self.assertFalse(could_say_i(chapter, "oliver", {"oliver": oliver}))

    def test_adjacent_named_narrators_stay_distinct_beside_an_unnamed_one(self):
        cast = self._anonymous_cast(["narrator"], (3,))
        chapters = [_story(1, _FIRST_PERSON, {1: "oliver", 2: "bettie", 3: "bettie"}),
                    _story(2, _FIRST_PERSON, {1: "bettie", 2: "oliver", 3: "oliver"}),
                    self._anonymous_story(3, "narrator")]
        for n in (1, 2):
            cast["chapters"][f"h{n}"] = {"number": n, "lines": {}}
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        # The named narrators stay apart; the unnamed "I" beside them is the nearest one's story.
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)], ["bettie", "oliver", "oliver"])
        self.assertIn("bettie", cast["characters"])
        self.assertIn("oliver", cast["characters"])
        self.assertNotIn("narrator", cast["characters"])

    def test_a_run_named_once_keeps_that_narrator_where_later_chapters_only_say_i(self):
        # Seen live: chapters 1-2 named the narrator, the model left 3-9 as "I", the book tone (offered
        # the placeholder) picked "The Narrator", and the book's narrator split in two at chapter 3.
        cast = self._anonymous_cast(["narrator", "narrator 2"], (2, 3))
        cast["chapters"]["h1"] = {"number": 1, "lines": {"1": "bettie", "2": "oliver", "3": "oliver"}}
        chapters = [_story(1, _FIRST_PERSON, {1: "bettie", 2: "oliver", 3: "oliver"}),
                    self._anonymous_story(2, "narrator"), self._anonymous_story(3, "narrator 2")]
        found = chapter_narrators(cast, chapters, None)
        apply_chapter_narrators(cast, chapters, found)
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)], ["oliver"] * 3)
        self.assertEqual([dict(c.lines) for c in chapters], [{1: "bettie", 2: "oliver", 3: "oliver"}] * 3)
        self.assertNotIn("narrator", cast["characters"])
        self.assertNotIn("narrator 2", cast["characters"])
        self.assertEqual(cast["book_tone"]["pov_key"], "oliver")

    def test_the_book_tone_is_never_offered_or_given_the_unnamed_placeholder(self):
        characters = dict(_ANTHOLOGY_CHARACTERS, narrator={"name": "The Narrator", "aliases": [], "gender": "female",
                                                           "age": "adult", "lines": 50})
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="The Narrator")))
        tone = _ask_tone(characters, [_story(1, _FIRST_PERSON, {})], chat, logging.getLogger("test"))
        self.assertNotIn("The Narrator", chat.prompts[0][1]["content"])
        self.assertIsNone(tone["pov_key"])

    def _other_story(self, mention=""):
        """A first-person story whose "I said" lines the model gave to Oliver, with people of its own."""
        text = (f"I walked into the lab at dawn and the machines hummed. {mention}" * 20 + M
                + f'"Sit down," Elgin said.{M}"Why?" I asked.{M}"Because," Elgin said.{M}'
                + f'"Coffee?" Mabel asked.{M}"Later," I said.{M}"Fine," said Dora.')
        chapter = _story(3, text, {})
        ids = [seg.line_id for p in chapter.paragraphs for seg in p if seg.kind == "dialogue"]
        chapter.lines.update(dict(zip(ids, ["elgin", "oliver", "elgin", "mabel", "oliver", "dora"])))
        return chapter

    def _collection(self, mention=""):
        cast = self._cast(1, 2, 3)
        for key in ("elgin", "mabel", "dora"):
            cast["characters"][key] = {"name": key.title(), "aliases": [], "gender": "unknown", "lines": 1}
        chapters = [_story(1, _FIRST_PERSON, dict(self.FIRST)), _story(2, _FIRST_PERSON, dict(self.FIRST)),
                    self._other_story(mention)]
        for chapter in chapters:
            cast["chapters"][f"h{chapter.number}"]["lines"] = {str(k): v for k, v in chapter.lines.items()}
        return cast, chapters

    def test_the_next_story_in_a_collection_gets_its_own_narrator(self):
        # Seen live: the model gave the next story's "I said" lines to the last story's narrator.
        cast, chapters = self._collection()
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        narrators = [cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)]
        self.assertEqual(narrators[:2], ["oliver", "oliver"])
        self.assertNotEqual(narrators[2], "oliver")
        self.assertEqual(cast["characters"][narrators[2]]["name"], "The Narrator")
        self.assertEqual(sorted({v for v in cast["chapters"]["h3"]["lines"].values()}),
                         sorted({"elgin", "mabel", "dora", narrators[2]}))
        self.assertEqual(cast["book_tone"]["pov_key"], "oliver")

    def test_generic_descriptions_do_not_connect_unrelated_stories(self):
        cast, chapters = self._collection("The girl waited beside me. ")
        for number in (1, 2):
            chapters[number - 1] = _story(number, _FIRST_PERSON + M + "The girl waited beside me.", dict(self.FIRST))
        for key in ("the girl", "the girl 2"):
            cast["characters"][key] = {"name": "The girl", "aliases": [], "gender": "female",
                                        "age": "unknown", "lines": 1, "reference_scope": "chapter"}
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2)], ["oliver", "oliver"])
        self.assertNotEqual(cast["chapters"]["h3"]["narrator"], "oliver")

    def test_neighbouring_stories_that_never_name_their_i_keep_their_own_unnamed_narrators(self):
        keys = ["narrator", "narrator 2", "narrator 3"]
        cast = self._anonymous_cast(keys, (1, 2, 3))
        for key in ("elgin", "mabel", "dora"):
            cast["characters"][key] = {"name": key.title(), "aliases": [], "gender": "unknown", "lines": 1}
        other = self._other_story()
        other.lines.update({k: "narrator 3" for k, v in other.lines.items() if v == "oliver"})
        cast["chapters"]["h3"]["lines"] = {str(k): v for k, v in other.lines.items()}
        chapters = [self._anonymous_story(1, "narrator"), self._anonymous_story(2, "narrator 2"), other]
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual([cast["chapters"][f"h{n}"]["narrator"] for n in (1, 2, 3)],
                         ["narrator", "narrator", "narrator 3"])
        self.assertEqual(cast["characters"]["narrator 3"]["name"], "The Narrator")
        self.assertNotIn("narrator 2", cast["characters"])

    def test_a_narrator_named_in_the_chapter_or_too_few_others_keeps_it(self):
        cast, chapters = self._collection(mention="Oliver, they called me. ")
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual(cast["chapters"]["h3"]["narrator"], "oliver")
        cast, chapters = self._collection()
        for key in ("mabel", "dora"):
            del cast["characters"][key]
        chapters[2].lines.update({k: "elgin" for k, v in chapters[2].lines.items() if v in ("mabel", "dora")})
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, None))
        self.assertEqual(cast["chapters"]["h3"]["narrator"], "oliver")  # one other person proves nothing

    def test_conflicting_chapter_vote_is_corrected_by_narration_only(self):
        cast, chapter, tagged = self._disputed_chapter()
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Polly")))
        found = chapter_narrators(cast, [chapter], self._book_tone(), chat)
        self.assertGreaterEqual(len(tagged), 2)
        self.assertEqual(found[1]["narrator"], "polly")
        prompt = chat.prompts[0][1]["content"]
        self.assertIn("I walked her home", prompt)
        self.assertNotIn("You worry too much", prompt)

    def test_an_untagged_chapter_does_not_inherit_an_unconfirmed_later_vote(self):
        cast = self._cast(1, 2)
        cast["characters"].update({
            "polly": {"name": "Polly", "aliases": [], "gender": "female", "age": "adult", "lines": 10},
            "dermatologist": {"name": "Self-proclaimed dermatologist", "aliases": [], "gender": "female",
                              "age": "adult", "lines": 6},
        })
        untagged = _FIRST_PERSON.replace(" I told her", "").replace(" I said", "")
        chapters = [_story(1, untagged, {1: "dermatologist", 2: "dermatologist", 3: "dermatologist"}),
                    _story(2, _FIRST_PERSON, {1: "bettie", 2: "dermatologist", 3: "dermatologist"})]
        reply = json.dumps(dict(TestBookTone.TONE, pov_character="Polly"))
        chat = ScriptedChat(reply, reply)

        found = chapter_narrators(cast, chapters, self._book_tone(), chat)

        self.assertEqual([found[n]["narrator"] for n in (1, 2)], ["polly", "polly"])
        self.assertEqual(len(chat.prompts), 2)
        self.assertIn("(Chapter 1)", chat.prompts[0][1]["content"])
        self.assertIn("(Chapter 2)", chat.prompts[1][1]["content"])

    def test_unconfirmed_conflicting_chapter_vote_falls_back_to_book_narrator(self):
        replies = (None, ScriptedChat("not json", "still not json"),
                   ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Bettie"))))
        for chat in replies:
            with self.subTest(chat=chat):
                cast, chapter, _ = self._disputed_chapter()
                self.assertEqual(chapter_narrators(cast, [chapter], self._book_tone(), chat)[1]["narrator"], "polly")

    def test_applying_narrator_corrects_tagged_lines_and_continuations_once(self):
        cast = self._cast(1)
        cast["characters"]["dermatologist"] = {"name": "Self-proclaimed dermatologist", "lines": 0}
        cast["characters"]["dermatologist"]["profile"] = {"description": "stale"}
        cast["characters"]["oliver"]["profile"] = {"description": "stale"}
        cast["characters"]["oliver"]["lines"] = 0
        text = (f'“Stay, I told her,” I said, “and then took the long way home{M}'
                f'“past the river and back again.”{M}'
                f'“What about you?” Bettie asked.')
        chapter = _story(1, text, {})
        segments = [s for p in chapter.paragraphs for s in p if s.kind == "dialogue"]
        tagged = set(first_person_tagged(chapter.paragraphs))
        continuation = next(s.line_id for s in segments if s.continues)
        self.assertTrue(tagged)
        self.assertIn(continuation - 1, tagged)
        chapter.lines.update({s.line_id: "bettie" for s in segments})
        chapter.lines.update({line_id: "dermatologist" for line_id in tagged | {continuation}})
        cast["characters"]["dermatologist"]["lines"] = len(tagged) + 1
        unrelated = next(s.line_id for s in segments if s.line_id not in tagged | {continuation})
        cast["characters"]["bettie"]["lines"] = len(segments) - len(tagged) - 1
        cast["chapters"]["h1"]["lines"] = {str(line_id): speaker for line_id, speaker in chapter.lines.items()}
        found = {1: {"point_of_view": "first", "narrator": "oliver"}}

        apply_chapter_narrators(cast, [chapter], found)
        expected = tagged | {continuation}
        self.assertTrue(all(chapter.lines[line_id] == "oliver" for line_id in expected))
        self.assertTrue(all(cast["chapters"]["h1"]["lines"][str(line_id)] == "oliver" for line_id in expected))
        self.assertEqual(chapter.lines[unrelated], "bettie")
        self.assertEqual(cast["chapters"]["h1"]["lines"][str(unrelated)], "bettie")
        self.assertEqual(cast["characters"]["dermatologist"]["lines"], 0)
        self.assertEqual(cast["characters"]["oliver"]["lines"], len(expected))
        self.assertNotIn("profile", cast["characters"]["dermatologist"])
        self.assertNotIn("profile", cast["characters"]["oliver"])
        counts = {key: value["lines"] for key, value in cast["characters"].items()}
        apply_chapter_narrators(cast, [chapter], found)
        self.assertEqual({key: value["lines"] for key, value in cast["characters"].items()}, counts)

    def test_an_untagged_story_next_to_a_tagged_one_is_asked_about_not_handed_its_teller(self):
        # Review finding: the next story of a collection has other people, so its neighbour's
        # teller (who doesn't speak in it) is no answer; the LLM is asked about that chapter alone.
        untagged = _FIRST_PERSON.replace(" I told her", "").replace(" I said", "")
        chapters = [_story(1, _FIRST_PERSON, self.FIRST), _story(2, untagged, {1: "tom", 2: "tom", 3: "tom"})]
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Tom")))
        found = chapter_narrators(self._cast(1, 2), chapters, self._book_tone("Oliver"), chat)
        self.assertEqual([found[n]["narrator"] for n in (1, 2)], ["oliver", "tom"])
        self.assertIn("(Chapter 2)", chat.prompts[0][1]["content"])
        self.assertEqual(len(chat.prompts), 1)
        # A novel's untagged chapter where the "I" speaks follows its neighbour without asking.
        chapters[1] = _story(2, untagged, {1: "bettie", 2: "oliver", 3: "oliver"})
        chat = ScriptedChat()
        self.assertEqual(chapter_narrators(self._cast(1, 2), chapters, None, chat)[2]["narrator"], "oliver")
        self.assertEqual(chat.prompts, [])

    def test_a_chapter_too_short_to_tell_keeps_the_books_narrator(self):
        # Review finding: an undecided chapter used to save "no narrator", hiding the book's own.
        from audiobook_generator.core import cast as cast_store
        cast = self._cast(1)
        cast["book_tone"] = {"point_of_view": "first", "pov_key": "tom", "pov_character": "Tom"}
        chapters = [ChapterText(1, chapter_segments(CHAPTER), dict(LINES))]  # too little narration
        found = chapter_narrators(cast, chapters, cast["book_tone"])
        self.assertEqual(found, {})
        apply_chapter_narrators(cast, chapters, found)
        self.assertNotIn("narrator", cast["chapters"]["h1"])
        self.assertEqual((cast["book_tone"]["pov_key"], cast_store.chapter_narrator(cast, "h1")), ("tom", "tom"))

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
        chat = ScriptedChat(json.dumps(dict(TestBookTone.TONE, pov_character="Oliver")))
        apply_chapter_narrators(cast, chapters, chapter_narrators(cast, chapters, cast["book_tone"], chat))
        self.assertEqual(cast["book_tone"]["point_of_view"], "third")
        self.assertNotIn("pov_key", cast["book_tone"])
        self.assertEqual(cast["chapters"]["h2"]["narrator"], "oliver")  # still read as Oliver's in its chapter
        self.assertEqual(len(chat.prompts), 1)


if __name__ == "__main__":
    unittest.main()


class TestTurnTaking(unittest.TestCase):
    """Two people take turns in paragraphs that are one bare quotation each."""

    CHARACTERS = {"ada": {"name": "Ada", "aliases": []}, "tom": {"name": "Tom", "aliases": []},
                  "kit": {"name": "Kit", "aliases": []}}

    def _fix(self, text, answers):
        paragraphs = chapter_segments(text)
        ids = [s.line_id for p in paragraphs for s in p if s.kind == "dialogue"]
        return turn_taking(paragraphs, dict(zip(ids, answers)), self.CHARACTERS)

    def test_an_exchange_shifted_by_one_line_is_put_back(self):
        text = M.join(['"Ready?" Ada asked.', '"Yes."', '"Then go."', '"Going."', '"Wait," Ada said.'])
        # the model slipped one turn: Tom's "Yes." and Ada's "Then go." swapped
        self.assertEqual(self._fix(text, ["ada", "ada", "tom", "tom", "ada"]), {2: "tom", 3: "ada"})

    def test_an_exchange_collapsed_onto_one_person_gets_the_other_back(self):
        text = M.join(['"Morning," Tom said.', 'The kettle hissed.', '"Sit," Ada said.', '"Why?"', '"Because."', '"Fine."'])
        self.assertEqual(self._fix(text, ["tom", "ada", "ada", "ada", "ada"]), {3: "tom", 5: "tom"})

    def test_left_alone_with_a_third_speaker_a_contradicting_far_end_or_a_new_beat(self):
        third = M.join(['"Ready?" Ada asked.', '"Yes."', '"Me too."', '"Go."'])
        self.assertEqual(self._fix(third, ["ada", "tom", "kit", "tom"]), {})
        far = M.join(['"Ready?" Ada asked.', '"Yes."', '"Go."', '"Now," Ada said.'])
        self.assertEqual(self._fix(far, ["ada", "ada", "ada", "ada"]), {})  # Ada twice in a row at the end
        beat = M.join(['"One."', '"Two."', 'He nodded, and they went in. "Look," Tom said.'])
        self.assertEqual(self._fix(beat, ["tom", "tom", "tom"]), {})

    def test_a_name_is_addressed_as_written(self):
        hope, man = {"name": "Hope", "aliases": []}, {"name": "Man", "aliases": []}
        self.assertTrue(addresses('"Hope, wait!"', hope))
        self.assertTrue(addresses('"Hey Hope."', hope))
        self.assertFalse(addresses('"Well, hope is all we have."', hope))
        self.assertFalse(addresses('"Oh man, that hurt."', man))

    def test_a_line_never_goes_to_the_person_it_addresses(self):
        text = M.join(['"Ready?" Ada asked.', '"Yes, Tom."', '"Go."'])
        self.assertEqual(self._fix(text, ["ada", "ada", "ada"]), {})
