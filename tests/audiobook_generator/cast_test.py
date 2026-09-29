"""Cast persistence, voice gender metadata and automatic voice suggestions."""
import os
import tempfile
import unittest

from audiobook_generator.core import cast as cast_store


def _cast(characters):
    cast = cast_store.new_cast("abc", "/x.epub", "Invented", "Nobody", "chatterbox", "Narrator.wav", [1, 2])
    for key, (lines, gender, voice) in characters.items():
        cast["characters"][key] = {"name": key.title(), "aliases": [], "gender": gender, "age": "adult",
                                   "lines": lines, "voice": voice}
    return cast


class TestPersistence(unittest.TestCase):

    def test_key_is_a_hash_of_the_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (os.path.join(tmp, name) for name in ("a.epub", "b.epub", "c.epub"))
            for path, content in ((a, b"same"), (b, b"same"), (c, b"other")):
                with open(path, "wb") as f:
                    f.write(content)
            self.assertEqual(cast_store.cast_key(a), cast_store.cast_key(b))
            self.assertNotEqual(cast_store.cast_key(a), cast_store.cast_key(c))
            self.assertEqual(len(cast_store.cast_key(a)), 16)

    def test_save_and_load_round_trip_including_per_line_attributions(self):
        cast = _cast({"ada": (3, "female", "Ada.wav")})
        cast["chapters"]["deadbeef"] = {"number": 1, "title": "One", "lines": {"1": "ada", "2": None}, "unknown": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = cast_store.cast_path("abc", tmp)
            cast_store.save_cast(path, cast)
            self.assertEqual(path, os.path.join(tmp, "abc.json"))
            loaded = cast_store.load_cast(path)
        self.assertEqual(loaded, cast)
        self.assertEqual(cast_store.chapter_lines(loaded, "deadbeef"), {1: "ada", 2: None})
        self.assertIsNone(cast_store.chapter_lines(loaded, "unknownhash"))
        self.assertEqual(cast_store.character_voice(loaded, "ada"), "Ada.wav")
        self.assertIsNone(cast_store.character_voice(loaded, None))
        self.assertIsNone(cast_store.character_voice(loaded, "nobody"))

    def test_missing_or_corrupt_files_load_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cast_store.load_cast(os.path.join(tmp, "missing.json")))
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as f:
                f.write("{not json")
            self.assertIsNone(cast_store.load_cast(bad))
            with open(bad, "w") as f:
                f.write('{"characters": []}')
            self.assertIsNone(cast_store.load_cast(bad))
        self.assertIsNone(cast_store.load_cast(None))

    def test_save_leaves_no_temp_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            cast_store.save_cast(os.path.join(tmp, "sub", "k.json"), _cast({}))
            self.assertEqual(os.listdir(os.path.join(tmp, "sub")), ["k.json"])

    def test_text_hash_matches_the_chapter_manifests_hash(self):
        import hashlib
        self.assertEqual(cast_store.text_hash("abc"), hashlib.sha1(b"abc").hexdigest())

    def test_progress(self):
        cast = _cast({})
        cast["chapters_done"] = 1
        self.assertEqual(cast_store.analysis_progress(cast), (1, 2))
        self.assertEqual(cast_store.analysis_progress(None), (0, 0))


class TestVoiceGenders(unittest.TestCase):

    def test_kokoro_gender_comes_from_the_prefix(self):
        self.assertEqual([cast_store.kokoro_voice_gender(v) for v in ("af_heart", "am_adam", "bf_emma", "bm_george", "zf_x")],
                         ["female", "male", "female", "male", "neutral"])

    def test_chatterbox_genders_are_saved_by_the_owner_and_default_to_neutral(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "voice_genders.json")
            self.assertEqual(cast_store.load_voice_genders(path), {})
            cast_store.save_voice_gender("Ada.wav", "female", path)
            cast_store.save_voice_gender("Tom.wav", "male", path)
            cast_store.save_voice_gender("Bad.wav", "robot", path)  # not a gender: forgotten, not saved
            genders = cast_store.load_voice_genders(path)
            self.assertEqual(genders, {"Ada.wav": "female", "Tom.wav": "male"})
            self.assertEqual(cast_store.voice_gender("chatterbox", "Ada.wav", genders), "female")
            self.assertEqual(cast_store.voice_gender("chatterbox", "Other.wav", genders), "neutral")
            cast_store.save_voice_gender("Ada.wav", None, path)
            self.assertEqual(cast_store.load_voice_genders(path), {"Tom.wav": "male"})

    def test_kokoro_gender_is_read_from_the_id_not_the_mapping(self):
        self.assertEqual(cast_store.voice_gender("kokoro", "am_adam", {"am_adam": "female"}), "male")


VOICES = [("Ada.wav", "female"), ("Bea.wav", "female"), ("Cal.wav", "male"), ("Dan.wav", "male"),
          ("Eve.wav", "neutral")]


class TestSuggestions(unittest.TestCase):

    def test_main_characters_get_distinct_voices_of_their_gender(self):
        cast = _cast({"anne": (50, "female", None), "bob": (40, "male", None), "cara": (30, "female", None),
                      "dave": (20, "male", None)})
        suggestions = cast_store.suggest_voices(cast, VOICES, "Narrator.wav")
        self.assertEqual(len(set(suggestions.values())), 4)
        self.assertIn(suggestions["anne"], ("Ada.wav", "Bea.wav"))
        self.assertIn(suggestions["cara"], ("Ada.wav", "Bea.wav"))
        self.assertIn(suggestions["bob"], ("Cal.wav", "Dan.wav"))
        self.assertIn(suggestions["dave"], ("Cal.wav", "Dan.wav"))

    def test_the_narrators_voice_is_never_suggested(self):
        cast = _cast({f"c{n}": (10 - n, "male", None) for n in range(6)})
        suggestions = cast_store.suggest_voices(cast, VOICES, "Cal.wav")
        self.assertNotIn("Cal.wav", suggestions.values())
        self.assertEqual(len(suggestions), 6)

    def test_unknown_gender_takes_any_unused_voice_and_neutral_voices_fit_anyone(self):
        cast = _cast({"x": (9, "unknown", None), "y": (8, "female", None), "z": (7, "female", None)})
        suggestions = cast_store.suggest_voices(cast, VOICES, "Narrator.wav")
        self.assertEqual(len(set(suggestions.values())), 3)
        third_female = _cast({"a": (9, "female", None), "b": (8, "female", None), "c": (7, "female", None)})
        third = cast_store.suggest_voices(third_female, VOICES, "Narrator.wav")
        self.assertEqual(third["c"], "Eve.wav")  # both female voices taken: the neutral one, not a male one

    def test_a_known_gender_is_served_before_an_unknown_one_that_could_take_anything(self):
        # One female and one male voice. The unknown-gender character has more lines, but must not
        # take the only female voice from the female character (who would then have to share it
        # while the male voice sits unused).
        cast = _cast({"x": (10, "unknown", None), "ada": (5, "female", None)})
        suggestions = cast_store.suggest_voices(cast, [("F.wav", "female"), ("M.wav", "male")], "Narrator.wav")
        self.assertEqual(suggestions, {"ada": "F.wav", "x": "M.wav"})

    def test_voices_are_shared_only_when_none_are_left_and_the_least_used_first(self):
        cast = _cast({f"c{n}": (20 - n, "female", None) for n in range(5)})
        suggestions = cast_store.suggest_voices(cast, [("Ada.wav", "female"), ("Bea.wav", "female")], "Narrator.wav")
        self.assertEqual([suggestions[f"c{n}"] for n in range(5)], ["Ada.wav", "Bea.wav", "Ada.wav", "Bea.wav", "Ada.wav"])

    def test_voices_the_owner_chose_are_kept_and_count_as_taken(self):
        cast = _cast({"anne": (50, "female", "Bea.wav"), "cara": (30, "female", None)})
        suggestions = cast_store.suggest_voices(cast, VOICES, "Narrator.wav")
        self.assertEqual(suggestions, {"cara": "Ada.wav"})

    def test_reanalysis_keeps_earlier_voice_picks_by_key_or_unambiguous_name(self):
        previous = _cast({"ada marsh": (5, "female", "Bea.wav"), "tom": (4, "male", "Cal.wav")})
        previous["characters"]["tom"]["aliases"] = ["Thomas"]
        fresh = _cast({"ada marsh": (6, "unknown", None), "thomas": (4, "male", None), "new": (2, "female", None)})
        carried = cast_store.carry_voice_choices(previous, fresh["characters"])
        self.assertEqual(carried, 2)
        self.assertEqual((fresh["characters"]["ada marsh"]["voice"], fresh["characters"]["ada marsh"]["gender"]),
                         ("Bea.wav", "female"))
        self.assertEqual(fresh["characters"]["thomas"]["voice"], "Cal.wav")
        self.assertIsNone(fresh["characters"]["new"].get("voice"))

    def test_an_ambiguous_name_match_is_left_for_review(self):
        previous = _cast({"ann": (5, "female", "Ada.wav"), "anne": (4, "female", "Bea.wav")})
        previous["characters"]["ann"]["aliases"] = ["Miss Gray"]
        previous["characters"]["anne"]["aliases"] = ["Miss Gray"]
        fresh = _cast({"miss gray": (3, "female", None)})
        self.assertEqual(cast_store.carry_voice_choices(previous, fresh["characters"]), 0)
        self.assertIsNone(fresh["characters"]["miss gray"].get("voice"))

    def test_no_voices_means_no_suggestions(self):
        self.assertEqual(cast_store.suggest_voices(_cast({"a": (1, "male", None)}), [], None), {})
        self.assertEqual(cast_store.suggest_voices(_cast({"a": (1, "male", None)}), [("N.wav", "male")], "N.wav"), {})


def _wants(cast, key, pitch=None, quality=None, delivery=None):
    cast["characters"][key]["profile"] = {"voice_targets": {"pitch": pitch, "quality": quality, "delivery": delivery}}


# Five female voices from deep to light; Husky.wav is the one husky voice, Lively.wav the lively one.
FEMALE = [("Deep.wav", "female"), ("Husky.wav", "female"), ("Mid.wav", "female"), ("Lively.wav", "female"),
          ("Light.wav", "female")]
TRAITS = {
    "Deep.wav": {"pitch": 0.0, "husky": 0.2, "expressive": 0.5},
    "Husky.wav": {"pitch": 0.25, "husky": 1.0, "expressive": 0.3},
    "Mid.wav": {"pitch": 0.5, "husky": 0.5, "expressive": 0.4},
    "Lively.wav": {"pitch": 0.75, "husky": 0.3, "expressive": 1.0},
    "Light.wav": {"pitch": 1.0, "husky": 0.0, "expressive": 0.0},
}


class TestMatching(unittest.TestCase):

    def test_each_character_gets_the_voice_that_fits_its_profile(self):
        cast = _cast({"boss": (50, "female", None), "kid": (40, "female", None), "vamp": (30, "female", None)})
        _wants(cast, "boss", pitch="low", quality="clear")
        _wants(cast, "kid", pitch="high", delivery="expressive")
        _wants(cast, "vamp", pitch="low", quality="husky")
        suggestions = cast_store.suggest_voices(cast, FEMALE, "Narrator.wav", TRAITS)
        self.assertEqual(suggestions, {"boss": "Deep.wav", "kid": "Lively.wav", "vamp": "Husky.wav"})

    def test_the_wanted_pitch_band_comes_before_huskiness_and_liveliness(self):
        # Measured live: "medium, clear, even" got a high voice that was clear and even.
        cast = _cast({"a": (5, "female", None)})
        _wants(cast, "a", pitch="medium", quality="clear", delivery="even")
        traits = {"Mid.wav": {"pitch": 0.5, "husky": 1.0, "expressive": 1.0},
                  "Light.wav": {"pitch": 0.75, "husky": 0.0, "expressive": 0.0}}
        suggestions = cast_store.suggest_voices(cast, [("Light.wav", "female"), ("Mid.wav", "female")], "N.wav", traits)
        self.assertEqual(suggestions, {"a": "Mid.wav"})

    def test_the_most_spoken_character_chooses_first_and_voices_stay_distinct(self):
        cast = _cast({"lead": (50, "female", None), "second": (10, "female", None)})
        _wants(cast, "lead", pitch="low")
        _wants(cast, "second", pitch="low")
        suggestions = cast_store.suggest_voices(cast, FEMALE, "Narrator.wav", TRAITS)
        # A band's middle fits best, so the extreme voice isn't everyone's first choice.
        self.assertEqual(suggestions["lead"], "Husky.wav")
        self.assertEqual(suggestions["second"], "Deep.wav")  # the other low voice, not a shared one

    def test_group_match_keeps_a_scarce_clear_voice_for_the_character_who_needs_it(self):
        cast = _cast({"lead": (50, "female", None), "second": (10, "female", None)})
        _wants(cast, "lead", pitch="high")
        _wants(cast, "second", pitch="high", quality="clear")
        voices = [("Clear.wav", "female"), ("Husky.wav", "female")]
        traits = {"Clear.wav": {"pitch": 0.8, "husky": 0.0, "expressive": 0.5},
                  "Husky.wav": {"pitch": 0.9, "husky": 1.0, "expressive": 0.5}}
        # Taking Clear for the lead's tiny pitch advantage would leave the second with a poor fit.
        self.assertEqual(cast_store.suggest_voices(cast, voices, "Narrator.wav", traits),
                         {"lead": "Husky.wav", "second": "Clear.wav"})

    def test_a_lead_no_voice_fits_does_not_stop_the_others_being_matched_together(self):
        cast = _cast({"lead": (90, "male", None), "second": (50, "female", None), "third": (10, "female", None)})
        _wants(cast, "second", pitch="high")
        _wants(cast, "third", pitch="high", quality="clear")
        voices = [("Clear.wav", "female"), ("Husky.wav", "female")]
        traits = {"Clear.wav": {"pitch": 0.8, "husky": 0.0, "expressive": 0.5},
                  "Husky.wav": {"pitch": 0.9, "husky": 1.0, "expressive": 0.5}}
        suggestions = cast_store.suggest_voices(cast, voices, "Narrator.wav", traits)
        self.assertEqual((suggestions["second"], suggestions["third"]), ("Husky.wav", "Clear.wav"))

    def test_without_targets_or_measurements_the_list_order_decides_as_before(self):
        cast = _cast({"a": (5, "female", None)})
        self.assertEqual(cast_store.suggest_voices(cast, FEMALE, "Narrator.wav", TRAITS), {"a": "Deep.wav"})
        _wants(cast, "a", pitch="high")
        self.assertEqual(cast_store.suggest_voices(cast, FEMALE, "Narrator.wav"), {"a": "Deep.wav"})

    def test_a_measured_fit_beats_an_unmeasured_voice(self):
        cast = _cast({"a": (5, "female", None)})
        _wants(cast, "a", pitch="high")
        voices = [("Unmeasured.wav", "female"), *FEMALE]
        self.assertEqual(cast_store.suggest_voices(cast, voices, "Narrator.wav", TRAITS), {"a": "Lively.wav"})

    def test_age_stands_in_for_a_missing_pitch_target(self):
        cast = _cast({"kid": (5, "female", None), "gran": (4, "female", None)})
        cast["characters"]["kid"]["age"], cast["characters"]["gran"]["age"] = "child", "elderly"
        self.assertEqual(cast_store.voice_targets(cast["characters"]["kid"])["pitch"], "high")
        suggestions = cast_store.suggest_voices(cast, FEMALE, "Narrator.wav", TRAITS)
        self.assertEqual((suggestions["kid"], suggestions["gran"]), ("Lively.wav", "Husky.wav"))

    def test_match_cost_is_zero_inside_the_wanted_band_centre(self):
        character = {"profile": {"voice_targets": {"pitch": "medium"}}}
        self.assertLess(cast_store.match_cost(character, {"pitch": 0.5, "husky": 0.5, "expressive": 0.5}), 0.01)
        self.assertGreater(cast_store.match_cost(character, {"pitch": 1.0, "husky": 0.5, "expressive": 0.5}), 0.6)
        self.assertEqual(cast_store.match_cost(character, None), cast_store.UNMEASURED_COST)
        self.assertEqual(cast_store.match_cost({}, None), 0.0)

    def test_suggest_again_clears_only_the_voices_the_owner_did_not_pick(self):
        cast = _cast({"a": (5, "female", "Deep.wav"), "b": (4, "female", "Mid.wav"), "c": (3, "female", None)})
        cast["characters"]["a"]["voice_picked"] = True
        self.assertEqual(cast_store.clear_suggested_voices(cast), 1)
        self.assertEqual([cast["characters"][k]["voice"] for k in "abc"], ["Deep.wav", None, None])

    def test_reanalysis_can_carry_only_the_voices_the_owner_picked(self):
        previous = _cast({"ann": (5, "female", "Deep.wav"), "bea": (4, "female", "Mid.wav")})
        previous["characters"]["ann"]["voice_picked"] = True
        fresh = _cast({"ann": (5, "female", None), "bea": (4, "female", None)})
        self.assertEqual(cast_store.carry_voice_choices(previous, fresh["characters"], picked_only=True), 1)
        self.assertEqual((fresh["characters"]["ann"]["voice"], fresh["characters"]["ann"]["voice_picked"]),
                         ("Deep.wav", True))
        self.assertIsNone(fresh["characters"]["bea"].get("voice"))


class TestDelivery(unittest.TestCase):

    def test_a_characters_delivery_comes_from_the_owner_then_the_profile(self):
        character = {"profile": {"voice_targets": {"delivery": "expressive"}}}
        self.assertEqual(cast_store.exaggeration_offset(character), cast_store.CHARACTER_EXAGGERATION_STEP)
        self.assertEqual(cast_store.exaggeration_offset({**character, "delivery": "even"}),
                         -cast_store.CHARACTER_EXAGGERATION_STEP)
        self.assertEqual(cast_store.exaggeration_offset({**character, "delivery": "book"}), 0.0)
        self.assertEqual(cast_store.exaggeration_offset({**character, "delivery": "auto"}),
                         cast_store.CHARACTER_EXAGGERATION_STEP)
        self.assertEqual(cast_store.exaggeration_offset({}), 0.0)

    def test_profile_deliveries_are_partly_centred_and_the_owners_are_kept(self):
        # Measured live: nearly every character of a dramatic book came back "expressive".
        cast = _cast({f"c{n}": (10, "female", None) for n in range(5)})
        cast["characters"]["calm"] = {"name": "Calm", "aliases": [], "gender": "male", "age": "adult", "lines": 10}
        for n in range(5):
            _wants(cast, f"c{n}", delivery="expressive")
        _wants(cast, "calm", delivery="even")
        offsets = cast_store.exaggeration_offsets(cast)
        self.assertEqual(offsets["c0"], 0.08)    # expressive, like most: a little above the book
        self.assertEqual(offsets["calm"], -0.12)  # the one even character stands out (capped at the step)
        cast["characters"]["c0"]["delivery"] = "expressive"  # the owner's own setting is applied as it is
        self.assertEqual(cast_store.exaggeration_offsets(cast)["c0"], 0.12)
        cast["characters"]["extra"] = {"name": "Extra", "aliases": [], "gender": "male", "age": "adult", "lines": 2}
        self.assertEqual(cast_store.exaggeration_offsets(cast)["extra"], 0.0)  # no profile: as the book

    def test_an_all_expressive_cast_keeps_a_small_boost(self):
        cast = _cast({"lead": (93, "male", None), "second": (62, "female", None),
                      "third": (61, "female", None), "fourth": (57, "female", None)})
        for key in cast["characters"]:
            _wants(cast, key, delivery="expressive")
        self.assertEqual(cast_store.exaggeration_offsets(cast),
                         {key: 0.06 for key in cast["characters"]})

    def test_the_narrators_sliders_follow_the_books_intensity_and_pace(self):
        base = (0.73, 0.5, 0.61)
        self.assertEqual(cast_store.narrator_delivery({"intensity": "dramatic", "pace": "brisk"}, base), (0.83, 0.55, 0.61))
        self.assertEqual(cast_store.narrator_delivery({"intensity": "restrained", "pace": "slow"}, base), (0.63, 0.45, 0.61))
        self.assertEqual(cast_store.narrator_delivery(None, base), base)
        self.assertEqual(cast_store.narrator_delivery({"intensity": "restrained"}, (0.3, 0.12, 0.5))[0], 0.25)

    def test_the_narrator_is_the_measured_voice_that_fits_the_books_tone(self):
        cast = _cast({"tom": (9, "male", None)})
        cast["book_tone"] = {"narrator": {"gender": "either", "pitch": "high", "quality": "clear", "delivery": None}}
        voices = [*FEMALE, ("Man.wav", "male")]
        traits = {**TRAITS, "Man.wav": {"pitch": 0.9, "husky": 0.0, "expressive": 0.5}}
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits), "Man.wav")
        cast["book_tone"]["narrator"]["gender"] = "female"
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits), "Light.wav")
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits, exclude=("Light.wav",)), "Lively.wav")
        # A first-person book: the viewpoint character's gender wins.
        cast["book_tone"].update(point_of_view="first", pov_key="tom")
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits), "Man.wav")
        self.assertIsNone(cast_store.suggest_narrator(cast, voices, {}))  # nothing measured: keep the owner's
        cast["book_tone"]["narrator"] = {"gender": "male"}
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits), "Man.wav")

    def test_gender_only_narrator_prefers_matching_gender_then_neutral(self):
        cast = _cast({})
        cast["book_tone"] = {"narrator": {"gender": "female"}}
        voices = [("Neutral.wav", "neutral"), ("Male.wav", "male"), ("Female.wav", "female")]
        traits = {voice: {"pitch": 0.5, "husky": 0.5, "expressive": 0.5} for voice, _ in voices}
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits), "Female.wav")
        self.assertEqual(cast_store.suggest_narrator(cast, voices[:1], traits), "Neutral.wav")

    def test_gender_only_narrator_keeps_the_owners_voice_when_it_has_that_gender(self):
        cast = _cast({})
        cast["book_tone"] = {"narrator": {"gender": "female"}}
        voices = [("Male.wav", "male"), ("Ada.wav", "female"), ("Elena.wav", "female")]
        traits = {voice: {"pitch": 0.5, "husky": 0.5, "expressive": 0.5} for voice, _ in voices}
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits, current="Elena.wav"), "Elena.wav")
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits, current="Male.wav"), "Ada.wav")
        self.assertEqual(cast_store.suggest_narrator(cast, voices, traits, exclude=("Elena.wav",),
                                                     current="Elena.wav"), "Ada.wav")

    def test_narrator_without_gender_or_voice_targets_has_no_suggestion(self):
        cast = _cast({})
        cast["book_tone"] = {"narrator": {"gender": "either"}}
        self.assertIsNone(cast_store.suggest_narrator(cast, FEMALE, TRAITS))


class TestFirstPerson(unittest.TestCase):
    """A first-person book: the "I" character's lines are read in the narrator's voice."""

    def _cast(self):
        cast = _cast({"me": (50, "female", None), "friend": (30, "female", None)})
        cast["book_tone"] = {"point_of_view": "first", "pov_key": "me",
                             "narrator": {"gender": "either", "pitch": "high", "quality": None, "delivery": None}}
        return cast

    def test_the_narrating_character_is_the_pov_unless_the_owner_gave_them_a_voice(self):
        cast = self._cast()
        self.assertEqual((cast_store.pov_character(cast), cast_store.narrating_character(cast)), ("me", "me"))
        cast["characters"]["me"].update(voice="Deep.wav", voice_picked=True)
        self.assertEqual((cast_store.pov_character(cast), cast_store.narrating_character(cast)), ("me", None))
        cast["book_tone"]["point_of_view"] = "third"
        self.assertIsNone(cast_store.pov_character(cast))

    def test_a_chapter_can_have_its_own_narrator_or_none(self):
        cast = self._cast()
        cast["chapters"] = {"old": {"number": 1, "lines": {}},
                            "a": {"number": 2, "lines": {}, "point_of_view": "first", "narrator": "friend"},
                            "b": {"number": 3, "lines": {}, "point_of_view": "third", "narrator": None}}
        self.assertEqual(cast_store.chapter_narrator(cast, "old"), "me")  # analysed before chapters had one
        self.assertEqual(cast_store.chapter_narrator(cast, "a"), "friend")
        self.assertIsNone(cast_store.chapter_narrator(cast, "b"))
        cast["characters"]["friend"]["voice_picked"] = True  # the owner's own voice for her wins
        self.assertIsNone(cast_store.chapter_narrator(cast, "a"))

    def test_another_characters_first_person_story_is_narrated_in_their_own_voice(self):
        cast = self._cast()
        cast["characters"]["friend"]["voice"] = "Mid.wav"
        cast["chapters"] = {"mine": {"number": 1, "narrator": "me"}, "hers": {"number": 2, "narrator": "friend"},
                            "third": {"number": 3, "narrator": None}, "old": {"number": 4}}
        self.assertIsNone(cast_store.chapter_narrator_voice(cast, "mine"))  # the book's own "I": its narrator voice
        self.assertEqual(cast_store.chapter_narrator_voice(cast, "hers"), "Mid.wav")
        self.assertIsNone(cast_store.chapter_narrator_voice(cast, "third"))
        self.assertIsNone(cast_store.chapter_narrator_voice(cast, "old"))
        cast["characters"]["friend"]["voice"] = None  # no voice yet: the book's narrator
        self.assertIsNone(cast_store.chapter_narrator_voice(cast, "hers"))

    def test_the_narrator_gets_no_suggested_voice_and_gives_one_back(self):
        cast = self._cast()
        self.assertEqual(cast_store.suggest_voices(cast, FEMALE, "Narrator.wav"), {"friend": "Deep.wav"})
        cast["characters"]["me"]["voice"] = "Mid.wav"  # suggested before the book was known to be first person
        self.assertTrue(cast_store.release_narrating_voice(cast))
        self.assertIsNone(cast["characters"]["me"]["voice"])
        self.assertFalse(cast_store.release_narrating_voice(cast))

    def test_the_narrator_voice_follows_the_narrating_characters_own_profile(self):
        cast = self._cast()
        _wants(cast, "me", pitch="low", quality="husky")  # the tone asked for a high voice
        self.assertEqual(cast_store.suggest_narrator(cast, FEMALE, TRAITS), "Husky.wav")

    def test_the_narrating_character_has_no_delivery_offset_and_isnt_averaged(self):
        cast = self._cast()
        _wants(cast, "me", delivery="expressive")
        _wants(cast, "friend", delivery="even")
        cast["characters"]["third"] = {"name": "Third", "aliases": [], "gender": "male", "age": "adult", "lines": 30,
                                       "profile": {"voice_targets": {"delivery": "expressive"}}}
        offsets = cast_store.exaggeration_offsets(cast)
        self.assertEqual((offsets["me"], offsets["friend"], offsets["third"]), (0.0, -0.12, 0.12))


class TestEngineCheck(unittest.TestCase):

    def test_voices_must_look_like_the_engines(self):
        cast = _cast({"a": (1, "male", "Cal.wav"), "b": (1, "female", "af_heart"), "c": (1, "male", None)})
        self.assertEqual(cast_store.voices_belong_to_engine(cast, "chatterbox"), ["af_heart"])
        self.assertEqual(cast_store.voices_belong_to_engine(cast, "kokoro"), ["Cal.wav"])
        self.assertEqual(cast_store.voices_belong_to_engine(cast, "chatterbox", known_voices=["Cal.wav"]), ["af_heart"])
        self.assertEqual(cast_store.voices_belong_to_engine(cast, "chatterbox", known_voices=["Other.wav"]),
                         ["Cal.wav", "af_heart"])

    def test_name_normalisation(self):
        self.assertEqual(cast_store.normalize_name("Mr. Thomas  Baker"), "thomas baker")
        self.assertEqual(cast_store.normalize_name("MRS MARSH"), "marsh")
        self.assertEqual(cast_store.normalize_name("O'Brien"), "o'brien")
        self.assertEqual(cast_store.normalize_name("  "), "")
