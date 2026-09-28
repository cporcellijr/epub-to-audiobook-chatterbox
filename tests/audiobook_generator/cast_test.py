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

    def test_voices_are_shared_only_when_none_are_left_and_the_least_used_first(self):
        cast = _cast({f"c{n}": (20 - n, "female", None) for n in range(5)})
        suggestions = cast_store.suggest_voices(cast, [("Ada.wav", "female"), ("Bea.wav", "female")], "Narrator.wav")
        self.assertEqual([suggestions[f"c{n}"] for n in range(5)], ["Ada.wav", "Bea.wav", "Ada.wav", "Bea.wav", "Ada.wav"])

    def test_voices_the_owner_chose_are_kept_and_count_as_taken(self):
        cast = _cast({"anne": (50, "female", "Bea.wav"), "cara": (30, "female", None)})
        suggestions = cast_store.suggest_voices(cast, VOICES, "Narrator.wav")
        self.assertEqual(suggestions, {"cara": "Ada.wav"})

    def test_no_voices_means_no_suggestions(self):
        self.assertEqual(cast_store.suggest_voices(_cast({"a": (1, "male", None)}), [], None), {})
        self.assertEqual(cast_store.suggest_voices(_cast({"a": (1, "male", None)}), [("N.wav", "male")], "N.wav"), {})


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
